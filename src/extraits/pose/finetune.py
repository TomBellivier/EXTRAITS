"""Fine-tuning: adapt the pose models to a small set of new annotated images.

Inputs (`finetune.*` of the config; `extraits finetune` sets them):
  - images_dir + annotations: Label Studio JSON exports or CSV files in the columns of
    annotation_data.csv (image_name, group, width, height, <kp>_x, <kp>_y, <kp>_v);
    without annotations, the layout <images_dir>/<order>/{*.json, images/};
  - the base ensemble: base_models (default <models>/pose/), every member or one.

Two modes:
  produce   (`cmd_finetune_produce`) every chosen member is fine-tuned on ALL the new
            images (a val share only picks the checkpoint) for the number of epochs it
            was trained with; the members form a new ensemble written to
            <output>/pose/ with their model cards and ensemble.json, usable with
            `extraits process --models-dir <output>` (scale_bar/ and
            measurement_validity/ are copied next to it). The base models and the
            models of the workspace are never written.
  evaluate  (`cmd_finetune_evaluate`) the images are split into `n_folds` folds; per
            fold, ONE base member is evaluated as it is ("before"), then fine-tuned on
            the train of the fold ("after"); both on the same untouched test images.
            The comparison goes to results/pose/finetune/<name>/: detection metrics on
            every test image, point metrics only on the images detected both before AND
            after (see `write_comparison`). Nothing is exported.

Everything this module writes (canonical annotations, splits, runs) lives under its own
folders, data/pose/finetune/<name>/ and runs/finetune/<name>/: the main datasets,
runs and report are never touched.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from extraits.pose.contracts import DATASETS, ContractError
from extraits.pose.data.splits import build_splits, write_splits
from extraits.pose.evaluation.evaluator import evaluate_predictions, evaluate_run
from extraits.pose.paths import ProjectPaths
from extraits.pose.registry import ADAPTERS
from extraits.pose.utils.io import read_json, read_parquet
from extraits.pose.utils.logging import get_logger

log = get_logger("finetune")

CARD_NAME = "model_card.json"
WEIGHTS_NAME = "best.pt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
# Metrics printed in the console summary (all of them are in the CSV files).
SUMMARY_METRICS = ("detection_rate", "oks_ap", "pck@0.25_thorax_width", "nme",
                   "measurement_mape_median")


# --- base models -----------------------------------------------------------------
@dataclass(frozen=True)
class Member:
    """One model of an ensemble folder: <folder>/best.pt + model_card.json."""

    folder: Path
    card: dict[str, Any]

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def weights(self) -> Path:
        return self.folder / WEIGHTS_NAME


def ensemble_members(pose_dir: Path) -> list[Member]:
    """Every member of an ensemble folder (sub-folders with weights and a model card)."""
    members = [Member(card_path.parent, read_json(card_path))
               for card_path in sorted(Path(pose_dir).glob(f"*/{CARD_NAME}"))
               if (card_path.parent / WEIGHTS_NAME).is_file()]
    if not members:
        raise FileNotFoundError(
            f"No model (<member>/{WEIGHTS_NAME} + {CARD_NAME}) in {pose_dir}. Download the "
            "official models with `extraits download-models`, or set --base-models.")
    return members


def select_members(pose_dir: Path, base: str | None, single: bool) -> list[Member]:
    """The base members to fine-tune.

    `base`: None or 'all' (every member; with `single`, the fold-0 one), a member folder
    name, or a variant_hash (its members; with `single`, its fold-0 one, or its only
    member). `single` is the evaluate mode, which compares ONE model.
    """
    members = ensemble_members(pose_dir)
    if base not in (None, "", "all"):
        chosen = [m for m in members if m.name == base]
        if not chosen:
            chosen = [m for m in members if str(m.card.get("variant_hash")) == str(base)]
        if not chosen:
            raise ContractError(
                f"No member named or with variant_hash '{base}' in {pose_dir}. "
                f"Members: {[m.name for m in members]}; variant_hash: "
                f"{sorted({str(m.card.get('variant_hash')) for m in members})}")
        members = chosen
    if single and len(members) > 1:
        fold0 = [m for m in members if int(m.card.get("fold", -1)) == 0]
        if len(fold0) != 1:
            raise ContractError(
                f"{len(members)} candidate base models and not exactly one fold 0: choose "
                "one with base=<member folder name>.")
        members = fold0
    return members


def find_base_model(retained_pose_dir: Path, variant: str) -> tuple[Path, dict[str, Any]]:
    """Weights + model card of the fold-0 member of the ensemble `variant` (or its only
    member)."""
    member = select_members(retained_pose_dir, variant, single=True)[0]
    return member.weights, member.card


def base_epochs(weights: Path) -> int:
    """Number of epochs the base model was trained with (Ultralytics train_args)."""
    from ultralytics import YOLO

    train_args = (getattr(YOLO(str(weights)), "ckpt", None) or {}).get("train_args") or {}
    if "epochs" not in train_args:
        raise ContractError(f"No 'epochs' in the train_args of {weights}: pass train.epochs=N.")
    return int(train_args["epochs"])


# --- data ------------------------------------------------------------------------
def _labelstudio_rows(json_path: Path, group: str | None) -> list[dict[str, Any]]:
    """Rows of a Label Studio export (the converter of `extraits annotate to-csv`)."""
    from extraits.annotation.labelstudio import rows_from_export

    return list(rows_from_export(json_path, group))


@dataclass
class FinetuneData:
    """The new annotated images, in the columns of annotation_data.csv.

    `table.image_name` is relative to `images_root`.
    """

    table: pd.DataFrame
    images_root: Path
    sources: list[str]

    @property
    def orders(self) -> list[str]:
        return sorted(self.table["group"].astype(str).str.lower().unique())


def _image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(file.name, []).append(file.relative_to(root))
    return index


def _locate(name: str, root: Path, index: dict[str, list[Path]]) -> str | None:
    """Path of an image relative to `root`: as written, else found by its file name."""
    name = str(name).replace("\\", "/")
    if (root / name).is_file():
        return name
    found = index.get(Path(name).name, [])
    if len(found) > 1:
        raise ContractError(f"Image name '{Path(name).name}' found several times under {root}: "
                            f"{[str(p) for p in found[:3]]}. Give image_name as a relative path.")
    return found[0].as_posix() if found else None


def load_finetune_data(images_dir: Path, annotations: list[str | Path] | None = None,
                       group: str | None = None) -> FinetuneData:
    """Read the annotations of the new images into one table. No side effect.

    `annotations`: Label Studio JSON exports (.json) and/or CSV files in the columns of
    annotation_data.csv (.csv). None: the layout <images_dir>/<order>/{*.json, images/},
    where the order is the folder name. `group` gives the insect order of the rows that
    carry none.
    """
    images_dir = Path(images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Fine-tuning image folder missing: {images_dir}")
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    if not annotations:
        orders = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
        if not orders or any(o not in DATASETS for o in orders):
            raise ContractError(
                f"{images_dir}: without --annotations, it must hold one sub-folder per insect "
                f"order among {list(DATASETS)}, each with the Label Studio JSON export(s) and "
                f"an images/ folder. Found: {orders}")
        for order in orders:
            exports = sorted((images_dir / order).glob("*.json"))
            if not exports:
                raise FileNotFoundError(f"No Label Studio JSON export in {images_dir / order}")
            for export in exports:
                rows = pd.DataFrame(_labelstudio_rows(export, order))
                rows["image_name"] = [f"{order}/images/{Path(str(n)).name}"
                                      for n in rows["image_name"]]
                frames.append(rows)
                sources.append(str(export))
    else:
        for item in annotations:
            path = Path(item).expanduser().resolve()
            if path.suffix.lower() == ".json":
                folder_group = path.parent.name if path.parent.name in DATASETS else None
                frames.append(pd.DataFrame(_labelstudio_rows(path, group or folder_group)))
            elif path.suffix.lower() == ".csv":
                frames.append(pd.read_csv(path, low_memory=False))
            else:
                raise ContractError(f"Unknown annotation file type: {path} (.json or .csv).")
            sources.append(str(path))

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if table.empty or "image_name" not in table.columns:
        raise ContractError(f"No annotated image read from {sources or images_dir}.")
    if "group" not in table.columns:
        table["group"] = group
    if group is not None:
        table["group"] = table["group"].where(table["group"].notna() & (table["group"] != ""),
                                              group)
    table["group"] = table["group"].astype(str).str.lower()
    unknown = sorted(set(table["group"]) - set(DATASETS))
    if unknown:
        raise ContractError(f"Unknown insect order(s) {unknown}: expected {list(DATASETS)}. "
                            "Fill the `group` column, or pass --group.")

    index = _image_index(images_dir)
    located = [_locate(n, images_dir, index) for n in table["image_name"]]
    missing = [n for n, loc in zip(table["image_name"], located) if loc is None]
    if missing:
        log.warning("%d annotated image(s) not found under %s, left out: %s", len(missing),
                    images_dir, ", ".join(map(str, missing[:5])))
    table["image_name"] = located
    table = table[table["image_name"].notna()].reset_index(drop=True)
    if table.empty:
        raise ContractError(f"None of the annotated images is under {images_dir}.")

    for col in ("width", "height"):
        if col not in table.columns:
            table[col] = np.nan
        table[col] = pd.to_numeric(table[col], errors="coerce")
    unknown_size = table["width"].isna() | table["height"].isna()
    if unknown_size.any():
        from PIL import Image

        for i in table.index[unknown_size]:
            with Image.open(images_dir / table.at[i, "image_name"]) as image:
                table.at[i, "width"], table.at[i, "height"] = image.size
    return FinetuneData(table=table, images_root=images_dir, sources=sources)


def prepare_finetune(cfg: DictConfig, paths: ProjectPaths, data: FinetuneData) -> list[str]:
    """New images -> canonical annotations of each order present (contract 1).

    `paths.images` must be `data.images_root` (image_path is relative to it).
    Side effect: <processed>/finetune_annotations.csv and <processed>/<order>/annotations.parquet.
    """
    paths.processed.mkdir(parents=True, exist_ok=True)
    csv_path = paths.processed / "finetune_annotations.csv"
    data.table.to_csv(csv_path, index=False)
    for order in data.orders:
        options = {
            "csv_path": str(csv_path),
            "group_column": "group",
            "images_subdir": ".",               # image_name is relative to paths.images
            "margin": float(cfg.data.adapter_options.get("margin", 10)),
            "keypoint_schema": str(cfg.data.keypoint_schema),
            "skip_missing_images": True,
            "project_root": str(paths.root),
            "configs_dir": str(paths.configs),
            "data_dir": str(paths.images),
        }
        adapter = ADAPTERS.get("annotation_csv")(dataset=order, source_dir=data.images_root,
                                                 options=options)
        log.info("[%s] %d annotated image(s) -> %s", order,
                 int((data.table["group"] == order).sum()), adapter.run(paths))
    return data.orders


def _finetune_data_from_cfg(cfg: DictConfig, main_paths: ProjectPaths) -> FinetuneData:
    images_dir = (main_paths.root / str(cfg.finetune.images_dir)).resolve()
    annotations = cfg.finetune.get("annotations")
    if annotations is not None and not isinstance(annotations, (list, tuple)):
        annotations = list(annotations) if OmegaConf.is_list(annotations) else [annotations]
    group = cfg.finetune.get("group")
    return load_finetune_data(images_dir, [str(a) for a in annotations or []] or None,
                              str(group) if group else None)


def _work_cfg(cfg: DictConfig, main_paths: ProjectPaths, name: str,
              data: FinetuneData) -> tuple[DictConfig, ProjectPaths]:
    """Config whose data, splits, runs and images are the fine-tuning ones."""
    cfg = cfg.copy()
    work = main_paths.data / "finetune" / name
    for key, value in {"paths.processed": work / "processed", "paths.splits": work / "splits",
                       "paths.interim": work / "interim",
                       "paths.runs": main_paths.runs / "finetune" / name,
                       "paths.images": data.images_root}.items():
        OmegaConf.update(cfg, key, str(value), force_add=True)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_writable_dirs()
    return cfg, paths


def _base_dir(cfg: DictConfig, main_paths: ProjectPaths) -> Path:
    base_models = cfg.finetune.get("base_models")
    return (Path(str(base_models)).expanduser().resolve() if base_models
            else main_paths.retained / "pose")


# --- comparison ------------------------------------------------------------------
# Threshold-free metrics that measure the DETECTION itself: they keep every test image
# of the fold, as written by the evaluator. All the others (point placement) are
# recomputed only on the images that carry a skeleton both before AND after
# fine-tuning, otherwise the images the base model misses count as point failures and
# bias the "before" downwards. `kpt_coverage` is left out: it depends on the
# acceptance threshold of the evaluator, replaced here by `detection_rate`.
DETECTION_METRICS = ("det_ap@0.5", "det_ap@[.5:.95]", "oks_ar")
SKIPPED_POINT_METRICS = ("kpt_coverage", "pck_normalizer_fallback_rate")


def score_threshold(cfg: DictConfig) -> float:
    """finetune.score_threshold, else the detection threshold of `extraits process`."""
    value = cfg.finetune.get("score_threshold")
    if value is None:
        from extraits.process.config import POSE_CONF_THRESHOLD

        value = POSE_CONF_THRESHOLD
    return float(value)


def analysis_eval_cfg(cfg: DictConfig) -> DictConfig:
    """Evaluation config of the comparison: "detected" = carries a skeleton.

    Like the pipeline, a prediction is kept when its box score reaches
    `finetune.score_threshold`, and its points are then evaluated whatever their
    distance to the GT (match_oks_threshold = 0): a wrong skeleton lowers the PCK and
    raises the NME instead of disappearing from the point metrics.
    """
    eval_cfg = cfg.eval.copy()
    OmegaConf.update(eval_cfg, "score_threshold_pointwise", score_threshold(cfg))
    OmegaConf.update(eval_cfg, "match_oks_threshold", 0.0)
    return eval_cfg


def _test_metrics(paths: ProjectPaths, run_id: str) -> pd.DataFrame:
    metrics = read_parquet(paths.metrics(run_id))
    return metrics[(metrics["split"] == "test") & (metrics["scope"] == "overall")]


def _test_predictions(paths: ProjectPaths, run_id: str, fold: int) -> pd.DataFrame:
    file = paths.run_dir(run_id) / "predictions" / f"test_fold{fold}.parquet"
    return read_parquet(file, artifact="predictions", validate=True)


def _prepared(pred: pd.DataFrame, gt: pd.DataFrame, eval_cfg: Any
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GT and predictions filtered exactly as `evaluate_predictions` does."""
    pred = pred[pred["bbox_score"] >= float(eval_cfg.score_threshold_curves)]
    return gt.reset_index(drop=True), pred.reset_index(drop=True)


def detected_images(pred: pd.DataFrame, gt: pd.DataFrame, schemas: dict[str, Any],
                    eval_cfg: Any) -> set[str]:
    """Images where at least one GT insect is matched by a prediction.

    Same matching as the point metrics (score_threshold_pointwise + match_oks_threshold):
    an image counted here is exactly an image whose points are evaluated. With
    `analysis_eval_cfg`, it is an image that carries a skeleton in the pipeline.
    """
    from extraits.pose.evaluation.bundle import EvalBundle
    from extraits.pose.evaluation.matching import build_pairs

    if pred.empty or gt.empty:
        return set()
    gt, pred = _prepared(pred, gt, eval_cfg)
    pairs = build_pairs(gt, pred, schemas, area_source=str(eval_cfg.oks.area_source))
    bundle = EvalBundle(gt=gt, pred=pred, pairs=pairs, schemas=schemas, cfg=eval_cfg)
    return set(bundle.matched_instances()["image_id"])


def _metrics_on(pred: pd.DataFrame, gt: pd.DataFrame, images: set[str],
                schemas: dict[str, Any], eval_cfg: Any) -> pd.Series:
    """Overall metrics of `pred` restricted to `images` (empty if no image)."""
    if not images:
        return pd.Series(dtype=float)
    metrics = evaluate_predictions(pred[pred["image_id"].isin(images)],
                                   gt[gt["image_id"].isin(images)], schemas, eval_cfg)
    return metrics[metrics["scope"] == "overall"].set_index("metric")["value"]


def _summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    per_fold["delta"] = per_fold["after"] - per_fold["before"]
    summary = (per_fold.groupby("metric")[["before", "after", "delta"]]
               .agg(["mean", "std"]).reset_index())
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    return summary


def write_comparison(paths: ProjectPaths, runs: dict[int, dict[str, str]],
                     test_images: dict[int, list[str]], out_dir: Path,
                     annotations: pd.DataFrame, schemas: dict[str, Any],
                     eval_cfg: Any) -> pd.DataFrame:
    """Per-fold and mean before/after table, from the saved predictions only.

    `eval_cfg` is the one of `analysis_eval_cfg` (a "detected" image carries a skeleton).
    - detection metrics (DETECTION_METRICS + n/rate of detected images): every test image;
    - every other metric: only the images detected before AND after (`evaluated_on`).

    Side effects in `out_dir`: comparison.csv / comparison_per_fold.csv (corrected) and
    comparison_all_images*.csv (every metric on every test image, for traceability).
    """
    rows, raw_rows = [], []
    for fold, pair in runs.items():
        fold_images = set(test_images[fold])
        gt = annotations[annotations["image_id"].isin(fold_images)]
        preds = {when: _test_predictions(paths, pair[when], fold) for when in ("before", "after")}
        detected = {when: detected_images(preds[when], gt, schemas, eval_cfg)
                    for when in preds}
        common = detected["before"] & detected["after"]

        full = {when: _test_metrics(paths, pair[when]).set_index("metric")["value"]
                for when in preds}
        points = {when: _metrics_on(preds[when], gt, common, schemas, eval_cfg)
                  for when in preds}
        base = {"fold": fold, "test_images": " ".join(test_images[fold])}

        for metric in sorted(set(full["before"].index) | set(full["after"].index)):
            raw_rows.append({**base, "metric": metric, "before": full["before"].get(metric),
                             "after": full["after"].get(metric)})

        n = len(fold_images)
        for metric, b, a in [
            ("n_images_detected", len(detected["before"]), len(detected["after"])),
            ("detection_rate", len(detected["before"]) / n if n else float("nan"),
             len(detected["after"]) / n if n else float("nan")),
        ]:
            rows.append({**base, "evaluated_on": "all_test_images", "n_images": n,
                         "metric": metric, "before": b, "after": a})
        for metric in DETECTION_METRICS:
            if metric in full["before"].index or metric in full["after"].index:
                rows.append({**base, "evaluated_on": "all_test_images", "n_images": n,
                             "metric": metric, "before": full["before"].get(metric),
                             "after": full["after"].get(metric)})
        for metric in sorted(set(points["before"].index) | set(points["after"].index)):
            if metric in DETECTION_METRICS or metric in SKIPPED_POINT_METRICS:
                continue
            rows.append({**base, "evaluated_on": "detected_before_and_after",
                         "n_images": len(common), "metric": metric,
                         "before": points["before"].get(metric),
                         "after": points["after"].get(metric)})
        log.info("Fold %d: %d test image(s), detected before %d / after %d / both %d.",
                 fold, n, len(detected["before"]), len(detected["after"]), len(common))

    per_fold = pd.DataFrame(rows)
    summary = _summarise(per_fold)
    raw = pd.DataFrame(raw_rows)
    raw_summary = _summarise(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out_dir / "comparison_per_fold.csv", index=False)
    summary.to_csv(out_dir / "comparison.csv", index=False)
    raw.to_csv(out_dir / "comparison_all_images_per_fold.csv", index=False)
    raw_summary.to_csv(out_dir / "comparison_all_images.csv", index=False)
    return summary


# --- figures ---------------------------------------------------------------------
_GT_COLOR = (255, 255, 255)
_SKELETON_COLOR = (40, 40, 40)


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _conf_rgb(value: float) -> tuple[int, int, int]:
    """Confidence -> RdYlGn colour, as `extraits analyze --print` (red 0 -> green 1)."""
    import matplotlib

    r, g, b, _ = matplotlib.colormaps["RdYlGn"](min(max(float(value), 0.0), 1.0))
    return int(r * 255), int(g * 255), int(b * 255)


def _panel(image_path: Path, gt_rows: pd.DataFrame, pred: pd.DataFrame, title: str,
           schemas: dict[str, Any], eval_cfg: Any, threshold: float) -> Any:
    """One annotated copy of the image: GT in white, skeleton coloured by confidence.

    The skeleton is drawn only if the box score reaches `threshold`, like the pipeline.
    """
    from PIL import Image, ImageDraw

    from extraits.pose.evaluation.matching import build_pairs

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    radius = max(2.0, 0.005 * max(width, height))
    line = max(1, int(radius / 2))
    schema = schemas[str(gt_rows["keypoint_schema"].iloc[0])]

    for gt in gt_rows.itertuples(index=False):
        kpts = np.asarray(gt.kpts_xy, dtype=float).reshape(-1, 2)
        vis = np.asarray(gt.kpts_vis, dtype=int) > 0
        small = radius * 0.6
        for (x, y) in kpts[vis]:
            draw.ellipse([x - small, y - small, x + small, y + small],
                         outline=_GT_COLOR, width=max(1, line // 2))

    lines = [title]
    if pred.empty:
        lines.append("no prediction")
    else:
        top = pred.sort_values("bbox_score", ascending=False).head(1).reset_index(drop=True)
        score = float(top.loc[0, "bbox_score"])
        pairs = build_pairs(gt_rows.reset_index(drop=True), top, schemas,
                            area_source=str(eval_cfg.oks.area_source))
        oks = float(pairs[0].oks.max()) if pairs and pairs[0].oks.size else float("nan")
        if score >= threshold:
            kpts = np.asarray(top.loc[0, "kpts_xy"], dtype=float).reshape(-1, 2)
            conf = np.asarray(top.loc[0, "kpts_score"], dtype=float)
            for a, b in schema.skeleton:
                if a < len(kpts) and b < len(kpts):
                    draw.line([tuple(kpts[a]), tuple(kpts[b])], fill=_SKELETON_COLOR,
                              width=line)
            for (x, y), c in zip(kpts, conf):
                draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                             fill=_conf_rgb(c), outline=(0, 0, 0))
            lines.append(f"box score {score:.2f} | OKS {oks:.2f}")
        else:
            lines.append(f"box score {score:.2f} < {threshold:g}: no skeleton")

    size = max(12, int(0.025 * max(width, height)))
    font = _font(size)
    step = int(size * 1.3)
    box_w = max(draw.textbbox((0, 0), t, font=font)[2] for t in lines) + size
    draw.rectangle([0, 0, box_w, step * len(lines) + size // 2], fill=(0, 0, 0))
    for i, text in enumerate(lines):
        draw.text((size // 2, size // 4 + i * step), text, fill=(255, 255, 255), font=font)
    return image


def export_figures(paths: ProjectPaths, runs: dict[int, dict[str, str]],
                   test_images: dict[int, list[str]], annotations: pd.DataFrame,
                   schemas: dict[str, Any], eval_cfg: Any, out_dir: Path) -> list[Path]:
    """One `before | after` image per test image, in the style of `extraits analyze --print`.

    White circles: GT points. Coloured points: predicted skeleton, red (confidence 0) to
    green (1), drawn only when the box score reaches the threshold of the pipeline.
    Side effect: writes out_dir/fold<k>__<image>.png.
    """
    from PIL import Image

    threshold = float(eval_cfg.score_threshold_pointwise)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fold, pair in runs.items():
        preds = {when: _test_predictions(paths, pair[when], fold) for when in ("before", "after")}
        gt = annotations[annotations["image_id"].isin(set(test_images[fold]))]
        for image_id, gt_rows in gt.groupby("image_id"):
            image_path = paths.images / str(gt_rows["image_path"].iloc[0])
            if not image_path.is_file():
                log.warning("Image missing, figure skipped: %s", image_path)
                continue
            panels = [
                _panel(image_path, gt_rows, preds[when][preds[when]["image_id"] == image_id],
                       f"{when} fine-tuning (fold {fold})", schemas, eval_cfg, threshold)
                for when in ("before", "after")
            ]
            gap = max(4, panels[0].width // 100)
            canvas = Image.new("RGB", (panels[0].width * 2 + gap, panels[0].height),
                               (255, 255, 255))
            canvas.paste(panels[0], (0, 0))
            canvas.paste(panels[1], (panels[0].width + gap, 0))
            name = str(image_id).replace("/", "_").replace("\\", "_")
            path = out_dir / f"fold{fold}__{Path(name).stem}.png"
            canvas.save(path)
            written.append(path)
    log.info("%d before | after figure(s) written to %s", len(written), out_dir)
    return written


# --- verbs -----------------------------------------------------------------------
def cmd_finetune(cfg: DictConfig) -> Path:
    """`finetune` verb: produce (default) or evaluate (finetune.evaluate=true)."""
    if bool(cfg.finetune.get("evaluate", False)):
        return cmd_finetune_evaluate(cfg)
    return cmd_finetune_produce(cfg)


def _same_folder(a: Path, b: Path) -> bool:
    return a.expanduser().resolve() == b.expanduser().resolve()


def cmd_finetune_produce(cfg: DictConfig) -> Path:
    """Fine-tune every chosen base member on ALL the new images; export the new ensemble.

    Returns <output>/pose/. Side effects: data/pose/finetune/<name>/,
    runs/finetune/<name>/, <output>/ (pose/ + copies of scale_bar/, measurement_validity/).
    """
    from extraits.pose.pipeline import cmd_fit_full
    from extraits.pose.retain import ENSEMBLE_NAME, clear_retained, write_ensemble_card

    if not cfg.finetune.get("output"):
        raise SystemExit("Fine-tuning needs an output models folder: --output <folder> "
                         "(finetune.output), or --evaluate for the before/after comparison.")
    main_paths = ProjectPaths.from_config(cfg)
    output = Path(str(cfg.finetune.output)).expanduser().resolve()
    base_dir = _base_dir(cfg, main_paths)
    for protected, what in ((base_dir.parent, "the base models"),
                            (main_paths.retained, "the models of the workspace")):
        if _same_folder(output, protected):
            raise SystemExit(f"--output {output} is {what}: fine-tuning never overwrites them. "
                             "Choose another folder.")
    out_pose = output / "pose"
    if out_pose.is_dir() and any(not p.name.startswith(".") for p in out_pose.iterdir()):
        if not bool(cfg.finetune.get("overwrite", False)):
            raise SystemExit(f"{out_pose} is not empty: pass --overwrite to replace it.")

    members = select_members(base_dir, cfg.finetune.get("base"), single=False)
    data = _finetune_data_from_cfg(cfg, main_paths)
    name = str(cfg.finetune.get("name") or output.name)
    work_cfg, paths = _work_cfg(cfg, main_paths, name, data)
    OmegaConf.update(work_cfg, "paths.retained", str(output), force_add=True)
    paths = ProjectPaths.from_config(work_cfg)
    orders = prepare_finetune(work_cfg, paths, data)
    log.info("Fine-tuning %d base model(s) of %s on %d image(s) (%s) -> %s",
             len(members), base_dir, len(data.table), ", ".join(orders), out_pose)

    for key, value in {"data.scope": "finetune", "data.datasets": orders,
                       "retain.enabled": True, "folds": None}.items():
        OmegaConf.update(work_cfg, key, value, force_add=True)
    clear_retained(paths)
    trained: list[str] = []
    for member in members:
        member_cfg = work_cfg.copy()
        epochs = (int(cfg.finetune.epochs) if cfg.finetune.get("epochs")
                  else base_epochs(member.weights))
        OmegaConf.update(member_cfg, "approach.weights", str(member.weights))
        OmegaConf.update(member_cfg, "train.epochs", epochs)
        OmegaConf.update(member_cfg, "tag", f"ft-{name}-from-{member.card.get('fold', 'x')}",
                         force_add=True)
        card = {"finetune": {
            "base_member": member.name,
            "base_models": str(base_dir),
            "base_variant_hash": member.card.get("variant_hash"),
            "base_primary_metric": member.card.get("primary_metric"),
            "base_primary_value": member.card.get("primary_value"),
            "n_images": int(len(data.table)),
            "orders": orders,
            "annotations": data.sources,
            "epochs": epochs,
        }}
        log.info("Base member %s | %d epochs", member.name, epochs)
        ctx = cmd_fit_full(member_cfg, extra={"role_in_protocol": "finetune",
                                              "base_model": str(member.weights)}, card=card)
        trained.append(ctx.run_id)

    write_ensemble_card(paths, {"source": "finetune", "name": name,
                                "base_models": str(base_dir),
                                "base_members": [m.name for m in members],
                                "n_images": int(len(data.table)), "orders": orders,
                                "runs": trained})
    if bool(cfg.finetune.get("copy_other_models", True)):
        for sub in ("scale_bar", "measurement_validity"):
            source, target = base_dir.parent / sub, output / sub
            if source.is_dir() and not target.exists():
                shutil.copytree(source, target)
                log.info("Copied %s -> %s", source, target)
    log.info("Fine-tuned ensemble: %d model(s) in %s (%s). Use it with:\n"
             "    extraits process --models-dir %s ...", len(trained), out_pose,
             ENSEMBLE_NAME, output)
    return out_pose


def cmd_finetune_evaluate(cfg: DictConfig) -> Path:
    """Fine-tune ONE base member fold by fold on the new images, and compare before/after."""
    from ultralytics import YOLO

    from extraits.pose.pipeline import _load_context_data, _prepare_run, cmd_train

    main_paths = ProjectPaths.from_config(cfg)
    member = select_members(_base_dir(cfg, main_paths), cfg.finetune.get("base"),
                            single=True)[0]
    weights, card = member.weights, member.card
    epochs = int(cfg.finetune.epochs) if cfg.finetune.get("epochs") else base_epochs(weights)
    name = str(cfg.finetune.get("name") or card.get("variant_hash") or member.name)
    log.info("Base model: %s (run %s, fold %s) | %d epochs", weights, card.get("run_id"),
             card.get("fold"), epochs)

    data_in = _finetune_data_from_cfg(cfg, main_paths)
    cfg, paths = _work_cfg(cfg, main_paths, name, data_in)
    orders = prepare_finetune(cfg, paths, data_in)
    n_folds = int(cfg.finetune.n_folds)
    overrides = {
        "data.scope": "finetune", "data.datasets": orders,
        "cv.name": f"kfold{n_folds}_grouped", "cv.n_folds": n_folds,
        "approach.weights": str(weights), "train.epochs": epochs,
        "retain.enabled": False, "folds": None,
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)

    annotations, schemas = _load_context_data(cfg, paths)
    table, meta = build_splits(annotations, cfg)
    write_splits(table, meta, paths)
    log.info("Split '%s': %d images, %d folds.", meta["split_id"], meta["n_images"], n_folds)

    runs: dict[int, dict[str, str]] = {}
    test_images: dict[int, list[str]] = {}
    for fold in range(n_folds):
        fold_cfg = cfg.copy()
        OmegaConf.update(fold_cfg, "fold", fold)

        # BEFORE: the base model as it is, predicted on the test of the fold.
        before_cfg = fold_cfg.copy()
        OmegaConf.update(before_cfg, "tag", f"ft-{name}-before", force_add=True)
        ctx, data, approach = _prepare_run(before_cfg, {"role_in_protocol": "finetune_before",
                                                        "base_model": str(weights)})
        test_images[fold] = sorted(data.role("test").images["image_id"])
        if not ctx.is_complete() or bool(cfg.force):
            ctx.setup()
            approach.model = YOLO(str(weights))
            approach.predict(data.role("test"), ctx, "test")
            evaluate_run(ctx.run_id, paths, annotations, schemas, cfg.eval,
                         approach=str(cfg.approach.name), split_id=ctx.split_id)
            ctx.write_manifest()

        # AFTER: fine-tuned on the train of the fold, evaluated on the same test.
        after_cfg = fold_cfg.copy()
        OmegaConf.update(after_cfg, "tag", f"ft-{name}", force_add=True)
        after = cmd_train(after_cfg, extra={"role_in_protocol": "finetune",
                                            "base_model": str(weights)})
        runs[fold] = {"before": ctx.run_id, "after": after.run_id}

    out_dir = main_paths.results / "finetune" / name
    eval_cfg = analysis_eval_cfg(cfg)
    log.info("Analysis: an image is detected when its box score is >= %g (skeleton drawn).",
             float(eval_cfg.score_threshold_pointwise))
    summary = write_comparison(paths, runs, test_images, out_dir, annotations, schemas,
                               eval_cfg)
    if bool(cfg.finetune.get("figures", True)):
        export_figures(paths, runs, test_images, annotations, schemas, eval_cfg,
                       out_dir / "figures")
    shown = summary[summary["metric"].isin(SUMMARY_METRICS)]
    log.info("Before / after fine-tuning, mean over %d folds (test):\n%s",
             n_folds, shown.to_string(index=False))
    log.info("Comparison written to %s", out_dir)
    return out_dir
