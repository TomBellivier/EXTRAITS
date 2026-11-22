"""Publish the EXTRAITS models on the Hugging Face Hub (maintainers only).

Uploads a models folder (pose/, scale_bar/, measurement_validity/) to a Hub model
repository in one commit, together with a generated model card (README.md),
tags that commit, and rewrites the package manifest src/extraits/models.yaml
(repository, commit hash, tag, sha256 of every file) that
`extraits download-models` reads. Commit the new manifest afterwards.

    # look at the model card and the manifest first: nothing is uploaded
    python tools/upload_models.py --models-dir <models> --tag v0.1.0 --dry-run
    # log in once (`hf auth login`, or HF_TOKEN), then publish
    python tools/upload_models.py --models-dir <models> --tag v0.1.0

The repository is created private unless --public is given; make it public from
the Hub settings once the model card is checked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from extraits import workspace  # noqa: E402
from extraits.models import COMPONENTS, MANIFEST_PATH, sha256_of  # noqa: E402

DEFAULT_REPO_ID = "TomBellivier/extraits-models"


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def collect_files(models_dir: Path) -> list[str]:
    """Relative posix paths of every model file to publish."""
    pose = models_dir / workspace.POSE_SUBDIR
    validity = models_dir / workspace.VALIDITY_SUBDIR
    found: list[Path] = []
    found += sorted(pose.glob("*/best.pt")) + sorted(pose.glob("*/model_card.json"))
    if (pose / "ensemble.json").is_file():
        found.append(pose / "ensemble.json")
    if (models_dir / workspace.SCALE_BAR_SUBPATH).is_file():
        found.append(models_dir / workspace.SCALE_BAR_SUBPATH)
    found += sorted(validity.glob("*.joblib"))
    if (validity / workspace.VALIDITY_METRICS_NAME).is_file():
        found.append(validity / workspace.VALIDITY_METRICS_NAME)
    rels = sorted(p.relative_to(models_dir).as_posix() for p in found)
    for c, sub in COMPONENTS.items():
        if not any(r.startswith(sub + "/") for r in rels):
            raise SystemExit(f"No {c} model under {models_dir / sub}.")
    return rels


# Keys of a pose model_card.json that describe the private training machine
# (local paths, commit of the private repository): not published.
PRIVATE_CARD_KEYS = ("source_run", "git")


def payload(models_dir: Path, rel: str) -> bytes | Path:
    """What is uploaded for `rel`: the file itself, or a cleaned copy of a model card."""
    path = models_dir / rel
    if not rel.endswith("/model_card.json"):
        return path
    card = json.loads(path.read_text(encoding="utf-8"))
    for key in PRIVATE_CARD_KEYS:
        card.pop(key, None)
    return (json.dumps(card, indent=2) + "\n").encode("utf-8")


def file_table(models_dir: Path, rels: list[str]) -> dict[str, dict]:
    table = {}
    for rel in rels:
        data = payload(models_dir, rel)
        if isinstance(data, Path):
            table[rel] = {"sha256": sha256_of(data), "size": data.stat().st_size}
        else:
            table[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        print(f"  {table[rel]['sha256'][:12]}  {table[rel]['size']:>11,d}  {rel}")
    return table


# --------------------------------------------------------------------------- #
# model card
# --------------------------------------------------------------------------- #
def _pose_section(models_dir: Path) -> str:
    pose = models_dir / workspace.POSE_SUBDIR
    lines = []
    ens_path = pose / "ensemble.json"
    ens = json.loads(ens_path.read_text()) if ens_path.is_file() else {}
    cards = [json.loads(p.read_text()) for p in sorted(pose.glob("*/model_card.json"))]
    first = cards[0] if cards else {}
    cv = ens.get("cv_estimate") or {}
    if cv:
        lines.append(f"Cross-validated {cv['metric']} of the ensemble members on their "
                     f"held-out folds: **{cv['mean']:.3f} ± {cv['std']:.3f}** "
                     f"({cv['n_folds']} folds).\n")
    if cards:
        lines.append("| Member | Held-out fold | " + first.get("primary_metric", "metric") + " |")
        lines.append("|---|---|---|")
        for c in cards:
            lines.append(f"| `{c['run_id']}` | {c.get('fold')} | {c.get('primary_value', float('nan')):.3f} |")
        lines.append("")
        lines.append(f"- Keypoint schema: `{first.get('keypoint_schema')}`, "
                     f"{len(first.get('keypoint_names', []))} keypoints.")
        lines.append(f"- Input size: {first.get('image_size')}.")
        lines.append(f"- Insect orders: {', '.join(first.get('datasets', []))}.")
        lines.append(f"- Cross-validation split: `{first.get('split_id')}`.")
    return "\n".join(lines)


def _validity_section(models_dir: Path) -> str:
    path = models_dir / workspace.VALIDITY_SUBDIR / workspace.VALIDITY_METRICS_NAME
    if not path.is_file():
        return ""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = ["| Measurement | n | MCC | Average precision | Balanced accuracy |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['measure']} | {r['n']} | {float(r['mcc']):.3f} | "
                     f"{float(r['average_precision']):.3f} | {float(r['balanced_accuracy']):.3f} |")
    return "\n".join(lines)


def _data_section(annotation_data: Path | None) -> str:
    if annotation_data is None or not annotation_data.is_file():
        return ""
    with open(annotation_data, newline="", encoding="utf-8") as f:
        groups = [row.get("group", "") for row in csv.DictReader(f)]
    counts = {g: groups.count(g) for g in sorted(set(groups)) if g}
    table = ", ".join(f"{g} {n}" for g, n in counts.items())
    return f"Annotated images: {len(groups)} ({table}).\n"


def model_card(models_dir: Path, tag: str, annotation_data: Path | None) -> str:
    header = {
        "license": "agpl-3.0",
        "library_name": "ultralytics",
        "pipeline_tag": "keypoint-detection",
        "tags": ["insects", "morphometrics", "pose-estimation", "yolo", "biology", "extraits"],
    }
    return f"""---
{yaml.safe_dump(header, sort_keys=False).strip()}
---

# EXTRAITS models ({tag})

Models used by [EXTRAITS](https://github.com/TomBellivier/EXTRAITS) to measure insects
on dorsal images:

| Folder | Model | Role |
|---|---|---|
| `pose/` | ensemble of YOLO-pose models (one per cross-validation fold) | 42 keypoints, from which 27 measurements are derived |
| `scale_bar/` | YOLO detector | finds the scale bar; its value is read by EasyOCR |
| `measurement_validity/` | one random forest per measurement (scikit-learn) | says whether each measurement is valid |

Download them with the EXTRAITS command line, which checks every file against
the sha256 pinned in the package:

```bash
extraits download-models            # into <workspace>/models
extraits process -i <images folder> -o results.csv
```

## Training data

The pose models and the validity classifiers were trained on dorsal images of
insects from several natural-history image databases, annotated for the GRIT
project. The images are not redistributable, so neither the images nor the
annotations are published.

{_data_section(annotation_data)}
The pose models are fine-tuned from Ultralytics `yolo26n-pose.pt`. The scale-bar
detector was trained outside the EXTRAITS repository and is published as weights
only.

## Evaluation

### Pose ensemble

{_pose_section(models_dir)}

### Measurement validity

Cross-validated metrics of each classifier ("unmeasurable" is the positive class
for the average precision):

{_validity_section(models_dir)}

## Limitations

- Trained on four insect orders (Coleoptera, Diptera, Hymenoptera, Lepidoptera)
  photographed dorsally; other orders or views are out of distribution.
- `extraits finetune` adapts the pose models to new images.

## Licence

These weights are licensed under the **GNU Affero General Public License v3.0**
(AGPL-3.0), because they derive from Ultralytics YOLO models, themselves AGPL-3.0.
The EXTRAITS source code is MIT-licensed.

The `.joblib` files are Python pickles: load them only from a trusted source
(`extraits download-models` verifies their sha256).

## Citation

Please cite EXTRAITS, the GRIT project, the image-database providers and
Ultralytics (see `CITATION.cff` in the EXTRAITS repository).

_Model card generated on {date.today().isoformat()}._
"""


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def manifest_text(repo_id: str, revision: str | None, tag: str, files: dict) -> str:
    head = MANIFEST_PATH.read_text(encoding="utf-8").split("repo_id:")[0]
    body = yaml.safe_dump({"repo_id": repo_id, "revision": revision, "tag": tag,
                           "files": files}, sort_keys=False, allow_unicode=True)
    return head + body


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models-dir", required=True, type=Path,
                   help="Folder with pose/, scale_bar/ and measurement_validity/.")
    p.add_argument("--tag", required=True, help="Release name of the models, e.g. v0.1.0.")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--annotation-data", type=Path, default=None,
                   help="annotation_data.csv, to count the training images per order "
                        "in the model card (optional; the table itself is not uploaded).")
    p.add_argument("--public", action="store_true",
                   help="Create the repository public (default: private).")
    p.add_argument("--dry-run", action="store_true",
                   help="Write the model card and the manifest to --out and upload nothing.")
    p.add_argument("--out", type=Path, default=Path("upload_preview"),
                   help="Folder for the --dry-run output (default: ./upload_preview).")
    args = p.parse_args(argv)

    models_dir = args.models_dir.expanduser().resolve()
    print(f"Model files under {models_dir}:")
    rels = collect_files(models_dir)
    files = file_table(models_dir, rels)
    card = model_card(models_dir, args.tag, args.annotation_data)

    if args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "README.md").write_text(card, encoding="utf-8")
        (args.out / "models.yaml").write_text(
            manifest_text(args.repo_id, "<commit hash after upload>", args.tag, files),
            encoding="utf-8")
        print(f"\nDry run: model card and manifest written to {args.out.resolve()}. "
              f"Nothing was uploaded.")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8"))]
    for rel in rels:
        data = payload(models_dir, rel)
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=(
            str(data) if isinstance(data, Path) else data)))
    print(f"\nUploading {len(rels)} files to {args.repo_id} ...")
    info = api.create_commit(args.repo_id, operations=ops, repo_type="model",
                             commit_message=f"EXTRAITS models {args.tag}")
    revision = info.oid
    api.create_tag(args.repo_id, tag=args.tag, revision=revision, repo_type="model")
    MANIFEST_PATH.write_text(manifest_text(args.repo_id, revision, args.tag, files),
                             encoding="utf-8")
    print(f"Published {args.repo_id}@{args.tag} (commit {revision}).\n"
          f"Manifest updated: {MANIFEST_PATH}. Commit it, then check with:\n"
          f"  extraits download-models --models-dir /tmp/extraits-models-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
