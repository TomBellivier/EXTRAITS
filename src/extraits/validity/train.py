"""`extraits train validity`: train the measurement-validity classifiers.

For each measurement (27), trains a random forest ("rf_related": relative coordinates
of the keypoints of the anatomical neighbourhood of the measurement + insect order
one-hot, see extraits/validity/features.py) that predicts whether the automatic
measurement is measurable, and saves

    <models>/measurement_validity/rf_related_<measure>.joblib
    <models>/measurement_validity/metrics.csv      (their decision thresholds)

the files `extraits process` loads (extraits/validity/predict.py). The evaluation
outputs (PR curves, confusion matrices, importances, summaries) go to
<workspace>/results/meas_classifier/training/.

Convention: the positive class is "non measurable" (minority class).
Protocol: stratified k-fold (5), out-of-fold predictions; the decision threshold
(`threshold_median` of metrics.csv) is chosen on an inner partition of the train set
(MCC maximisation), never on the test set. The saved model of a measurement is the one
fitted on the inner partition of the LAST outer fold (behaviour of the original
notebook, kept so that the published models can be reproduced).

The comparison of the 8 approaches evaluated originally (thresholds, XGBoost, random
forests on other feature sets) is research: research/validity/.
"""

from __future__ import annotations

import gc
import re
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no window: PNG files are written
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from extraits.kp_infos import MEAS_TO_KP, MEASUREMENTS, POINTS, related_entities
from extraits.validity.dataset import STATUS_SUFFIX, load_annotation_data
from extraits.validity.features import (
    GROUP_COLUMNS,
    NA_FILL,
    REL_X_SUFFIX,
    REL_Y_SUFFIX,
    relative_coordinates,
    x_column,
    y_column,
)

# --- protocol -----------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 0
MIN_MINORITY = 20   # safeguard: measurement skipped below this number


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def add_relative_coordinates(frame: pd.DataFrame, columns) -> dict:
    """Add to `frame` the relative coordinates of each resolved keypoint
    (extraits.validity.features). Returns {point: (x_column, y_column)}."""
    points = [p for p in POINTS if p in columns.x and p in columns.y]
    if not points:
        raise KeyError("No keypoint coordinate column (<point>_x / <point>_y) in the table.")
    rel_x, rel_y = relative_coordinates(
        frame[[columns.x[p] for p in points]].to_numpy(dtype=float),
        frame[[columns.y[p] for p in points]].to_numpy(dtype=float))
    coords = {}
    for i, point in enumerate(points):
        frame[x_column(point)] = rel_x[:, i]
        frame[y_column(point)] = rel_y[:, i]
        coords[point] = (x_column(point), y_column(point))
    n_degenerate = int(np.isnan(rel_x).all(axis=1).sum())
    if n_degenerate:
        print(f"  /!\\ {n_degenerate} image(s) without a usable keypoint box: "
              f"geometric features set to NaN")
    return coords


def make_coord_columns(coords: dict, points) -> list:
    """Relative coordinate columns (x then y) for a list of keypoints."""
    return [column for p in points if p in coords for column in coords[p]]


def make_conf_columns(columns, points) -> list:
    """Existing confidence columns for a list of keypoints."""
    return [columns.conf[p] for p in points if p in columns.conf]


def make_feature_sets(columns, coords: dict, all_coords: list, measure: str) -> dict:
    """Feature sets of a measurement: its keypoints, its neighbourhood, or all of them.

    ``*_conf`` stays available for the "rule" approaches (threshold on the
    confidence), which only make sense on confidences.
    """
    return {
        "direct": make_coord_columns(coords, MEAS_TO_KP[measure]),
        "related": make_coord_columns(coords, related_entities(measure)[0]),
        "all": all_coords,
        "direct_conf": make_conf_columns(columns, MEAS_TO_KP[measure]),
        "related_conf": make_conf_columns(columns, related_entities(measure)[0]),
    }


def make_target(frame: pd.DataFrame, status_suffix: str, measure: str) -> np.ndarray:
    """1 = non measurable (positive, minority class)."""
    return 1 - frame[f"{measure}{status_suffix}"].astype(int).to_numpy()


def make_rf(y_train: np.ndarray, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        class_weight="balanced",                  # imbalance
        n_jobs=-1, random_state=random_state,
    )


# (key, type, feature set, model factory). Production: a single approach, the models
# `extraits process` loads. research/validity/ compares it with 7 others.
PRODUCTION_APPROACHES = [
    ("rf_related", "model", "related", make_rf),
]


def names_of(approaches: list) -> list:
    return [a[0] for a in approaches]


# --------------------------------------------------------------------------- #
# Score, threshold and cross-validation
# --------------------------------------------------------------------------- #
def rule_score(values: pd.DataFrame, how: str) -> np.ndarray:
    data = np.nan_to_num(values.to_numpy(dtype=float), nan=0.0)  # missing kp -> conf 0
    agg = data.mean(axis=1) if how == "mean" else data.min(axis=1)
    return -agg   # low confidence -> high score -> non measurable


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising the MCC, searched on data not seen during training."""
    candidates = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 201)))
    best_t, best_m = candidates[0], -np.inf
    for t in candidates:
        m = matthews_corrcoef(y, (score >= t).astype(int))
        if m > best_m:
            best_t, best_m = t, m
    return float(best_t)


def evaluate_measure(
    frame: pd.DataFrame,
    columns,
    coords: dict,
    all_coords: list,
    group_cols: list,
    measure: str,
    status_suffix: str,
    n_folds: int,
    random_state: int,
    na_fill: float,
    approaches: list,
    models_dir: Path | None = None,
) -> dict:
    """Out-of-fold predictions of the ``approaches`` for one measurement.

    If ``models_dir`` is given, saves the "rf_related" model (if it is part of
    ``approaches``) trained on the inner partition (``inner_fit``) of EACH outer fold
    as ``models_dir/rf_related_<measure>.joblib``: the final file is therefore that of
    the last outer fold processed (same behaviour as the notebook).
    """
    names = names_of(approaches)
    y = make_target(frame, status_suffix, measure)
    sets = make_feature_sets(columns, coords, all_coords, measure)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scores = {name: np.zeros(len(y)) for name in names}
    preds = {name: np.zeros(len(y), dtype=int) for name in names}
    thresholds = {name: [] for name in names}

    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        # inner partition: only used to set the decision threshold
        inner_fit, inner_val = train_test_split(
            train_idx, test_size=0.25, stratify=y[train_idx], random_state=random_state
        )
        for name, kind, which, factory in approaches:
            # A threshold applies to a confidence, not to a coordinate.
            cols = sets[f"{which}_conf"] if kind == "rule" else sets[which]
            if not cols:
                continue
            if kind == "rule":
                how = "mean" if name.endswith("mean") else "min"
                val_score = rule_score(frame.iloc[inner_val][cols], how)
                test_score = rule_score(frame.iloc[test_idx][cols], how)
            else:
                data = frame[cols + group_cols]
                if name.startswith("rf"):
                    data = data.fillna(na_fill)   # the RF does not handle NaN
                model = factory(y[inner_fit], random_state)
                model.fit(data.iloc[inner_fit], y[inner_fit])
                if name == "rf_related" and models_dir is not None:
                    joblib.dump(model, models_dir / f"rf_related_{measure}.joblib")
                val_score = model.predict_proba(data.iloc[inner_val])[:, 1]
                del model
                gc.collect()
                model = factory(y[train_idx], random_state)
                model.fit(data.iloc[train_idx], y[train_idx])
                test_score = model.predict_proba(data.iloc[test_idx])[:, 1]
                del model
                gc.collect()

            threshold = best_threshold(y[inner_val], val_score)
            scores[name][test_idx] = test_score
            preds[name][test_idx] = (test_score >= threshold).astype(int)
            thresholds[name].append(threshold)

    return {"y": y, "scores": scores, "preds": preds, "thresholds": thresholds}


# --------------------------------------------------------------------------- #
# Metrics and figures
# --------------------------------------------------------------------------- #
def metric_rows(measure: str, result: dict, approaches: list) -> list:
    y = result["y"]
    prev = float(y.mean())
    rows = []
    for name in names_of(approaches):
        score, pred = result["scores"][name], result["preds"][name]
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        ap = float(average_precision_score(y, score))
        rows.append({
            "measure": measure,
            "model": name,
            "n": len(y),
            "prevalence_unmeasurable": prev,
            "mcc": float(matthews_corrcoef(y, pred)),
            "average_precision": ap,
            "average_precision_norm": (ap - prev) / (1 - prev) if prev < 1 else np.nan,
            "accuracy": float((tp + tn) / len(y)),
            "accuracy_unmeasurable": float(tp / (tp + fn)) if (tp + fn) else np.nan,
            "accuracy_measurable": float(tn / (tn + fp)) if (tn + fp) else np.nan,
            "balanced_accuracy": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
            "precision_unmeasurable": float(tp / (tp + fp)) if (tp + fp) else np.nan,
            "threshold_median": float(np.median(result["thresholds"][name])),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return rows


def plot_pr(out_dir: Path, measure: str, result: dict, approaches: list) -> None:
    y = result["y"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in names_of(approaches):
        score = result["scores"][name]
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(recall, precision, lw=1.6, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y.mean(), color="grey", ls="--", lw=1, label=f"chance ({y.mean():.3f})")
    ax.set_xlabel("Recall (non measurable)")
    ax.set_ylabel("Precision (non measurable)")
    ax.set_title(f"Precision-recall curve: {measure}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves" / f"{slug(measure)}.png", dpi=140)
    plt.close(fig)


def plot_confusion(out_dir: Path, measure: str, result: dict, approaches: list) -> None:
    y = result["y"]
    names = names_of(approaches)
    labels = ["measurable", "non meas."]
    n_cols = min(len(names), 4) or 1
    n_rows = -(-len(names) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.25 * n_cols, 3.5 * n_rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        matrix = confusion_matrix(y, result["preds"][name], labels=[0, 1])
        normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]}\n{normalised[i, j]:.0%}",
                        ha="center", va="center", fontsize=9,
                        color="white" if normalised[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{name}\nMCC={matthews_corrcoef(y, result['preds'][name]):.3f}", fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices (out-of-fold): {measure}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion" / f"{slug(measure)}.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Production training
# --------------------------------------------------------------------------- #
APPROACHES = PRODUCTION_APPROACHES         # a single approach: rf_related
MODEL_NAME, _, MODEL_FEATURES, MODEL_FACTORY = APPROACHES[0]


def run_training(annotation_data: str | Path, models_dir: str | Path, out_dir: str | Path,
                 min_minority: int = MIN_MINORITY, n_folds: int = N_FOLDS) -> Path:
    """Train, evaluate and save the classifiers. Returns `models_dir`.

    Side effects: `models_dir`/rf_related_<measure>.joblib + metrics.csv (a measurement
    below the safeguard loses its old model), and the evaluation outputs in `out_dir`.
    """
    warnings.filterwarnings("ignore")
    annotation_data = Path(annotation_data)
    MODELS_DIR = Path(models_dir)
    OUT_DIR = Path(out_dir)
    group_cols = list(GROUP_COLUMNS)

    for sub in ("pr_curves", "confusion", "importance"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data loading -------------------------------------------------------------
    if not annotation_data.is_file():
        raise SystemExit(
            f"Annotation table missing: {annotation_data}\n"
            "Build it with: extraits annotate build-table"
        )
    frame, columns = load_annotation_data(annotation_data)
    print(f"{len(frame)} images | {columns.summary()}")

    # The out-of-fold predictions are stacked measurement by measurement: every
    # measurement must cover the same rows. An image with a single unannotated
    # measurement is therefore left out, rather than counted as non measurable.
    status_columns = list(columns.status.values())
    incomplete = frame[status_columns].isna().any(axis=1)
    if incomplete.any():
        print(f"{int(incomplete.sum())} image(s) with an incomplete annotation left out "
              f"({len(frame) - int(incomplete.sum())} kept)")
        frame = frame.loc[~incomplete].reset_index(drop=True)

    # Prevalence per measurement: used by the safeguard and to read the accuracies.
    prevalence = []
    for measure in MEASUREMENTS:
        status = f"{measure}{STATUS_SUFFIX}"
        if status not in frame.columns:
            continue
        y = 1 - frame[status].astype(int).to_numpy()   # 1 = non measurable
        prevalence.append({
            "measure": measure,
            "n": len(y),
            "n_unmeasurable": int(y.sum()),
            "prevalence_unmeasurable": float(y.mean()),
            "kept": bool(min(y.sum(), len(y) - y.sum()) >= min_minority),
        })
    prevalence = pd.DataFrame(prevalence)
    prevalence.to_csv(OUT_DIR / "prevalence.csv", index=False)

    # A measurement gets its classifier as soon as EACH class (measurable / non
    # measurable) counts at least --min-per-class images: it appears by itself at the next
    # training, without touching anything else (`extraits process` loads every .joblib present).
    kept = prevalence.loc[prevalence["kept"], "measure"].tolist()
    skipped = prevalence.loc[~prevalence["kept"]]
    print(f"{len(kept)} measurements kept, {len(skipped)} skipped (< {min_minority} minority examples)")
    for row in skipped.itertuples(index=False):
        minority = min(row.n_unmeasurable, row.n - row.n_unmeasurable)
        print(f"  skipped: {row.measure} ({minority}/{min_minority} examples of the minority class)")

    # A measurement that falls back below the threshold loses its old model: otherwise
    # `extraits process` would keep loading it, without a decision threshold in the new metrics.csv.
    for measure in MEASUREMENTS:
        stale = MODELS_DIR / f"rf_related_{measure}.joblib"
        if measure not in kept and stale.is_file():
            stale.unlink()
            print(f"  stale model removed: {stale.name}")

    # Features: the keypoint geometry, brought back into its bounding box (see
    # extraits.validity.features). The pose-model confidences are only used by the
    # "rule" approaches of the research comparison.
    coords = add_relative_coordinates(frame, columns)
    all_coords = make_coord_columns(coords, POINTS)
    print(f"{len(coords)}/{len(POINTS)} keypoints in relative coordinates "
          f"({len(all_coords)} geometric features)")

    # --- Main loop: training + saving + metrics -------------------------------------
    rows, oof = [], {}
    for i, measure in enumerate(kept, start=1):
        print(f"[{i:2d}/{len(kept)}] {measure}", flush=True)
        result = evaluate_measure(
            frame, columns, coords, all_coords, group_cols, measure, STATUS_SUFFIX,
            n_folds, RANDOM_STATE, NA_FILL, APPROACHES, MODELS_DIR,
        )
        rows.extend(metric_rows(measure, result, APPROACHES))
        plot_pr(OUT_DIR, measure, result, APPROACHES)
        plot_confusion(OUT_DIR, measure, result, APPROACHES)
        oof[measure] = {"y": result["y"], MODEL_NAME: result["preds"][MODEL_NAME]}
        del result
        gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"\n{len(metrics)} rows written to {OUT_DIR / 'metrics.csv'}")
    # The decision threshold is part of the model: it goes with the .joblib files,
    # otherwise `extraits process` falls back to 0.5 for every measurement.
    metrics.to_csv(MODELS_DIR / "metrics.csv", index=False)
    print(f"Production models and thresholds written to {MODELS_DIR}")

    # --- Performance summary (a single model: no ranking) ----------------------------
    summary = (
        metrics
        .rename(columns={
            "precision_unmeasurable": "precision",
            "accuracy_unmeasurable": "recall",
        })
        [["mcc", "precision", "recall"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.to_csv(OUT_DIR / "summary_mcc_precision_recall.csv")
    print(summary)

    # --- Practical consequence: usable measurements per image ------------------------
    y_true = np.column_stack([oof[m]["y"] for m in kept])        # 1 = non measurable
    valid = y_true == 0                                          # measurement actually usable

    practical_rows, per_image = [], {
        "image_name": frame["image_name"], "group": frame["group"].astype(str),
    }

    def summarise(label, rejected):
        keep = ~rejected
        keep_valid = keep & valid
        per_image[f"n_kept_{label}"] = keep.sum(axis=1)
        per_image[f"n_valid_kept_{label}"] = keep_valid.sum(axis=1)
        practical_rows.append({
            "filtering": label,
            "kept_measures_per_image": keep.sum(axis=1).mean(),
            "of_which_actually_valid": keep_valid.sum(axis=1).mean(),
            "valid_lost_per_image": (valid & rejected).sum(axis=1).mean(),
            "valid_retention": keep_valid.sum() / valid.sum(),
            "kept_contamination": (keep & ~valid).sum() / max(keep.sum(), 1),
        })

    summarise("no_filtering", np.zeros_like(y_true, dtype=bool))
    predicted = np.column_stack([oof[m][MODEL_NAME] for m in kept])
    summarise(MODEL_NAME, predicted == 1)

    practical = pd.DataFrame(practical_rows).set_index("filtering").round(3)
    practical.to_csv(OUT_DIR / "filtering_impact.csv")
    pd.DataFrame(per_image).to_csv(OUT_DIR / "measures_per_image.csv", index=False)
    print(f"\n{len(kept)} measurements evaluated, {valid.mean():.1%} actually usable\n")

    # --- Label distribution per measurement (stacked bars) ----------------------------
    order = prevalence.sort_values("prevalence_unmeasurable")
    n_ok = (order["n"] - order["n_unmeasurable"]).to_numpy()
    n_ko = order["n_unmeasurable"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 0.34 * len(order) + 2))
    ax.barh(order["measure"], n_ok, color="#4c9f70", label="measurable")
    ax.barh(order["measure"], n_ko, left=n_ok, color="#c0504d", label="non measurable")
    for i, (a, b) in enumerate(zip(n_ok, n_ko)):
        ax.text(a + b + order["n"].max() * 0.01, i, f"{b / (a + b):.0%}", va="center", fontsize=7)
    ax.set_xlim(0, order["n"].max() * 1.09)
    ax.set_xlabel("number of images")
    ax.set_title("Label repartition per measure (sorted by non-measurable ratio)")
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "labels_per_measure.png", dpi=140)
    plt.close(fig)
    print("written:", OUT_DIR / "labels_per_measure.png")

    # --- Feature importance -----------------------------------------------------------
    # The "rf_related" model is retrained on the whole data, measurement by
    # measurement. These importances are descriptive: they show which keypoints carry
    # the signal, they do not measure a performance.
    def pretty(column: str) -> str:
        """'head-top kp_x_rel' -> 'head-top x', 'coleoptera_one_hot' -> 'coleoptera'."""
        for suffix, axis in ((REL_X_SUFFIX, " x"), (REL_Y_SUFFIX, " y")):
            if column.endswith(suffix):
                return column[: -len(suffix)] + axis
        return column.replace("_one_hot", "")

    series = []
    for measure in kept:
        cols = make_feature_sets(columns, coords, all_coords, measure)[MODEL_FEATURES]
        if not cols:
            continue
        y = make_target(frame, STATUS_SUFFIX, measure)
        data = frame[cols + group_cols].fillna(NA_FILL)

        model = MODEL_FACTORY(y, RANDOM_STATE)
        model.fit(data, y)
        values = pd.Series(model.feature_importances_, index=[pretty(c) for c in data.columns])
        del model
        gc.collect()
        series.append(values.rename(measure))

        top = values.sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(top) + 1.6))
        ax.barh(top.index, top.to_numpy(), color="steelblue")
        ax.set_xlabel("importance")
        ax.set_title(f"{MODEL_NAME}: {measure}", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "importance" / f"{slug(measure)}.png", dpi=140)
        plt.close(fig)

    importance = pd.concat(series, axis=1).T
    importance.to_csv(OUT_DIR / "feature_importance.csv")
    print(f"{len(importance)} measurements, {importance.shape[1]} variables")

    mean_importance = importance.mean().sort_values().tail(20)
    coverage = importance.notna().sum()

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(mean_importance) + 2))
    ax.barh(mean_importance.index, mean_importance.to_numpy(), color="#4472c4")
    for i, name in enumerate(mean_importance.index):
        ax.text(mean_importance[name], i, f"  {coverage[name]}/{len(importance)}",
                va="center", fontsize=7, color="grey")
    ax.set_xlim(0, float(mean_importance.max()) * 1.12)
    ax.set_xlabel("mean importance (grey number: number of measures where the variable exists)")
    ax.set_title(f"Most used variables ({MODEL_NAME})")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_importance.png", dpi=140)
    plt.close(fig)
    print("written:", OUT_DIR / "mean_importance.png")

    # --- Measurements whose label is almost determined by the taxonomic group ---------
    suspects = []
    for measure in kept:
        y = pd.Series(make_target(frame, STATUS_SUFFIX, measure))
        by_group = y.groupby(frame["group"].astype(str).to_numpy()).mean()
        if ((by_group < 0.05) | (by_group > 0.95)).all():
            suspects.append({"measure": measure, **by_group.round(2).to_dict()})

    suspects = pd.DataFrame(suspects)
    if not suspects.empty:
        suspects.to_csv(OUT_DIR / "group_determined_measures.csv", index=False)
        print("Non-measurable rate per group (trivial measurements):")
        print(suspects)
    return MODELS_DIR
