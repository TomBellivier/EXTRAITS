"""Measurement validity: inference of the per-measurement classifiers.

Scores each measurement of a processed image with its pre-trained validity
classifier ("does this automatic measurement make morphological sense?").
Inference only: one random forest per measurement, fed the keypoint COORDINATES
of its anatomical neighbourhood -- expressed inside the bounding box of the
instance's keypoints -- plus the insect's order (one-hot), saved as

    <models>/measurement_validity/rf_related_<measure>.joblib
    <models>/measurement_validity/metrics.csv        (decision thresholds)

The anatomical neighbourhood ("related" keypoints) and the insect orders come
from kp_infos.yaml, like at training time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extraits import workspace
from extraits.kp_infos import KEYPOINT_INDEX, KEYPOINT_NAMES, MEASUREMENT_NAMES
from extraits.validity.features import (
    GROUP_COLUMNS,
    NA_FILL,
    group_features,
    related_columns,
    relative_keypoints,
    x_column,
    y_column,
)

DEFAULT_THRESHOLD = 0.5        # fallback when no tuned threshold is available

# The features (keypoint geometry inside its own box + insect order) are defined ONCE,
# in extraits/validity/features.py, shared with the training. The guard in
# load_measurement_classifiers() catches models whose feature NAMES this code cannot
# build (e.g. models trained on keypoint confidences).

VALID_PROBA_SUFFIX = " [valid_proba]"
VALID_PRED_SUFFIX = " [valid_pred]"


def _load_thresholds(metrics_csv: Path) -> dict[str, float]:
    """Per-measurement decision threshold (median over CV folds, rf_related)."""
    if not metrics_csv.is_file():
        print(f"[measure-clf] no metrics CSV at {metrics_csv} -> using the default "
              f"threshold ({DEFAULT_THRESHOLD}) for every measurement.")
        return {}
    df = pd.read_csv(metrics_csv)
    sub = df[df["model"] == "rf_related"]
    return dict(zip(sub["measure"], sub["threshold_median"]))


class MeasurementClassifiers:
    """Loaded once, shared read-only across worker threads.

    Unlike the YOLO models (which need one instance per thread, see worker.py),
    a fitted RandomForestClassifier's predict_proba() does not mutate the
    estimator, so the same models can be scored concurrently from every thread.
    """

    def __init__(self, models: dict[str, object], thresholds: dict[str, float]):
        self.models = models
        self.thresholds = thresholds
        # Precomputed once: which feature columns feed each measurement's model.
        self._related_columns = {name: related_columns(name, KEYPOINT_NAMES) for name in models}

    def score(self, keypoints, group: str | None) -> dict[str, dict[str, float]]:
        """{'<measure>': {'proba': float, 'pred': 0/1}} for every scored measure.

        'proba' is the probability the measurement IS trustworthy (measurable);
        'pred' is that same call thresholded at the measure's tuned cutoff.
        """
        out: dict[str, dict[str, float]] = {}
        if keypoints is None or not self.models:
            return out

        groups = group_features(group)
        # One frame per image: every measurement of this instance is expressed in
        # the same box, exactly as at training time.
        relative = relative_keypoints(keypoints, KEYPOINT_NAMES)
        for name, model in self.models.items():
            features = {column: relative[column] for column in self._related_columns[name]
                        if column in relative}
            features.update(groups)

            row = pd.DataFrame(
                [[features.get(col, np.nan) for col in model.feature_names_in_]],
                columns=model.feature_names_in_,
            ).fillna(NA_FILL)

            proba_unmeasurable = float(model.predict_proba(row)[0, 1])
            threshold = self.thresholds.get(name, DEFAULT_THRESHOLD)
            out[name] = {
                "proba": 1.0 - proba_unmeasurable,
                "pred": 0 if proba_unmeasurable >= threshold else 1,
            }
        return out


def load_measurement_classifiers(models_dir, metrics_csv=None) -> MeasurementClassifiers:
    """Load every classifier of `models_dir` (a measurement_validity/ folder)."""
    models_dir = Path(models_dir)
    thresholds = _load_thresholds(Path(metrics_csv or models_dir / workspace.VALIDITY_METRICS_NAME))

    models: dict[str, object] = {}
    if not models_dir.is_dir():
        print(f"[measure-clf] models dir not found: {models_dir} -> "
              f"measurement-validity columns will be empty.")
        return MeasurementClassifiers(models, thresholds)

    buildable = {column for p in KEYPOINT_INDEX for column in (x_column(p), y_column(p))}
    buildable |= set(GROUP_COLUMNS)
    stale: list[str] = []

    for name in MEASUREMENT_NAMES:
        path = models_dir / f"rf_related_{name}.joblib"
        if path.is_file():
            model = joblib.load(path)
            # A model whose features this pipeline cannot build would be scored on
            # a row of NaN -> NA_FILL: a confident answer to a question it was
            # never asked. Models trained on keypoint confidences (the feature set
            # used before the switch to geometry) land here.
            unknown = [c for c in getattr(model, "feature_names_in_", []) if c not in buildable]
            if unknown:
                stale.append(name)
                del model
                continue
            # Saved with n_jobs=-1 (useful for batch training, not for scoring
            # one row at a time): left as-is, every predict_proba() call spins
            # up its own internal thread pool across cores purely for
            # overhead, and 16 pipeline workers x 26 measures makes that
            # contention real. Forcing n_jobs=1 is ~4x faster per call.
            model.n_jobs = 1
            models[name] = model

    if stale:
        print(f"[measure-clf] /!\\ {len(stale)} classifier(s) in {models_dir} expect features "
              f"this pipeline no longer produces (e.g. keypoint confidences): they are "
              f"IGNORED, and their CSV columns will be empty. Retrain them with "
              f"`extraits train validity`. First: {stale[:3]}")
    print(f"[measure-clf] loaded {len(models)}/{len(MEASUREMENT_NAMES)} "
          f"measurement-validity classifiers from {models_dir}")
    return MeasurementClassifiers(models, thresholds)
