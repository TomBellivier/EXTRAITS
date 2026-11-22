"""Measurement-validity classifiers: shared features, training, and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from extraits.kp_infos import INSECT_GROUPS, KEYPOINT_NAMES, MEASUREMENTS
from extraits.validity import features
from extraits.validity.dataset import STATUS_SUFFIX


def _random_keypoints(rng, n_images: int, missing: float = 0.2) -> np.ndarray:
    xy = rng.uniform(10, 500, (n_images, len(KEYPOINT_NAMES), 2))
    xy[rng.random((n_images, len(KEYPOINT_NAMES))) < missing] = 0.0   # undetected point
    return xy


def test_inference_and_training_features_are_the_same():
    """One image at a time (inference) == the whole table at once (training)."""
    rng = np.random.default_rng(0)
    xy = _random_keypoints(rng, 20)
    xy[3, :, :] = 0.0                    # no keypoint at all
    xy[4, 1:, :] = 0.0                   # a single point: degenerate box
    xy[4, 0, :] = (50.0, 60.0)
    rel_x, rel_y = features.relative_coordinates(xy[:, :, 0], xy[:, :, 1])
    for i in range(len(xy)):
        one = features.relative_keypoints(xy[i], KEYPOINT_NAMES)
        for k, name in enumerate(KEYPOINT_NAMES):
            if np.isfinite(rel_x[i, k]):
                assert one[features.x_column(name)] == rel_x[i, k]
                assert one[features.y_column(name)] == rel_y[i, k]
            else:
                assert features.x_column(name) not in one
    assert features.relative_keypoints(xy[3], KEYPOINT_NAMES) == {}
    assert features.relative_keypoints(xy[4], KEYPOINT_NAMES) == {}


def test_relative_coordinates_live_in_the_unit_box():
    rng = np.random.default_rng(1)
    rel_x, rel_y = features.relative_coordinates(*_random_keypoints(rng, 50).transpose(2, 0, 1))
    assert np.nanmax(np.abs(rel_x)) <= 0.5 + 1e-12 and np.nanmax(np.abs(rel_y)) <= 0.5 + 1e-12


def _annotation_table(path, n_images: int = 120):
    """Synthetic annotation table: a measurement is 'non measurable' when one of its
    keypoints is missing, so the classifiers have a signal to learn."""
    from extraits.kp_infos import MEAS_TO_KP

    rng = np.random.default_rng(2)
    xy = _random_keypoints(rng, n_images, missing=0.15)
    rows = []
    for i in range(n_images):
        row = {"image_name": f"img{i}.jpg", "group": INSECT_GROUPS[i % len(INSECT_GROUPS)]}
        for k, name in enumerate(KEYPOINT_NAMES):
            row[f"{name}_x"], row[f"{name}_y"] = xy[i, k]
            row[f"{name}_v"] = 0 if (xy[i, k] == 0).all() else 2
        for measure in MEASUREMENTS:
            index = [KEYPOINT_NAMES.index(p) for p in MEAS_TO_KP[measure]]
            ok = not any((xy[i, j] == 0).all() for j in index)
            row[f"{measure}{STATUS_SUFFIX}"] = "measurable" if ok else "non_measurable"
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_train_validity_then_inference(tmp_path, monkeypatch):
    """`extraits train validity` writes classifiers that `extraits process` loads."""
    from extraits.cli import main
    from extraits.validity.predict import load_measurement_classifiers

    # Never let a test write into the real models folder of the environment.
    monkeypatch.delenv("EXTRAITS_MODELS", raising=False)
    (tmp_path / "annotations").mkdir()
    _annotation_table(tmp_path / "annotations" / "annotation_data.csv")
    assert main(["train", "--workspace", str(tmp_path), "validity", "--min-per-class", "5",
                 "--n-folds", "3", "--models-dir", str(tmp_path / "models")]) == 0
    models = tmp_path / "models" / "measurement_validity"
    metrics = pd.read_csv(models / "metrics.csv")
    assert set(metrics["model"]) == {"rf_related"} and len(metrics) > 0
    assert (tmp_path / "results" / "meas_classifier" / "training" / "metrics.csv").is_file()
    assert len(sorted(models.glob("rf_related_*.joblib"))) == len(metrics)

    classifiers = load_measurement_classifiers(models)
    assert len(classifiers.models) == len(metrics)
    keypoints = np.column_stack([_random_keypoints(np.random.default_rng(3), 1)[0],
                                 np.ones(len(KEYPOINT_NAMES))])
    scores = classifiers.score(keypoints, "coleoptera")
    assert set(scores) == set(classifiers.models)
    assert all(0.0 <= s["proba"] <= 1.0 and s["pred"] in (0, 1) for s in scores.values())
