# models

Trained models that `extraits process` runs on: `<workspace>/models/` (or
`--models-dir` / `$EXTRAITS_MODELS`). Nothing is trained in this folder and nothing is
edited by hand. `extraits download-models` fills it with the official models.

```
models/
├── pose/                              <- ENSEMBLE of YOLO-pose models (extraits train pose)
│   ├── <run_id>/
│   │   ├── best.pt                    ... one model of the ensemble
│   │   └── model_card.json            ... run_id, approach, fold, metrics, keypoint schema
│   ├── <run_id>/ ...                  ... one folder per model (one per fold trained)
│   └── ensemble.json                  ... members + cross-validation estimate of their folds
├── scale_bar/
│   └── best.pt                        <- YOLO scale-bar detector
└── measurement_validity/              <- measurement-validity classifiers
    ├── rf_related_<measure>.joblib    ... one random forest per measurement
    └── metrics.csv                    ... their per-measurement decision thresholds
```

## Who writes what

| Folder | Produced by | How |
|--------|-------------|-----|
| everything | `extraits download-models` | the official models (pinned revision, checksums verified) |
| `pose/` | `extraits train pose` | the folder is **replaced** by one model per fold trained |
| `pose/` | `extraits train pose --optimise` | the folder is **replaced** by one model per outer fold (5, or those of `--folds`) |
| `pose/` | `extraits pose evaluate run_id=...` | the folder is **replaced** by that single model |
| `measurement_validity/` | `extraits train validity` | the classifiers and their `metrics.csv` are **replaced** |
| `scale_bar/` | trained outside EXTRAITS | drop the `best.pt` file in by hand |

`--no-export` (or `retain.enabled=false`) leaves `pose/` untouched. Only `yolo_pooled`
runs are exported (`retain.approaches` in `extraits/pose/configs/config.yaml`): one
YOLO-pose model on the full 42-keypoint schema of `kp_infos.yaml`.

## Who reads it

`extraits process` runs **every** `*.pt` under `pose/` on each image, matches the same
insect across the models, and writes the mean and the standard deviation of every
keypoint coordinate (0 with a single model). Measurements, confidences and the
measurement-validity classifiers all work on the mean keypoints. `--pose-models
<folder or .pt>` points it at another ensemble.

Each model's `model_card.json` says which run produced it and what it scored on its
held-out fold. `ensemble.json` holds `cv_estimate`: the mean and the standard
deviation of the primary metric over the folds of the ensemble.

Model files (`*.pt`, `*.joblib`) are never in git.
