# Training the models

To adapt the official models to a few images of your own, prefer
[fine-tuning](finetune.md). Training from scratch needs a few thousand annotated
images ([annotation.md](annotation.md)) and, for the pose models, a GPU.

```bash
pip install "extraits[train]"
```

Inputs, in the workspace ([workspace.md](workspace.md)):
`annotations/annotation_data.csv` and the images in `images/<order>/`.

## Pose models

```bash
extraits train pose                       # fold 0 only, fixed hyperparameters
extraits train pose --folds all           # one model per fold: the 5-model ensemble
extraits train pose --optimise            # Optuna search (20 trials), then the 5 folds
extraits report                           # tables and figures of every run -> results/pose/
```

`extraits train pose` prepares the annotations, builds the 5 cross-validation folds
(shared by every run), trains one YOLO-pose model per fold (starting from Ultralytics
`yolo26n-pose.pt`, 640 x 640, 100 epochs with early stopping), evaluates each on its
held-out fold, and **replaces** `<models>/pose/` with the new ensemble.
`ensemble.json` then holds the cross-validated estimate of the primary metric (OKS AP).

| Option | Meaning |
|---|---|
| `--folds all\|0,2` | folds to train (default: fold 0; every fold with `--optimise`) |
| `--epochs N` | training epochs |
| `--no-export` | keep the models in `runs/` only, leave `<models>/pose/` untouched |
| `--models-dir DIR` | export to another models folder |
| `--experiment NAME` | another frozen experiment (`exp_a_yolo_pooled` by default) |
| `--skip-prepare` | reuse the prepared annotations and folds |
| `key=value` | any setting of the framework, e.g. `train.batch_size=8 tuning.n_trials=40` |

A run is identified by its configuration and data (`run_id`): running the same
command again skips the runs already complete (`force=true` retrains them).

`extraits pose <verb>` gives access to every step (`prepare`, `split`, `train`,
`tune`, `evaluate run_id=...`, `predict`, `report`, `finetune`). The framework, its
contracts and its protocol are described in [pose/README.md](pose/README.md),
[pose/CONVENTIONS.md](pose/CONVENTIONS.md) and [pose/DECISIONS.md](pose/DECISIONS.md).
The other approaches compared in the paper are in `research/`.

## Measurement-validity classifiers

```bash
extraits train validity
```

trains one random forest per measurement on the `<measure>_status` columns of the
annotation table (a measurement needs at least 20 images of each class,
`--min-per-class`), evaluates it with a stratified 5-fold cross-validation
(`--n-folds`), tunes its decision threshold on held-out data, and **replaces**
`<models>/measurement_validity/`. The evaluation (PR curves, confusion matrices,
feature importances, filtering impact) goes to `results/meas_classifier/training/`.

The features are the keypoint coordinates of the anatomical neighbourhood of the
measurement, expressed in the bounding box of the keypoints, plus the insect order
(`src/extraits/validity/features.py`).

## Everything at once

`extraits run` chains annotations, checks, both trainings, measuring and analyses as
set in `<workspace>/run.yaml` (`extraits run --init` writes a commented one).
