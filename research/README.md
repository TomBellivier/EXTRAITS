# EXTRAITS research

The experiments behind the EXTRAITS paper: the pose approaches compared with the
retained `yolo_pooled`, the comparison scripts, and the evaluation of the ruler
detector. None of this is needed to use EXTRAITS; it is here to reproduce the
comparisons and to try new approaches.

## Install

```bash
pip install -e ".[train]"      # EXTRAITS with the training framework (repository root)
pip install -e research/       # this package: extraits-research
```

`extraits-research` registers itself with the pose framework through the
`extraits.pose_plugins` entry point: its approaches and their Hydra configs become
available to `extraits pose` and `extraits train pose --experiment ...` with no other
setting.

## Contents

| Path | What |
|---|---|
| `src/extraits_research/approaches/` | `yolo_per_dataset`, `detect_then_pose`, `lora`, `group_bn`, `yolo_pooled_reduced`, `head_only`, `lora_per_dataset` |
| `src/extraits_research/configs/` | their `approach/` configs and the frozen experiments `exp_b` ... `exp_h` |
| `src/extraits_research/models/`, `training/`, `data/` | building blocks used only by these approaches (per-group BatchNorm, patched trainers, crops) |
| `scripts/compare_surface.sh` | every approach on one fold, no HPO: a quick screening |
| `scripts/compare_base_models.sh` | five starting weights on `yolo_pooled` (ADR-0033) |
| `scripts/compare_models.py` | heatmaps and cost figures from `results/pose/master.parquet` |
| `scripts/plot_optuna.py` | diagnostics of the Optuna studies |
| `scripts/check_schema.py` | prepared annotations whose keypoint count differs from the schema |
| `ruler/` | evaluation of the ruler detector (script + 2 notebooks) |
| `validity/` | the 8 measurement-validity approaches compared (`compare_measure_validity_approaches.py`, notebook) |
| `tests/` | tests of the research approaches (fake Ultralytics, no GPU needed) |

`yolo_pooled` and the `mean_pose` baseline are part of EXTRAITS itself
(`src/extraits/pose/`). The protocol (shared folds, frozen metrics, HPO budget) is in
`docs/pose/CONVENTIONS.md`; the decisions are in `docs/pose/DECISIONS.md`.

## Workspace

Every command reads and writes in the workspace: `--workspace` (`--root` for the
scripts), else `$EXTRAITS_WORKSPACE`, else the current folder. It must hold
`annotations/annotation_data.csv` and `images/<order>/` (see `docs/pose/README.md`).
The images are not public: the comparisons can only be reproduced with access to them.

## Reproducing the comparisons of the paper

```bash
export EXTRAITS_WORKSPACE=~/EXTRAITS-workspace

extraits pose prepare
extraits pose split

# every approach: HPO (tune_once, 20 trials), then the 5 outer folds
for exp in exp_a_yolo_pooled exp_b_yolo_per_dataset exp_c_detect_then_pose exp_d_lora \
           exp_e_group_bn exp_f_yolo_reduced exp_g_head_only exp_h_lora_per_dataset; do
    extraits pose tune experiment=$exp folds=all retain.enabled=false
done
extraits pose train experiment=exp_ref_mean_pose folds=all   # floor baseline

extraits report                                   # tables -> results/pose/
python research/scripts/compare_models.py         # heatmaps, cost figures
python research/scripts/plot_optuna.py            # HPO diagnostics
```

`retain.enabled=false` keeps the retained ensemble (`<workspace>/models/pose/`)
untouched: only `yolo_pooled` is ever exported. A quick screening on one fold, without
HPO: `bash research/scripts/compare_surface.sh 0 30` (fold 0, 30 epochs).

## Ruler detector

```bash
python research/ruler/ruler_detection_evaluation.py   # -> results/ruler_detection/
```

then open `ruler/ruler_detection_evaluation.ipynb`. `ruler_fourier_analysis.ipynb`
explains the Fourier analysis the detector (`extraits/scale/ruler.py`) relies on.
The notebooks still read the images from their former location; adapt the first cell
to your workspace.

## Measurement-validity approaches

```bash
python research/validity/compare_measure_validity_approaches.py   # -> results/meas_classifier/comparison/
```

compares thresholds on the pose confidence, XGBoost and random forests on three
feature sets, with the protocol of `extraits train validity` (the retained approach is
`rf_related`). `measure_validity_classifiers.ipynb` is the original notebook; it still
reads the files from their former location.

The notebooks are published without their outputs (they showed images of the
non-public databases).

## Tests

```bash
pytest research/tests
```
