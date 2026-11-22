# Pose framework (`extraits.pose`)

Framework that prepares the annotations, builds the cross-validation folds, trains,
tunes and evaluates pose approaches on the 4 insect orders (Coleoptera, Diptera,
Hymenoptera, Lepidoptera), and exports the retained ensemble to `<models>/pose/`.

**Read `CONVENTIONS.md` (next to this file) before any contribution.** It holds the
whole doctrine (contracts, rules, protocol); `DECISIONS.md` holds the ADRs.

## Installation

```bash
pip install -e ".[train]"          # the framework: yolo_pooled and mean_pose
pip install -e research/           # optional: the research approaches (research/README.md)
```

torch and ultralytics are first-rank dependencies (ADR-0019). Everything adapts to the
machine: `train.device: auto` uses GPU 0 if there is one, else the Apple GPU (`mps`), else
the CPU; `train.num_workers: auto` takes one data-loading worker per CPU minus one (capped at
8, 0 on a single-CPU machine); mixed precision is on on GPU, off on CPU and in `mode: debug`.
`train.batch_size` stays fixed at 16: two runs only compare at an equal batch size.

## Workspace

Every path is in the workspace (`--workspace`, else `$EXTRAITS_WORKSPACE`, else the current
folder; `configs/paths.yaml`):

```
<workspace>/annotations/annotation_data.csv   annotation table (input, one row per image)
<workspace>/images/<order>/                   annotated images (input)
<workspace>/data/pose/                        prepared annotations (processed/) and folds (splits/)
<workspace>/runs/<run_id>/                    one folder per run (manifest, metrics, weights, figures)
<workspace>/results/pose/                     aggregated tables and figures (`extraits report`)
<workspace>/models/pose/                      retained ensemble (--models-dir / $EXTRAITS_MODELS)
```

`prepare` reads the annotation table (`annotation_csv` adapter) and derives the canonical
annotations of contract 1 from it, filtered on the `group` column. An annotated image
missing from `images/<order>/` is left out with a warning (`skip_missing_images`). The
`coco` and `yolo` adapters remain available (`data.adapter=yolo`, with
`paths.images=${paths.data}`).

No path enters the `run_id` or the `variant_hash` (`context.py`): a run keeps its
identity on another machine or in another workspace.

## Usual use

```bash
extraits train pose                       # prepare + split + train fold 0, export it
extraits train pose --folds all           # one model per fold = the ensemble
extraits train pose --optimise            # Optuna search, then every outer fold
extraits train pose --epochs 30 --no-export train.batch_size=8   # any Hydra override
extraits report                           # aggregate every run -> results/pose/
```

## Every step: `extraits pose <verb>`

```bash
extraits pose prepare                     # data=coleoptera prepares a single order
extraits pose split                       # folds shared by ALL the approaches (contract 2)
extraits pose train experiment=exp_a_yolo_pooled fold=0
extraits pose train experiment=exp_a_yolo_pooled folds=[0,1,2]
extraits pose tune experiment=exp_a_yolo_pooled folds=all
extraits pose evaluate run_id=<run_id>    # re-evaluate (and retain) a run without retraining
extraits pose predict run_id=<run_id> split=test
extraits pose finetune finetune.evaluate=true base=<variant_hash>   # see docs/finetune.md
extraits pose report
```

## Retained models: an ensemble, used by `extraits process`

`<models>/pose/` holds an **ensemble** of `yolo_pooled` models (`retain.approaches`), one
sub-folder per model with its `model_card.json`. `extraits process` runs ALL of them on every
image and writes, for every keypoint coordinate, the mean and the standard deviation over the
ensemble. Every command REPLACES the previous ensemble, so that two trainings never mix:

```
train                  -> 1 model + ensemble.json
train folds=[0,1,2]    -> 3 models + ensemble.json   (one per fold, ADR-0039)
tune     ├─ HPO trials (inner folds)           -> not exported
         └─ reruns on the outer folds          -> EXPORTED: 5 models (or those of `folds`) + ensemble.json
evaluate run_id=<run_id> -> 1 model
```

`--no-export` (`retain.enabled=false`) leaves the ensemble in place intact.
`ensemble.json` carries `cv_estimate`: the mean and the standard deviation of the primary
metric over the folds trained, each one measured on a test it has not seen. Running `tune`
again skips the folds already complete. `tuning.final_full_fit=true` ALSO trains a model on
all the images and ADDS it to the ensemble. A `mode=smoke` run, an HPO trial or an approach
outside `retain.approaches` is never exported.

## Keypoints and measurements

The `insect42_v1` schema (points, difficulties, symmetries, skeleton) and the 27 measurements
are read from `extraits/kp_infos.yaml`, the single definition of EXTRAITS.

## Adding an approach

Copy `src/extraits/pose/approaches/TEMPLATE.py.txt` and follow `CONVENTIONS.md` §11. An
approach outside the package registers itself through the `extraits.pose_plugins` entry
point (see `research/pyproject.toml`). The smoke test is parametrised on the registry: a
registered approach enters it automatically.

## Status

Complete generic framework (contracts, registry, splits, evaluation, tuning, reporting, CLI).

| Approach                | Status                                                                  |
| ----------------------- | ----------------------------------------------------------------------- |
| `mean_pose`           | implemented (core) - reference and floor baseline (GT bbox, diagnostic)        |
| `yolo_pooled`         | **implemented** (core) - CUDA GPU, AMP, FP16 at inference; the retained model |
| `yolo_per_dataset`    | **implemented** - N models routed by dataset (ADR-0023)                |
| `detect_then_pose`    | **implemented** - pooled detector + pose on a crop (ADR-0024)          |
| `lora`                | **implemented** - adapters on the neck, trainable heads (ADR-0025)     |
| `group_bn`            | **implemented** - BatchNorm per dataset, mixed batches (ADR-0026)      |
| `yolo_pooled_reduced` | **implemented** - A without legs nor hind wings (ADR-0027)             |

`head_only` and `lora_per_dataset` are implemented too. Every approach but `mean_pose`
and `yolo_pooled` lives in `research/` (`pip install -e research/`).

An approach whose heavy dependency is missing is **skipped** by the smoke test
(`availability()` mechanism), never failed.

Every run produces: `manifest.json`, resolved `config.yaml`, `predictions/`,
`metrics.parquet`, `logs/` and `figures/` (12 pred vs GT examples including the 6 worst cases).

### What `mean_pose` is for

It is not a candidate model: for each instance it predicts the **mean pose of the train set**
placed back into the ground-truth bbox. It plays three roles:

1. **floor baseline** - a trained model that does not clearly beat it has a problem
   (convergence, malformed labels, wrong keypoint order). It is a sanity test, not a
   competitor;
2. **template** - it is the reference implementation of the `Approach` protocol, to copy when
   writing a new approach;
3. **smoke test** - it runs in a few seconds, without a GPU nor a heavy dependency, which
   validates the whole `train -> predict -> evaluate -> figures` chain at every
   `pytest -m smoke`.

It uses the GT bboxes (`bbox_source: gt`): its numbers are therefore **not comparable** with
those of the end-to-end approaches and must never appear in the same table (§9.3).

## Frozen protocol

Every protocol decision is settled (ADR-0006 to 0039, see `DECISIONS.md`):

| Point          | Value                                                                  |
| -------------- | ---------------------------------------------------------------------- |
| Keypoints      | `insect42_v1`: 42 points, common to the 4 datasets, union = identity |
| OKS sigmas     | `difficulty x 0.0025` (10 -> 0.025 ... 40 -> 0.100)                  |
| PCK            | `alpha x thorax width`, reference alpha = 0.25                        |
| Primary metric | `oks_ap` (overridable without re-evaluation)                         |
| Measurements   | 27 morphometric measurements + 9 symmetry pairs                        |
| Folds          | 5 outer folds, group_id = image_id (one image = one specimen)          |
| HPO            | `tune_once`: search on the inner folds of outer fold 0, 20 trials (ADR-0031) |
| Resolution     | 640x640 for every approach (strict guard)                              |

**Cost of the fully nested HPO** (`tuning.mode=nested`): `n_folds x n_trials x inner_folds`
trainings per approach. To be calibrated before launching a heavy approach, then frozen
identically for all of them (fair budget).
