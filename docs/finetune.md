# Fine-tuning the pose models on your images

The official pose models were trained on dorsal images of Coleoptera, Diptera,
Hymenoptera and Lepidoptera from a few natural-history collections. If your images look
different (other photographic setup, other taxa within these orders), a few dozen
annotated images are often enough to adapt the models.

```bash
pip install "extraits[train]"
extraits download-models                     # the base models, if not done yet

# 1. optional: estimate the gain first (k-fold before/after comparison, nothing exported)
extraits finetune --evaluate --images my_images/ --annotations my_annotations.csv --group hymenoptera

# 2. fine-tune and export a new ensemble
extraits finetune --images my_images/ --annotations my_annotations.csv --group hymenoptera \
                  --output my_models/

# 3. measure with it
extraits process -i new_images/ -o results.csv --models-dir my_models/
```

## What it does

`extraits finetune --output DIR` starts from **each model of the base ensemble** (by
default the official models, `<models>/pose/`; `--base-models` points at another one)
and fine-tunes it on **all** your annotated images. A small share of them (15 %) is held
out only to pick the best checkpoint. Each model is trained for as many epochs as the
base model was (100, with early stopping; `--epochs` changes it).

The fine-tuned models form a new ensemble in `DIR/pose/`, with a `model_card.json` per
model (base model, number of images, annotation files) and an `ensemble.json`. The
scale-bar detector and the measurement-validity classifiers are copied next to it
(`--pose-only` to skip), so `DIR` is a complete models folder for
`extraits process --models-dir DIR`.

The base models are **never** modified: `--output` cannot be the base models folder nor
the models folder of the workspace, and a non-empty `DIR/pose/` needs `--overwrite`.

Options:

| Option | Meaning |
|---|---|
| `--base all` (default) | fine-tune every model of the base ensemble |
| `--base <member folder>` | fine-tune a single model (faster, no ensemble spread) |
| `--epochs N` | training epochs (default: those of the base model) |
| `--name NAME` | name of the work folders `runs/finetune/NAME/` and `data/pose/finetune/NAME/` |
| `key=value` | any setting of the training framework, e.g. `train.batch_size=8` |

The work files (prepared annotations, training runs) go to the workspace
(`--workspace`, else `$EXTRAITS_WORKSPACE`, else the current folder).

## Estimating the gain: `--evaluate`

`extraits finetune --evaluate` splits your images into 10 folds (`--folds N`). For each
fold, one base model (the fold-0 member, or `--base`) is evaluated as it is on the test
images of the fold ("before"), then fine-tuned on the other images and evaluated again
("after"). The comparison is written to `<workspace>/results/pose/finetune/<name>/`:

- `comparison.csv`: mean and standard deviation over the folds of every metric, before,
  after and their difference;
- `comparison_per_fold.csv`: the same, per fold;
- `figures/`: one `before | after` image per annotated image (ground truth in white,
  predicted skeleton coloured by confidence).

An image counts as detected when the model draws a skeleton on it, i.e. when its box
score reaches the detection threshold of `extraits process` (0.25). Detection metrics
use every test image; point metrics (PCK, NME, OKS AP, measurement errors) only the
images detected both before and after, so that a missed image does not count as a
badly placed skeleton. With few images, expect a large spread between folds.

## Annotations

### Label Studio

Use the labelling template of EXTRAITS (`extraits annotate label-studio` prints its path)
and export the project as JSON. Pass the export(s) with `--annotations export.json`, and
the insect order with `--group` (or put each export in a folder named after the order).

Without `--annotations`, `--images` may also follow this layout:

```
my_images/
└── hymenoptera/              one folder per insect order
    ├── export.json           Label Studio export(s)
    └── images/               the images
```

### CSV

One row per image, in the columns of the EXTRAITS annotation table:

| Column | Content |
|---|---|
| `image_name` | file name of the image (or its path relative to `--images`) |
| `group` | insect order: `coleoptera`, `diptera`, `hymenoptera` or `lepidoptera` (or `--group`) |
| `width`, `height` | image size in pixels (optional: read from the file) |
| `<keypoint>_x`, `<keypoint>_y` | position in pixels, origin at the top-left corner |
| `<keypoint>_v` | `2` = annotated, `0` = not annotated (absent, hidden, or not applicable) |

for each of the 42 keypoints, in any order:

```
head-top, head-left, head-right, left-eye, right-eye, neck, thorax-left, thorax-right,
thorax-bottom, body-left, body-right, body-tip, left-antenna-0, left-antenna-1,
left-antenna-2, right-antenna-0, right-antenna-1, right-antenna-2, left-forewing-base,
left-forewing-tip, left-forewing-front, left-forewing-rear, right-forewing-base,
right-forewing-tip, right-forewing-front, right-forewing-rear, left-hindwing-base,
left-hindwing-tip, left-hindwing-front, left-hindwing-rear, right-hindwing-base,
right-hindwing-tip, right-hindwing-front, right-hindwing-rear, left-leg-0, left-leg-1,
left-leg-2, left-leg-3, right-leg-0, right-leg-1, right-leg-2, right-leg-3
```

Every keypoint column must be present; a point that is not annotated has `_v = 0` (its
`_x` / `_y` are then ignored). The definition of each point is in
`src/extraits/kp_infos.yaml` and in the annotation protocol. Images are looked up under
`--images`, recursively, by file name; an annotated image that is not found is left
out with a warning.
