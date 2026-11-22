# Measuring images: `extraits process`

`extraits process` measures every insect image of a folder (or of a Hugging Face
dataset) and writes **one CSV row per image**: 42 keypoints, 27 measurements in pixels
and in millimetres, a confidence for each, the scale and how it was found, and whether
each measurement is valid. The columns are described in
[output_columns.md](output_columns.md).

```bash
extraits download-models                          # once: the official models -> <workspace>/models
extraits process -i my_images/ -o results.csv     # measure
extraits analyze -i results.csv                   # figures and summary -> results_analysis/
```

The images must show one insect, **in dorsal view**, of one of the four orders the
models were trained on: Coleoptera, Diptera, Hymenoptera, Lepidoptera.

## What happens to each image

1. **Pose.** Every model of the pose ensemble (`<models>/pose/`, 5 YOLO-pose models by
   default) places the 42 keypoints. The insect found by each model is matched with
   the others (box IoU >= 0.5), and every keypoint coordinate is averaged over the
   models; their standard deviation is kept (`[kp_x_std]`, `[kp_y_std]`). An image
   where no model detects an insect (box score < 0.25) gets an empty row.
2. **Measurements.** Each measurement is the length of a chain of keypoints (e.g. the
   femur, the whole hind leg, the wing length), defined in `src/extraits/kp_infos.yaml`.
3. **Scale.** With `--scale auto` (default), a YOLO detector looks for a **scale bar**
   and EasyOCR reads its value (e.g. "2 mm"); when there is none, the image is searched
   for a **ruler**, whose graduation spacing is found by Fourier analysis. `--scale
   scale_bar`, `--scale ruler` or `--scale none` restrict it. Without a scale, the
   millimetre columns are empty.
4. **Validity.** For each measurement, a random forest trained on human judgements
   says whether the automatic measurement makes morphological sense (e.g. a leg length
   on a leg that is folded under the body does not). It uses the geometry of the
   keypoints and the insect order (see below). `--no-measurement-classifier` skips it.
5. **Confidences.** See below.

## Options

| Option | Meaning |
|---|---|
| `-i, --input DIR` | folder of images, searched recursively |
| `-o, --output FILE` | output CSV (default `extraits_results.csv`) |
| `--models-dir DIR` | models folder (default `$EXTRAITS_MODELS`, else `<workspace>/models`) |
| `--pose-models DIR\|FILE` | another pose ensemble (e.g. a fine-tuned one), or a single `.pt` |
| `--scale auto\|scale_bar\|ruler\|none` | how to find the scale (default `auto`) |
| `--all-scale-methods` | run both scale methods on every image (diagnostics) |
| `--no-measurement-classifier` | no validity columns |
| `--group ORDER` | insect order of every image |
| `--groups-csv FILE` | insect order per image (columns `image_name`, `group`) |
| `--image-list FILE` | only the images listed (one name per line) |
| `--source hf --dataset USER/NAME [--hf-folders 1 2]` | read a Hugging Face dataset into memory instead of a folder (`HF_TOKEN` for a private one) |
| `--workers N`, `--buffer N`, `--torch-threads N` | force the parallelism (default: sized to the machine) |
| `--flush-every N` | write the CSV to disk every N rows (crash safety) |

**Insect order.** The validity classifiers use the order of the insect. It is taken,
by priority, from `--groups-csv`, then `--group`, then the first folder of the image
path named after an order (`my_images/hymenoptera/...`). Without any of them, the
order one-hot columns are 0 and the classifiers work on the geometry alone.

**Hardware.** The run is sized to the machine at start-up (printed on the `Hardware`
line): every CUDA GPU (FP16), else the Apple GPU, else the CPU; one or more worker
threads, each with its own copy of the models, capped so that they fit in memory.

## Confidences

Every confidence is in [0, 1]; higher means more trustworthy. They are monotone
signals, **not calibrated probabilities**: a 0.9 does not guarantee a given error.

| Column | Definition |
|---|---|
| `<measure> [conf]` | the lowest confidence of the keypoints of the measurement (a distance is only as good as its weakest endpoint) |
| `overall_pose_confidence` | box score of the insect × mean keypoint confidence |
| `scale_bar_confidence` | bar box score × text box score × OCR reliability |
| `ruler_confidence` | separation between the main graduation frequency and the others (1 = a single clear graduation) |
| `scale_confidence` | the confidence of the scale method used |
| `needs_review` | `overall_pose_confidence` or `scale_confidence` below 0.5 |
| `<measure> [valid_proba]` | probability that the measurement is valid (validity classifier) |

The defaults and the alternatives (e.g. a test-time-augmentation confidence) are in
`src/extraits/process/config.py` and `confidence.py`.

## Analysing a results CSV: `extraits analyze`

```bash
extraits analyze -i results.csv                                         # -> results_analysis/
extraits analyze -i results.csv --annotation-data annotation_data.csv   # + real errors
extraits analyze -i results.csv --print --images-dir my_images/         # + annotated copies
```

It draws the distributions of the confidences and of the measurements, the scale
methods used, the share of rows to review and the disagreement of the ensemble. With an
annotation table (the `<kp>_x/_y/_v` columns of [annotation.md](annotation.md)), it
also measures the real error on the annotated images (OKS, per-keypoint error against
confidence, per-measurement error). Images the models were trained on give optimistic
errors: the cross-validated estimate is `cv_estimate` in `<models>/pose/ensemble.json`.

## Your own models

`--models-dir` takes any folder with the layout of [models.md](models.md):
fine-tuned models ([finetune.md](finetune.md)) or models you trained
([training.md](training.md)).
