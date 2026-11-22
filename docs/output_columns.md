# Columns of the `extraits process` CSV

One row per image. `<measure>` stands for each of the 27 measurements and `<kp>` for
each of the 42 keypoints (their definitions: `src/extraits/kp_infos.yaml`). Empty cells
mean "not available" (no insect detected, no scale found, keypoint missing...).

## Identification

| Column | Content |
|---|---|
| `image_name` | file name of the image |
| `image_width`, `image_height` | size of the image, in pixels |

## Measurements

| Column | Content |
|---|---|
| `<measure> [px]` | length in pixels: sum of the segments of its chain of keypoints |
| `<measure> [mm]` | length in millimetres (`[px]` / `scale_px_per_mm`) |
| `<measure> [conf]` | confidence of the measurement: lowest confidence of its keypoints |
| `<measure> [valid_proba]` | probability that the measurement is valid, from its validity classifier |
| `<measure> [valid_pred]` | 1 = valid, 0 = not valid (`valid_proba` against the classifier's tuned threshold) |

A measurement whose keypoints are missing has empty `[px]`, `[mm]` and `[conf]`. The
validity columns are empty for the measurements that have no classifier (too few
annotated examples) or with `--no-measurement-classifier`.

## Pose

| Column | Content |
|---|---|
| `overall_pose_confidence` | box score of the insect × mean keypoint confidence |
| `detection_confidence` | box score of the insect (mean over the ensemble) |
| `n_pose_models` | number of ensemble models averaged for this image (a model whose insect does not overlap the others is left out) |
| `<kp> [kp_x]`, `<kp> [kp_y]` | keypoint position in pixels (origin top-left), mean over the ensemble |
| `<kp> [kp_conf]` | keypoint confidence, mean over the ensemble |
| `<kp> [kp_x_std]`, `<kp> [kp_y_std]` | standard deviation of the position over the ensemble (0 with a single model): a large value flags a doubtful point |

## Scale

| Column | Content |
|---|---|
| `scale_px_per_mm` | pixels per millimetre |
| `scale_method` | `scale_bar`, `ruler` or `none` |
| `scale_confidence` | confidence of the method used |
| `scale_bar_confidence` | bar box score × text box score × OCR reliability |
| `scale_bar_box` | scale-bar box, `x1,y1,x2,y2` |
| `scale_text_box` | scale-bar text box, `x1,y1,x2,y2` |
| `scale_ocr_text` | raw text read by EasyOCR |
| `ruler_confidence` | separation of the main graduation frequency from the others |
| `ruler_line` | row (or column) of the image where the ruler was read |
| `ruler_orientation` | `horizontal` or `vertical` |

## Review and context

| Column | Content |
|---|---|
| `needs_review` | `True` when `overall_pose_confidence` or `scale_confidence` is below 0.5 |
| `coleoptera_one_hot`, `diptera_one_hot`, `hymenoptera_one_hot`, `lepidoptera_one_hot` | insect order given to the validity classifiers (all 0 if unknown) |

## Timings

`scale time [s]`, `pose time [s]`, `measurements time [s]`, `confidence time [s]`,
`mm_conversion time [s]`, `measurement_classifier time [s]`, `total time [s]`: time
spent on each step for this image.

The optional columns can be switched off in `OPTIONAL_COLUMNS` and
`EXPORT_KEYPOINTS` of `src/extraits/process/config.py`.
