# Annotating images

Annotations are needed to train or fine-tune the models, and to measure their real
error. Three kinds of annotations feed EXTRAITS:

| Annotation | Tool | File (in `<workspace>/annotations/`) |
|---|---|---|
| 42 keypoints per insect | Label Studio | `label_studio_annotations/<order>/*.json` -> `pose/pose_annotations.csv` |
| validity of each measurement | `extraits annotate validate-measurements` | `meas_classifier/<batch>_measurements.csv` |
| scale (optional) | by hand | `scale/scale_annotations.csv` |

They are merged into one table, `annotation_data.csv`, which every training reads.

```bash
pip install "extraits[annotation]"      # Label Studio and the desktop app (PySide6)
```

The protocol (where to place each keypoint, what makes a measurement valid) is
`src/extraits/annotation/annotation_protocol.pdf`; the keypoints and measurements are
defined in `src/extraits/kp_infos.yaml`.

## 1. Keypoints, in Label Studio

```bash
extraits annotate label-studio --document-root /path/to/images
```

starts Label Studio with local image serving and prints the path of the labelling
interface (`label_studio_template.txt`). In a new project, paste it under
*Settings > Labeling Interface > Code*, add the images (*Settings > Cloud Storage >
Add Source Storage > Local files*, under the document root), and annotate. Export the
project as JSON into `annotations/label_studio_annotations/<order>/`.

An export made by someone else, on another machine, points at their image paths. Make
it point at your copy of the images before importing it:

```bash
extraits annotate import-volunteer --json-file their_export.json --image-folders /path/to/images
```

Then convert every export to the pose CSV:

```bash
extraits annotate to-csv          # label_studio_annotations/ -> pose/pose_annotations.csv
```

## 2. Validity of the measurements

```bash
extraits annotate validate-measurements --annotations-csv annotations/pose/pose_annotations.csv
```

opens a desktop app showing each annotated image with its measurements drawn. Mark
each one measurable or not (a leg folded under the body, a wing hidden by another...);
the result is written to `annotations/meas_classifier/<csv name>_measurements.csv`.
Closing and reopening the app on the same CSV resumes where you stopped.

## 3. Scale (optional)

`annotations/scale/scale_annotations.csv`, one row per image: `image_name`,
`scale_type` (`scale_bar` or `ruler`), `scale_px_per_mm`, and, for a scale bar,
`scale_bar_bbox_x`, `scale_bar_bbox_y`, `scale_bar_text`, for a ruler,
`ruler_direction`, `ruler_line_min`, `ruler_line_max`. It is used to evaluate the scale
detection, not to train the models.

## 4. Merge and check

```bash
extraits annotate build-table     # -> annotations/annotation_data.csv
extraits annotate check           # -> results/annotation_check/issues.csv, presence.csv
```

`check` reads every file, on its own and against the others, and reports images
present in one source and missing from another, invalid values, keypoints outside the
image, an out-of-date merged table, annotated images missing from `images/<order>/`,
and the measurements that have too few examples of each class to get a validity
classifier. Errors make it exit with code 1 (`--strict`: warnings too).

## The annotation table

`annotation_data.csv` has one row per image: `image_name`, `group` (insect order),
`width`, `height`, `<kp>_x`, `<kp>_y`, `<kp>_v` for each keypoint (`v` = 2 annotated, 0
not), the scale columns, and `<measure>_status` (`measurable` / `non_measurable`) for
each measurement. The CSV format of `extraits finetune` is a subset of it
([finetune.md](finetune.md)).
