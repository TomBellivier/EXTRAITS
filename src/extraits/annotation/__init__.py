"""Annotation tools of EXTRAITS (`extraits annotate ...`).

Where the annotation files live, in the workspace (all overridable on the command line):

    <workspace>/annotations/
    ├── label_studio_annotations/<order>/*.json   Label Studio exports (keypoints)
    ├── pose/pose_annotations.csv                 the exports converted to CSV (to-csv)
    ├── scale/scale_annotations.csv               scale annotations (hand-written CSV)
    ├── meas_classifier/*_measurements.csv        measurement statuses (validate-measurements)
    ├── measurement_validation/                   presets, resume state and logs of the app
    └── annotation_data.csv                       the merged table (build-table)
    <workspace>/images/<order>/                   the annotated images
    <workspace>/results/annotation_check/         reports of `extraits annotate check`

Files shipped with the package: label_studio_template.txt (the labelling interface)
and annotation_protocol.pdf (how to place each keypoint).
"""

from __future__ import annotations

from pathlib import Path

from extraits import workspace

PACKAGE_DIR = Path(__file__).resolve().parent
LABEL_STUDIO_TEMPLATE = PACKAGE_DIR / "label_studio_template.txt"
ANNOTATION_PROTOCOL = PACKAGE_DIR / "annotation_protocol.pdf"


def annotations_dir(ws: str | Path | None = None) -> Path:
    return workspace.workspace_dir(ws) / "annotations"


def label_studio_dir(ws: str | Path | None = None) -> Path:
    return annotations_dir(ws) / "label_studio_annotations"


def pose_csv(ws: str | Path | None = None) -> Path:
    return annotations_dir(ws) / "pose" / "pose_annotations.csv"


def scale_csv(ws: str | Path | None = None) -> Path:
    return annotations_dir(ws) / "scale" / "scale_annotations.csv"


def measurements_dir(ws: str | Path | None = None) -> Path:
    return annotations_dir(ws) / "meas_classifier"


def app_dir(ws: str | Path | None = None) -> Path:
    """Presets, resume state and logs of the measurement-validation app."""
    return annotations_dir(ws) / "measurement_validation"


def images_dir(ws: str | Path | None = None) -> Path:
    return workspace.workspace_dir(ws) / "images"


def check_report_dir(ws: str | Path | None = None) -> Path:
    return workspace.results_dir(ws) / "annotation_check"
