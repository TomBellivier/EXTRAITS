"""Where EXTRAITS reads and writes its data.

EXTRAITS never assumes that data lives inside the source tree. Everything a user
owns (images, annotations, trained or downloaded models, training runs, results)
lives in a *workspace* folder:

    <workspace>/
    ├── models/                  trained models, the layout `extraits process` reads:
    │   ├── pose/                    ensemble of YOLO-pose models (every *.pt under it)
    │   ├── scale_bar/best.pt        YOLO scale-bar detector
    │   └── measurement_validity/    one random forest per measurement + metrics.csv
    ├── annotations/             annotation tables (annotation_data.csv, ...)
    ├── images/<group>/...       annotated image databases, one folder per insect order
    ├── runs/                    training and fine-tuning runs
    └── results/                 analyses

Resolution order:
    workspace : --workspace, else $EXTRAITS_WORKSPACE, else the current directory
    models    : --models-dir, else $EXTRAITS_MODELS, else <workspace>/models
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ENV = "EXTRAITS_WORKSPACE"
MODELS_ENV = "EXTRAITS_MODELS"

# Layout of a models folder (shared by `process`, `download-models` and the trainers).
POSE_SUBDIR = "pose"
SCALE_BAR_SUBPATH = Path("scale_bar") / "best.pt"
VALIDITY_SUBDIR = "measurement_validity"
VALIDITY_METRICS_NAME = "metrics.csv"


def workspace_dir(override: str | os.PathLike | None = None) -> Path:
    """The workspace folder (not created)."""
    value = override or os.environ.get(WORKSPACE_ENV)
    return Path(value).expanduser().resolve() if value else Path.cwd()


def models_dir(override: str | os.PathLike | None = None,
               workspace: str | os.PathLike | None = None) -> Path:
    """The folder holding pose/, scale_bar/ and measurement_validity/ (not created)."""
    value = override or os.environ.get(MODELS_ENV)
    if value:
        return Path(value).expanduser().resolve()
    return workspace_dir(workspace) / "models"


def annotation_table(workspace: str | os.PathLike | None = None) -> Path:
    """The merged annotation table read by the trainers and by `extraits analyze`."""
    return workspace_dir(workspace) / "annotations" / "annotation_data.csv"


def results_dir(workspace: str | os.PathLike | None = None) -> Path:
    """Root of every analysis written by EXTRAITS."""
    return workspace_dir(workspace) / "results"
