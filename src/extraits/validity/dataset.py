"""Training data of the measurement-validity classifiers: the annotation table.

`load_annotation_data` reads <workspace>/annotations/annotation_data.csv (pose, scale
and measurement-validity annotations merged, one row per image; built by
`extraits annotate build-table`) and keeps the images that carry measurement-validity
labels. The keypoints are human annotations: there is no confidence column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from extraits.kp_infos import INSECT_GROUPS, MEASUREMENTS, POINTS
from extraits.validity.features import GROUP_SUFFIX

LOGGER = logging.getLogger(__name__)

STATUS_SUFFIX = "_status"
MEASURABLE_LABEL = "measurable"
MEAS_TOKEN = "px"


@dataclass
class ColumnIndex:
    """Columns of the table holding each keypoint coordinate, measurement and status.

    `conf` stays empty for the annotation table (human annotations); it is used by the
    confidence-threshold approaches of the research comparison on pose outputs.
    """

    conf: dict[str, str] = field(default_factory=dict)
    x: dict[str, str] = field(default_factory=dict)
    y: dict[str, str] = field(default_factory=dict)
    measure: dict[str, str] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{len(self.conf)}/{len(POINTS)} conf, "
            f"{len(self.x)}/{len(POINTS)} x, "
            f"{len(self.y)}/{len(POINTS)} y, "
            f"{len(self.measure)}/{len(MEASUREMENTS)} measurements, "
            f"{len(self.status)}/{len(MEASUREMENTS)} statuses"
        )


def load_annotation_data(path: str | Path) -> tuple[pd.DataFrame, ColumnIndex]:
    """Read the annotation table and build its column index.

    Only the images carrying measurement-validity labels are kept: an empty status is
    "not annotated", never "not measurable". Statuses become 1.0 (measurable), 0.0 or
    NaN; the insect order becomes a categorical `group` plus one-hot columns.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    if "image_name" not in frame.columns:
        raise KeyError(f"'image_name' column missing from {path}")

    status_columns = [f"{m}{STATUS_SUFFIX}" for m in MEASUREMENTS
                      if f"{m}{STATUS_SUFFIX}" in frame.columns]
    if not status_columns:
        raise KeyError(
            f"No '<measure>{STATUS_SUFFIX}' column in {path}: this table carries no "
            "measurement-validity annotation, there is nothing to train on."
        )

    annotated = frame[status_columns].notna().any(axis=1)
    LOGGER.info("%d/%d images carry measurement-validity labels", int(annotated.sum()), len(frame))
    frame = frame.loc[annotated].reset_index(drop=True)

    for column in status_columns:
        values = frame[column]
        frame[column] = (
            values.astype(str).str.strip().str.lower().eq(MEASURABLE_LABEL)
            .astype(float).where(values.notna())      # never 0 by default: NaN = not annotated
        )

    index = ColumnIndex()
    for point in POINTS:
        if f"{point}_x" in frame.columns and f"{point}_y" in frame.columns:
            index.x[point] = f"{point}_x"
            index.y[point] = f"{point}_y"
    for measure in MEASUREMENTS:
        if f"{measure}_{MEAS_TOKEN}" in frame.columns:
            index.measure[measure] = f"{measure}_{MEAS_TOKEN}"
        if f"{measure}{STATUS_SUFFIX}" in frame.columns:
            index.status[measure] = f"{measure}{STATUS_SUFFIX}"

    if "group" not in frame.columns:
        raise KeyError(f"'group' column missing from {path}")
    frame["group"] = pd.Categorical(frame["group"].astype(str).str.lower(),
                                    categories=INSECT_GROUPS)
    unknown = int(frame["group"].isna().sum())
    if unknown:
        LOGGER.warning("%d image(s) with an unknown taxonomic group: group one-hot all zero",
                       unknown)
    for group in INSECT_GROUPS:
        frame[f"{group}{GROUP_SUFFIX}"] = (frame["group"] == group).astype(int)

    LOGGER.info("Annotation table: %d images | %s", len(frame), index.summary())
    return frame, index
