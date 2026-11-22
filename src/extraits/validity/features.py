"""Features of the measurement-validity classifiers: the ONLY definition.

Used identically at training time (`extraits.validity.train`, on the annotated
keypoints of the annotation table) and at inference (`extraits.validity.predict`, on
the keypoints predicted by `extraits process`). A model fed features that no longer
mean what they meant at training time returns confident nonsense: never define them
anywhere else.

Feature space: the GEOMETRY of the keypoints, not the pose model's confidences (two
stacked points or a point sent outside the body show in the coordinates, not in a
score). Raw coordinates are not comparable across images (a 278x679 database and a
4000x3000 one have nothing in common), so each point is expressed inside the bounding
box of the keypoints of ITS image,

    x_rel = (x - cx) / w        y_rel = (y - cy) / h

which makes the features invariant to the resolution and to the framing (values
around [-0.5, 0.5]). A keypoint that is missing (non-finite) or exactly at (0, 0) --
what an undetected point looks like -- is left out of the box and comes out NaN; so do
all the points of an image whose box is degenerate (fewer than two usable points, or
all aligned). The random forests see NaN as NA_FILL. The model of a measurement uses
the relative coordinates of the keypoints of its anatomical neighbourhood
(`kp_infos.related_entities`) plus the insect order, one-hot.
"""

from __future__ import annotations

import numpy as np

from extraits.kp_infos import INSECT_GROUPS, related_entities

REL_X_SUFFIX = " kp_x_rel"
REL_Y_SUFFIX = " kp_y_rel"
GROUP_SUFFIX = "_one_hot"
NA_FILL = -1.0                 # absence sentinel of the random forests (no NaN support)
GROUP_COLUMNS = [f"{g}{GROUP_SUFFIX}" for g in INSECT_GROUPS]


def x_column(point: str) -> str:
    return f"{point}{REL_X_SUFFIX}"


def y_column(point: str) -> str:
    return f"{point}{REL_Y_SUFFIX}"


def relative_coordinates(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative coordinates of (n_images, n_points) pixel coordinates.

    Returns two arrays of the same shape; NaN where a point is missing or the box of
    its image is degenerate.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    missing = (~np.isfinite(xs)) | (~np.isfinite(ys)) | ((xs == 0) & (ys == 0))
    xs = np.where(missing, np.nan, xs)
    ys = np.where(missing, np.nan, ys)

    with np.errstate(invalid="ignore"):
        # all-NaN on a row (no keypoint): nanmin warns, NaN is wanted.
        empty = np.isnan(xs).all(axis=1)
        x_min = np.nanmin(np.where(empty[:, None], 0.0, xs), axis=1)
        x_max = np.nanmax(np.where(empty[:, None], 0.0, xs), axis=1)
        y_min = np.nanmin(np.where(empty[:, None], 0.0, ys), axis=1)
        y_max = np.nanmax(np.where(empty[:, None], 0.0, ys), axis=1)

    width, height = x_max - x_min, y_max - y_min
    degenerate = empty | ~(width > 0) | ~(height > 0)   # 0 or 1 point, or all aligned
    width = np.where(degenerate, np.nan, width)
    height = np.where(degenerate, np.nan, height)
    cx, cy = x_min + width / 2.0, y_min + height / 2.0
    return (xs - cx[:, None]) / width[:, None], (ys - cy[:, None]) / height[:, None]


def relative_keypoints(keypoints, names: list[str]) -> dict[str, float]:
    """Inference: {feature column: value} of ONE instance, keypoints (n, >=2) in the
    order of `names`. Missing points (and every point of a degenerate box) are left out.
    """
    xy = np.asarray(keypoints, dtype=float)[:, :2]
    n = min(len(xy), len(names))
    rel_x, rel_y = relative_coordinates(xy[None, :n, 0], xy[None, :n, 1])
    features: dict[str, float] = {}
    for i in range(n):
        if np.isfinite(rel_x[0, i]) and np.isfinite(rel_y[0, i]):
            features[x_column(names[i])] = float(rel_x[0, i])
            features[y_column(names[i])] = float(rel_y[0, i])
    return features


def related_columns(measure: str, available_points) -> list[str]:
    """Feature columns of the keypoints of a measurement's neighbourhood (x then y)."""
    available = set(available_points)
    return [column for p in related_entities(measure)[0] if p in available
            for column in (x_column(p), y_column(p))]


def group_features(group: str | None) -> dict[str, float]:
    """One-hot of the insect order (all zero for an unknown order)."""
    return {f"{g}{GROUP_SUFFIX}": (1.0 if g == group else 0.0) for g in INSECT_GROUPS}
