"""
definitions.py
==============

Keypoints, measurements and anatomy of the pipeline, re-exported from the single
definition of the project, `extraits/kp_infos.yaml` (loaded by `extraits.kp_infos`).

    * KEYPOINT_NAMES      -> the canonical keypoint order. THIS ORDER MUST MATCH the
                             order the model was trained with, because YOLO returns
                             keypoints as an array indexed by this order.
    * KEYPOINT_COLORS     -> RGB colour of each keypoint (overlays).
    * MEASUREMENTS        -> for each measurement, the ordered list of keypoints
                             whose consecutive segments are summed.
    * FLIP_INDEX          -> permutation that swaps left/right keypoints, used by
                             the horizontal-flip test-time augmentation.
"""

from extraits.kp_infos import (  # noqa: F401  (re-exported)
    FLIP_INDEX,
    INSECT_GROUPS,
    KEYPOINT_COLORS_RGB as KEYPOINT_COLORS,
    KEYPOINT_INDEX,
    KEYPOINT_NAMES,
    MEAS_TO_KP as MEASUREMENTS,
    MEASUREMENT_INDICES,
    MEASUREMENT_NAMES,
    NUM_KEYPOINTS,
    SYMMETRIC_MEASUREMENT_PAIRS,
    related_entities,
)
