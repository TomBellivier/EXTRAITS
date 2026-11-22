"""The 8 measurement-validity approaches compared in the paper (research only).

Everything shared with the production training (features, cross-validation,
thresholds, metrics, figures) comes from extraits.validity.train; this module only adds
the approaches that are not in production: thresholds on the pose confidence and
XGBoost (`pip install xgboost`, a dependency of extraits-research).
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from extraits.validity.train import (  # noqa: F401  (re-exported for the comparison)
    REL_X_SUFFIX,
    REL_Y_SUFFIX,
    add_relative_coordinates,
    evaluate_measure,
    make_coord_columns,
    make_feature_sets,
    make_rf,
    make_target,
    metric_rows,
    names_of,
    plot_confusion,
    plot_pr,
    slug,
)


def make_xgb(y_train: np.ndarray, random_state: int) -> XGBClassifier:
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=n_neg / n_pos,          # imbalance
        eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=random_state,
    )


# (key, type, feature set, model factory): thresholds, XGBoost and random forests on 3
# feature sets (direct / related / all). "rf_related" is the production approach.
ALL_APPROACHES = [
    ("rule_conf_mean", "rule", "direct", None),
    ("rule_conf_min", "rule", "direct", None),
    ("xgb_direct", "model", "direct", make_xgb),
    ("rf_direct", "model", "direct", make_rf),
    ("xgb_related", "model", "related", make_xgb),
    ("rf_related", "model", "related", make_rf),
    ("xgb_all", "model", "all", make_xgb),
    ("rf_all", "model", "all", make_rf),
]
