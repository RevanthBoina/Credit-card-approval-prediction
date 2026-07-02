"""XGBoost model training with a graceful sklearn fallback."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from utils.config import RANDOM_STATE


def build_xgboost_model(y_train=None):
    """Create an XGBoost classifier, falling back if xgboost is unavailable.

    Args:
        y_train: Optional target vector used to estimate class imbalance.

    Returns:
        An unfitted classifier implementing fit and predict.
    """
    try:
        from xgboost import XGBClassifier

        scale_pos_weight = 1.0
        if y_train is not None:
            positives = float(np.sum(y_train == 1))
            negatives = float(np.sum(y_train == 0))
            if positives > 0:
                scale_pos_weight = negatives / positives

        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        )


def train_xgboost(X_train, y_train):
    """Train and return an XGBoost-style gradient boosting model."""
    model = build_xgboost_model(y_train)
    model.fit(X_train, y_train)
    return model

