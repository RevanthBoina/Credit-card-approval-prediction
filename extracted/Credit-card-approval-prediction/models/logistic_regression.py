"""Logistic Regression model training."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from utils.config import RANDOM_STATE


def build_logistic_regression_model() -> LogisticRegression:
    """Create a Logistic Regression classifier configured for imbalanced data."""
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train and return a Logistic Regression model."""
    model = build_logistic_regression_model()
    model.fit(X_train, y_train)
    return model

