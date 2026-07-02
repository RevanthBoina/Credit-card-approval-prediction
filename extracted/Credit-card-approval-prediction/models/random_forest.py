"""Random Forest model training."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from utils.config import RANDOM_STATE


def build_random_forest_model() -> RandomForestClassifier:
    """Create a Random Forest classifier with conservative defaults."""
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train and return a Random Forest model."""
    model = build_random_forest_model()
    model.fit(X_train, y_train)
    return model

