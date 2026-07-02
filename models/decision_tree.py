"""Decision Tree model training."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from utils.config import RANDOM_STATE


def build_decision_tree_model() -> DecisionTreeClassifier:
    """Create a regularized Decision Tree classifier."""
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_leaf=25,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Train and return a Decision Tree model."""
    model = build_decision_tree_model()
    model.fit(X_train, y_train)
    return model

