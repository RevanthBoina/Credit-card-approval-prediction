"""Model evaluation and comparison utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _positive_class_scores(model: Any, X_test) -> list[float] | None:
    """Return positive-class probabilities or decision scores when available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def evaluate_model(model: Any, X_test, y_test) -> dict[str, Any]:
    """Evaluate a classifier with standard binary classification metrics."""
    predictions = model.predict(X_test)
    scores = _positive_class_scores(model, X_test)
    roc_auc = roc_auc_score(y_test, scores) if scores is not None else None

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def compare_models(models: dict[str, Any], X_test, y_test) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate multiple models and return a leaderboard plus detailed metrics."""
    detailed_metrics = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}
    rows = []
    for model_name, metrics in detailed_metrics.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"] if metrics["roc_auc"] is not None else 0.0,
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(["f1_score", "roc_auc"], ascending=False).reset_index(drop=True)
    return leaderboard, detailed_metrics


def select_best_model(models: dict[str, Any], leaderboard: pd.DataFrame) -> tuple[str, Any]:
    """Select the best model by F1-score and ROC-AUC."""
    best_model_name = str(leaderboard.iloc[0]["model"])
    return best_model_name, models[best_model_name]

