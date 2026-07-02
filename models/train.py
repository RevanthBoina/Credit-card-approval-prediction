"""End-to-end training pipeline for the credit card approval model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from models.decision_tree import train_decision_tree
from models.logistic_regression import train_logistic_regression
from models.model_comparison import compare_models, select_best_model
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from preprocessing.encoding import fit_transform_features
from preprocessing.feature_engineering import create_modeling_dataset, split_features_and_target
from preprocessing.load_data import load_raw_datasets
from preprocessing.split_data import create_train_test_split
from utils.config import (
    ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
)


def save_training_artifacts(model, encoders, scaler, feature_columns: list[str], metrics: dict) -> None:
    """Persist the trained model, preprocessing objects, and metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    with Path(METRICS_PATH).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def train_pipeline() -> dict:
    """Run loading, feature engineering, training, evaluation, and artifact saving."""
    application_df, credit_df = load_raw_datasets()
    modeling_df = create_modeling_dataset(application_df, credit_df)
    features, target = split_features_and_target(modeling_df)
    transformed_features, encoders, scaler = fit_transform_features(features)
    feature_columns = transformed_features.columns.tolist()

    X_train, X_test, y_train, y_test = create_train_test_split(transformed_features, target)

    trained_models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Decision Tree": train_decision_tree(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    leaderboard, detailed_metrics = compare_models(trained_models, X_test, y_test)
    best_model_name, best_model = select_best_model(trained_models, leaderboard)
    metrics = {
        "best_model": best_model_name,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "detailed_metrics": detailed_metrics,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_columns": feature_columns,
        "target_distribution": target.value_counts().sort_index().to_dict(),
    }
    save_training_artifacts(best_model, encoders, scaler, feature_columns, metrics)
    return metrics


if __name__ == "__main__":
    pipeline_metrics = train_pipeline()
    print(json.dumps(pipeline_metrics["leaderboard"], indent=2))
    print(f"Best model: {pipeline_metrics['best_model']}")

