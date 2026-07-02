"""Prediction utilities for the Flask application."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from preprocessing.encoding import transform_features
from utils.config import ENCODER_PATH, FEATURE_COLUMNS_PATH, MODEL_PATH, SCALER_PATH
from utils.preprocess_input import preprocess_form_input


@lru_cache(maxsize=1)
def load_prediction_artifacts() -> tuple[Any, dict, Any, list[str]]:
    """Load trained model and preprocessing artifacts from disk."""
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, encoders, scaler, feature_columns


def artifact_files_exist() -> bool:
    """Check whether all prediction artifacts are available."""
    return all(path.exists() for path in [MODEL_PATH, ENCODER_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH])


def predict_credit_card_approval(form_data) -> dict[str, object]:
    """Predict approval status from submitted applicant form data."""
    model, encoders, scaler, feature_columns = load_prediction_artifacts()
    raw_features = preprocess_form_input(form_data)
    transformed_features = transform_features(raw_features, encoders, scaler, feature_columns)
    prediction = int(model.predict(transformed_features)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(transformed_features)[0][1])

    return {
        "approved": bool(prediction == 1),
        "label": "Approved" if prediction == 1 else "Rejected",
        "probability": probability,
        "confidence_percent": round(probability * 100, 2) if probability is not None else None,
    }

