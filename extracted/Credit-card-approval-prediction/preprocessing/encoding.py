"""Encoding and scaling helpers shared by training and inference."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def fit_label_encoders(features: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit label encoders for all categorical columns present in features."""
    encoders: dict[str, LabelEncoder] = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in features.columns:
            continue
        encoder = LabelEncoder()
        values = features[column].fillna("Unknown").astype(str)
        if "Unknown" not in set(values):
            values = pd.concat([values, pd.Series(["Unknown"])], ignore_index=True)
        encoder.fit(values)
        encoders[column] = encoder
    return encoders


def transform_with_label_encoders(features: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """Transform categorical columns while mapping unseen values to Unknown."""
    transformed = features.copy()
    for column, encoder in encoders.items():
        if column not in transformed.columns:
            transformed[column] = "Unknown"
        known_values = set(encoder.classes_)
        values = transformed[column].fillna("Unknown").astype(str)
        values = values.where(values.isin(known_values), "Unknown")
        transformed[column] = encoder.transform(values)
    return transformed


def prepare_numeric_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric model columns to numeric values and fill missing values."""
    prepared = features.copy()
    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            prepared[column] = prepared[column].fillna(prepared[column].median())
    return prepared


def fit_transform_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder], StandardScaler]:
    """Fit encoders and scaler, then return transformed training features."""
    prepared = prepare_numeric_columns(features)
    encoders = fit_label_encoders(prepared)
    encoded = transform_with_label_encoders(prepared, encoders)
    scaler = StandardScaler()
    encoded[encoded.columns] = scaler.fit_transform(encoded[encoded.columns])
    return encoded, encoders, scaler


def transform_features(
    features: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    scaler: StandardScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Apply saved encoders and scaler to new feature rows."""
    aligned = features.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = "Unknown" if column in CATEGORICAL_COLUMNS else 0
    aligned = aligned[feature_columns]
    aligned = prepare_numeric_columns(aligned)
    encoded = transform_with_label_encoders(aligned, encoders)
    scaled_values = scaler.transform(encoded[feature_columns])
    return pd.DataFrame(scaled_values, columns=feature_columns, index=encoded.index)


def describe_artifacts(encoders: dict[str, Any], scaler: StandardScaler, feature_columns: list[str]) -> dict[str, Any]:
    """Return a serializable description of preprocessing artifacts."""
    return {
        "encoded_columns": sorted(encoders.keys()),
        "scaled_columns": feature_columns,
        "scaler_type": scaler.__class__.__name__,
    }

