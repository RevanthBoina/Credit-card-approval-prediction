"""Persistent storage for submitted prediction requests."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from utils.config import PREDICTION_HISTORY_PATH, STORAGE_DIR


def _normalise_value(value: object) -> str:
    """Convert stored values to CSV-safe strings."""
    if value is None:
        return ""
    return str(value)


def build_history_record(form_data: Mapping[str, object], result: Mapping[str, object]) -> dict[str, str]:
    """Build one prediction-history row from applicant input and model output."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prediction_label": _normalise_value(result.get("label")),
        "approved": _normalise_value(result.get("approved")),
        "approval_probability": _normalise_value(result.get("probability")),
        "confidence_percent": _normalise_value(result.get("confidence_percent")),
    }
    for key, value in form_data.items():
        record[key] = _normalise_value(value)
    return record


def append_prediction_history(form_data: Mapping[str, object], result: Mapping[str, object]) -> Path:
    """Append the submitted applicant data and prediction result to CSV storage."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    record = build_history_record(form_data, result)
    file_exists = PREDICTION_HISTORY_PATH.exists()

    existing_columns: list[str] = []
    if file_exists:
        with PREDICTION_HISTORY_PATH.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            existing_columns = next(reader, [])

    fieldnames = existing_columns or list(record.keys())
    for column in record:
        if column not in fieldnames:
            fieldnames.append(column)

    rows: list[dict[str, str]] = []
    if file_exists and existing_columns != fieldnames:
        with PREDICTION_HISTORY_PATH.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    with PREDICTION_HISTORY_PATH.open("w" if rows else "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists or rows:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow(record)

    return PREDICTION_HISTORY_PATH


def read_prediction_history(limit: int = 100) -> list[dict[str, str]]:
    """Read recent prediction-history rows for display in the website."""
    if not PREDICTION_HISTORY_PATH.exists():
        return []
    dataframe = pd.read_csv(PREDICTION_HISTORY_PATH).fillna("")
    dataframe = dataframe.tail(limit).iloc[::-1]
    return dataframe.to_dict(orient="records")
