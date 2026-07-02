"""Utilities for prediction and Flask form handling."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import joblib
import pandas as pd

from extracted.Credit-card-approval-prediction.utils.config import (  # type: ignore
    ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    MODEL_FEATURE_COLUMNS,
)
from extracted.Credit-card-approval-prediction.utils.predict import (  # type: ignore
    artifact_files_exist,
    predict_credit_card_approval,
)
from extracted.Credit-card-approval-prediction.utils.prediction_history import (  # type: ignore
    append_prediction_history,
)
from extracted.Credit-card-approval-prediction.utils.preprocess_input import (  # type: ignore
    get_form_options,
    preprocess_form_input,
)


__all__ = [
    "artifact_files_exist",
    "predict_credit_card_approval",
    "append_prediction_history",
    "get_form_options",
]


