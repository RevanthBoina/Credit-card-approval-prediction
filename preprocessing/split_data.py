"""Train-test splitting helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import RANDOM_STATE, TEST_SIZE


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train-test split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

