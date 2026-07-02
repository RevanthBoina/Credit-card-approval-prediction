"""Feature engineering and target construction."""

from __future__ import annotations

import pandas as pd

from preprocessing.cleaning import clean_application_data, clean_credit_data
from utils.config import MODEL_FEATURE_COLUMNS, STATUS_ORDER, TARGET_COLUMN


def create_credit_history_features(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly repayment history into one row per applicant.

    The output is used to derive the approval target and to support EDA. These
    fields should not be used as prediction features in the production model
    because they are derived from the same repayment statuses used for labels.
    """
    credit = clean_credit_data(credit_df)
    credit["OVERDUE_LEVEL"] = credit["STATUS"].map(STATUS_ORDER)
    credit["IS_PAID_ON_TIME"] = credit["STATUS"].eq("0").astype(int)
    credit["IS_CLOSED"] = credit["STATUS"].eq("C").astype(int)
    credit["IS_NO_LOAN"] = credit["STATUS"].eq("X").astype(int)
    credit["IS_LATE_1"] = credit["STATUS"].eq("1").astype(int)
    credit["IS_LATE_2_PLUS"] = credit["STATUS"].isin(["2", "3", "4", "5"]).astype(int)
    credit["IS_GOOD_STATUS"] = credit["STATUS"].isin(["X", "C", "0"]).astype(int)

    grouped = credit.groupby("ID").agg(
        TOTAL_CREDIT_MONTHS=("STATUS", "size"),
        MONTHS_PAID_ON_TIME=("IS_PAID_ON_TIME", "sum"),
        MONTHS_CLOSED=("IS_CLOSED", "sum"),
        MONTHS_NO_LOAN=("IS_NO_LOAN", "sum"),
        MONTHS_LATE_1=("IS_LATE_1", "sum"),
        MONTHS_LATE_2_PLUS=("IS_LATE_2_PLUS", "sum"),
        MAX_OVERDUE_STATUS=("OVERDUE_LEVEL", "max"),
        GOOD_MONTHS=("IS_GOOD_STATUS", "sum"),
    )

    grouped["EVER_LATE"] = ((grouped["MONTHS_LATE_1"] + grouped["MONTHS_LATE_2_PLUS"]) > 0).astype(int)
    grouped["EVER_SEVERELY_LATE"] = (grouped["MONTHS_LATE_2_PLUS"] > 0).astype(int)
    grouped["LATE_PAYMENT_RATIO"] = (
        (grouped["MONTHS_LATE_1"] + grouped["MONTHS_LATE_2_PLUS"]) / grouped["TOTAL_CREDIT_MONTHS"]
    ).round(4)
    grouped["GOOD_PAYMENT_RATIO"] = (grouped["GOOD_MONTHS"] / grouped["TOTAL_CREDIT_MONTHS"]).round(4)

    grouped[TARGET_COLUMN] = (
        (grouped["MONTHS_LATE_2_PLUS"] == 0) & (grouped["MONTHS_LATE_1"] < 3)
    ).astype(int)

    grouped = grouped.drop(columns=["GOOD_MONTHS"])
    return grouped.reset_index()


def create_modeling_dataset(application_df: pd.DataFrame, credit_df: pd.DataFrame) -> pd.DataFrame:
    """Create the merged modeling dataset with engineered target labels."""
    applications = clean_application_data(application_df)
    credit_features = create_credit_history_features(credit_df)
    merged = applications.merge(credit_features, on="ID", how="inner")
    merged = merged.dropna(subset=[TARGET_COLUMN])
    return merged


def split_features_and_target(modeling_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model input features and target without credit-history leakage."""
    available_columns = [column for column in MODEL_FEATURE_COLUMNS if column in modeling_df.columns]
    features = modeling_df[available_columns].copy()
    target = modeling_df[TARGET_COLUMN].astype(int).copy()
    return features, target
