"""Dataset loading and inspection utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from utils.config import (
    APPLICATION_DATA_PATH,
    APPLICATION_REQUIRED_COLUMNS,
    CREDIT_DATA_PATH,
    CREDIT_REQUIRED_COLUMNS,
)


def validate_columns(dataframe: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    """Validate that a dataframe contains all required columns.

    Args:
        dataframe: DataFrame to validate.
        required_columns: Columns expected by the pipeline.
        dataset_name: Human-readable dataset name used in error messages.

    Raises:
        ValueError: If any required column is missing.
    """
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


def read_application_data(path: str | None = None) -> pd.DataFrame:
    """Read applicant demographic and financial data from CSV."""
    csv_path = path or APPLICATION_DATA_PATH
    dataframe = pd.read_csv(csv_path)
    validate_columns(dataframe, APPLICATION_REQUIRED_COLUMNS, "application_record.csv")
    return dataframe


def read_credit_data(path: str | None = None) -> pd.DataFrame:
    """Read monthly repayment history data from CSV."""
    csv_path = path or CREDIT_DATA_PATH
    dataframe = pd.read_csv(csv_path)
    validate_columns(dataframe, CREDIT_REQUIRED_COLUMNS, "credit_record.csv")
    return dataframe


def load_raw_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw project datasets."""
    return read_application_data(), read_credit_data()


def dataset_overview(dataframe: pd.DataFrame) -> dict[str, object]:
    """Return a compact overview used by notebooks and CLI checks."""
    return {
        "shape": dataframe.shape,
        "columns": dataframe.columns.tolist(),
        "head": dataframe.head(),
        "tail": dataframe.tail(),
        "info": dataframe.info(),
        "describe": dataframe.describe(include="all").transpose(),
        "missing_values": dataframe.isna().sum(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def print_dataset_overview(dataframe: pd.DataFrame, dataset_name: str) -> None:
    """Print shape, columns, sample rows, summary statistics, and quality checks."""
    print(f"\n========== {dataset_name} ==========")
    print(f"Shape: {dataframe.shape}")
    print("\nColumns:")
    print(dataframe.columns.tolist())
    print("\nHead:")
    print(dataframe.head())
    print("\nTail:")
    print(dataframe.tail())
    print("\nInfo:")
    dataframe.info()
    print("\nDescribe:")
    print(dataframe.describe(include="all").transpose())
    print("\nMissing values:")
    print(dataframe.isna().sum())
    print(f"\nDuplicate rows: {dataframe.duplicated().sum()}")


if __name__ == "__main__":
    application_df, credit_df = load_raw_datasets()
    print_dataset_overview(application_df, "Application Record")
    print_dataset_overview(credit_df, "Credit Record")

