"""Cleaning routines for application and credit datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import STATUS_ORDER


def clean_application_data(application_df: pd.DataFrame) -> pd.DataFrame:
    """Clean applicant records and create stable application-level fields.

    Cleaning decisions:
    - Exact duplicates are removed because repeated identical rows add no signal.
    - Repeated applicant IDs are reduced to the first record after duplicate
      removal, keeping one application profile per applicant.
    - Missing occupation is filled with "Unknown" because absence of occupation
      can be meaningful and dropping those rows would remove too much data.
    - DAYS_BIRTH is converted from negative days to positive age in years.
    - DAYS_EMPLOYED is converted to positive years; the sentinel 365243 is
      treated as not currently employed or unknown employment duration.
    - FLAG_MOBIL is removed because it is constant in this dataset.
    """
    cleaned = application_df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values("ID").drop_duplicates(subset="ID", keep="first")

    cleaned["OCCUPATION_TYPE"] = cleaned["OCCUPATION_TYPE"].fillna("Unknown")

    cleaned["AGE_YEARS"] = (-cleaned["DAYS_BIRTH"] / 365.25).round(2)

    employed_mask = cleaned["DAYS_EMPLOYED"] < 0
    cleaned["IS_EMPLOYED"] = employed_mask.astype(int)
    cleaned["YEARS_EMPLOYED"] = np.where(employed_mask, -cleaned["DAYS_EMPLOYED"] / 365.25, 0.0)
    cleaned["YEARS_EMPLOYED"] = cleaned["YEARS_EMPLOYED"].round(2)

    cleaned["CNT_CHILDREN"] = cleaned["CNT_CHILDREN"].clip(lower=0)
    cleaned["CNT_FAM_MEMBERS"] = cleaned["CNT_FAM_MEMBERS"].clip(lower=1)
    cleaned["AMT_INCOME_TOTAL"] = cleaned["AMT_INCOME_TOTAL"].clip(lower=0)

    cleaned["INCOME_PER_FAMILY_MEMBER"] = (
        cleaned["AMT_INCOME_TOTAL"] / cleaned["CNT_FAM_MEMBERS"].replace(0, np.nan)
    ).fillna(cleaned["AMT_INCOME_TOTAL"])
    cleaned["CHILDREN_RATIO"] = cleaned["CNT_CHILDREN"] / cleaned["CNT_FAM_MEMBERS"].replace(0, np.nan)
    cleaned["CHILDREN_RATIO"] = cleaned["CHILDREN_RATIO"].fillna(0).clip(lower=0, upper=1)
    cleaned["HAS_CONTACT_METHOD"] = (
        (cleaned["FLAG_WORK_PHONE"] == 1) | (cleaned["FLAG_PHONE"] == 1) | (cleaned["FLAG_EMAIL"] == 1)
    ).astype(int)

    cleaned = cleaned.drop(columns=["FLAG_MOBIL", "DAYS_BIRTH", "DAYS_EMPLOYED"], errors="ignore")
    return cleaned


def clean_credit_data(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Clean credit history records and validate repayment status values."""
    cleaned = credit_df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned["STATUS"] = cleaned["STATUS"].astype(str).str.strip()

    invalid_statuses = sorted(set(cleaned["STATUS"]) - set(STATUS_ORDER))
    if invalid_statuses:
        raise ValueError(f"Unexpected credit STATUS values found: {invalid_statuses}")

    cleaned["MONTHS_BALANCE"] = pd.to_numeric(cleaned["MONTHS_BALANCE"], errors="coerce")
    cleaned = cleaned.dropna(subset=["ID", "MONTHS_BALANCE", "STATUS"])
    return cleaned

