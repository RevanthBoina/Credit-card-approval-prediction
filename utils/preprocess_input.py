"""User-input preprocessing for Flask predictions."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from utils.config import MODEL_FEATURE_COLUMNS


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert a form value to float with a safe default."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    """Convert a form value to int with a safe default."""
    return int(round(_to_float(value, default)))


def preprocess_form_input(form_data: Mapping[str, object]) -> pd.DataFrame:
    """Convert submitted form data into the model feature dataframe.

    Args:
        form_data: Flask form data or any mapping with matching input names.

    Returns:
        A one-row DataFrame containing the same raw feature columns used during
        model training.
    """
    children = max(_to_int(form_data.get("CNT_CHILDREN"), 0), 0)
    income = max(_to_float(form_data.get("AMT_INCOME_TOTAL"), 0.0), 0.0)
    family_members = max(_to_float(form_data.get("CNT_FAM_MEMBERS"), 1.0), 1.0)
    age_years = max(_to_float(form_data.get("AGE_YEARS"), 18.0), 18.0)
    years_employed = max(_to_float(form_data.get("YEARS_EMPLOYED"), 0.0), 0.0)
    work_phone = _to_int(form_data.get("FLAG_WORK_PHONE"), 0)
    phone = _to_int(form_data.get("FLAG_PHONE"), 0)
    email = _to_int(form_data.get("FLAG_EMAIL"), 0)

    row = {
        "CODE_GENDER": str(form_data.get("CODE_GENDER", "F")),
        "FLAG_OWN_CAR": str(form_data.get("FLAG_OWN_CAR", "N")),
        "FLAG_OWN_REALTY": str(form_data.get("FLAG_OWN_REALTY", "Y")),
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": str(form_data.get("NAME_INCOME_TYPE", "Working")),
        "NAME_EDUCATION_TYPE": str(form_data.get("NAME_EDUCATION_TYPE", "Secondary / secondary special")),
        "NAME_FAMILY_STATUS": str(form_data.get("NAME_FAMILY_STATUS", "Married")),
        "NAME_HOUSING_TYPE": str(form_data.get("NAME_HOUSING_TYPE", "House / apartment")),
        "FLAG_WORK_PHONE": 1 if work_phone else 0,
        "FLAG_PHONE": 1 if phone else 0,
        "FLAG_EMAIL": 1 if email else 0,
        "OCCUPATION_TYPE": str(form_data.get("OCCUPATION_TYPE", "Unknown")),
        "CNT_FAM_MEMBERS": family_members,
        "AGE_YEARS": age_years,
        "YEARS_EMPLOYED": years_employed,
        "IS_EMPLOYED": 1 if years_employed > 0 else 0,
        "INCOME_PER_FAMILY_MEMBER": income / family_members,
        "CHILDREN_RATIO": min(children / family_members, 1.0),
        "HAS_CONTACT_METHOD": 1 if any([work_phone, phone, email]) else 0,
    }
    return pd.DataFrame([{column: row[column] for column in MODEL_FEATURE_COLUMNS}])


def get_form_options() -> dict[str, list[str]]:
    """Return categorical options used by the prediction form."""
    return {
        "CODE_GENDER": ["F", "M"],
        "FLAG_OWN_CAR": ["N", "Y"],
        "FLAG_OWN_REALTY": ["Y", "N"],
        "NAME_INCOME_TYPE": [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Student",
        ],
        "NAME_EDUCATION_TYPE": [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
        "NAME_FAMILY_STATUS": [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow",
        ],
        "NAME_HOUSING_TYPE": [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ],
        "OCCUPATION_TYPE": [
            "Unknown",
            "Laborers",
            "Core staff",
            "Sales staff",
            "Managers",
            "Drivers",
            "High skill tech staff",
            "Accountants",
            "Medicine staff",
            "Security staff",
            "Cooking staff",
            "Cleaning staff",
            "Private service staff",
            "Low-skill Laborers",
            "Waiters/barmen staff",
            "Secretaries",
            "Realty agents",
            "HR staff",
            "IT staff",
        ],
    }

