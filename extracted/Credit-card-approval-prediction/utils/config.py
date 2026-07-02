"""Central configuration for paths, columns, and modeling rules.

This module intentionally keeps project constants in one place so training,
prediction, notebooks, and documentation use the same definitions.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
APPLICATION_DATA_PATH = DATASET_DIR / "application_record.csv"
CREDIT_DATA_PATH = DATASET_DIR / "credit_record.csv"

MODEL_DIR = BASE_DIR / "models" / "saved"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoders.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

STORAGE_DIR = BASE_DIR / "storage"
PREDICTION_HISTORY_PATH = STORAGE_DIR / "prediction_history.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

APPLICATION_REQUIRED_COLUMNS = [
    "ID",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
]

CREDIT_REQUIRED_COLUMNS = ["ID", "MONTHS_BALANCE", "STATUS"]

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

NUMERIC_COLUMNS = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "CNT_FAM_MEMBERS",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "IS_EMPLOYED",
    "INCOME_PER_FAMILY_MEMBER",
    "CHILDREN_RATIO",
    "HAS_CONTACT_METHOD",
]

TARGET_COLUMN = "APPROVED"

# The target is derived from credit STATUS. These summary columns are useful for
# reports, but excluded from training to avoid target leakage.
CREDIT_LEAKAGE_COLUMNS = [
    "TOTAL_CREDIT_MONTHS",
    "MONTHS_PAID_ON_TIME",
    "MONTHS_CLOSED",
    "MONTHS_NO_LOAN",
    "MONTHS_LATE_1",
    "MONTHS_LATE_2_PLUS",
    "MAX_OVERDUE_STATUS",
    "EVER_LATE",
    "EVER_SEVERELY_LATE",
    "LATE_PAYMENT_RATIO",
    "GOOD_PAYMENT_RATIO",
]

MODEL_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

STATUS_ORDER = {"X": -1, "C": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
