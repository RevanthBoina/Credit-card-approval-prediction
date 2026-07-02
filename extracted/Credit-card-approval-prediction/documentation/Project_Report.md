# Credit Card Approval Prediction System - Project Report

## Objective

The objective is to build a machine learning system that predicts whether an applicant should be approved for a credit card using demographic, financial, employment, housing, and contact information.

## Data Sources

Two datasets are used:

- `application_record.csv`: one or more application profile rows per applicant.
- `credit_record.csv`: monthly repayment behavior per applicant.

The target variable is not directly available. It is engineered from `credit_record.csv`.

## Data Understanding

`application_record.csv` includes gender, income, education, family status, housing type, employment days, birth days, occupation, children, and family member counts.

`credit_record.csv` includes `ID`, `MONTHS_BALANCE`, and `STATUS`. Each applicant can have multiple monthly credit records, so credit data must be grouped before merging.

## Target Variable Creation

The `STATUS` field is mapped into repayment severity:

| Status | Meaning | Risk Level |
|---|---|---|
| X | No loan | Neutral |
| C | Loan closed | Good |
| 0 | Paid on time | Good |
| 1 | 1-30 days overdue | Mild risk |
| 2-5 | 31+ days overdue | High risk |

The applicant is rejected if they ever have severe overdue behavior or repeated mild overdue behavior. Otherwise, the applicant is approved.

## Cleaning

The cleaning process removes duplicates, fills missing occupations, handles invalid values, converts negative day fields into positive year fields, and removes constant columns.

## Feature Engineering

Application features:

- `AGE_YEARS`
- `YEARS_EMPLOYED`
- `IS_EMPLOYED`
- `INCOME_PER_FAMILY_MEMBER`
- `CHILDREN_RATIO`
- `HAS_CONTACT_METHOD`

Credit history features:

- `TOTAL_CREDIT_MONTHS`
- `MONTHS_PAID_ON_TIME`
- `MONTHS_LATE_1`
- `MONTHS_LATE_2_PLUS`
- `MAX_OVERDUE_STATUS`
- `LATE_PAYMENT_RATIO`
- `GOOD_PAYMENT_RATIO`

Credit history features are used for label engineering and analysis, not for production prediction inputs.

## Models

The following models are trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## System Design

The project follows clean modular architecture. Data loading, cleaning, feature engineering, encoding, splitting, model training, prediction utilities, Flask routes, frontend templates, and documentation are separated into independent modules.

## Conclusion

The final system provides a reproducible ML pipeline and a web interface for approval prediction. The project is ready for local execution, GitHub submission, and IBM Cloud style deployment.

