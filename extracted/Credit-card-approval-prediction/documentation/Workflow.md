# Project Workflow

## 1. Load Data

The pipeline reads `application_record.csv` and `credit_record.csv` from the `dataset/` directory. Required columns are validated before processing begins.

## 2. Inspect Data

The loading module supports shape, columns, head, tail, info, describe, missing values, and duplicate checks.

## 3. Clean Application Records

- Remove exact duplicate rows.
- Keep one row per applicant `ID`.
- Fill missing `OCCUPATION_TYPE` with `Unknown`.
- Convert `DAYS_BIRTH` into `AGE_YEARS`.
- Convert valid negative `DAYS_EMPLOYED` into `YEARS_EMPLOYED`.
- Convert the `365243` employment sentinel into zero employment years and `IS_EMPLOYED = 0`.
- Remove `FLAG_MOBIL` because it is constant.

## 4. Engineer Credit Labels

The monthly credit table is grouped by `ID`. For each applicant, the pipeline calculates late-payment counts, good-payment ratio, maximum overdue level, and final approval label.

## 5. Merge Data

The cleaned application table is inner-joined with applicant-level credit features on `ID`. Only applicants present in both files are used for supervised learning.

## 6. Prevent Leakage

Credit history features are not used as model inputs because they directly influence the target label. The model receives only applicant profile features available at application time.

## 7. Encode And Scale

Categorical columns are label encoded. Numeric and encoded columns are scaled with `StandardScaler` for consistent model input.

## 8. Train And Compare

The pipeline trains Logistic Regression, Decision Tree, Random Forest, and XGBoost. The comparison report includes accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report.

## 9. Save Artifacts

The best model and preprocessing artifacts are saved under `models/saved/` for Flask inference.

## 10. Serve Predictions

The Flask app loads saved artifacts once, preprocesses submitted form input, and returns an approval decision with probability when the selected model supports it.

