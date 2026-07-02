# Credit Card Approval Prediction System

A production-style machine learning and Flask project that predicts credit card approval from applicant demographic, financial, housing, employment, and contact information. The target label is engineered from repayment history in `credit_record.csv`; it is not assumed to exist.

## Project Structure

```text
dataset/                 Raw CSV datasets
notebooks/               Exploratory data analysis notebook
preprocessing/           Loading, cleaning, feature engineering, encoding, splitting
models/                  Model training, comparison, and saved artifacts
templates/               Flask HTML templates
static/                  CSS, JavaScript, and image assets
utils/                   App configuration and prediction utilities
documentation/           Report, workflow, ER diagram, deployment notes
app.py                   Flask entry point
requirements.txt         Python dependencies
```

## Dataset

The system uses:

- `application_record.csv`: applicant demographic and financial profile.
- `credit_record.csv`: monthly repayment history with `STATUS`.

The two files are merged by `ID` after credit history is grouped to applicant level.

## Target Engineering

`STATUS` values are converted into an applicant-level approval label:

- Good or neutral repayment: `X`, `C`, `0`
- Mild overdue: `1`
- Severe overdue: `2`, `3`, `4`, `5`

The final label is:

- `1`: approved applicant
- `0`: rejected or risky applicant

Applicants with any severe overdue month are marked risky. Applicants with three or more mild overdue months are also marked risky.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train The Model

```bash
python -m models.train
```

This creates:

- `models/saved/model.pkl`
- `models/saved/label_encoders.pkl`
- `models/saved/scaler.pkl`
- `models/saved/feature_columns.pkl`
- `models/saved/metrics.json`

Prediction submissions are stored for future reference in:

- `storage/prediction_history.csv`

Prediction records are stored internally for reference.

## Run The Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost, with a scikit-learn fallback if XGBoost is unavailable

The best model is selected by F1-score and ROC-AUC.

## Important Design Choice

Credit repayment aggregates are used to create the target label and support EDA. They are intentionally excluded from the model input features to prevent target leakage.
