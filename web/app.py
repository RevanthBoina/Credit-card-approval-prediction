import os
import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "final_credit_model_pipeline.pkl")

_model = None

def get_model():
    """Lazy-load the model once and cache it in the module-level variable."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Train the model using train_approval_model.py to generate it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

# ---------------------------------------------------------------------------
# Feature columns – must match the order used during training exactly.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 
    'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 
    'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS'
]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")

@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  – render the prediction form.
    POST – extract form fields, run model inference, render result inline.
    """
    if request.method == "POST":
        errors = {}
        form_values = {}
        raw_values = {}

        # 1. Categorical fields
        categorical_mappings = {
            'gender': ('CODE_GENDER', {'Male': 'M', 'Female': 'F'}),
            'own_car': ('FLAG_OWN_CAR', {'Yes': 'Y', 'No': 'N'}),
            'own_realty': ('FLAG_OWN_REALTY', {'Yes': 'Y', 'No': 'N'}),
            'income_type': ('NAME_INCOME_TYPE', None),
            'education': ('NAME_EDUCATION_TYPE', None),
            'family_status': ('NAME_FAMILY_STATUS', None),
            'housing_type': ('NAME_HOUSING_TYPE', None),
            'occupation': ('OCCUPATION_TYPE', None),
        }

        for form_key, (col_name, mapping) in categorical_mappings.items():
            val = request.form.get(form_key, "").strip()
            form_values[form_key] = val
            if val == "":
                errors[form_key] = "This field is required."
            else:
                if mapping:
                    raw_values[col_name] = mapping.get(val, val)
                else:
                    raw_values[col_name] = val

        # 2. Numerical fields
        numerical_inputs = {
            'num_children': ('CNT_CHILDREN', int, 0),
            'annual_income': ('AMT_INCOME_TOTAL', float, 0),
            'family_members': ('CNT_FAM_MEMBERS', float, 1),
        }

        for form_key, (col_name, val_type, min_val) in numerical_inputs.items():
            val = request.form.get(form_key, "").strip()
            form_values[form_key] = val
            if val == "":
                errors[form_key] = "This field is required."
            else:
                try:
                    num_val = val_type(val)
                    if num_val < min_val:
                        errors[form_key] = f"Value must be at least {min_val}."
                    else:
                        raw_values[col_name] = num_val
                except ValueError:
                    errors[form_key] = "Must be a valid number."

        # 3. Special transformations (Age and Employment Years)
        # Age -> DAYS_BIRTH (-Age * 365.25)
        age_val = request.form.get('age', "").strip()
        form_values['age'] = age_val
        if age_val == "":
            errors['age'] = "Age is required."
        else:
            try:
                age_float = float(age_val)
                if age_float < 18:
                    errors['age'] = "Age must be at least 18."
                else:
                    raw_values['DAYS_BIRTH'] = -age_float * 365.25
            except ValueError:
                errors['age'] = "Must be a valid number."

        # Employment years -> DAYS_EMPLOYED (-Years * 365.25)
        emp_val = request.form.get('employment_years', "").strip()
        form_values['employment_years'] = emp_val
        if emp_val == "":
            errors['employment_years'] = "Employment duration is required."
        else:
            try:
                emp_float = float(emp_val)
                if emp_float < 0:
                    errors['employment_years'] = "Cannot be negative."
                else:
                    raw_values['DAYS_EMPLOYED'] = -emp_float * 365.25
            except ValueError:
                errors['employment_years'] = "Must be a valid number."

        # 4. Binary flag inputs (mapped to 1/0)
        flag_inputs = {
            'mobile': ('FLAG_MOBIL', {'Yes': 1, 'No': 0}),
            'work_phone': ('FLAG_WORK_PHONE', {'Yes': 1, 'No': 0}),
            'phone': ('FLAG_PHONE', {'Yes': 1, 'No': 0}),
            'email': ('FLAG_EMAIL', {'Yes': 1, 'No': 0}),
        }

        for form_key, (col_name, mapping) in flag_inputs.items():
            val = request.form.get(form_key, "").strip()
            form_values[form_key] = val
            if val == "":
                errors[form_key] = "This field is required."
            else:
                raw_values[col_name] = mapping.get(val, 0)

        if errors:
            return render_template(
                "predict.html",
                errors=errors,
                form=form_values,
            )

        # Construct DataFrame in the exact same column order as FEATURE_COLUMNS
        input_df = pd.DataFrame([raw_values], columns=FEATURE_COLUMNS)

        try:
            model = get_model()
            prediction_int = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])
        except FileNotFoundError as exc:
            return render_template(
                "predict.html",
                model_error=str(exc),
                form=form_values,
            )
        except Exception as exc:
            return render_template(
                "predict.html",
                model_error=f"Prediction failed: {exc}",
                form=form_values,
            )

        # Mapping output class: 1 -> Approved, 0 -> Rejected
        label = "Approved" if prediction_int == 1 else "Rejected"
        label_class = "result--safe" if prediction_int == 1 else "result--fraud"

        return render_template(
            "result.html",
            prediction=label,
            probability=round(probability, 4),
            income=raw_values.get("AMT_INCOME_TOTAL"),
            label_class=label_class,
        )

    return render_template("predict.html")

@app.route("/result")
def result():
    """Direct GET of /result redirects back to the form."""
    return redirect(url_for('predict'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
