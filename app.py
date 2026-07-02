"""Flask application for credit card approval prediction."""

from __future__ import annotations

import os

from flask import Flask, redirect, render_template, request, session, url_for

from utils.predict import artifact_files_exist, predict_credit_card_approval
from utils.prediction_history import append_prediction_history
from utils.preprocess_input import get_form_options


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "credit-card-approval-secret-key")

    @app.route("/")
    def home():
        """Render the application home screen."""
        return render_template("home.html", artifacts_ready=artifact_files_exist())

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        """Render the prediction form and process submitted applications."""
        options = get_form_options()
        if request.method == "POST":
            if not artifact_files_exist():
                return render_template(
                    "predict.html",
                    options=options,
                    error="Model artifacts are missing. Run python -m models.train first.",
                )
            result = predict_credit_card_approval(request.form)
            form_data = request.form.to_dict(flat=True)
            for checkbox_name in ["FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"]:
                form_data.setdefault(checkbox_name, "0")
            append_prediction_history(form_data, result)
            session["prediction_result"] = result
            session["prediction_form"] = form_data
            return redirect(url_for("result"))
        return render_template("predict.html", options=options, error=None)

    @app.route("/result")
    def result():
        """Render the latest prediction result without resubmitting on refresh."""
        result_data = session.get("prediction_result")
        form_data = session.get("prediction_form")
        if not result_data or not form_data:
            return redirect(url_for("predict"))
        return render_template("result.html", result=result_data, form=form_data)

    return app


app = create_app()


if __name__ == "__main__":
    # IBM Cloud (Cloud Foundry / Code Engine) injects the port via $PORT.
    port = int(os.getenv("PORT", "5000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    app.run(host=host, port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
