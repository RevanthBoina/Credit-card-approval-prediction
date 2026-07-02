"""Flask web application entry point."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, redirect, render_template, request, session, url_for

from src.utils import (
    artifact_files_exist,
    append_prediction_history,
    get_form_options,
    predict_credit_card_approval,
)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "credit-card-approval-secret-key")

    @app.route("/")
    def home() -> Any:
        return render_template("home.html", artifacts_ready=artifact_files_exist())

    @app.route("/predict", methods=["GET", "POST"])
    def predict() -> Any:
        options = get_form_options()

        if request.method == "POST":
            if not artifact_files_exist():
                return render_template(
                    "predict.html",
                    options=options,
                    error="Model artifacts are missing. Run python -m src.model first.",
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
    def result() -> Any:
        result_data = session.get("prediction_result")
        form_data = session.get("prediction_form")
        if not result_data or not form_data:
            return redirect(url_for("predict"))
        return render_template("result.html", result=result_data, form=form_data)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=os.getenv("FLASK_DEBUG", "0") == "1")

