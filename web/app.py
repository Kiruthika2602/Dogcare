# web/app.py
"""
Flask app for DogCare AI with BMI + Risk endpoint.

Run with:
    python -m web.app

Features:
 - Runs global evaluator at startup (saved_models/metrics.json)
 - Ensures BMI risk models/metrics exist (saved_models/bmi_metrics.json)
 - Exposes pages: /, /weight, /calorie, /activity, /bmi
"""

import os
import json
import runpy
from flask import Flask, render_template, request, flash

# Project root (one level above web/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Try to run the global evaluator module at startup (so saved_models/metrics.json exists)
try:
    runpy.run_module("src.models.evaluate_and_save_metrics", run_name="__main__")
except Exception as e:
    # Don't crash the app if evaluator fails — log the error
    print("Warning: running global evaluator at startup failed:", e)

# Ensure BMI models/metrics exist by running bmi module if needed
try:
    # This will train/save BMI models and metrics if missing
    runpy.run_module("src.models.bmi_risk", run_name="__main__")
except Exception as e:
    print("Warning: running bmi_risk trainer at startup failed (if files already exist, it's ok):", e)

# Now import predictor functions (after the optional model creation)
from src.models.predictor import predict_weight_only, predict_calorie_only, predict_activity_only
from src.models.bmi_risk import predict_bmi_and_risks, METRICS_JSON as BMI_METRICS_JSON
# near other imports
from src.models.life_activity import predict_life_expectancy, predict_activity_recommendation

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "change-me"

METRICS_PATH = os.path.join(PROJECT_ROOT, "saved_models", "metrics.json")
BMI_METRICS_PATH = os.path.join(PROJECT_ROOT, "saved_models", "bmi_metrics.json")

def load_json(path):
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_metrics():
    return load_json(METRICS_PATH)

def load_bmi_metrics():
    # load bmi_metrics.json created by src/models/bmi_risk
    return load_json(BMI_METRICS_PATH)

# -------------------------
# Index / Home (endpoint name: index)
# -------------------------
@app.route("/")
def index():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics)

# -------------------------
# Weight page
# -------------------------
@app.route("/weight", methods=["GET", "POST"])
def weight_page():
    metrics = load_metrics().get("weight_model", {})
    if request.method == "POST":
        data = {
            "breed": request.form.get("breed"),
            "sex": request.form.get("sex"),
            "age_months": request.form.get("age_months") or None,
            "height_cm": request.form.get("height_cm") or None,
            "length_cm": request.form.get("length_cm") or None,
            "avg_daily_play_min": request.form.get("avg_daily_play_min") or 0,
            "avg_daily_walk_min": request.form.get("avg_daily_walk_min") or 0,
            "sleep_hours": request.form.get("sleep_hours") or 0,
            "weight_kg": None
        }
        # convert numeric fields to floats where provided
        for k in ["age_months", "height_cm", "length_cm", "avg_daily_play_min", "avg_daily_walk_min", "sleep_hours"]:
            if data.get(k) is not None and data.get(k) != "":
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = None
        out = predict_weight_only(data)
        mae = metrics.get("mae")
        ci_low = out["weight_kg"] - mae if mae is not None else None
        ci_high = out["weight_kg"] + mae if mae is not None else None
        return render_template("weight_result.html", result=out, submitted=data, metrics=metrics, ci=(ci_low, ci_high))
    return render_template("weight.html", metrics=metrics)

# -------------------------
# Calorie page
# -------------------------
@app.route("/calorie", methods=["GET", "POST"])
def calorie_page():
    metrics = load_metrics().get("calorie_model", {})
    if request.method == "POST":
        data = {
            "breed": request.form.get("breed"),
            "sex": request.form.get("sex"),
            "age_months": request.form.get("age_months") or None,
            "height_cm": request.form.get("height_cm") or None,
            "length_cm": request.form.get("length_cm") or None,
            "avg_daily_play_min": request.form.get("avg_daily_play_min") or 0,
            "avg_daily_walk_min": request.form.get("avg_daily_walk_min") or 0,
            "sleep_hours": request.form.get("sleep_hours") or 0,
            "weight_kg": request.form.get("weight_kg") or None
        }
        for k in ["age_months", "height_cm", "length_cm", "avg_daily_play_min", "avg_daily_walk_min", "sleep_hours", "weight_kg"]:
            if data.get(k) is not None and data.get(k) != "":
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = None
        out = predict_calorie_only(data)
        mae = metrics.get("mae")
        ci_low = out["calories_per_day"] - mae if mae is not None else None
        ci_high = out["calories_per_day"] + mae if mae is not None else None
        return render_template("calorie_result.html", result=out, submitted=data, metrics=metrics, ci=(ci_low, ci_high))
    return render_template("calorie.html", metrics=metrics)

# -------------------------
# Activity page
# -------------------------
@app.route("/activity", methods=["GET", "POST"])
def activity_page():
    metrics = load_metrics().get("activity_cluster", {})
    if request.method == "POST":
        data = {
            "breed": request.form.get("breed"),
            "sex": request.form.get("sex"),
            "age_months": request.form.get("age_months") or None,
            "avg_daily_play_min": request.form.get("avg_daily_play_min") or 0,
            "avg_daily_walk_min": request.form.get("avg_daily_walk_min") or 0,
            "sleep_hours": request.form.get("sleep_hours") or 0,
            "weight_kg": None
        }
        for k in ["age_months", "avg_daily_play_min", "avg_daily_walk_min", "sleep_hours"]:
            if data.get(k) is not None and data.get(k) != "":
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = None
        out = predict_activity_only(data)
        return render_template("activity_result.html", result=out, submitted=data, metrics=metrics)
    return render_template("activity.html", metrics=metrics)

# -------------------------
# BMI + Risk page
# -------------------------
@app.route("/bmi", methods=["GET", "POST"])
def bmi_page():
    # bmi_metrics loads the evaluation metrics saved by the bmi_risk module
    bmi_metrics = load_bmi_metrics()
    # The UI will use bmi_metrics to show model performance for obesity/joint/heart risk models
    if request.method == "POST":
        # collect and coerce inputs
        data = {
            "weight_kg": request.form.get("weight_kg") or None,
            "height_cm": request.form.get("height_cm") or None,
            "age_years": request.form.get("age_years") or None,
            "avg_daily_play_min": request.form.get("avg_daily_play_min") or 0,
            "avg_daily_walk_min": request.form.get("avg_daily_walk_min") or 0,
            "size_group": request.form.get("size_group") or None,
            "sex": request.form.get("sex") or None
        }
        # convert numeric
        for k in ["weight_kg", "height_cm", "age_years", "avg_daily_play_min", "avg_daily_walk_min"]:
            if data.get(k) is not None and data.get(k) != "":
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = None

        # Predict using bmi_risk
        try:
            out = predict_bmi_and_risks(data)
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")
            return render_template("bmi.html", metrics=bmi_metrics, submitted=data)

        # expose metrics (for each binary classifier we have accuracy/precision/recall/f1/roc_auc/n_test)
        return render_template("bmi_result.html", result=out, submitted=data, metrics=bmi_metrics)

    return render_template("bmi.html", metrics=bmi_metrics)

# Life Expectancy page
@app.route("/life", methods=["GET", "POST"])
def life_page():
    # load metrics for display (optional)
    try:
        with open(os.path.join(PROJECT_ROOT, "saved_models", "life_activity_metrics.json"), "r", encoding="utf8") as f:
            life_metrics = json.load(f).get("life", {})
    except:
        life_metrics = {}
    if request.method == "POST":
        data = {
            "age_years": request.form.get("age_years") or None,
            "height_cm": request.form.get("height_cm") or None,
            "length_cm": request.form.get("length_cm") or None,
            "weight_kg": request.form.get("weight_kg") or None,
            "avg_daily_play_min": request.form.get("avg_daily_play_min") or 0,
            "avg_daily_walk_min": request.form.get("avg_daily_walk_min") or 0,
            "sleep_hours": request.form.get("sleep_hours") or 0,
            "size_group": request.form.get("size_group") or None
        }
        # cast numeric
        for k in ["age_years","height_cm","length_cm","weight_kg","avg_daily_play_min","avg_daily_walk_min","sleep_hours"]:
            if data.get(k) not in (None, ""):
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = None
        out = predict_life_expectancy(data)
        return render_template("life_result.html", result=out, submitted=data, metrics=life_metrics)
    return render_template("life_form.html", metrics=life_metrics)

# Activity recommendation page
@app.route("/activity-reco", methods=["GET", "POST"])
def activity_reco_page():
    try:
        with open(os.path.join(PROJECT_ROOT, "saved_models", "life_activity_metrics.json"), "r", encoding="utf8") as f:
            act_metrics = json.load(f).get("activity", {})
    except:
        act_metrics = {}
    if request.method == "POST":
        data = {
            "age_years": request.form.get("age_years") or None,
            "height_cm": request.form.get("height_cm") or None,
            "length_cm": request.form.get("length_cm") or None,
            "weight_kg": request.form.get("weight_kg") or None,
            "avg_daily_play_min": request.form.get("avg_daily_play_min") or 0,
            "avg_daily_walk_min": request.form.get("avg_daily_walk_min") or 0,
            "sleep_hours": request.form.get("sleep_hours") or 0,
            "size_group": request.form.get("size_group") or None
        }
        for k in ["age_years","height_cm","length_cm","weight_kg","avg_daily_play_min","avg_daily_walk_min","sleep_hours"]:
            if data.get(k) not in (None, ""):
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = None
        out = predict_activity_recommendation(data)
        return render_template("activity_reco_result.html", result=out, submitted=data, metrics=act_metrics)
    return render_template("activity_reco_form.html", metrics=act_metrics)


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
