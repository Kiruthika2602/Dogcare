"""
src/models/life_activity.py

Train & expose predictors for:
 - life expectancy (regression)
 - activity recommendations (play & walk minutes) (multi-output regression)

Usage:
    python -m src.models.life_activity

This script:
 - generates a synthetic dataset (reasonable heuristics)
 - trains lightweight models (RandomForest / MultiOutputRegressor)
 - saves models to saved_models/
 - writes metrics (MAE, RMSE, R2) to saved_models/life_activity_metrics.json

Also exposes two predict functions:
 - predict_life_expectancy(input_dict)
 - predict_activity_recommendation(input_dict)
"""
import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# file paths (project-root relative)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

LIFE_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "life_expectancy_model.pkl")
ACTIVITY_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "activity_reco_model.pkl")
METRICS_PATH = os.path.join(SAVED_MODELS_DIR, "life_activity_metrics.json")

# reproducible RNG
RNG = np.random.RandomState(42)

# ----------------------------
# Synthetic dataset generator
# ----------------------------
def _make_synthetic_dataset(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    breeds = ["Mixed", "Labrador", "German Shepherd", "Beagle", "Pomeranian", "Bulldog", "Golden Retriever",
              "Pug", "Cocker Spaniel", "Great Dane"]
    size_groups = ["small", "medium", "large"]
    sexes = ["M", "F"]
    for _ in range(n):
        size = rng.choice(size_groups, p=[0.25,0.6,0.15])
        breed = rng.choice(breeds)
        sex = rng.choice(sexes)
        # age in years 0.5 - 15
        age_years = float(round(rng.uniform(0.5, 15.0), 2))
        # height / length heuristics by size
        if size == "small":
            height = round(rng.uniform(20, 35), 1)
            length = round(height + rng.uniform(5,15),1)
            ideal_weight = rng.uniform(4,9)
        elif size == "medium":
            height = round(rng.uniform(35, 55), 1)
            length = round(height + rng.uniform(10,25),1)
            ideal_weight = rng.uniform(10, 25)
        else:
            height = round(rng.uniform(55, 85), 1)
            length = round(height + rng.uniform(10,40),1)
            ideal_weight = rng.uniform(25, 55)

        # weight around ideal, with noise & occasional obesity
        weight = round(ideal_weight * rng.normal(1.0, 0.15), 2)
        if rng.rand() < 0.12:
            weight *= rng.uniform(1.2, 1.6)  # obese cases

        # activity minutes
        play = float(round(rng.normal(30 if size!="small" else 20, 15), 1))
        walk = float(round(rng.normal(40 if size!="large" else 30, 20), 1))
        play = max(0.0, play)
        walk = max(0.0, walk)
        activity_minutes = round(play + walk, 1)
        sleep_hours = round(rng.uniform(8, 14), 1)

        # BMI-like metric for dog (demo): weight / (height*length) * scale (bigger => larger)
        bmi = (weight / (max(1.0, height) * max(1.0, length))) * 1000.0

        # compute life expectancy target using heuristic:
        # base_by_size - age_penalty - bmi_penalty + activity_bonus + noise
        base = {"small": 13.5, "medium": 12.0, "large": 10.0}[size]
        age_penalty = (age_years * 0.35)
        bmi_penalty = max(0.0, (bmi - 50.0) * 0.05)  # higher bmi reduces expectancy
        activity_bonus = min(2.0, (activity_minutes / 60.0) * 0.5)  # more activity slightly boosts
        noise = rng.normal(0.0, 1.0)
        expected_life = base - age_penalty - bmi_penalty + activity_bonus + noise
        # remaining years cannot be negative
        years_left = max(0.1, round(expected_life, 2))

        # activity recommendations heuristic (targets)
        # recommended play & walk depend on BMI and age:
        play_target = float(round(max(5, 30 + (50 - bmi) * 0.2 - (age_years * 0.3) ), 1))
        walk_target = float(round(max(5, 40 + (50 - bmi) * 0.25 - (age_years * 0.4) ), 1))

        # if obese bump recommendations a bit
        if bmi > 70:
            play_target += 10
            walk_target += 15

        rows.append({
            "breed": breed,
            "size_group": size,
            "sex": sex,
            "age_years": age_years,
            "height_cm": height,
            "length_cm": length,
            "weight_kg": round(weight,2),
            "avg_daily_play_min": round(play,1),
            "avg_daily_walk_min": round(walk,1),
            "activity_minutes": round(activity_minutes,1),
            "sleep_hours": sleep_hours,
            "bmi_like": round(bmi,3),
            "years_left": years_left,
            "play_target": max(0.0, round(play_target,1)),
            "walk_target": max(0.0, round(walk_target,1))
        })
    return pd.DataFrame(rows)

# ----------------------------
# Train & save
# ----------------------------
def train_and_save(n_samples=5000, dump=True):
    print("Generating synthetic dataset...")
    df = _make_synthetic_dataset(n_samples, seed=42)

    # simple feature set for both tasks
    feat_cols = ["age_years", "weight_kg", "height_cm", "length_cm", "activity_minutes", "sleep_hours", "bmi_like"]
    X = df[feat_cols].fillna(0.0).astype(float)
    y_life = df["years_left"].values.ravel()
    y_activity = df[["play_target", "walk_target"]].values

    # train/test split
    X_train, X_test, y_life_train, y_life_test, y_act_train, y_act_test = train_test_split(
        X, y_life, y_activity, test_size=0.15, random_state=42
    )

    # life expectancy regressor
    life_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
    life_model.fit(X_train, y_life_train)
    life_pred = life_model.predict(X_test)

    # activity multi-output regressor
    act_base = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    activity_model = MultiOutputRegressor(act_base)
    activity_model.fit(X_train, y_act_train)
    act_pred = activity_model.predict(X_test)

    # metrics
    metrics = {
        "life": {
            "mae": float(mean_absolute_error(y_life_test, life_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_life_test, life_pred))),
            "r2": float(r2_score(y_life_test, life_pred)),
            "n_test": int(len(y_life_test))
        },
        "activity": {
            "mae_play": float(mean_absolute_error(y_act_test[:,0], act_pred[:,0])),
            "mae_walk": float(mean_absolute_error(y_act_test[:,1], act_pred[:,1])),
            "rmse_play": float(np.sqrt(mean_squared_error(y_act_test[:,0], act_pred[:,0]))),
            "rmse_walk": float(np.sqrt(mean_squared_error(y_act_test[:,1], act_pred[:,1]))),
            "r2_play": float(r2_score(y_act_test[:,0], act_pred[:,0])),
            "r2_walk": float(r2_score(y_act_test[:,1], act_pred[:,1])),
            "n_test": int(len(y_act_test))
        }
    }

    if dump:
        joblib.dump(life_model, LIFE_MODEL_PATH)
        joblib.dump(activity_model, ACTIVITY_MODEL_PATH)
        with open(METRICS_PATH, "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved models to:", LIFE_MODEL_PATH, "and", ACTIVITY_MODEL_PATH)
        print("Saved metrics to:", METRICS_PATH)
        print("Metrics summary:")
        print(json.dumps(metrics, indent=2))
    return life_model, activity_model, metrics

# ----------------------------
# Utilities / Predictors (runtime)
# ----------------------------
# load models lazily
_life_model = None
_activity_model = None
_metrics = None

def _load_models_if_needed():
    global _life_model, _activity_model, _metrics
    if _life_model is None:
        if os.path.exists(LIFE_MODEL_PATH):
            _life_model = joblib.load(LIFE_MODEL_PATH)
        else:
            raise FileNotFoundError("Life expectancy model not found. Run trainer: python -m src.models.life_activity")
    if _activity_model is None:
        if os.path.exists(ACTIVITY_MODEL_PATH):
            _activity_model = joblib.load(ACTIVITY_MODEL_PATH)
        else:
            raise FileNotFoundError("Activity reco model not found. Run trainer: python -m src.models.life_activity")
    if _metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf8") as f:
            _metrics = json.load(f)

def _prepare_minimal_input(data: dict):
    # similar to your other predictors: ensure numeric and derived features
    d = dict(data) if data else {}
    def _n(k, fallback=0.0):
        v = d.get(k, None)
        if v is None or v == "":
            return float(fallback)
        try:
            return float(v)
        except:
            return float(fallback)
    age_years = _n("age_years", 0.0)
    weight_kg = _n("weight_kg", 0.0)
    height_cm = _n("height_cm", 0.0)
    length_cm = _n("length_cm", 0.0)
    play = _n("avg_daily_play_min", 0.0)
    walk = _n("avg_daily_walk_min", 0.0)
    activity_minutes = round(play + walk, 1)
    sleep_hours = _n("sleep_hours", 10.0)
    bmi_like = 0.0
    if height_cm > 0 and length_cm > 0:
        bmi_like = (weight_kg / (height_cm * length_cm)) * 1000.0
    else:
        # fallback: approximate by size (if provided)
        size_group = d.get("size_group", "").lower() if d.get("size_group") else ""
        if size_group == "small":
            bmi_like = 45.0
        elif size_group == "large":
            bmi_like = 65.0
        else:
            bmi_like = 52.0
    row = {
        "age_years": age_years,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "length_cm": length_cm,
        "activity_minutes": activity_minutes,
        "sleep_hours": sleep_hours,
        "bmi_like": round(float(bmi_like),3)
    }
    return pd.DataFrame([row])

def predict_life_expectancy(input_dict: dict):
    """
    Returns:
      {
        "years_left": float,
        "life_category": "Short"|"Normal"|"Long",
        "metrics": metrics['life'] or {}
      }
    """
    _load_models_if_needed()
    X = _prepare_minimal_input(input_dict)
    pred = _life_model.predict(X)[0]
    pred = float(max(0.0, round(pred, 2)))
    # categorize
    if pred < 2.5:
        cat = "Short"
    elif pred < 6.0:
        cat = "Normal"
    else:
        cat = "Long"
    return {
        "years_left": pred,
        "life_category": cat,
        "metrics": _metrics.get("life") if _metrics else {}
    }

def predict_activity_recommendation(input_dict: dict):
    """
    Returns:
      {
        "recommended_play_min": float,
        "recommended_walk_min": float,
        "metrics": metrics['activity'] or {}
      }
    """
    _load_models_if_needed()
    X = _prepare_minimal_input(input_dict)
    out = _activity_model.predict(X)[0]
    play = float(max(0.0, round(out[0],1)))
    walk = float(max(0.0, round(out[1],1)))
    return {
        "recommended_play_min": play,
        "recommended_walk_min": walk,
        "metrics": _metrics.get("activity") if _metrics else {}
    }

# ----------------------------
# CLI trainer
# ----------------------------
if __name__ == "__main__":
    train_and_save(n_samples=6000, dump=True)
