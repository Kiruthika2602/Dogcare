# src/models/bmi_risk.py
"""
Dog BMI Calculator + Health Risk Checker

Usage:
- Place this file at src/models/bmi_risk.py
- Run (from project root) to (re)train synthetic models and save artifacts:
    python -m src.models.bmi_risk
- Import in Flask or other code:
    from src.models.bmi_risk import predict_bmi_and_risks, load_saved_models
    out = predict_bmi_and_risks({
        "weight_kg": 12.5,
        "height_cm": 35,
        "age_years": 3.0,
        "avg_daily_play_min": 45,
        "avg_daily_walk_min": 30,
        "breed": "Beagle",
        "size_group": "medium",
        "sex": "M"
    })
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_DIR = os.path.join(PROJECT_ROOT, "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)
MODEL_PKL = os.path.join(SAVED_DIR, "bmi_risk_models.pkl")
METRICS_JSON = os.path.join(SAVED_DIR, "bmi_metrics.json")

# -------------------------
# BMI and category helpers
# -------------------------
def compute_bmi(weight_kg: float, height_cm: float) -> float:
    """
    Simple BMI-like metric for dogs using kg / m^2.
    Note: canine 'BMI' is not universally standardized — this is for demo/relative scoring.
    """
    try:
        w = float(weight_kg)
        h = float(height_cm)
    except Exception:
        raise ValueError("weight_kg and height_cm must be numeric")

    if h <= 0 or w <= 0:
        raise ValueError("weight_kg and height_cm must be > 0")

    h_m = h / 100.0
    bmi = w / (h_m ** 2)
    return round(float(bmi), 3)


def categorize_bmi(bmi: float, size_group: str = "medium") -> str:
    """
    Size-aware thresholds (empirical, for demo).
    small dogs: lower absolute healthy weight -> BMI thresholds a bit higher.
    Use three groups: small / medium / large
    """
    sg = (size_group or "medium").lower()
    if sg not in {"small", "medium", "large"}:
        sg = "medium"

    # thresholds chosen empirically for demo (not vet guidelines)
    if sg == "small":
        # small dogs are shorter → BMI numeric may be higher; thresholds adjusted
        if bmi < 15:
            return "Underweight"
        if bmi < 23:
            return "Healthy"
        if bmi < 28:
            return "Overweight"
        return "Obese"
    elif sg == "large":
        # large breeds: taller → BMI numeric lower for same weight
        if bmi < 9:
            return "Underweight"
        if bmi < 14:
            return "Healthy"
        if bmi < 18:
            return "Overweight"
        return "Obese"
    else:  # medium
        if bmi < 12:
            return "Underweight"
        if bmi < 18:
            return "Healthy"
        if bmi < 22:
            return "Overweight"
        return "Obese"


# -------------------------
# Synthetic training data generator
# -------------------------
def generate_synthetic_dataset(n=3000, random_state=42):
    """
    Generate a synthetic dataset consistent with earlier app features.
    Columns:
      - weight_kg, height_cm, age_years, avg_daily_play_min, avg_daily_walk_min, activity_minutes, activity_score
      - size_group (small/medium/large), sex
      - bmi (computed), bmi_cat
      - target_obesity (binary), target_joint_risk (binary), target_heart_risk (binary)
    The targets are generated with plausible heuristics + noise.
    """
    rng = np.random.RandomState(random_state)

    # sample size_group with imbalanced realistic distribution
    size_group = rng.choice(["small", "medium", "large"], size=n, p=[0.35, 0.45, 0.20])
    sex = rng.choice(["M", "F"], size=n, p=[0.5, 0.5])
    age_years = np.clip(rng.normal(loc=5.0, scale=3.5, size=n), 0.1, 16.0)

    # heights per size_group (cm)
    height_cm = np.array([
        rng.normal(loc=28, scale=6) if sg == "small" else (rng.normal(45, 6) if sg == "medium" else rng.normal(65, 8))
        for sg in size_group
    ])
    height_cm = np.clip(height_cm, 15, 100)

    # weights correlate with height and size_group, add noise
    base_w = []
    for sg, h in zip(size_group, height_cm):
        if sg == "small":
            base_w.append( (h/50.0)*6.0 )  # small baseline
        elif sg == "medium":
            base_w.append( (h/50.0)*14.0 )
        else:
            base_w.append( (h/50.0)*30.0 )
    base_w = np.array(base_w)
    weight_kg = base_w + rng.normal(0, base_w*0.12)

    # activity
    avg_play = np.clip(rng.normal(35, 20, size=n), 0, 180)
    avg_walk = np.clip(rng.normal(40, 25, size=n), 0, 240)
    activity_minutes = avg_play + avg_walk
    activity_score = (activity_minutes / (24*60)) * 100

    # bmi
    bmi = np.array([w / ((h/100.0)**2) for w,h in zip(weight_kg, height_cm)])

    # derive targets using heuristics + noise
    # obesity probability increases with bmi, low activity, older age, and small/medium breeds
    obesity_logit = (
        0.7*(bmi - np.median(bmi)) / np.std(bmi)
        - 0.5*(activity_minutes - np.mean(activity_minutes))/np.std(activity_minutes)
        + 0.15*(age_years - np.mean(age_years))/np.std(age_years)
    )
    # add size_group effect
    obesity_logit += np.where(size_group == "small", 0.3, 0.0)
    obesity_prob = 1 / (1 + np.exp(-obesity_logit))
    target_obesity = (rng.rand(n) < obesity_prob).astype(int)

    # joint risk: increases strongly with weight and age, moderate with activity (low activity -> higher risk)
    joint_logit = (
        0.6*(weight_kg - np.mean(weight_kg))/np.std(weight_kg)
        + 0.5*(age_years - np.mean(age_years))/np.std(age_years)
        - 0.4*(activity_minutes - np.mean(activity_minutes))/np.std(activity_minutes)
    )
    joint_prob = 1 / (1 + np.exp(-joint_logit))
    target_joint = (rng.rand(n) < joint_prob).astype(int)

    # heart risk: increases with age and low activity, large breeds slightly higher
    heart_logit = (
        0.5*(age_years - np.mean(age_years))/np.std(age_years)
        - 0.3*(activity_minutes - np.mean(activity_minutes))/np.std(activity_minutes)
    )
    heart_logit += np.where(size_group == "large", 0.35, 0.0)
    heart_prob = 1 / (1 + np.exp(-heart_logit))
    target_heart = (rng.rand(n) < heart_prob).astype(int)

    df = pd.DataFrame({
        "weight_kg": np.round(weight_kg, 2),
        "height_cm": np.round(height_cm, 1),
        "age_years": np.round(age_years, 2),
        "avg_daily_play_min": np.round(avg_play, 1),
        "avg_daily_walk_min": np.round(avg_walk, 1),
        "activity_minutes": np.round(activity_minutes, 1),
        "activity_score": np.round(activity_score, 3),
        "size_group": size_group,
        "sex": sex,
        "bmi": np.round(bmi, 3),
        "target_obesity": target_obesity,
        "target_joint": target_joint,
        "target_heart": target_heart
    })
    return df


# -------------------------
# Training / evaluation
# -------------------------
def _train_binary_model(X, y):
    """
    Train a small logistic regression with sensible defaults.
    Returns model and metrics dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None,
        "n_test": int(len(y_test))
    }
    return model, metrics


def train_and_save_models(save=True):
    """
    Generate synthetic data, train three binary classifiers and save artifacts + metrics.
    """
    df = generate_synthetic_dataset(n=3000)
    # features to use
    # We'll create simple numeric features and one-hot size_group
    X_base = df[["weight_kg", "height_cm", "age_years", "activity_minutes", "activity_score"]].copy()
    X_base["bmi"] = df["bmi"]
    # one-hot encode size_group
    X_base = pd.concat([X_base, pd.get_dummies(df["size_group"], prefix="size")], axis=1)

    results = {}
    models = {}

    # obesity
    mod_obs, metrics_obs = _train_binary_model(X_base, df["target_obesity"])
    models["obesity"] = mod_obs
    results["obesity_metrics"] = metrics_obs

    # joint
    mod_joint, metrics_joint = _train_binary_model(X_base, df["target_joint"])
    models["joint"] = mod_joint
    results["joint_metrics"] = metrics_joint

    # heart
    mod_heart, metrics_heart = _train_binary_model(X_base, df["target_heart"])
    models["heart"] = mod_heart
    results["heart_metrics"] = metrics_heart

    # Save models + metadata
    if save:
        joblib.dump({
            "models": models,
            "feature_columns": X_base.columns.tolist()
        }, MODEL_PKL)
        with open(METRICS_JSON, "w", encoding="utf8") as f:
            json.dump(results, f, indent=2)
    return models, results


# -------------------------
# Loading helper
# -------------------------
def load_saved_models():
    if not os.path.exists(MODEL_PKL) or not os.path.exists(METRICS_JSON):
        # if missing, train and save
        train_and_save_models(save=True)
    payload = joblib.load(MODEL_PKL)
    with open(METRICS_JSON, "r", encoding="utf8") as f:
        metrics = json.load(f)
    return payload["models"], payload["feature_columns"], metrics


# -------------------------
# Prediction wrapper for Flask
# -------------------------
def predict_bmi_and_risks(data: dict):
    """
    data must include:
      - weight_kg (optional if you want to compute from elsewhere)
      - height_cm
      - age_years (optional, default 0)
      - avg_daily_play_min (optional)
      - avg_daily_walk_min (optional)
      - size_group (optional: small/medium/large)
      - sex (optional)
    Returns:
      {
        "bmi": float,
        "bmi_category": str,
        "risk_probs": {"obesity": float, "joint": float, "heart": float},
        "risk_preds": {"obesity": int, "joint": int, "heart": int},
        "metrics": metrics_loaded_from_file
      }
    """
    # defensive parsing & defaulting
    def _num(k, default=0.0):
        v = data.get(k, None)
        try:
            if v is None or (isinstance(v, str) and v.strip() == ""):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    weight = _num("weight_kg", default=np.nan)
    height = _num("height_cm", default=np.nan)
    age = _num("age_years", default=0.0)
    play = _num("avg_daily_play_min", default=0.0)
    walk = _num("avg_daily_walk_min", default=0.0)
    activity_minutes = round(play + walk, 1)
    activity_score = round((activity_minutes / (24*60)) * 100, 3)
    size_group = (data.get("size_group") or data.get("size") or "medium").lower()
    if size_group not in {"small", "medium", "large"}:
        size_group = "medium"

    # compute bmi (if height and weight present)
    if np.isnan(weight) or np.isnan(height):
        bmi_val = None
        bmi_cat = "Unknown (missing weight/height)"
    else:
        bmi_val = compute_bmi(weight, height)
        bmi_cat = categorize_bmi(bmi_val, size_group=size_group)

    # load models
    models, feat_cols, metrics = load_saved_models()

    # build feature vector in same order as training
    X = {}
    # numeric features
    X["weight_kg"] = 0.0 if np.isnan(weight) else float(weight)
    X["height_cm"] = 0.0 if np.isnan(height) else float(height)
    X["age_years"] = float(age)
    X["activity_minutes"] = float(activity_minutes)
    X["activity_score"] = float(activity_score)
    X["bmi"] = float(bmi_val) if bmi_val is not None else 0.0
    # one-hot sizes
    for sg in ["size_small", "size_medium", "size_large"]:
        X[sg] = 1.0 if sg == f"size_{size_group}" else 0.0

    X_df = pd.DataFrame([X], columns=feat_cols)

    risk_probs = {}
    risk_preds = {}
    for key, mdl in models.items():
        prob = float(mdl.predict_proba(X_df)[0, 1])
        pred = int(mdl.predict(X_df)[0])
        risk_probs[key] = round(prob, 3)
        risk_preds[key] = int(pred)

    out = {
        "bmi": bmi_val,
        "bmi_category": bmi_cat,
        "risk_probs": risk_probs,
        "risk_preds": risk_preds,
        "metrics": metrics
    }
    return out


# -------------------------
# CLI: train/evaluate/save when module run
# -------------------------
if __name__ == "__main__":
    print("Training BMI + risk models on synthetic data (this may take a moment)...")
    models, results = train_and_save_models(save=True)
    print("Saved models to:", MODEL_PKL)
    print("Saved metrics to:", METRICS_JSON)
    print("Metrics summary:")
    print(json.dumps(results, indent=2))
