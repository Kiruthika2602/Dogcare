"""
src/models/predictor.py

Defensive, robust predictor module with focused functions:
  - predict_weight_only(data)
  - predict_calorie_only(data)
  - predict_activity_only(data)

Also includes predict_all(data) wrapper for backward compatibility.

This file:
 - Avoids FutureWarnings (uses .iloc[0] when reading single-element Series)
 - Treats missing weight as NaN so predictor can decide to predict
 - Uses a sensible fallback when the weight model returns non-finite or <= 0
 - Provides a small sklearn-version-mismatch fallback that sets missing
   `monotonic_cst` attributes on tree estimators so older/newer sklearns don't crash.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple

# -------------------------
# Paths (project-root relative)
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

WEIGHT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "weight_model.pkl")
CALORIE_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "calorie_model_pipeline.pkl")
ACTIVITY_KMEANS_PATH = os.path.join(SAVED_MODELS_DIR, "activity_kmeans.pkl")
ACTIVITY_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "activity_scaler.pkl")
ACTIVITY_META_PATH = os.path.join(SAVED_MODELS_DIR, "activity_cluster_metadata.pkl")

# -------------------------
# Load models (fail fast with clear message)
# -------------------------
def _load_or_raise(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required model file not found: {path}")
    return joblib.load(path)

weight_model = _load_or_raise(WEIGHT_MODEL_PATH)
calorie_model = _load_or_raise(CALORIE_MODEL_PATH)
kmeans_model = _load_or_raise(ACTIVITY_KMEANS_PATH)
activity_scaler = _load_or_raise(ACTIVITY_SCALER_PATH)
activity_meta = _load_or_raise(ACTIVITY_META_PATH)

activity_cols = activity_meta.get("activity_cols", ["avg_daily_play_min", "avg_daily_walk_min", "sleep_hours"])
cluster_map = activity_meta.get("cluster_map", {})


# -------------------------
# Helpers
# -------------------------
def _ensure_numeric(value, fallback=0.0) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return float(fallback)
        return float(value)
    except Exception:
        return float(fallback)


def _to_real_float(x, name="value") -> float:
    """Safe float conversion with clear errors."""
    if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
        if len(x) == 0:
            raise ValueError(f"{name} is empty")
        x = x[0]

    try:
        return float(x)
    except (TypeError, ValueError):
        pass

    try:
        c = complex(x)
        if abs(c.imag) > 1e-6:
            raise ValueError(f"{name} has non-zero imaginary part ({c.imag})")
        return float(c.real)
    except Exception as e:
        raise ValueError(f"Could not convert {name} ({x}) to float: {e}")


# -------------------------
# Input preparation
# -------------------------
def prepare_input_df(data: dict) -> pd.DataFrame:
    """
    Convert input dict to DataFrame and compute derived features.
    Important: do not coerce missing weight to zero — leave as NaN so calling code
    can decide whether to predict weight.
    """
    df = pd.DataFrame([data]).copy()

    # Age_years from age_months if needed
    if ("age_years" not in df.columns or pd.isna(df.loc[0, "age_years"])) and "age_months" in df.columns:
        try:
            df["age_years"] = pd.to_numeric(df["age_months"], errors="coerce") / 12.0
        except Exception:
            df["age_years"] = 0.0
    if "age_years" not in df.columns:
        df["age_years"] = 0.0

    # Play/walk -> activity_minutes
    play = df.get("avg_daily_play_min", [np.nan])[0]
    walk = df.get("avg_daily_walk_min", [np.nan])[0]
    play = _ensure_numeric(play, 0.0)
    walk = _ensure_numeric(walk, 0.0)
    df["avg_daily_play_min"] = play
    df["avg_daily_walk_min"] = walk
    df["activity_minutes"] = round(play + walk, 1)

    # activity_score scaled 0-100 (use iloc[0] to avoid FutureWarning)
    try:
        act_min_val = df["activity_minutes"].iloc[0]
        act_min_val = 0.0 if pd.isna(act_min_val) else float(act_min_val)
        df["activity_score"] = round((act_min_val / (24 * 60)) * 100, 3)
    except Exception:
        df["activity_score"] = 0.0

    # sleep_hours
    if "sleep_hours" not in df.columns or pd.isna(df.loc[0, "sleep_hours"]):
        df["sleep_hours"] = 0.0
    else:
        df["sleep_hours"] = _ensure_numeric(df.loc[0, "sleep_hours"], 0.0)

    # weight_kg: leave NaN if missing or blank; coerce numeric otherwise
    if "weight_kg" in df.columns:
        raw_w = df.loc[0, "weight_kg"]
        if raw_w is None or (isinstance(raw_w, str) and raw_w.strip() == ""):
            df["weight_kg"] = np.nan
        else:
            try:
                df["weight_kg"] = float(raw_w)
            except Exception:
                df["weight_kg"] = np.nan
    else:
        df["weight_kg"] = np.nan

    # breed_reduced / size_group / sex / dimensions defaults
    if "breed_reduced" not in df.columns:
        df["breed_reduced"] = df.loc[0, "breed"] if "breed" in df.columns and pd.notna(df.loc[0, "breed"]) else "Other"
    if "size_group" not in df.columns:
        df["size_group"] = "medium"
    if "sex" not in df.columns or pd.isna(df.loc[0, "sex"]):
        df["sex"] = "M"
    df["height_cm"] = _ensure_numeric(df.get("height_cm", [0])[0], 0.0)
    df["length_cm"] = _ensure_numeric(df.get("length_cm", [0])[0], 0.0)

    df = df.replace({None: np.nan})
    return df


# -------------------------
# Activity helpers
# -------------------------
def compute_activity_label_and_insert(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, str]:
    """
    Use activity_scaler + kmeans_model to compute cluster id and label.
    Insert activity_label into df (string) for calorie pipeline.
    """
    # Ensure activity columns exist
    for c in activity_cols:
        if c not in df.columns:
            df[c] = 0.0

    act_input = df[activity_cols].copy().fillna(0)
    scaled = activity_scaler.transform(act_input)
    cluster_id = int(kmeans_model.predict(scaled)[0])
    label = cluster_map.get(cluster_id, f"cluster_{cluster_id}")

    df["activity_label"] = label
    return df, cluster_id, label


# -------------------------
# RER & MER computation (defensive)
# -------------------------
def compute_rer_mer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RER and MER multiplier used by calorie model.
    RER = 70 * weight^0.75
    MER multiplier heuristic:
      - <40 => 1.2
      - 40-89 => 1.5
      - >=90 => 1.8
    This version validates and coerces inputs and raises ValueError with clear message on failure.
    """
    raw_w = df.get("weight_kg", [None])[0] if "weight_kg" in df.columns else None
    if raw_w is None or (isinstance(raw_w, float) and np.isnan(raw_w)):
        raise ValueError("weight_kg must be present and numeric to compute RER/MER")

    try:
        weight = _to_real_float(raw_w, name="weight_kg")
    except ValueError as e:
        raise ValueError(f"Failed to compute RER/MER: {e}")

    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("weight_kg must be a positive finite number to compute RER")

    rer = 70.0 * (weight ** 0.75)
    df["rer"] = float(rer)

    raw_act = df.get("activity_minutes", [0])[0] if "activity_minutes" in df.columns else 0
    try:
        act_min = _to_real_float(raw_act, name="activity_minutes")
    except ValueError as e:
        raise ValueError(f"Failed to compute RER/MER: {e}")

    if act_min < 40:
        df["mer_multiplier"] = 1.2
    elif act_min < 90:
        df["mer_multiplier"] = 1.5
    else:
        df["mer_multiplier"] = 1.8

    return df


# -------------------------
# Size-based fallback for invalid predicted weights
# -------------------------
def _fallback_weight_from_size(df: pd.DataFrame) -> float:
    """
    Estimate a fallback weight based on size_group, height_cm, or defaults:
     - small: 7.0 kg
     - medium: 15.0 kg
     - large: 25.0 kg
     - If height present: <35cm -> small, <55cm -> medium, else large
     - Final default: 10.0 kg
    """
    size_defaults = {"small": 7.0, "medium": 15.0, "large": 25.0}
    size_group = None
    if "size_group" in df.columns and pd.notna(df.loc[0, "size_group"]):
        size_group = str(df.loc[0, "size_group"]).lower()
        if size_group in size_defaults:
            return size_defaults[size_group]

    # try height heuristic
    height = _ensure_numeric(df.get("height_cm", [0])[0], 0.0)
    if height > 0:
        if height < 35:
            return size_defaults["small"]
        if height < 55:
            return size_defaults["medium"]
        return size_defaults["large"]

    # fallback default
    return 10.0


# -------------------------
# Focused predictions
# -------------------------
def predict_weight_only(data: dict):
    """
    Predict only weight or return supplied positive weight.
    Returns: {"weight_kg": float, "weight_source": "predicted"|"user_given"|"fallback_size_default"}
    Raises RuntimeError on catastrophic failure.
    """
    df = prepare_input_df(data)

    supplied_w = df.get("weight_kg", [None])[0] if "weight_kg" in df.columns else None
    if supplied_w is not None and not (isinstance(supplied_w, float) and np.isnan(supplied_w)):
        try:
            numeric_w = _to_real_float(supplied_w, name="weight_kg")
            if numeric_w > 0 and np.isfinite(numeric_w):
                return {"weight_kg": round(numeric_w, 2), "weight_source": "user_given"}
            # else fall through to prediction
        except Exception:
            pass

    # Predict using model
    try:
        pred = weight_model.predict(df)
        pred_val = float(pred[0])
    except Exception:
        # If prediction call fails entirely, try fallback
        fallback = _fallback_weight_from_size(df)
        return {"weight_kg": round(fallback, 2), "weight_source": "fallback_size_default"}

    # Validate predicted weight
    if not np.isfinite(pred_val) or pred_val <= 0:
        # compute fallback based on size/height
        fallback = _fallback_weight_from_size(df)
        return {"weight_kg": round(fallback, 2), "weight_source": "fallback_size_default"}

    return {"weight_kg": round(pred_val, 2), "weight_source": "predicted"}


def predict_activity_only(data: dict):
    """
    Predict activity cluster and label.
    Returns: {"activity_cluster": int, "activity_label": str}
    """
    df = prepare_input_df(data)
    df, cluster_id, label = compute_activity_label_and_insert(df)
    return {"activity_cluster": cluster_id, "activity_label": label}


def _apply_monotonic_patch_to_forest(pipeline_or_model) -> bool:
    """
    Try to find an ensemble estimator (RandomForest/Forest) and ensure every tree
    estimator has attribute 'monotonic_cst' (set to None) to avoid sklearn internal
    attribute errors when versions differ.
    Returns True if patch applied (estimators found), False otherwise.
    """
    try:
        model = None
        # pipeline with named_steps
        if hasattr(pipeline_or_model, "named_steps") and "model" in pipeline_or_model.named_steps:
            model = pipeline_or_model.named_steps["model"]
        # pipeline.steps (older sklearn)
        elif hasattr(pipeline_or_model, "steps"):
            model = dict(pipeline_or_model.steps).get("model")
        # maybe it's directly the ensemble
        elif hasattr(pipeline_or_model, "estimators_") or hasattr(pipeline_or_model, "estimator_"):
            model = pipeline_or_model

        if model is None:
            return False

        # If ensemble has fitted estimators_, patch each
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
            return True

        # Some wrappers may store the final estimator differently
        if hasattr(model, "estimator_") and hasattr(model.estimator_, "estimators_"):
            for est in model.estimator_.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
            return True

        return False
    except Exception:
        return False


def predict_calorie_only(data: dict):
    """
    Predict only calories.
    Steps:
      - ensure weight exists (user-provided or predicted)
      - compute activity_label (required by calorie pipeline)
      - compute RER and MER multiplier
      - call calorie_model.predict
    Returns:
      {
        "calories_per_day": float,
        "calorie_label": str,
        "weight_used": float,
        "weight_source": str,
        "activity_cluster": int,
        "activity_label": str
      }
    """
    df = prepare_input_df(data)

    # Ensure weight: if missing/NaN/<=0 -> predict
    supplied_w = df.get("weight_kg", [np.nan])[0] if "weight_kg" in df.columns else np.nan
    use_predicted_weight = False
    try:
        if supplied_w is None or (isinstance(supplied_w, float) and np.isnan(supplied_w)):
            use_predicted_weight = True
        else:
            numeric_w = _to_real_float(supplied_w, name="weight_kg")
            if numeric_w <= 0:
                use_predicted_weight = True
            else:
                weight_used = numeric_w
                weight_source = "user_given"
                df["weight_kg"] = weight_used
    except Exception:
        use_predicted_weight = True

    if use_predicted_weight:
        winfo = predict_weight_only(data)
        weight_used = float(winfo["weight_kg"])
        weight_source = winfo["weight_source"]
        # Validate predicted/fallback weight
        if not np.isfinite(weight_used) or weight_used <= 0:
            raise RuntimeError(f"Predicted/fallback weight invalid: {weight_used}")
        df["weight_kg"] = weight_used

    # Compute activity label and insert
    try:
        df, cluster_id, activity_label = compute_activity_label_and_insert(df)
    except Exception as e:
        raise RuntimeError(f"Failed to compute activity label: {e}")

    # Compute RER & MER defensively
    try:
        df = compute_rer_mer(df)
    except Exception as e:
        raise RuntimeError(f"Failed to compute RER/MER: {e}")

    # Predict calories (with defensive retry/patch for sklearn version mismatch)
    try:
        # primary attempt
        pred_cal_arr = calorie_model.predict(df)
        pred_val = float(pred_cal_arr[0])
    except Exception as e_primary:
        # Attempt to patch missing internal attributes on tree estimators (monotonic_cst)
        patched = _apply_monotonic_patch_to_forest(calorie_model)
        if patched:
            try:
                pred_cal_arr = calorie_model.predict(df)
                pred_val = float(pred_cal_arr[0])
            except Exception as e_retry:
                # If still failing, surface helpful error (include original error)
                raise RuntimeError(f"Calorie model predict failed after patch: {e_retry}; original: {e_primary}")
        else:
            # No patch possible — re-raise original as RuntimeError
            raise RuntimeError(f"Calorie model predict failed: {e_primary}")

    if not np.isfinite(pred_val) or pred_val <= 0:
        raise RuntimeError(f"Calorie model returned invalid value: {pred_val}")

    pred_cal = round(pred_val, 1)
    if pred_cal < 500:
        cal_label = "Low Energy Diet"
    elif pred_cal < 900:
        cal_label = "Normal Diet"
    else:
        cal_label = "High Energy Diet"

    return {
        "calories_per_day": pred_cal,
        "calorie_label": cal_label,
        "weight_used": round(weight_used, 2),
        "weight_source": weight_source,
        "activity_cluster": cluster_id,
        "activity_label": activity_label
    }


# -------------------------
# Backwards-compatible wrapper
# -------------------------
def predict_all(data: dict):
    """
    Compatibility wrapper used by web/app.py — combines focused predictions.
    Returns a combined dictionary:
      - weight_kg, weight_source
      - calories_per_day, calorie_label
      - activity_cluster, activity_label
      - weight_used_for_calorie, weight_source_for_calorie
    """
    if data is None:
        raise ValueError("No input data provided to predict_all()")

    w = predict_weight_only(data)
    a = predict_activity_only(data)
    c = predict_calorie_only(data)

    combined = {
        "weight_kg": w.get("weight_kg"),
        "weight_source": w.get("weight_source"),
        "calories_per_day": c.get("calories_per_day"),
        "calorie_label": c.get("calorie_label"),
        "activity_cluster": a.get("activity_cluster"),
        "activity_label": a.get("activity_label"),
        "weight_used_for_calorie": c.get("weight_used"),
        "weight_source_for_calorie": c.get("weight_source"),
    }
    return combined
