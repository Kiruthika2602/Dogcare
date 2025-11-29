# src/models/train_calorie.py
"""
Train a RandomForestRegressor to predict calories_per_day.

Usage (from project root):
    python -m src.models.train_calorie

Outputs:
- saved_models/calorie_model_pipeline.pkl    (pipeline containing preprocessor + RF)
- saved_models/calorie_preprocessor_metadata.pkl  (dictionary with numeric/categorical features)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Paths (project-root relative)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train_df.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test_df.csv")
OUT_MODEL = os.path.join(PROJECT_ROOT, "saved_models", "calorie_model_pipeline.pkl")
OUT_META = os.path.join(PROJECT_ROOT, "saved_models", "calorie_preprocessor_metadata.pkl")

RANDOM_STATE = 42


def load_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Processed train/test CSVs not found at:\n  {TRAIN_PATH}\n  {TEST_PATH}"
        )
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def _safe_onehot(dense=True):
    """
    Return OneHotEncoder compatible across sklearn versions.

    dense=True  -> produce dense output (OneHotEncoder(..., sparse_output=False) or sparse=False)
    dense=False -> produce sparse output (OneHotEncoder(..., sparse_output=True) or sparse=True)

    We try the new arg name first (sparse_output) and fall back to old (sparse).
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=(not dense))
    except TypeError:
        # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=(not dense))


def _patch_forest_monotonic_attribute(pipeline_or_estimator):
    """
    Ensure tree estimators inside RandomForest-like objects have attribute
    'monotonic_cst' so predictions won't fail across sklearn version mismatches.

    Returns True if a patch was applied (or False otherwise). Non-fatal.
    """
    try:
        model = None
        if hasattr(pipeline_or_estimator, "named_steps") and "model" in pipeline_or_estimator.named_steps:
            model = pipeline_or_estimator.named_steps["model"]
        elif hasattr(pipeline_or_estimator, "steps"):
            model = dict(pipeline_or_estimator.steps).get("model")
        else:
            model = pipeline_or_estimator

        if model is None:
            return False

        # Most RandomForest implementations expose estimators_
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
            return True

        # Some wrappers might use estimator_.estimators_
        if hasattr(model, "estimator_") and hasattr(model.estimator_, "estimators_"):
            for est in model.estimator_.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
            return True

        return False
    except Exception:
        # Don't fail training for this compatibility patch
        return False


def train_calorie_model():
    print("Loading processed data...")
    train_df, test_df = load_data()

    target = "calories_per_day"
    if target not in train_df.columns:
        raise RuntimeError(f"{target} not found in {TRAIN_PATH}")

    # Numeric / categorical candidate lists
    numeric_candidates = [
        "weight_kg", "age_years", "height_cm", "length_cm",
        "activity_minutes", "activity_score", "sleep_hours", "rer", "mer_multiplier"
    ]
    categorical_candidates = ["breed_reduced", "sex", "size_group", "activity_label"]

    numeric_features = [c for c in numeric_candidates if c in train_df.columns]
    categorical_features = [c for c in categorical_candidates if c in train_df.columns]

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Drop rows missing required target and weight (we expect weight for training)
    train_df = train_df.dropna(subset=[target, "weight_kg"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[target, "weight_kg"]).reset_index(drop=True)

    # Build preprocessors
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    # produce dense one-hot output for easier joblib/pickle portability
    categorical_transformer = Pipeline([("onehot", _safe_onehot(dense=True))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop", sparse_threshold=0)

    # Prepare X / y (fillna with reasonable defaults to avoid unexpected missing values)
    feature_cols = numeric_features + categorical_features
    if len(feature_cols) == 0:
        raise RuntimeError("No numeric or categorical features found for calorie model training.")

    X_train = train_df[feature_cols].copy().fillna(0)
    y_train = train_df[target].values
    X_test = test_df[feature_cols].copy().fillna(0)
    y_test = test_df[target].values

    # Build pipeline and grid
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf)])

    param_grid = {
        "model__n_estimators": [100],
        "model__max_depth": [20, None],
        "model__min_samples_leaf": [1, 2]
    }

    print("Starting GridSearchCV (cv=3) — this can take a little while...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Evaluate on test set
    preds = best.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Compatible RMSE calculation (works across sklearn versions)
    mse = mean_squared_error(y_test, preds)   # ALWAYS available
    rmse = float(np.sqrt(mse))

    r2 = r2_score(y_test, preds)

    print("\n===== Calorie Model Performance =====")
    print(f"MAE  : {mae:.2f} kcal/day")
    print(f"RMSE : {rmse:.2f} kcal/day")
    print(f"R²   : {r2:.4f}")

    # Patch tree estimators to include monotonic_cst attribute for cross-version compatibility
    patched = _patch_forest_monotonic_attribute(best)
    if patched:
        print("Patched tree estimators with 'monotonic_cst = None' for sklearn compatibility.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

    # Save pipeline (contains preprocessor + model)
    joblib.dump(best, OUT_MODEL, compress=3)
    print(f"\nSaved calorie pipeline -> {OUT_MODEL}")

    # Save metadata (feature lists) for inference
    meta = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_params": grid.best_params_
    }
    joblib.dump(meta, OUT_META, compress=3)
    print(f"Saved calorie preprocessor metadata -> {OUT_META}")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_test": int(len(y_test)),
        "best_params": grid.best_params_
    }


if __name__ == "__main__":
    metrics = train_calorie_model()
    print("\nDone. Summary metrics returned:")
    print(metrics)
