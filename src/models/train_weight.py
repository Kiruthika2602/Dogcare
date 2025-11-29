"""
Corrected train_weight.py

This trains a Ridge model for weight prediction WITHOUT using weight_kg as an input feature.
It also saves the preprocessor used for inference alongside the model.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_processing.preprocess import build_preprocessor


TRAIN_PATH = "data/processed/train_df.csv"
TEST_PATH = "data/processed/test_df.csv"
OUTPUT_MODEL_PATH = "saved_models/weight_model.pkl"
OUTPUT_PREPROC_PATH = "saved_models/weight_preprocessor_for_weight_model.pkl"
RANDOM_STATE = 42

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def train_weight_model():
    print("Loading data...")
    train_df, test_df = load_data()

    # Drop rows missing the target
    train_df = train_df.dropna(subset=["weight_kg"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["weight_kg"]).reset_index(drop=True)

    # Build a preprocessor that DOES NOT include weight_kg as an input.
    # We call build_preprocessor on a DataFrame where 'weight_kg' column is removed,
    # so the returned preprocessor will not expect it.
    train_for_preproc = train_df.drop(columns=["weight_kg"], errors="ignore")
    preprocessor, numeric_features, categorical_features = build_preprocessor(train_for_preproc)

    print("Fitting preprocessor on train (without weight_kg)...")
    preprocessor.fit(train_for_preproc)

    # Save the preprocessor used for inference for the weight model
    joblib.dump({
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }, OUTPUT_PREPROC_PATH)
    print(f"Saved weight-model preprocessor -> {OUTPUT_PREPROC_PATH}")

    # Prepare X/y for training the weight model
    X_train = train_for_preproc.copy()
    y_train = train_df["weight_kg"].values

    # For evaluation, make sure we transform test set in the same way:
    X_test = test_df.drop(columns=["weight_kg"], errors="ignore").copy()
    y_test = test_df["weight_kg"].values

    # Build pipeline
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print("Training Ridge model for weight prediction...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print("\n===== Weight Model Performance (no leakage) =====")
    print(f"MAE  : {mae:.4f} kg")
    print(f"RMSE : {rmse:.4f} kg")
    print(f"R²   : {r2:.4f}")

    # Save full pipeline (preprocessor + model)
    joblib.dump(pipeline, OUTPUT_MODEL_PATH)
    print(f"Saved weight prediction pipeline -> {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    train_weight_model()
