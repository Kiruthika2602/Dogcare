# preprocessing functions
"""
src/data_processing/preprocess.py

Usage:
    python src/data_processing/preprocess.py

This script expects the raw CSV at:
    data/raw/dogcare_synthetic.csv

If you downloaded the dataset to a different path (e.g. /mnt/data/dogcare_synthetic.csv),
the script will attempt to copy it into data/raw/ automatically.

Outputs:
- data/processed/train_df.csv
- data/processed/test_df.csv
- saved_models/preprocessor_pipeline.pkl
- saved_models/activity_scaler.pkl
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

RANDOM_STATE = 42

# Paths (relative to project root)
DEFAULT_RAW = "data/raw/dogcare_synthetic.csv"
# Known location from generation step (if user didn't move file)
FALLBACK_RAW = "/mnt/data/dogcare_synthetic.csv"

PROCESSED_DIR = "data/processed"
SAVED_MODELS_DIR = "saved_models"


def ensure_dirs():
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    Path(SAVED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)


def _copy_fallback_raw_if_needed():
    """
    If DEFAULT_RAW does not exist but FALLBACK_RAW exists (from earlier generation),
    copy it into data/raw/.
    """
    if not os.path.exists(DEFAULT_RAW) and os.path.exists(FALLBACK_RAW):
        print(f"Copying fallback raw file {FALLBACK_RAW} -> {DEFAULT_RAW}")
        shutil.copy(FALLBACK_RAW, DEFAULT_RAW)


def load_raw(path=DEFAULT_RAW):
    if not os.path.exists(path):
        _copy_fallback_raw_if_needed()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found. Place your CSV at {path} or at {FALLBACK_RAW}."
        )
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: ensure data types, fix column names, drop duplicates (if present)
    """
    df = df.copy()

    # Standardize column names (lowercase)
    df.columns = [c.strip() for c in df.columns]

    # Expected columns (these were created in synthetic generator). Handle missing gracefully.
    expected = [
        "dog_id",
        "date",
        "breed",
        "sex",
        "age_months",
        "age_years",
        "height_cm",
        "length_cm",
        "weight_kg",
        "avg_daily_play_min",
        "avg_daily_walk_min",
        "activity_minutes",
        "activity_score_percent",
        "sleep_hours",
        "rer",
        "mer_multiplier",
        "calories_per_day",
        "activity_cluster",
        "activity_label",
    ]

    # If some expected columns are missing, continue but warn
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print("Warning: missing expected columns:", missing)

    # Cast numeric columns safely
    numeric_cols = [
        "age_months",
        "age_years",
        "height_cm",
        "length_cm",
        "weight_kg",
        "avg_daily_play_min",
        "avg_daily_walk_min",
        "activity_minutes",
        "activity_score_percent",
        "sleep_hours",
        "rer",
        "mer_multiplier",
        "calories_per_day",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sex: standardize
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.upper().str.strip().replace({"MALE": "M", "FEMALE": "F"})
        df.loc[~df["sex"].isin(["M", "F"]), "sex"] = np.nan

    # breed: strip whitespace
    if "breed" in df.columns:
        df["breed"] = df["breed"].astype(str).str.strip()

    # date: parse
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            # keep as is if can't parse
            pass

    # drop exact duplicates (if any)
    before = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < before:
        print(f"Dropped {before - df.shape[0]} duplicate rows")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features used by models:
    - age_years (if missing)
    - activity_minutes (play + walk)
    - activity_score (0-100)
    - size_group: small / medium / large (based on height_cm if available else weight)
    """
    df = df.copy()

    if ("age_years" not in df.columns or df["age_years"].isna().all()) and "age_months" in df.columns:
        df["age_years"] = (df["age_months"] / 12.0).round(3)

    # activity_minutes (fallback to avg_daily_play_min + avg_daily_walk_min)
    if "activity_minutes" not in df.columns or df["activity_minutes"].isna().all():
        play = df["avg_daily_play_min"] if "avg_daily_play_min" in df.columns else 0
        walk = df["avg_daily_walk_min"] if "avg_daily_walk_min" in df.columns else 0
        df["activity_minutes"] = (play.fillna(0) + walk.fillna(0)).round(1)

    if "activity_score_percent" not in df.columns or df["activity_score_percent"].isna().all():
        df["activity_score_percent"] = ((df["activity_minutes"] / (24 * 60)) * 100).round(3)

    # size_group: use height if present, else weight heuristics
    def size_group_row(r):
        if pd.notna(r.get("height_cm")):
            h = r["height_cm"]
            if h < 30:
                return "small"
            elif h < 50:
                return "medium"
            else:
                return "large"
        elif pd.notna(r.get("weight_kg")):
            w = r["weight_kg"]
            if w < 7:
                return "small"
            elif w < 25:
                return "medium"
            else:
                return "large"
        else:
            return "medium"

    df["size_group"] = df.apply(size_group_row, axis=1)

    # keep limited breed categories: top N breeds + 'other' to reduce cardinality
    if "breed" in df.columns:
        top_breeds = df["breed"].value_counts().nlargest(12).index.tolist()
        df["breed_reduced"] = df["breed"].where(df["breed"].isin(top_breeds), other="Other")
    else:
        df["breed_reduced"] = "Other"

    # sanity bounds
    if "weight_kg" in df.columns:
        df["weight_kg"] = df["weight_kg"].clip(lower=0.4)  # avoid zeros or negatives

    # Drop/rename if needed for consistency
    # e.g., rename activity_score_percent -> activity_score
    if "activity_score_percent" in df.columns:
        df = df.rename(columns={"activity_score_percent": "activity_score"})

    return df


def build_preprocessor(df: pd.DataFrame):
    """
    Build a ColumnTransformer pipeline:
    - onehot encode breed_reduced and sex
    - standard scale numeric features
    Returns (preprocessor, numeric_features, categorical_features)
    """
    # Features we plan to use for weight and calorie models
    numeric_features = []
    for c in [
        "age_years",
        "height_cm",
        "length_cm",
        "weight_kg",  # used as feature for calorie model; for weight model this will be target
        "activity_minutes",
        "activity_score",
        "sleep_hours",
    ]:
        if c in df.columns:
            numeric_features.append(c)

    categorical_features = []
    for c in ["breed_reduced", "sex", "size_group"]:
        if c in df.columns:
            categorical_features.append(c)

    # numeric pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return preprocessor, numeric_features, categorical_features


def split_and_save(df: pd.DataFrame, test_size=0.2, random_state=RANDOM_STATE):
    """
    Split the dataframe into train/test, save CSVs, and fit+save preprocessor.
    Returns paths to saved artifacts.
    """
    ensure_dirs()

    # Shuffle then split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)

    train_path = os.path.join(PROCESSED_DIR, "train_df.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_df.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved processed train ({train_df.shape[0]}) -> {train_path}")
    print(f"Saved processed test  ({test_df.shape[0]}) -> {test_path}")

    # Build and fit preprocessor on train set
    preprocessor, numeric_features, categorical_features = build_preprocessor(train_df)
    print("Fitting preprocessor (this may take a moment)...")
    preprocessor.fit(train_df)

    preproc_path = os.path.join(SAVED_MODELS_DIR, "preprocessor_pipeline.pkl")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        preproc_path,
    )
    print(f"Saved preprocessor pipeline -> {preproc_path}")

    # Fit activity scaler (for clustering) on activity features
    from sklearn.preprocessing import StandardScaler

    activity_cols = [c for c in ["avg_daily_play_min", "avg_daily_walk_min", "sleep_hours"] if c in train_df.columns]
    activity_scaler = StandardScaler()
    if activity_cols:
        activity_scaler.fit(train_df[activity_cols].fillna(0))
        act_scaler_path = os.path.join(SAVED_MODELS_DIR, "activity_scaler.pkl")
        joblib.dump({"activity_cols": activity_cols, "scaler": activity_scaler}, act_scaler_path)
        print(f"Saved activity scaler -> {act_scaler_path}")
    else:
        print("No activity columns found to fit activity scaler.")

    return {
        "train_csv": train_path,
        "test_csv": test_path,
        "preprocessor_pkl": preproc_path,
        "activity_scaler_pkl": os.path.join(SAVED_MODELS_DIR, "activity_scaler.pkl") if activity_cols else None,
    }


def main():
    ensure_dirs()
    df = load_raw()
    df = clean_and_cast(df)
    df = feature_engineering(df)
    artifacts = split_and_save(df)
    print("Preprocessing complete. Artifacts:")
    for k, v in artifacts.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
