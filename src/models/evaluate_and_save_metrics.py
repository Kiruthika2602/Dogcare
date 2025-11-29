# src/models/evaluate_and_save_metrics.py
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_DIR = os.path.join(PROJECT_ROOT, "saved_models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_PATH = os.path.join(SAVED_DIR, "metrics.json")

# Load artifacts
weight_pipe = joblib.load(os.path.join(SAVED_DIR, "weight_model.pkl"))
calorie_pipe = joblib.load(os.path.join(SAVED_DIR, "calorie_model_pipeline.pkl"))
kmeans = joblib.load(os.path.join(SAVED_DIR, "activity_kmeans.pkl"))
activity_scaler = joblib.load(os.path.join(SAVED_DIR, "activity_scaler.pkl"))
activity_meta = joblib.load(os.path.join(SAVED_DIR, "activity_cluster_metadata.pkl"))
activity_cols = activity_meta.get("activity_cols", ["avg_daily_play_min", "avg_daily_walk_min", "sleep_hours"])

# Load processed data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_df.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_df.csv"))

metrics = {
    "weight_model": {},
    "calorie_model": {},
    "activity_cluster": {}
}

# ---- Weight regression metrics (use test set rows that have weight) ----
df_w = test_df.dropna(subset=["weight_kg"]).copy()
if not df_w.empty:
    Xw = df_w.drop(columns=["weight_kg"])
    yw = df_w["weight_kg"]
    preds_w = weight_pipe.predict(Xw)
    mae_w = float(mean_absolute_error(yw, preds_w))
    rmse_w = float(mean_squared_error(yw, preds_w, squared=False))
    r2_w = float(r2_score(yw, preds_w))
    metrics["weight_model"].update({
        "mae": mae_w,
        "rmse": rmse_w,
        "r2": r2_w,
        "n_test": int(len(yw)),
        "notes": "MAE in kg; use prediction +/- MAE as a simple confidence band"
    })
else:
    metrics["weight_model"]["note"] = "no weight_kg in test set"

# ---- Calorie regression metrics (if present) ----
if "calories_per_day" in test_df.columns:
    df_c = test_df.dropna(subset=["calories_per_day"]).copy()
    if not df_c.empty:
        Xc = df_c.drop(columns=["calories_per_day"])
        yc = df_c["calories_per_day"]
        preds_c = calorie_pipe.predict(Xc)
        mae_c = float(mean_absolute_error(yc, preds_c))
        rmse_c = float(mean_squared_error(yc, preds_c, squared=False))
        r2_c = float(r2_score(yc, preds_c))
        metrics["calorie_model"].update({
            "mae": mae_c,
            "rmse": rmse_c,
            "r2": r2_c,
            "n_test": int(len(yc)),
            "notes": "MAE in kcal/day; use prediction +/- MAE as simple confidence band"
        })
    else:
        metrics["calorie_model"]["note"] = "no calories_per_day in test set"
else:
    metrics["calorie_model"]["note"] = "calories_per_day not present in test_df"

# ---- Activity clustering metrics ----
# Use training data (or test) activity features
df_act = train_df.copy()
if not df_act.empty:
    X_act = df_act[activity_cols].fillna(0)
    Xs = activity_scaler.transform(X_act)
    labels = kmeans.predict(Xs)
    try:
        sil = float(silhouette_score(Xs, labels))
    except Exception:
        sil = None
    # cluster sizes and centroids
    unique, counts = np.unique(labels, return_counts=True)
    sizes = {int(u): int(c) for u, c in zip(unique, counts)}
    centroids = kmeans.cluster_centers_.tolist()
    metrics["activity_cluster"].update({
        "silhouette": sil,
        "inertia": float(kmeans.inertia_),
        "cluster_sizes": sizes,
        "centroids": centroids,
        "activity_cols": activity_cols
    })
else:
    metrics["activity_cluster"]["note"] = "no activity data"

# Save metrics JSON
with open(OUT_PATH, "w", encoding="utf8") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics to:", OUT_PATH)
