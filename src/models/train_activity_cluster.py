# training clustering model
"""
src/models/train_activity_cluster.py
Train KMeans clustering to categorize dog activity levels.

Usage:
    python -m src.models.train_activity_cluster
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "data/processed/train_df.csv"
OUT_KMEANS = "saved_models/activity_kmeans.pkl"
OUT_SCALER = "saved_models/activity_scaler.pkl"
OUT_META   = "saved_models/activity_cluster_metadata.pkl"

RANDOM_STATE = 42

def train_activity_cluster():
    print("Loading processed train data...")
    df = pd.read_csv(TRAIN_PATH)

    # Features used for clustering
    activity_cols = ["avg_daily_play_min", "avg_daily_walk_min", "sleep_hours"]

    for c in activity_cols:
        if c not in df.columns:
            raise RuntimeError(f"Column {c} not found in training CSV.")

    X = df[activity_cols].copy().fillna(0)

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training KMeans (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)

    # Interpret clusters using inverse transform of centroids
    centroids_raw = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroids_raw, columns=activity_cols)
    centroid_df["total_activity"] = centroid_df["avg_daily_play_min"] + centroid_df["avg_daily_walk_min"]
    print("\nCluster centroids:")
    print(centroid_df)

    # Determine label mapping based on total activity
    order = centroid_df["total_activity"].argsort().values
    cluster_map = {
        order[0]: "Low Activity",
        order[1]: "Normal",
        order[2]: "Hyperactive"
    }

    print("\nCluster label mapping:")
    print(cluster_map)

    # Save models
    joblib.dump(kmeans, OUT_KMEANS)
    joblib.dump(scaler, OUT_SCALER)
    joblib.dump({
        "activity_cols": activity_cols,
        "cluster_map": cluster_map,
        "centroids": centroid_df.to_dict()
    }, OUT_META)

    print(f"\nSaved: {OUT_KMEANS}")
    print(f"Saved: {OUT_SCALER}")
    print(f"Saved: {OUT_META}")
    print("\nActivity clustering model training complete.")

if __name__ == "__main__":
    train_activity_cluster()
