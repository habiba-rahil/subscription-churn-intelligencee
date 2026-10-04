"""
segmentation.py
===============
Customer segmentation using K-Means clustering.

WHAT IS K-MEANS CLUSTERING:
  K-Means divides customers into K groups (clusters) such that
  customers within the same group are as similar as possible.

  The algorithm:
    1. Randomly place K cluster centroids.
    2. Assign each customer to the nearest centroid.
    3. Move centroids to the mean of their assigned customers.
    4. Repeat steps 2-3 until the assignments stop changing.

WHY WE USE IT:
  Clustering is unsupervised — it finds natural groups in the
  data without needing labelled training data. This helps us
  discover customer segments we didn't pre-define.

IMPORTANT NOTE:
  Cluster labels (0, 1, 2...) have no inherent meaning.
  We analyse each cluster AFTER fitting to give it a name.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

MODELS_PATH = Path(__file__).parent.parent / "models"


def build_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and prepare features for clustering.

    WHAT FEATURES WE USE:
      - recency (days_since_login): How recently was the customer active?
      - frequency (usage_frequency): How often do they use the service?
      - monetary (monthly_fee): How much do they pay?
      - support_tickets: Risk signal
      - payment_failures: Risk signal
      - tenure_months: How long have they been subscribed?

    WHY STANDARDISE:
      K-Means uses Euclidean distance. If monthly_fee (range: 5-80) is
      not scaled, it will dominate over usage_frequency (range: 0-60).
      StandardScaler converts all features to mean=0, std=1 so they
      contribute equally.

    Returns
    -------
    pd.DataFrame
        Raw feature DataFrame (before scaling).
    """
    features = df[[
        "days_since_login",
        "usage_frequency",
        "monthly_fee",
        "support_tickets",
        "payment_failures",
        "tenure_months",
    ]].copy()

    # Drop rows with any NaN in these features
    features = features.dropna()

    return features


def find_optimal_k(features: pd.DataFrame, k_range: range = range(2, 9)) -> dict:
    """
    Find the optimal number of clusters using Elbow Method and Silhouette Score.

    ELBOW METHOD:
      Plot inertia (sum of squared distances to nearest centroid) for each K.
      The "elbow" — where adding more clusters stops helping much — is the
      optimal K. Think of it as the point of diminishing returns.

    SILHOUETTE SCORE:
      Measures how similar a customer is to its own cluster vs other clusters.
      Range: -1 (wrong cluster) to +1 (perfect cluster).
      Higher is better.

    Returns
    -------
    dict with 'inertias' and 'silhouette_scores' for each K.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)

    return {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }


def fit_kmeans(features: pd.DataFrame, n_clusters: int = 4) -> tuple:
    """
    Fit K-Means and return labels + the fitted model + scaler.

    Parameters
    ----------
    features : pd.DataFrame
        Raw (unscaled) feature DataFrame.
    n_clusters : int
        Number of clusters to fit.

    Returns
    -------
    tuple: (cluster_labels, kmeans_model, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Save models
    MODELS_PATH.mkdir(exist_ok=True)
    joblib.dump(km, MODELS_PATH / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_PATH / "kmeans_scaler.pkl")

    return labels, km, scaler


def profile_clusters(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Profile each cluster by computing mean feature values.

    WHAT THIS DOES:
      After clustering, we look at the average characteristics of each
      cluster to understand what type of customers are in it.

      Example output:
        Cluster 0: high usage, low payment failures → "Loyal / High Value"
        Cluster 1: low usage, high payment failures → "At Risk"

    Returns
    -------
    pd.DataFrame
        Cluster profile table.
    """
    profile = df.groupby("cluster")[feature_cols + ["churned"]].mean().round(2)
    profile["cluster_size"] = df.groupby("cluster")["customer_id"].count()
    profile["churn_rate_%"] = (profile["churned"] * 100).round(1)
    return profile.drop(columns=["churned"])


def assign_segment_labels(profile: pd.DataFrame) -> dict:
    """
    Assign human-friendly segment labels based on cluster profiles.

    HOW LABELS ARE ASSIGNED:
      We look at each cluster's characteristics and assign a label
      that best describes the business meaning of that group.

      The labels are assigned AFTER analysis, not before.
      This ensures labels are data-driven, not arbitrary.

    Returns
    -------
    dict: {cluster_id: segment_label}
    """
    labels = {}

    for cluster_id, row in profile.iterrows():
        usage = row.get("usage_frequency", 0)
        recency = row.get("days_since_login", 0)
        failures = row.get("payment_failures", 0)
        tenure = row.get("tenure_months", 0)
        churn_rate = row.get("churn_rate_%", 0)

        if tenure < 3:
            label = "New Customers"
        elif churn_rate >= 45 or (failures >= 2 and usage <= 5):
            label = "High Risk / At Risk"
        elif usage >= 15 and churn_rate <= 20:
            label = "Loyal / High Value"
        elif usage <= 5:
            label = "Low Engagement"
        else:
            label = "Average / Stable"

        labels[cluster_id] = label

    return labels
