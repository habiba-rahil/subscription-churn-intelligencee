"""
feature_engineering.py
=======================
Builds ML-ready features from the cleaned dataset.

WHY FEATURE ENGINEERING MATTERS:
  Raw columns like 'signup_date' or 'country' can't go directly
  into a machine learning model. This module transforms them into
  numerical representations that algorithms can learn from.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── RFM Feature Builder ─────────────────────────────────────────────────────────
def build_rfm_features(df: pd.DataFrame, reference_date: str = "2024-10-01") -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features.

    WHAT IS RFM:
      A classic customer analytics framework used in marketing.
      - Recency:   How recently did the customer log in?
                   Lower = better (customer is still engaged)
      - Frequency: How often does the customer use the service?
                   Higher = better
      - Monetary:  How much does the customer pay per month?
                   Higher = more valuable customer

    WHY WE USE IT:
      RFM captures the most important behavioural signals.
      It is well understood by business stakeholders.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer DataFrame.
    reference_date : str
        Date to calculate recency from (typically "today").

    Returns
    -------
    pd.DataFrame
        DataFrame with added R, F, M columns.
    """
    ref = pd.Timestamp(reference_date)
    df = df.copy()

    # Recency: days since last login (lower = more recent = better)
    if "days_since_login" not in df.columns:
        df["days_since_login"] = (ref - pd.to_datetime(df["last_login_date"])).dt.days.clip(lower=0)

    df["recency"] = df["days_since_login"]
    df["frequency"] = df["usage_frequency"]
    df["monetary"] = df["monthly_fee"]

    return df


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix for machine learning.

    FEATURES USED:
      Numeric:
        - tenure_months        (how long the customer has been subscribed)
        - usage_frequency      (sessions per month)
        - average_session_minutes
        - support_tickets      (more tickets → risk signal)
        - payment_failures     (strong churn predictor)
        - monthly_fee          (price sensitivity)
        - days_since_login     (recency proxy)
        - age

      Categorical (one-hot encoded):
        - plan
        - payment_method
        - country

    WHAT IS ONE-HOT ENCODING:
      Converts categories like "Basic", "Standard", "Premium"
      into separate binary columns (0 or 1). This is necessary
      because most ML algorithms only understand numbers.

    Returns
    -------
    pd.DataFrame
        Feature matrix (X) and target series (y) as a dict.
    """
    df = df.copy()

    # Numeric features
    numeric_features = [
        "tenure_months",
        "usage_frequency",
        "average_session_minutes",
        "support_tickets",
        "payment_failures",
        "monthly_fee",
        "days_since_login",
        "age",
    ]

    # Categorical features to encode
    categorical_features = ["plan", "payment_method", "country"]

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df[categorical_features], drop_first=False)

    # Combine numeric + encoded
    X = pd.concat([df[numeric_features].reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
    y = df["churned"].reset_index(drop=True)

    return X, y


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable age group column.

    Useful for EDA visualizations (e.g., churn rate by age group).
    """
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 24, 34, 44, 54, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    )
    return df


def add_usage_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple usage segment label based on usage_frequency.

    Segments:
      - Low:    0-4 sessions/month
      - Medium: 5-14 sessions/month
      - High:   15+ sessions/month
    """
    df = df.copy()
    df["usage_segment"] = pd.cut(
        df["usage_frequency"],
        bins=[-1, 4, 14, 9999],
        labels=["Low", "Medium", "High"],
    )
    return df
