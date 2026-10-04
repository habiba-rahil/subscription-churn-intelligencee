"""
data_processing.py
==================
Handles raw data generation and cleaning pipeline.

WHY THIS MODULE EXISTS:
  Real projects always separate raw data from clean data.
  This module generates the synthetic dataset and provides
  reusable cleaning functions used across notebooks and tests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "customers_raw.csv"
PROCESSED_DATA_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "customers_clean.csv"
)

PLANS = ["Basic", "Standard", "Premium", "Enterprise"]
PLAN_PRICES = {"Basic": 9.99, "Standard": 19.99, "Premium": 39.99, "Enterprise": 79.99}
COUNTRIES = ["USA", "UK", "Canada", "Germany", "France", "Australia", "India", "Brazil"]
PAYMENT_METHODS = ["Credit Card", "PayPal", "Bank Transfer", "Crypto"]
SUBSCRIPTION_STATUSES = ["active", "churned", "paused"]

RANDOM_SEED = 42


# ── Dataset Generation ──────────────────────────────────────────────────────────
def generate_synthetic_dataset(n_customers: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic customer dataset.

    WHAT IT DOES:
      Creates 5,000 fictional subscription customers with realistic
      relationships between features and churn probability.

    WHY REALISTIC RELATIONSHIPS:
      - Low usage → higher churn chance
      - Many payment failures → higher churn chance
      - More support tickets → higher churn chance
      - Higher-tier plans → somewhat lower churn (more invested)
      - Very new customers → different churn behaviour (honeymoon effect)

    Parameters
    ----------
    n_customers : int
        Number of customer records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw customer DataFrame.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    today = datetime(2024, 10, 1)

    # ── Core customer attributes ────────────────────────────────────────────────
    customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

    # Signup dates spread over 3 years
    signup_offsets = rng.integers(0, 3 * 365, n_customers)
    signup_dates = [today - timedelta(days=int(d)) for d in signup_offsets]

    ages = rng.integers(18, 75, n_customers)
    countries = rng.choice(COUNTRIES, n_customers, p=[0.35, 0.15, 0.12, 0.1, 0.08, 0.08, 0.07, 0.05])
    plans = rng.choice(PLANS, n_customers, p=[0.40, 0.30, 0.20, 0.10])
    monthly_fees = [PLAN_PRICES[p] + rng.uniform(-1, 1) for p in plans]
    payment_methods = rng.choice(PAYMENT_METHODS, n_customers, p=[0.50, 0.25, 0.15, 0.10])

    # ── Behavioural features ────────────────────────────────────────────────────
    # Usage frequency: sessions per month
    usage_frequency = np.clip(rng.normal(loc=12, scale=8, size=n_customers), 0, 60).astype(int)

    # Average session minutes
    avg_session_minutes = np.clip(
        rng.normal(loc=25, scale=15, size=n_customers), 1, 120
    ).round(1)

    # Support tickets submitted
    support_tickets = np.clip(rng.poisson(lam=1.5, size=n_customers), 0, 15).astype(int)

    # Payment failures
    payment_failures = np.clip(rng.poisson(lam=0.8, size=n_customers), 0, 10).astype(int)

    # ── Churn probability model ─────────────────────────────────────────────────
    # Build a logistic-like churn probability from features.
    # Higher value = more likely to churn.
    churn_score = np.zeros(n_customers)

    # Low usage increases churn risk
    churn_score += (20 - usage_frequency) * 0.04

    # Payment failures strongly predict churn
    churn_score += payment_failures * 0.30

    # Support tickets slightly predict churn
    churn_score += support_tickets * 0.08

    # Premium and Enterprise customers churn less
    plan_effect = np.array([
        0.15 if p == "Basic" else
        0.05 if p == "Standard" else
        -0.05 if p == "Premium" else
        -0.15
        for p in plans
    ])
    churn_score += plan_effect

    # Crypto payments correlate with higher churn
    payment_effect = np.array([
        0.10 if pm == "Crypto" else
        0.05 if pm == "Bank Transfer" else
        0.0
        for pm in payment_methods
    ])
    churn_score += payment_effect

    # Very new customers (honeymoon) — lower churn in first 60 days
    tenure_days = np.array([(today - sd).days for sd in signup_dates])
    new_customer_effect = np.where(tenure_days < 60, -0.20, 0.0)
    churn_score += new_customer_effect

    # Add random noise
    churn_score += rng.normal(0, 0.25, n_customers)

    # Convert to probability using sigmoid
    churn_prob = 1 / (1 + np.exp(-churn_score))

    # Determine churn outcome
    churned = rng.random(n_customers) < churn_prob

    # ── Derived fields for churned customers ───────────────────────────────────
    subscription_statuses = []
    churn_dates = []

    for i in range(n_customers):
        if churned[i]:
            subscription_statuses.append("churned")
            # Churn date = sometime after signup, before today
            days_to_churn = rng.integers(30, max(31, tenure_days[i]))
            cd = signup_dates[i] + timedelta(days=int(days_to_churn))
            churn_dates.append(cd.strftime("%Y-%m-%d"))
        else:
            subscription_statuses.append(
                rng.choice(["active", "paused"], p=[0.92, 0.08])
            )
            churn_dates.append(None)

    # Last login date
    last_login_offsets = np.where(
        churned,
        rng.integers(30, 365, n_customers),   # churned: longer since last login
        rng.integers(0, 60, n_customers),     # active: recent login
    )
    last_login_dates = [
        (today - timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in last_login_offsets
    ]

    # ── Introduce realistic data quality issues ─────────────────────────────────
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
        "plan": plans,
        "monthly_fee": [round(f, 2) for f in monthly_fees],
        "age": ages.tolist(),
        "country": countries.tolist(),
        "payment_method": payment_methods.tolist(),
        "usage_frequency": usage_frequency.tolist(),
        "average_session_minutes": avg_session_minutes.tolist(),
        "support_tickets": support_tickets.tolist(),
        "payment_failures": payment_failures.tolist(),
        "last_login_date": last_login_dates,
        "subscription_status": subscription_statuses,
        "churn_date": churn_dates,
        "churned": churned.astype(int).tolist(),
    })

    # Introduce ~3% missing values in key columns
    for col in ["age", "usage_frequency", "average_session_minutes", "last_login_date"]:
        mask = rng.random(n_customers) < 0.03
        df.loc[mask, col] = np.nan

    # Introduce ~1% duplicate rows
    n_dupes = int(n_customers * 0.01)
    dupe_indices = rng.integers(0, n_customers, n_dupes)
    df = pd.concat([df, df.iloc[dupe_indices]], ignore_index=True)

    # Introduce a few invalid ages
    invalid_age_idx = rng.integers(0, len(df), 20)
    df.loc[invalid_age_idx, "age"] = rng.choice([-5, 0, 130, 200], 20)

    return df


# ── Data Cleaning Pipeline ──────────────────────────────────────────────────────
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a systematic data cleaning pipeline to the raw dataset.

    CLEANING DECISIONS (documented):
      1. Remove exact duplicate rows.
      2. Parse date columns to datetime.
      3. Fix invalid age values (outside 16–100 → NaN).
      4. Impute missing numeric columns with median.
      5. Impute missing categorical columns with mode.
      6. Ensure correct dtypes.
      7. Add derived features (tenure_days, tenure_months).

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()

    # 1. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"  [Cleaning] Removed {before - len(df)} duplicate rows.")

    # 2. Parse dates
    for date_col in ["signup_date", "last_login_date", "churn_date"]:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 3. Fix invalid ages
    invalid_mask = (df["age"] < 16) | (df["age"] > 100)
    n_invalid = invalid_mask.sum()
    df.loc[invalid_mask, "age"] = np.nan
    print(f"  [Cleaning] Set {n_invalid} invalid age values to NaN.")

    # 4. Impute missing numeric values with median
    numeric_cols = ["age", "usage_frequency", "average_session_minutes"]
    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [Cleaning] Imputed {missing} missing '{col}' values with median ({median_val:.1f}).")

    # 5. Impute missing last_login_date with signup_date (conservative)
    missing_login = df["last_login_date"].isna().sum()
    if missing_login > 0:
        df["last_login_date"] = df["last_login_date"].fillna(df["signup_date"])
        print(f"  [Cleaning] Imputed {missing_login} missing 'last_login_date' with signup_date.")

    # 6. Add tenure features
    reference_date = pd.Timestamp("2024-10-01")
    df["tenure_days"] = (reference_date - df["signup_date"]).dt.days
    df["tenure_months"] = (df["tenure_days"] / 30.44).round(1)

    # 7. Add recency feature: days since last login
    df["days_since_login"] = (reference_date - df["last_login_date"]).dt.days.clip(lower=0)

    # 8. Ensure correct dtypes
    df["churned"] = df["churned"].astype(int)
    df["age"] = df["age"].round(0).astype(int)
    df["usage_frequency"] = df["usage_frequency"].round(0).astype(int)
    df["support_tickets"] = df["support_tickets"].astype(int)
    df["payment_failures"] = df["payment_failures"].astype(int)

    print(f"  [Cleaning] Final dataset: {len(df):,} rows, {df.shape[1]} columns.")
    return df


def calculate_churn_rate(df: pd.DataFrame) -> float:
    """
    Calculate the overall churn rate.

    WHAT IS CHURN RATE:
      The percentage of customers who have churned.
      Formula: churned_customers / total_customers * 100

    WHY IT MATTERS:
      The primary business KPI. Even a 5% reduction in churn
      can significantly improve revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer DataFrame with 'churned' column (0/1).

    Returns
    -------
    float
        Churn rate as a percentage (e.g., 28.5 means 28.5%).
    """
    if "churned" not in df.columns:
        raise ValueError("DataFrame must contain a 'churned' column.")
    return round(df["churned"].mean() * 100, 2)


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from disk. Generates it first if not present."""
    if not RAW_DATA_PATH.exists():
        print("Raw dataset not found. Generating...")
        df = generate_synthetic_dataset()
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Saved raw data to {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset from disk. Cleans from raw if not present."""
    if not PROCESSED_DATA_PATH.exists():
        print("Cleaned dataset not found. Running cleaning pipeline...")
        raw = load_raw_data()
        clean = clean_dataset(raw)
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Saved cleaned data to {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["signup_date", "last_login_date", "churn_date"])
    return df


if __name__ == "__main__":
    print("=== Generating and cleaning dataset ===")
    raw_df = generate_synthetic_dataset()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw data: {len(raw_df):,} rows saved to {RAW_DATA_PATH}")

    print("\nCleaning...")
    clean_df = clean_dataset(raw_df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Clean data: {len(clean_df):,} rows saved to {PROCESSED_DATA_PATH}")

    churn_rate = calculate_churn_rate(clean_df)
    print(f"\nOverall churn rate: {churn_rate:.1f}%")
