"""
cohort_analysis.py
==================
Implements customer cohort retention analysis.

WHAT IS COHORT ANALYSIS:
  Customers are grouped by the month they signed up (their "cohort").
  Then we track what percentage of each cohort is still active
  in subsequent months.

WHY IT MATTERS:
  Cohort analysis answers: "Are we getting better at retaining customers
  over time?" It separates the performance of different customer groups
  so you are not averaging across very different behaviours.

EXAMPLE:
  January 2023 cohort: 200 customers signed up.
  By month 3: 160 are still active → 80% retention.
  By month 6: 130 are still active → 65% retention.
"""

import pandas as pd
import numpy as np


def build_cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cohort retention table.

    WHAT IT PRODUCES:
      A matrix where:
        - Each ROW is a signup cohort (month-year).
        - Each COLUMN is the number of months since signup (0, 1, 2, ...).
        - Each CELL is the retention rate (%) for that cohort at that age.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer DataFrame. Must have 'signup_date' and 'churned'.

    Returns
    -------
    pd.DataFrame
        Cohort retention table as percentage values.
    """
    df = df.copy()

    # Create cohort label = year-month of signup
    df["cohort"] = df["signup_date"].dt.to_period("M")

    # Calculate the reference date
    reference_date = pd.Timestamp("2024-10-01")

    # For non-churned customers, they are still active at reference_date
    # For churned customers, their "last active month" is their churn_date
    df["last_active_date"] = df.apply(
        lambda row: row["churn_date"] if pd.notna(row.get("churn_date")) and row["churned"] == 1
        else reference_date,
        axis=1,
    )
    df["last_active_date"] = pd.to_datetime(df["last_active_date"])

    # Build monthly retention counts per cohort
    cohort_data = []

    cohorts = sorted(df["cohort"].unique())

    for cohort in cohorts:
        cohort_df = df[df["cohort"] == cohort]
        cohort_size = len(cohort_df)
        cohort_start = cohort.to_timestamp()

        # How many months is this cohort "alive"?
        max_months = int((reference_date - cohort_start).days / 30) + 1

        for month_num in range(min(max_months, 25)):  # cap at 24 months shown
            check_date = cohort_start + pd.DateOffset(months=month_num)
            # A customer is "retained" at month_num if they were still active at check_date
            retained = cohort_df[
                cohort_df["last_active_date"] >= check_date
            ]
            retention_rate = round(len(retained) / cohort_size * 100, 1)
            cohort_data.append({
                "cohort": str(cohort),
                "month_number": month_num,
                "cohort_size": cohort_size,
                "retained_count": len(retained),
                "retention_rate": retention_rate,
            })

    cohort_long = pd.DataFrame(cohort_data)

    # Pivot to wide format for heatmap
    cohort_pivot = cohort_long.pivot_table(
        index="cohort",
        columns="month_number",
        values="retention_rate",
    )

    # Rename columns to "Month 0", "Month 1", etc.
    cohort_pivot.columns = [f"Month {c}" for c in cohort_pivot.columns]

    return cohort_pivot


def build_cohort_sizes(df: pd.DataFrame) -> pd.Series:
    """
    Return the number of customers per signup cohort.

    Useful for annotating the heatmap.
    """
    df = df.copy()
    df["cohort"] = df["signup_date"].dt.to_period("M").astype(str)
    return df.groupby("cohort")["customer_id"].count().rename("cohort_size")


def get_monthly_churn_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall churn trend by month.

    Returns a DataFrame with columns:
      - month: year-month period
      - new_customers: signups that month
      - churned_customers: churns that month
      - churn_rate: churns / (total active at start of month)
    """
    df = df.copy()
    df["signup_month"] = df["signup_date"].dt.to_period("M")

    # Monthly new signups
    monthly_signups = (
        df.groupby("signup_month")["customer_id"]
        .count()
        .reset_index()
        .rename(columns={"customer_id": "new_customers"})
    )

    # Monthly churns
    churned_df = df[df["churned"] == 1].copy()
    churned_df["churn_month"] = pd.to_datetime(
        churned_df["churn_date"], errors="coerce"
    ).dt.to_period("M")
    monthly_churns = (
        churned_df.groupby("churn_month")["customer_id"]
        .count()
        .reset_index()
        .rename(columns={"customer_id": "churned_customers", "churn_month": "signup_month"})
    )

    trend = monthly_signups.merge(monthly_churns, on="signup_month", how="left")
    trend["churned_customers"] = trend["churned_customers"].fillna(0).astype(int)
    trend["churn_rate"] = (trend["churned_customers"] / trend["new_customers"] * 100).round(1)
    trend["month"] = trend["signup_month"].astype(str)

    return trend[["month", "new_customers", "churned_customers", "churn_rate"]]
