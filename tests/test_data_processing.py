"""
test_data_processing.py
=======================
Tests for data generation and cleaning pipeline.

WHY WE WRITE TESTS:
  Tests catch bugs early, especially when you refactor code.
  Even simple tests show interviewers that you think about code quality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing import (
    generate_synthetic_dataset,
    clean_dataset,
    calculate_churn_rate,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df():
    """Generate a small raw dataset for testing."""
    return generate_synthetic_dataset(n_customers=500, seed=42)


@pytest.fixture
def clean_df(raw_df):
    """Clean the raw dataset for testing."""
    return clean_dataset(raw_df)


# ── Dataset Generation Tests ────────────────────────────────────────────────────
class TestDataGeneration:
    def test_dataset_has_correct_columns(self, raw_df):
        """Dataset must have all required columns."""
        required_columns = [
            "customer_id", "signup_date", "plan", "monthly_fee",
            "age", "country", "payment_method", "usage_frequency",
            "average_session_minutes", "support_tickets",
            "payment_failures", "last_login_date",
            "subscription_status", "churn_date", "churned",
        ]
        for col in required_columns:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_dataset_size_approximate(self, raw_df):
        """Dataset should have approximately n_customers rows (plus duplicates)."""
        # We add ~1% duplicates in generation, so size > 500
        assert len(raw_df) >= 500

    def test_churned_is_binary(self, raw_df):
        """churned column must only contain 0 and 1."""
        unique_vals = set(raw_df["churned"].dropna().unique())
        assert unique_vals.issubset({0, 1}), f"Non-binary values in churned: {unique_vals}"

    def test_plans_are_valid(self, raw_df):
        """Plans must be one of the known plan names."""
        valid_plans = {"Basic", "Standard", "Premium", "Enterprise"}
        actual_plans = set(raw_df["plan"].unique())
        assert actual_plans.issubset(valid_plans), f"Invalid plans: {actual_plans - valid_plans}"

    def test_monthly_fee_is_positive(self, raw_df):
        """Monthly fee should always be positive."""
        assert (raw_df["monthly_fee"] > 0).all(), "Some monthly fees are not positive"

    def test_reproducible_with_seed(self):
        """Same seed should produce same dataset."""
        df1 = generate_synthetic_dataset(n_customers=100, seed=99)
        df2 = generate_synthetic_dataset(n_customers=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_dataset_has_realistic_churn_rate(self, raw_df):
        """Churn rate should be between 10% and 80% (plausible synthetic range)."""
        churn_rate = raw_df["churned"].mean() * 100
        assert 10 <= churn_rate <= 80, f"Churn rate {churn_rate:.1f}% outside realistic range"


# ── Data Cleaning Tests ─────────────────────────────────────────────────────────
class TestDataCleaning:
    def test_no_duplicate_customer_ids(self, clean_df):
        """After cleaning, there should be no duplicate customer_ids."""
        assert clean_df["customer_id"].duplicated().sum() == 0

    def test_no_invalid_ages(self, clean_df):
        """After cleaning, all ages should be between 16 and 100."""
        valid_mask = (clean_df["age"] >= 16) & (clean_df["age"] <= 100)
        assert valid_mask.all(), "Some ages are outside valid range after cleaning"

    def test_tenure_days_is_positive(self, clean_df):
        """Tenure in days should always be >= 0."""
        assert (clean_df["tenure_days"] >= 0).all()

    def test_tenure_columns_exist(self, clean_df):
        """Cleaned data should have tenure_days and tenure_months columns."""
        assert "tenure_days" in clean_df.columns
        assert "tenure_months" in clean_df.columns

    def test_no_missing_usage_frequency(self, clean_df):
        """usage_frequency should have no missing values after cleaning."""
        assert clean_df["usage_frequency"].isna().sum() == 0

    def test_no_missing_age(self, clean_df):
        """age should have no missing values after cleaning."""
        assert clean_df["age"].isna().sum() == 0

    def test_signup_date_is_datetime(self, clean_df):
        """signup_date should be datetime after cleaning."""
        assert pd.api.types.is_datetime64_any_dtype(clean_df["signup_date"])

    def test_churned_is_integer(self, clean_df):
        """churned should be integer type after cleaning."""
        assert clean_df["churned"].dtype in [np.int32, np.int64, int]


# ── Churn Rate Tests ────────────────────────────────────────────────────────────
class TestChurnRate:
    def test_churn_rate_is_float(self, clean_df):
        """calculate_churn_rate should return a float."""
        rate = calculate_churn_rate(clean_df)
        assert isinstance(rate, float)

    def test_churn_rate_range(self, clean_df):
        """Churn rate should be between 0 and 100."""
        rate = calculate_churn_rate(clean_df)
        assert 0 <= rate <= 100, f"Churn rate {rate} out of range"

    def test_churn_rate_all_churned(self):
        """If all customers churned, rate should be 100."""
        df = pd.DataFrame({"churned": [1, 1, 1, 1, 1]})
        assert calculate_churn_rate(df) == 100.0

    def test_churn_rate_none_churned(self):
        """If no customers churned, rate should be 0."""
        df = pd.DataFrame({"churned": [0, 0, 0, 0, 0]})
        assert calculate_churn_rate(df) == 0.0

    def test_churn_rate_missing_column(self):
        """Should raise ValueError if 'churned' column is missing."""
        df = pd.DataFrame({"age": [25, 30]})
        with pytest.raises(ValueError):
            calculate_churn_rate(df)
