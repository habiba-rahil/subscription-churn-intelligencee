"""
test_feature_engineering.py
============================
Tests for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing import generate_synthetic_dataset, clean_dataset
from feature_engineering import build_rfm_features, build_ml_features, add_age_group


@pytest.fixture
def clean_df():
    raw = generate_synthetic_dataset(n_customers=300, seed=42)
    return clean_dataset(raw)


class TestRFMFeatures:
    def test_rfm_columns_exist(self, clean_df):
        """RFM features should add recency, frequency, monetary columns."""
        df = build_rfm_features(clean_df)
        assert "recency" in df.columns
        assert "frequency" in df.columns
        assert "monetary" in df.columns

    def test_recency_is_non_negative(self, clean_df):
        """Recency (days since login) should be >= 0."""
        df = build_rfm_features(clean_df)
        assert (df["recency"] >= 0).all()

    def test_monetary_matches_monthly_fee(self, clean_df):
        """Monetary should equal monthly_fee."""
        df = build_rfm_features(clean_df)
        pd.testing.assert_series_equal(df["monetary"], df["monthly_fee"], check_names=False)


class TestMLFeatures:
    def test_returns_x_and_y(self, clean_df):
        """build_ml_features should return a tuple (X, y)."""
        result = build_ml_features(clean_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_x_has_no_missing_values(self, clean_df):
        """Feature matrix X should have no NaN values."""
        X, y = build_ml_features(clean_df)
        assert X.isna().sum().sum() == 0, "X contains NaN values"

    def test_y_is_binary(self, clean_df):
        """Target y should only contain 0 and 1."""
        X, y = build_ml_features(clean_df)
        assert set(y.unique()).issubset({0, 1})

    def test_x_y_same_length(self, clean_df):
        """X and y must have the same number of rows."""
        X, y = build_ml_features(clean_df)
        assert len(X) == len(y)

    def test_numeric_features_included(self, clean_df):
        """Core numeric features must be in X."""
        X, _ = build_ml_features(clean_df)
        for col in ["tenure_months", "usage_frequency", "payment_failures"]:
            assert col in X.columns, f"Missing feature: {col}"


class TestAgeGroup:
    def test_age_group_column_created(self, clean_df):
        """add_age_group should create an age_group column."""
        df = add_age_group(clean_df)
        assert "age_group" in df.columns

    def test_age_group_no_nulls(self, clean_df):
        """All ages should map to a valid age group."""
        df = add_age_group(clean_df)
        # After imputation, all ages should be valid
        assert df["age_group"].isna().sum() == 0
