"""
test_churn_model.py
===================
Tests for churn prediction model training and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing import generate_synthetic_dataset, clean_dataset
from feature_engineering import build_ml_features
from churn_model import (
    split_data,
    train_logistic_regression,
    evaluate_model,
    predict_customer_churn,
)


@pytest.fixture(scope="module")
def ml_data():
    """Prepare train/test data for model tests."""
    raw = generate_synthetic_dataset(n_customers=500, seed=42)
    clean = clean_dataset(raw)
    X, y = build_ml_features(clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_lr(ml_data):
    """Train a logistic regression model for testing."""
    X_train, X_test, y_train, y_test = ml_data
    return train_logistic_regression(X_train, y_train)


class TestTrainTestSplit:
    def test_split_sizes(self, ml_data):
        """Test set should be approximately 20% of total data."""
        X_train, X_test, y_train, y_test = ml_data
        total = len(X_train) + len(X_test)
        test_pct = len(X_test) / total
        assert 0.18 <= test_pct <= 0.22, f"Test size {test_pct:.2f} outside expected range"

    def test_no_overlap(self, ml_data):
        """Train and test sets should not share the same indices."""
        X_train, X_test, _, _ = ml_data
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx & test_idx) == 0, "Overlap between train and test sets"


class TestModelTraining:
    def test_model_has_predict_method(self, trained_lr):
        """Model should have a predict method (it's a Pipeline)."""
        assert hasattr(trained_lr, "predict")

    def test_model_has_predict_proba(self, trained_lr):
        """Model should be able to output probabilities."""
        assert hasattr(trained_lr, "predict_proba")


class TestModelEvaluation:
    def test_metrics_keys_exist(self, trained_lr, ml_data):
        """Evaluation should return all required metric keys."""
        X_train, X_test, y_train, y_test = ml_data
        metrics = evaluate_model(trained_lr, X_test, y_test, "LR Test")
        required_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_roc_auc_above_random(self, trained_lr, ml_data):
        """ROC-AUC should be above 0.5 (better than random)."""
        X_train, X_test, y_train, y_test = ml_data
        metrics = evaluate_model(trained_lr, X_test, y_test, "LR Test")
        assert metrics["roc_auc"] > 0.5, f"ROC-AUC {metrics['roc_auc']} is not above random"

    def test_metrics_in_valid_range(self, trained_lr, ml_data):
        """All metrics should be between 0 and 1."""
        X_train, X_test, y_train, y_test = ml_data
        metrics = evaluate_model(trained_lr, X_test, y_test, "LR Test")
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert 0 <= metrics[key] <= 1, f"{key} = {metrics[key]} is outside [0, 1]"


class TestPrediction:
    def test_prediction_has_probability(self, trained_lr, ml_data):
        """predict_customer_churn should return a churn_probability."""
        X_train, X_test, _, _ = ml_data
        result = predict_customer_churn(trained_lr, X_test.iloc[:1])
        assert "churn_probability" in result
        assert 0 <= result["churn_probability"] <= 1

    def test_prediction_has_risk_category(self, trained_lr, ml_data):
        """predict_customer_churn should return a risk_category."""
        X_train, X_test, _, _ = ml_data
        result = predict_customer_churn(trained_lr, X_test.iloc[:1])
        assert result["risk_category"] in ["Low", "Medium", "High"]
