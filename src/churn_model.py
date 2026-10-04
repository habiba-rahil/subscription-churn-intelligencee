"""
churn_model.py
==============
Trains, evaluates, and explains churn prediction models.

WHAT WE ARE DOING:
  Binary classification: predicting whether a customer will churn (1) or not (0).

  We train two models:
    1. Logistic Regression — simple, interpretable, good baseline.
    2. Random Forest — ensemble of decision trees, usually more accurate.

WHY TWO MODELS:
  Comparing models helps us understand whether complexity adds value.
  Logistic Regression gives a clear baseline. If Random Forest is
  significantly better, we know complex patterns exist in the data.

IMPORTANT CONCEPT — DATA LEAKAGE:
  We must NOT include 'churn_date' or 'subscription_status' as features
  because these are consequences of churn, not causes. Using them would
  make the model appear unrealistically accurate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

MODELS_PATH = Path(__file__).parent.parent / "models"
MODELS_PATH.mkdir(exist_ok=True)


# ── Train/Test Split ────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and test sets.

    WHAT IS TRAIN/TEST SPLIT:
      We hold back 20% of data as a "test set" that the model never sees
      during training. This gives us an unbiased estimate of how the model
      will perform on new, unseen customers.

      WHY 80/20:
        Common split. Large enough training set for the model to learn,
        large enough test set to get reliable performance estimates.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ── Model Training ──────────────────────────────────────────────────────────────
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train a Logistic Regression model inside a preprocessing pipeline.

    WHAT IS LOGISTIC REGRESSION:
      Despite the name, it is a classification algorithm.
      It models the probability that a customer churns using a
      sigmoid (S-shaped) function applied to a linear combination
      of features.

      Output: probability between 0 and 1 (e.g., 0.73 = 73% churn chance).

    WHY USE A PIPELINE:
      A Pipeline chains preprocessing (scaling) and the model together.
      This prevents data leakage — scaling parameters are learned only
      from training data and applied consistently to test data.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",  # handles class imbalance
        )),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODELS_PATH / "logistic_regression.pkl")
    print("  Logistic Regression trained and saved.")
    return pipeline


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train a Random Forest model inside a preprocessing pipeline.

    WHAT IS RANDOM FOREST:
      An ensemble of many decision trees. Each tree is trained on a
      random subset of data and features. The final prediction is
      the majority vote across all trees.

      WHY IT OFTEN OUTPERFORMS LOGISTIC REGRESSION:
        It can capture non-linear relationships and interactions
        between features (e.g., "payment failures are dangerous
        only when usage is also low").

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODELS_PATH / "random_forest.pkl")
    print("  Random Forest trained and saved.")
    return pipeline


# ── Model Evaluation ────────────────────────────────────────────────────────────
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> dict:
    """
    Evaluate a trained model and return all key metrics.

    METRICS EXPLAINED:

    ACCURACY:
      Percentage of all predictions that are correct.
      Problem: If 80% of customers don't churn, a model that always
      predicts "no churn" gets 80% accuracy but is useless.

    PRECISION:
      Of all customers the model PREDICTED would churn, what fraction
      actually churned?
      Business meaning: How often are our churn alerts correct?

    RECALL:
      Of all customers who ACTUALLY churned, what fraction did the
      model correctly identify?
      Business meaning: How many at-risk customers are we catching?

    F1-SCORE:
      Harmonic mean of precision and recall. Balances both concerns.
      Use this when both false positives and false negatives matter.

    ROC-AUC:
      Area Under the Receiver Operating Characteristic Curve.
      Measures how well the model separates churners from non-churners.
      Range: 0.5 (random) to 1.0 (perfect).

    WHY RECALL MATTERS FOR CHURN:
      Missing a churned customer (false negative) is usually more costly
      than incorrectly flagging a loyal customer (false positive). So we
      care a lot about recall in churn prediction.

    Returns
    -------
    dict with all evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    print(f"\n  === {model_name} Results ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    return metrics


def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Create a clean comparison table of model performance.

    Returns
    -------
    pd.DataFrame
        Model comparison DataFrame.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
        })
    return pd.DataFrame(rows).set_index("Model")


# ── SHAP Explainability ─────────────────────────────────────────────────────────
def compute_shap_values(model: Pipeline, X_sample: pd.DataFrame, n_samples: int = 500):
    """
    Compute SHAP values for the Random Forest model.

    WHAT IS SHAP:
      SHAP (SHapley Additive exPlanations) is a method to explain
      individual model predictions.

      For each prediction, SHAP answers:
        "How much did each feature PUSH this prediction towards
        'churn' or 'no churn'?"

    WHY SHAP:
      - Model-agnostic (works with any ML model)
      - Theoretically grounded in game theory
      - Provides both GLOBAL feature importance
        (which features matter most overall) and
        LOCAL explanations (why did THIS customer get THIS score?)

    HOW TO READ SHAP VALUES:
      - Positive SHAP value → pushes prediction towards CHURN
      - Negative SHAP value → pushes prediction towards NO CHURN
      - Larger magnitude → stronger effect

    Returns
    -------
    tuple: (shap_values, X_sample_scaled, explainer)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap package is required. Run: pip install shap")

    rf_model = model.named_steps["model"]
    scaler = model.named_steps["scaler"]

    # Use a sample for speed
    if len(X_sample) > n_samples:
        X_sample = X_sample.sample(n_samples, random_state=42)

    X_scaled = scaler.transform(X_sample)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_sample.columns)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled_df)

    # For binary classification, shap_values is a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
    else:
        shap_values_churn = shap_values

    return shap_values_churn, X_scaled_df, explainer


def predict_customer_churn(model: Pipeline, customer_features: pd.DataFrame) -> dict:
    """
    Predict churn probability for one or more customers.

    Returns
    -------
    dict with churn_probability and risk_category.
    """
    prob = model.predict_proba(customer_features)[:, 1][0]

    if prob < 0.35:
        risk = "Low"
    elif prob < 0.65:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(float(prob), 4),
        "churn_probability_pct": round(float(prob) * 100, 1),
        "risk_category": risk,
    }
