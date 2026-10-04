"""Quick script to train and save all models."""
import sys
sys.path.insert(0, 'src')

from data_processing import load_clean_data
from feature_engineering import build_ml_features
from churn_model import split_data, train_logistic_regression, train_random_forest, evaluate_model

print("Loading data...")
df = load_clean_data()
X, y = build_ml_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

print("Training Logistic Regression...")
lr = train_logistic_regression(X_train, y_train)

print("Training Random Forest...")
rf = train_random_forest(X_train, y_train)

print("Evaluating...")
lr_m = evaluate_model(lr, X_test, y_test, "Logistic Regression")
rf_m = evaluate_model(rf, X_test, y_test, "Random Forest")
print(f"LR  ROC-AUC: {lr_m['roc_auc']}")
print(f"RF  ROC-AUC: {rf_m['roc_auc']}")
print("Done. Models saved to models/")
