"""Quick check that the dashboard file is valid Python and all imports work."""
import sys, ast, pathlib
sys.path.insert(0, 'src')

code = pathlib.Path('dashboard/app.py').read_text(encoding='utf-8')
ast.parse(code)
print("✓ Syntax OK")

from data_processing import load_clean_data, calculate_churn_rate
from feature_engineering import build_ml_features, add_age_group
from cohort_analysis import build_cohort_table, get_monthly_churn_trend
from segmentation import build_segmentation_features, fit_kmeans, profile_clusters, assign_segment_labels
from churn_model import predict_customer_churn, split_data, train_logistic_regression, train_random_forest, evaluate_model
print("✓ All imports OK")

df = load_clean_data()
print(f"✓ Data loaded: {len(df):,} rows")

from sklearn.metrics import roc_curve
print("✓ sklearn metrics OK")
print("\n Dashboard is ready. Run:  streamlit run dashboard/app.py")
