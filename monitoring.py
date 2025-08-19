import pandas as pd
import numpy as np
import joblib
import shap
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.joblib")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Add synthetic sensitive feature
np.random.seed(42)
df['customer_region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(df))

target_col = 'Class'
# Only include features the model saw during training
model_features = [c for c in df.columns if c != target_col and c not in ['Time', 'customer_region']]
X = df[model_features]
y = df[target_col]

# --- Original Metrics ---
y_pred = model.predict(X)
metric_orig = MetricFrame(
    metrics={'TPR': true_positive_rate, 'FPR': false_positive_rate, 'SelectionRate': selection_rate},
    y_true=y, y_pred=y_pred,
    sensitive_features=df['customer_region']
)
print("=== Original Metrics ===")
print(metric_orig.by_group)

# --- Simulate Drift ---
X_drift = X.copy()
X_drift['V1'] = X_drift['V1'] * 1.5
X_drift['V2'] = X_drift['V2'] + 0.5

y_pred_drift = model.predict(X_drift)
metric_drift = MetricFrame(
    metrics={'TPR': true_positive_rate, 'FPR': false_positive_rate, 'SelectionRate': selection_rate},
    y_true=y, y_pred=y_pred_drift,
    sensitive_features=df['customer_region']
)
print("\n=== Metrics After Input Drift ===")
print(metric_drift.by_group)

# --- SHAP Explanation ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.iloc[:100])

# Summary plot
shap.summary_plot(shap_values, X.iloc[:100], show=False)
plt.savefig("shap_summary.png")
print("\n✅ SHAP summary plot saved as shap_summary.png")

# Force plot for first sample
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0,:],
    X.iloc[0,:],
    matplotlib=True
)
plt.savefig("shap_force_plot.png")
print("✅ SHAP force plot for first sample saved as shap_force_plot.png")

