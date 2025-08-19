import pandas as pd
import numpy as np
import joblib
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate

# Load trained model
model = joblib.load("model.joblib")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Synthetic sensitive feature
np.random.seed(42)
df['customer_region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(df))

# Features and target
target_col = 'Class'

# Only include features the model saw during training
model_features = [c for c in df.columns if c != target_col and c not in ['Time', 'customer_region']]
X = df[model_features]
y = df[target_col]

# Make predictions
y_pred = model.predict(X)

# Fairlearn metrics
metric_frame = MetricFrame(
    metrics={'TPR': true_positive_rate, 'FPR': false_positive_rate, 'SelectionRate': selection_rate},
    y_true=y,
    y_pred=y_pred,
    sensitive_features=df['customer_region']
)

# Print to console
print("=== Fairlearn Metrics by Region ===")
print(metric_frame.by_group)
print("\n=== Overall Metrics ===")
print(metric_frame.overall)

# Save metrics to file for GitHub Actions artifact
with open("fairlearn_metrics.txt", "w") as f:
    f.write("=== Fairlearn Metrics by Region ===\n")
    f.write(str(metric_frame.by_group))
    f.write("\n\n=== Overall Metrics ===\n")
    f.write(str(metric_frame.overall))

print("\n✅ Fairlearn metrics saved as fairlearn_metrics.txt")

