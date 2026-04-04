import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load predictions file
# -----------------------------
data_path = "outputs/data/predictions.csv"
df = pd.read_csv(data_path)

print(f"Loaded predictions from: {data_path}")
print(f"Dataset shape: {df.shape}")

# -----------------------------
# Define columns
# -----------------------------
actual_column = "admissions"
predicted_column = "predicted_admissions"

if actual_column not in df.columns:
    raise ValueError(f"Actual column '{actual_column}' not found")

if predicted_column not in df.columns:
    raise ValueError(f"Predicted column '{predicted_column}' not found")

# -----------------------------
# Extract values
# -----------------------------
y_true = df[actual_column]
y_pred = df[predicted_column]

# -----------------------------
# Calculate metrics
# -----------------------------
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_true, y_pred)

print("\nEvaluation Results")
print("-" * 30)
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# -----------------------------
# Save results
# -----------------------------
output_path = "outputs/data/evaluation_metrics.txt"

with open(output_path, "w") as f:
    f.write("Model Evaluation Results\n")
    f.write("-" * 30 + "\n")
    f.write(f"MAE:  {mae:.4f}\n")
    f.write(f"MSE:  {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2:   {r2:.4f}\n")

print(f"\nEvaluation saved to: {output_path}")