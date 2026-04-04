import os
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load model and feature schema
# -------------------------------
model_path = "model/artifacts/admissions_forecast_model.pkl"
features_path = "model/artifacts/feature_columns.txt"
data_path = "data/processed/final_dataset_clean.csv"

model = joblib.load(model_path)
print(f"Loaded model from: {model_path}")

with open(features_path, "r", encoding="utf-8") as f:
    expected_features = [line.strip() for line in f if line.strip()]

print(f"Loaded feature schema from: {features_path}")
print(f"Expected feature count: {len(expected_features)}")

# -------------------------------
# Load processed dataset
# -------------------------------
processed_df = pd.read_csv(data_path)
print(f"Loaded processed dataset from: {data_path}")
print(f"Dataset shape: {processed_df.shape}")

# -------------------------------
# Prepare date features
# -------------------------------
if "date" in processed_df.columns:
    processed_df["date"] = pd.to_datetime(processed_df["date"])
    processed_df["year"] = processed_df["date"].dt.year
    processed_df["month"] = processed_df["date"].dt.month
    processed_df["day"] = processed_df["date"].dt.day
    processed_df["day_of_week"] = processed_df["date"].dt.dayofweek
else:
    raise ValueError("The dataset must contain a 'date' column for inference.")

# -------------------------------
# One-hot encode ward_code
# -------------------------------
if "ward_code" in processed_df.columns:
    ward_dummies = pd.get_dummies(processed_df["ward_code"], prefix="ward_code")
    processed_df = pd.concat([processed_df, ward_dummies], axis=1)

# -------------------------------
# Create lag features
# -------------------------------
processed_df = processed_df.sort_values(by=["hospital_id", "date"])

processed_df["admissions_lag_1"] = processed_df.groupby("hospital_id")["admissions"].shift(1)
processed_df["admissions_lag_2"] = processed_df.groupby("hospital_id")["admissions"].shift(2)
processed_df["admissions_lag_3"] = processed_df.groupby("hospital_id")["admissions"].shift(3)

processed_df["admissions_rolling_mean_3"] = (
    processed_df.groupby("hospital_id")["admissions"]
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Drop rows with NaNs caused by lagging
processed_df = processed_df.dropna().reset_index(drop=True)

# -------------------------------
# Align inference features
# -------------------------------
for col in expected_features:
    if col not in processed_df.columns:
        processed_df[col] = 0

aligned_df = processed_df[expected_features].copy()
print(f"Aligned feature shape: {aligned_df.shape}")

# -------------------------------
# Predict full dataset
# -------------------------------
predictions = model.predict(aligned_df)

output_df = pd.DataFrame({
    "predicted_admissions": predictions
})

output_dir = "outputs/data"
os.makedirs(output_dir, exist_ok=True)

predictions_path = os.path.join(output_dir, "predictions.csv")
output_df.to_csv(predictions_path, index=False)

print(f"Predictions saved to: {predictions_path}")
print(output_df.head())

# -------------------------------
# Future forecast (next step)
# -------------------------------
next_input = pd.DataFrame([aligned_df.iloc[-1].copy()])
next_prediction = model.predict(next_input)[0]

print("\n--- FUTURE FORECAST (NEXT STEP) ---")
print(f"Next predicted admissions: {next_prediction:.2f}")

# -------------------------------
# Multi-step recursive forecast
# -------------------------------
print("\n--- MULTI-STEP FORECAST (NEXT 7 STEPS) ---")

n_steps = 7

last_row = aligned_df.iloc[-1].copy()
history = list(processed_df["admissions"].tail(3))

future_preds = []

for step in range(n_steps):
    last_row["admissions_lag_1"] = history[-1]
    last_row["admissions_lag_2"] = history[-2]
    last_row["admissions_lag_3"] = history[-3]
    last_row["admissions_rolling_mean_3"] = np.mean(history[-3:])

    input_df = pd.DataFrame([last_row])[expected_features]

    pred = model.predict(input_df)[0]
    future_preds.append(pred)
    history.append(pred)

    print(f"Step {step + 1}: {pred:.2f}")