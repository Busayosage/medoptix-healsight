import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# Load dataset
# -------------------------------
data_path = "data/processed/final_dataset_clean.csv"
df = pd.read_csv(data_path)

print(f"Loaded dataset from: {data_path}")
print(f"Dataset shape (original): {df.shape}")


# -------------------------------
# Prepare date features
# -------------------------------
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
else:
    raise ValueError("The dataset must contain a 'date' column for time-based forecasting.")


# -------------------------------
# One-hot encode ward_code
# -------------------------------
if "ward_code" in df.columns:
    ward_dummies = pd.get_dummies(df["ward_code"], prefix="ward_code")
    df = pd.concat([df, ward_dummies], axis=1)


# -------------------------------
# Create time-based features (lags)
# -------------------------------
df = df.sort_values(by=["hospital_id", "date"])

df["admissions_lag_1"] = df.groupby("hospital_id")["admissions"].shift(1)
df["admissions_lag_2"] = df.groupby("hospital_id")["admissions"].shift(2)
df["admissions_lag_3"] = df.groupby("hospital_id")["admissions"].shift(3)

df["admissions_rolling_mean_3"] = (
    df.groupby("hospital_id")["admissions"]
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Remove rows with NaN values caused by lagging
df = df.dropna().reset_index(drop=True)

print(f"Dataset shape (after lag features): {df.shape}")


# -------------------------------
# Prepare features & target
# -------------------------------
target_column = "admissions"

feature_columns = [
    "hospital_id",
    "base_beds",
    "effective_capacity",
    "staffing_index",
    "avg_wait_minutes",
    "day_of_week",
    "month",
    "year",
    "day",
    "ward_code_ICU",
    "ward_code_MED",
    "ward_code_SURG",
    "admissions_lag_1",
    "admissions_lag_2",
    "admissions_lag_3",
    "admissions_rolling_mean_3"
]

# Keep only columns that actually exist
available_feature_columns = [col for col in feature_columns if col in df.columns]

if target_column not in df.columns:
    raise ValueError(f"Missing required target column: {target_column}")

if not available_feature_columns:
    raise ValueError("No valid feature columns found in dataset.")

X = df[available_feature_columns].copy()
y = df[target_column].copy()

print(f"Using features: {available_feature_columns}")


# -------------------------------
# Time-based split
# -------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")


# -------------------------------
# Train model
# -------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -------------------------------
# Evaluate model
# -------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results (Time-Based Split)")
print("------------------------------------------")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# -------------------------------
# Save model artifacts
# -------------------------------
artifacts_dir = "model/artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "admissions_forecast_model.pkl")
features_path = os.path.join(artifacts_dir, "feature_columns.txt")

joblib.dump(model, model_path)

with open(features_path, "w", encoding="utf-8") as f:
    for col in available_feature_columns:
        f.write(f"{col}\n")

print(f"\nModel saved to: {model_path}")
print(f"Feature schema saved to: {features_path}")