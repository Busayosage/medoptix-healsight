import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/processed/final_dataset_clean.csv"
MODEL_OUTPUT_PATH = "model/artifacts/admissions_forecast_model.pkl"
FEATURES_OUTPUT_PATH = "model/artifacts/feature_columns.txt"

os.makedirs("model/artifacts", exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Convert date
# -------------------------------
df["date"] = pd.to_datetime(df["date"])

# -------------------------------
# Feature engineering
# -------------------------------

# Time features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Hospital encoding
hospital_dummies = pd.get_dummies(df["hospital_id"], prefix="hospital")
df = pd.concat([df, hospital_dummies], axis=1)

# Ward encoding
if "ward_code" in df.columns:
    ward_dummies = pd.get_dummies(df["ward_code"], prefix="ward_code")
    df = pd.concat([df, ward_dummies], axis=1)

# -------------------------------
# Sort for time series
# -------------------------------
df = df.sort_values(by=["hospital_id", "date"])

# -------------------------------
# Lag features
# -------------------------------
df["admissions_lag_1"] = df.groupby("hospital_id")["admissions"].shift(1)
df["admissions_lag_2"] = df.groupby("hospital_id")["admissions"].shift(2)
df["admissions_lag_3"] = df.groupby("hospital_id")["admissions"].shift(3)

df["admissions_rolling_mean_3"] = (
    df.groupby("hospital_id")["admissions"]
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)

# -------------------------------
# Drop NA from lagging
# -------------------------------
df = df.dropna().reset_index(drop=True)

# -------------------------------
# Define features
# -------------------------------
target = "admissions"

drop_cols = ["date", "admissions", "ward_code", "hospital_id"]
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df[target]

# -------------------------------
# Force numeric data only
# -------------------------------
X = pd.get_dummies(X)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_experiment("Hospital Admissions Forecasting")

n_estimators = 200
max_depth = 10
random_state = 42

with mlflow.start_run(run_name="RandomForest Admissions Model"):
    # -------------------------------
    # Model
    # -------------------------------
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # Evaluation
    # -------------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # -------------------------------
    # Save model
    # -------------------------------
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

    # -------------------------------
    # Save feature list
    # -------------------------------
    with open(FEATURES_OUTPUT_PATH, "w") as f:
        for col in X.columns:
            f.write(col + "\n")

    print(f"Feature list saved to: {FEATURES_OUTPUT_PATH}")
    print(f"Total features: {len(X.columns)}")

    # -------------------------------
    # Log to MLflow
    # -------------------------------
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("feature_count", len(X.columns))
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2_Score", r2)

    mlflow.sklearn.log_model(model, name="random_forest_model")
    mlflow.log_artifact(FEATURES_OUTPUT_PATH)

    # -------------------------------
    # Plot predictions vs actual
    # -------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Model Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Admissions")
    plt.legend()

    plot_path = "forecast_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()

    print("MLflow logging complete.")