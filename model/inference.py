import os
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ================================
# PATH SETUP
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "../data/processed/final_dataset_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts/admissions_forecast_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "artifacts/feature_columns.txt")

# ================================
# LOAD RANDOM FOREST MODEL + FEATURES
# ================================
rf_model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    expected_features = [line.strip() for line in f if line.strip()]

print(f"Loaded model from: {MODEL_PATH}")
print(f"Loaded {len(expected_features)} features")

# ================================
# GLOBAL SARIMAX CACHE
# ================================
sarimax_fitted_model = None
sarimax_training_series = None

# ================================
# DATA PREPARATION
# ================================
def prepare_inference_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in processed dataset.")

    df["date"] = pd.to_datetime(df["date"])

    # Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    if "week_of_year" not in df.columns:
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "date_diff_1_days" not in df.columns:
        df["date_diff_1_days"] = 1

    sort_cols = ["date"]
    if "hospital_id" in df.columns:
        sort_cols = ["hospital_id", "date"]

    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Lag features
    if "hospital_id" in df.columns:
        grp = df.groupby("hospital_id")["admissions"]
        df["admissions_lag_1"] = grp.shift(1)
        df["admissions_lag_2"] = grp.shift(2)
        df["admissions_lag_3"] = grp.shift(3)
        df["admissions_rolling_mean_3"] = (
            grp.rolling(window=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df["admissions_lag_1"] = df["admissions"].shift(1)
        df["admissions_lag_2"] = df["admissions"].shift(2)
        df["admissions_lag_3"] = df["admissions"].shift(3)
        df["admissions_rolling_mean_3"] = df["admissions"].rolling(window=3).mean()

    df = df.dropna().reset_index(drop=True)
    return df

# ================================
# FEATURE ALIGNMENT FOR RF
# ================================
def align_features(df):
    df_encoded = pd.get_dummies(df)

    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[expected_features]
    return df_encoded

# ================================
# CURRENT PREDICTION USING RANDOM FOREST
# ================================
def predict_from_inputs(capacity, staffing, wait_time):
    df = prepare_inference_data()
    last_row = df.iloc[-1].copy()

    if "effective_capacity" in last_row.index:
        last_row["effective_capacity"] = capacity

    if "staffing_index" in last_row.index:
        last_row["staffing_index"] = staffing

    if "avg_wait_minutes" in last_row.index:
        last_row["avg_wait_minutes"] = wait_time

    if "avg_wait_time" in last_row.index:
        last_row["avg_wait_time"] = wait_time

    input_df = pd.DataFrame([last_row])
    aligned_input = align_features(input_df)

    pred = rf_model.predict(aligned_input)[0]
    return float(pred)

# ================================
# BUILD DAILY SERIES FOR SARIMAX
# ================================
def build_sarimax_series():
    df = prepare_inference_data()

    # Use one hospital only for cleaner time-series signal
    if "hospital_id" in df.columns:
        first_hospital = df["hospital_id"].iloc[0]
        df_hospital = df[df["hospital_id"] == first_hospital].copy()
    else:
        df_hospital = df.copy()

    # Aggregate to one value per day
    daily_series = (
        df_hospital
        .groupby("date")["admissions"]
        .sum()
        .sort_index()
        .asfreq("D")
        .interpolate()
    )

    print("SARIMAX series head:")
    print(daily_series.head())
    print("SARIMAX series describe:")
    print(daily_series.describe())

    return daily_series

# ================================
# SARIMAX FORECAST
# ================================
def generate_sarimax_forecast(n_steps=30):
    global sarimax_fitted_model
    global sarimax_training_series

    try:
        daily_series = build_sarimax_series()

        # Train once, then reuse
        if sarimax_fitted_model is None:
            print("Training SARIMAX model (one-time)...")

            sarimax_model = SARIMAX(
                daily_series,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            sarimax_fitted_model = sarimax_model.fit(disp=False)
            sarimax_training_series = daily_series.copy()
            print("SARIMAX model trained successfully.")

        forecast = sarimax_fitted_model.forecast(steps=n_steps)
        forecast_values = [float(round(x, 2)) for x in forecast]

        print("SARIMAX Forecast:", forecast_values)

        return forecast_values

    except Exception as e:
        print("❌ SARIMAX ERROR:", str(e))
        raise e