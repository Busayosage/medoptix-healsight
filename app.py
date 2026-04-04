import joblib
import pandas as pd
from fastapi import FastAPI

# Load model and features
model = joblib.load("model/artifacts/admissions_forecast_model.pkl")

with open("model/artifacts/feature_columns.txt") as f:
    expected_features = [line.strip() for line in f]

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hospital Admissions Forecast API"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Ensure all features exist
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    pred = model.predict(df)[0]

    return {"predicted_admissions": float(pred)}