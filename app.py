from fastapi import FastAPI
from model.inference import generate_sarimax_forecast

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hospital Admissions API is running"}


@app.post("/predict")
def predict(data: dict):

    # Generate SARIMAX forecast
    future_forecast = generate_sarimax_forecast(n_steps=30)

    # Use first forecast value as prediction
    prediction = future_forecast[0]

    return {
        "prediction": float(round(prediction, 2)),
        "predicted_admissions": float(round(prediction, 2)),
        "future_forecast": future_forecast
    }