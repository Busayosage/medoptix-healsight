import streamlit as st
import requests
import pandas as pd

API_URL = "http://api:8000/predict"

st.set_page_config(page_title="Hospital Admissions Forecast", layout="centered")

st.title("🏥 Hospital Admissions Forecast")
st.caption("DASHBOARD VERSION: RF + SARIMAX (single-hospital forecast)")
st.subheader("Interactive admissions prediction dashboard")

# -------------------------------
# INPUTS
# -------------------------------
st.header("Input Parameters")

capacity = st.slider("Effective Capacity", 10, 100, 34)
staffing = st.slider("Staffing Index", 0.5, 2.0, 1.0)
wait_time = st.slider("Average Wait Time (minutes)", 10, 300, 200)

# -------------------------------
# SUMMARY
# -------------------------------
st.header("Current Input Summary")
st.write(f"Effective Capacity: {capacity}")
st.write(f"Staffing Index: {staffing}")
st.write(f"Average Wait Time: {wait_time}")

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Admissions"):
    try:
        payload = {
            "capacity": capacity,
            "staffing_index": staffing,
            "avg_wait_time": wait_time
        }

        response = requests.post(API_URL, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()

            prediction_value = result.get("predicted_admissions", result.get("prediction", 0))
            forecast_values = result.get("future_forecast", [])

            st.success("Prediction successful")

            st.header("Forecast Result")
            st.metric("Predicted Admissions", round(prediction_value, 2))

            if prediction_value < 90:
                st.info("🟢 Low expected admissions")
            elif prediction_value < 125:
                st.warning("🟡 Moderate expected admissions")
            else:
                st.error("🔴 High expected admissions")

            if forecast_values:
                avg_7_day = sum(forecast_values[:7]) / min(len(forecast_values), 7)
                peak_30_day = max(forecast_values)
                low_30_day = min(forecast_values)

                st.subheader("Forecast Insights")
                st.write(f"7-day average forecast: {avg_7_day:.2f}")
                st.write(f"30-day peak forecast: {peak_30_day:.2f}")
                st.write(f"30-day low forecast: {low_30_day:.2f}")

                # ✅ YOUR NEW LINE (EXPLANATION)
                st.write("This forecast reflects seasonal hospital demand patterns and recent admission trends.")

                forecast_df = pd.DataFrame({
                    "Day": list(range(1, len(forecast_values) + 1)),
                    "Predicted Admissions": forecast_values
                })

                st.subheader("30-Day Admission Trend (SARIMAX Forecast)")
                st.line_chart(forecast_df.set_index("Day"))
            else:
                st.warning("No forecast data received from API")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")