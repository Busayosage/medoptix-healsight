import requests
import streamlit as st
import requests

st.title("🏥 Hospital Admissions Forecast")

st.write("Adjust hospital parameters and predict admissions")

# Inputs
effective_capacity = st.slider("Effective Capacity", 10, 100, 34)
staffing_index = st.slider("Staffing Index", 0.5, 2.0, 1.0)
avg_wait_minutes = st.slider("Average Wait Time (minutes)", 50, 400, 200)

if st.button("Predict Admissions"):

    payload = {
        "effective_capacity": effective_capacity,
        "staffing_index": staffing_index,
        "avg_wait_minutes": avg_wait_minutes
    }

    response = requests.post(
        "http://api:8000/predict",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Admissions: {result['predicted_admissions']:.2f}")
    else:
        st.error("Error getting prediction")