import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ==========================
# Load Model & Scaler
# ==========================

model = load_model(
    "models/airline_lstm.keras"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================
# Streamlit Page
# ==========================

st.set_page_config(
    page_title="Airline Passenger Forecasting",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Airline Passenger Forecasting using LSTM")

st.markdown(
    """
    Enter passenger counts for the previous **12 months**
    and the LSTM model will predict the next month's passengers.
    """
)

# ==========================
# Input Fields
# ==========================

values = []

col1, col2 = st.columns(2)

for i in range(6):
    with col1:
        values.append(
            st.number_input(
                f"Month {i+1}",
                min_value=0,
                value=100
            )
        )

for i in range(6, 12):
    with col2:
        values.append(
            st.number_input(
                f"Month {i+1}",
                min_value=0,
                value=100
            )
        )

# ==========================
# Prediction
# ==========================

if st.button("Predict Next Month"):

    arr = np.array(values)

    arr = arr.reshape(-1, 1)

    arr = scaler.transform(arr)

    arr = arr.reshape(
        1,
        12,
        1
    )

    prediction = model.predict(
        arr,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    predicted_passengers = int(
        prediction[0][0]
    )

    st.success(
        f"Predicted Next Month Passengers: {predicted_passengers}"
    )

    st.metric(
        label="Forecast",
        value=f"{predicted_passengers:,}"
    )