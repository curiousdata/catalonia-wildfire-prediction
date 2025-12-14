import streamlit as st
import pandas as pd
import requests

# Title
st.title("Catalonia Wildfire Prediction Dashboard")

# Date selection
year = st.selectbox("Select Year", [2021, 2022, 2023])
month = st.selectbox("Select Month", list(range(1, 13)))
day = st.selectbox("Select Day", list(range(1, 32)))

# Trigger
if st.button("Predict Wildfire Risk"):
    try:
        # Call API
        response = requests.get(
            "http://backend:8000/predict",
            params={"year": year, "month": month, "day": day}
        )
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        predictions = pd.DataFrame(data)

        # Display predictions
        st.write(predictions)

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching predictions: {e}")
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")