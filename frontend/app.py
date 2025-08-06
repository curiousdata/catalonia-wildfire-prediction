import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium

# Streamlit app title
st.title("Catalonia Wildfire Risk Dashboard")

# Date input
date = st.date_input("Select a date")

# Example coordinates for Catalonia (replace with actual grid points)
latitudes = [41.5, 41.6, 41.7]
longitudes = [1.5, 1.6, 1.7]

# Generate heatmap
if st.button("Generate Heatmap"):
    # Call the backend API to fetch predictions
    try:
        response = requests.get(
            "http://localhost:8000/predict",  # Replace with the backend API URL
            params={"date": date, "latitudes": latitudes, "longitudes": longitudes}
        )
        response.raise_for_status()
        data = response.json()

        # Create a DataFrame for predictions
        predictions = pd.DataFrame({
            "latitude": latitudes,
            "longitude": longitudes,
            "risk": data["predictions"]
        })

        # Create a folium map
        m = folium.Map(location=[41.5, 1.5], zoom_start=8)
        for _, row in predictions.iterrows():
            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=500,
                color="red" if row["risk"] > 0.5 else "green",
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        # Display the map
        st_folium(m, width=700, height=500)

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching predictions: {e}")