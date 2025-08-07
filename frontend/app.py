import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from pyproj import Transformer

# Set up transformer
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

# Title
st.title("Catalonia Wildfire Risk Dashboard")

# Date selection
year = st.selectbox("Select Year", [2021, 2022, 2023])
month = st.selectbox("Select Month", list(range(1, 13)))
day = st.selectbox("Select Day", list(range(1, 32)))

# Trigger
if st.button("Generate Heatmap"):
    try:
        # Call API
        response = requests.get(
            "http://backend-service:8000/predict",
            params={"year": year, "month": month, "day": day}
        )
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        predictions = pd.DataFrame(data)

        # Transform coordinates EPSG:3035 -> WGS84
        x_vals = predictions["x"].values
        y_vals = predictions["y"].values
        lon_vals, lat_vals = transformer.transform(x_vals, y_vals)

        predictions["longitude"] = lon_vals
        predictions["latitude"] = lat_vals

        # Drop rows with invalid coordinates
        predictions.dropna(subset=["latitude", "longitude", "risk"], inplace=True)
        predictions = predictions[predictions["risk"] >= 0]
        if predictions["risk"].max() > 1:
            predictions["risk"] = predictions["risk"] / predictions["risk"].max()

        # Prepare heatmap data
        heat_data = predictions[["latitude", "longitude", "risk"]].values.tolist()

        # Center map
        map_center = [predictions["latitude"].mean(), predictions["longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=8)

        # Add HeatMap
        HeatMap(
            data=heat_data,
            
        ).add_to(m)
        if len(heat_data) == 0:
            st.warning("No heatmap data to display.")
            st.stop()
        # Show map
        st.success(f"{len(heat_data)} prediction points loaded.")
        st.write("Map successfully generated and ready to render.")
        st_folium(m, width=700, height=500)

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching predictions: {e}")
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")