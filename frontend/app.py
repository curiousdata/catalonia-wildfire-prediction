import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# Streamlit app title
st.title("Catalonia Wildfire Risk Dashboard")

# Date input
date = st.date_input("Select a date")

# Example wildfire prediction data (replace with actual model predictions)
data = pd.DataFrame({
    "latitude": [41.5, 41.6, 41.7],
    "longitude": [1.5, 1.6, 1.7],
    "risk": [0.8, 0.3, 0.6]  # Example risk values
})

# Generate heatmap
if st.button("Generate Heatmap"):
    m = folium.Map(location=[41.5, 1.5], zoom_start=8)
    for _, row in data.iterrows():
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=500,
            color="red" if row["risk"] > 0.5 else "green",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    # Display the map
    st_folium(m, width=700, height=500)