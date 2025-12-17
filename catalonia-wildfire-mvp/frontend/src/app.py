from __future__ import annotations

import base64
import os
from typing import List, Tuple

import streamlit as st
import requests


# -----------------------------
# Config
# -----------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")

st.set_page_config(page_title="Catalonia Wildfire MVP", layout="wide")


# -----------------------------
# Helpers
# -----------------------------

def _fetch_dates() -> List[str]:
    try:
        r = requests.get(f"{BACKEND_URL}/dates", timeout=30)
        r.raise_for_status()
        payload = r.json()
        return list(payload.get("dates", []))
    except Exception:
        return []


def _fetch_overlay(date: str, view: str) -> Tuple[bytes, List[List[float]]]:
    r = requests.get(
        f"{BACKEND_URL}/map",
        params={"date": date, "view": view},
        timeout=120,
    )
    r.raise_for_status()
    payload = r.json()

    image_b64 = payload["image_b64"]
    bounds = payload["bounds"]

    img_bytes = base64.b64decode(image_b64)
    return img_bytes, bounds

def _render_folium_overlay(img_bytes: bytes, bounds: List[List[float]]):
    import folium

    # bounds: [[lat_min, lon_min], [lat_max, lon_max]]
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    center = [(lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0]

    m = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    # Folium ImageOverlay accepts raw bytes via a data URL
    data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# -----------------------------
# UI
# -----------------------------

st.title("Catalonia Wildfire Prediction Dashboard (MVP)")

with st.sidebar:
    st.subheader("Controls")
    st.caption(f"Backend: {BACKEND_URL}")

    dates = _fetch_dates()
    if not dates:
        st.warning("No dates returned from backend (/dates). Check backend is running and dataset is mounted.")

    date = st.selectbox("Date", options=dates or [""], index=0)

    view_label = st.radio(
        "Show",
        options=["Prediction", "True label", "Both"],
        index=0,
        horizontal=False,
    )

    view = {
        "Prediction": "prediction",
        "True label": "label",
        "Both": "both",
    }[view_label]

    run = st.button("Render map")


# -----------------------------
# Render
# -----------------------------

if run:
    if not date:
        st.error("Please select a valid date.")
        st.stop()

    with st.spinner("Fetching overlay from backend..."):
        try:
            img_bytes, bounds = _fetch_overlay(date=date, view=view)
        except requests.HTTPError as e:
            # Show backend error message if present
            try:
                detail = e.response.json().get("detail")
            except Exception:
                detail = str(e)
            st.error(f"Backend error: {detail}")
            st.stop()
        except Exception as e:
            st.error(f"Failed to fetch overlay: {e}")
            st.stop()

    debug = st.checkbox("Debug overlay", value=True)
    if debug:
        st.caption("DEBUG: raw PNG returned by backend")
        st.image(img_bytes)
        st.caption("DEBUG: overlay bounds from backend")
        st.write(bounds)

    m = _render_folium_overlay(img_bytes, bounds)

    st.subheader(f"Overlay for {date} ({view})")

    # Render Folium map as HTML for stability (streamlit-folium can be flaky in containers).
    html = m.get_root().render()
    st.components.v1.html(html, height=650, scrolling=True)

    with st.expander("Debug"):
        st.json({"bounds": bounds, "backend": BACKEND_URL})
else:
    st.info("Pick a date and a layer, then click **Render map**.")