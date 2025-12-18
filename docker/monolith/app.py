from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Import your training-time dataset
from src.data.datasets import SimpleIberFireSegmentationDataset


# -----------------------------
# Config (env-driven)
# -----------------------------

@dataclass(frozen=True)
class Cfg:
    zarr_path: str
    stats_path: str
    model_path: str
    model_file: str
    label_var: str
    lead_time: int
    time_start: str
    time_end: str
    torch_device: str


def get_cfg() -> Cfg:
    return Cfg(
        zarr_path=os.getenv("IBERFIRE_ZARR_PATH", "/app/data/gold/IberFire_coarse8_time1.zarr"),
        stats_path=os.getenv("NORM_STATS_PATH", "/app/stats/simple_iberfire_stats_train.json"),
        model_path=os.getenv("MODEL_PATH", "/app/models"),
        model_file=os.getenv("MODEL_FILE", "resnet34_v9.torchscript"),
        label_var=os.getenv("LABEL_VAR", "is_fire"),
        lead_time=int(os.getenv("LEAD_TIME", "1")),
        time_start=os.getenv("TIME_START", "2008-01-01"),
        time_end=os.getenv("TIME_END", "2024-12-31"),
        torch_device=os.getenv("TORCH_DEVICE", "cpu"),
    )


# -----------------------------
# Feature vars (must match training!)
# -----------------------------
# IMPORTANT: Keep this identical to scripts/train.py feature_vars for the model you serve.
FEATURE_VARS: List[str] = [
    "FAPAR","FWI","LAI","LST","NDVI","RH_max","RH_mean","RH_min","RH_range",
    "SWI_001","SWI_005","SWI_010","SWI_020","is_holiday",
    "surface_pressure_max","surface_pressure_mean","surface_pressure_min","surface_pressure_range",
    "t2m_max","t2m_mean","t2m_min","t2m_range",
    "total_precipitation_mean","wind_direction_at_max_speed","wind_direction_mean",
    "wind_speed_max","wind_speed_mean",
    # CLC bases (year-aware)
    *[f"CLC_{i}" for i in range(1, 45)],
    "CLC_agricultural_proportion","CLC_arable_land_proportion","CLC_artificial_proportion",
    "CLC_artificial_vegetation_proportion","CLC_forest_and_semi_natural_proportion",
    "CLC_forest_proportion","CLC_heterogeneous_agriculture_proportion","CLC_industrial_proportion",
    "CLC_inland_waters_proportion","CLC_inland_wetlands_proportion","CLC_marine_waters_proportion",
    "CLC_maritime_wetlands_proportion","CLC_mine_proportion","CLC_open_space_proportion",
    "CLC_permanent_crops_proportion","CLC_scrub_proportion","CLC_urban_fabric_proportion",
    "CLC_waterbody_proportion","CLC_wetlands_proportion",
    # static
    "aspect_1","aspect_2","aspect_3","aspect_4","aspect_5","aspect_6","aspect_7","aspect_8","aspect_NODATA",
    "dist_to_railways_mean","dist_to_railways_stdev","dist_to_roads_mean","dist_to_roads_stdev",
    "dist_to_waterways_mean","dist_to_waterways_stdev",
    "elevation_mean","elevation_stdev","is_natura2000","is_sea","is_spain","is_waterbody",
    "roughness_mean","roughness_stdev","slope_mean","slope_stdev",
    # year-aware popdens base
    "popdens",
]


# -----------------------------
# Caching: dataset + model
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_dataset(cfg: Cfg) -> SimpleIberFireSegmentationDataset:
    # This is the “notebook way”: let dataset handle stats, NaNs, static/year-aware, lead_time, etc.
    ds = SimpleIberFireSegmentationDataset(
        zarr_path=Path(cfg.zarr_path),
        time_start=cfg.time_start,
        time_end=cfg.time_end,
        feature_vars=FEATURE_VARS,
        label_var=cfg.label_var,
        spatial_downsample=1,
        lead_time=cfg.lead_time,
        compute_stats=False,
        stats_path=Path(cfg.stats_path),
        mode="all",
    )
    return ds


@st.cache_resource(show_spinner=False)
def load_model(cfg: Cfg) -> torch.jit.ScriptModule:
    model_file = Path(cfg.model_path) / cfg.model_file
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    device = torch.device(cfg.torch_device)
    m = torch.jit.load(str(model_file), map_location=device)
    m.eval()
    return m


# -----------------------------
# Utilities
# -----------------------------

def list_available_dates(ds: SimpleIberFireSegmentationDataset) -> List[str]:
    # Use dataset time values (same as earlier backend approach)
    dates = []
    for i in range(len(ds)):
        t = ds.get_time_value(i)
        dates.append(str(t)[:10])
    return dates


def find_index_by_date(ds: SimpleIberFireSegmentationDataset, date_str: str) -> int:
    for i in range(len(ds)):
        if str(ds.get_time_value(i))[:10] == date_str:
            return i
    raise ValueError(f"Date not found in dataset: {date_str}")


def probs_to_rgba(prob: np.ndarray) -> np.ndarray:
    # prob: [H,W] in [0,1]
    p = np.asarray(prob, dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    red = (p * 255.0).astype(np.uint8)
    alpha = (p * 255.0).astype(np.uint8)
    alpha = np.where(p > 0, alpha, 0).astype(np.uint8)

    rgba = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = red
    rgba[..., 3] = alpha
    return rgba


def mask_to_rgba(mask01: np.ndarray, color: str) -> np.ndarray:
    # mask01: [H,W] in {0,1}
    m = (mask01 > 0.5).astype(np.uint8)
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)

    if color == "blue":
        rgba[..., 2] = m * 255
    elif color == "green":
        rgba[..., 1] = m * 255
    else:  # red
        rgba[..., 0] = m * 255

    rgba[..., 3] = m * 255
    return rgba


def alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    # Both RGBA uint8
    b = bottom.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0
    out = np.zeros_like(b)

    ta = t[..., 3:4]
    ba = b[..., 3:4]
    out_a = ta + ba * (1 - ta)
    out[..., :3] = (t[..., :3] * ta + b[..., :3] * ba * (1 - ta)) / np.clip(out_a, 1e-8, 1.0)
    out[..., 3:4] = out_a
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    img = Image.fromarray(rgba, mode="RGBA")
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_bounds(ds: SimpleIberFireSegmentationDataset) -> List[List[float]]:
    # Use underlying xarray dataset for coords (same logic as backend)
    xr_ds = ds.ds
    y = xr_ds["y"].values
    x = xr_ds["x"].values
    lat_min, lat_max = float(np.min(y)), float(np.max(y))
    lon_min, lon_max = float(np.min(x)), float(np.max(x))
    return [[lat_min, lon_min], [lat_max, lon_max]]


def render_folium(png_bytes: bytes, bounds: List[List[float]]):
    import folium
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    center = [(lat_min + lat_max)/2, (lon_min + lon_max)/2]

    m = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")
    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    return m


# -----------------------------
# App
# -----------------------------

st.set_page_config(page_title="Catalonia Wildfire Monolith MVP", layout="wide")
st.title("Catalonia Wildfire Prediction (Monolith MVP)")

cfg = get_cfg()

with st.sidebar:
    st.subheader("Runtime")
    st.caption(f"Zarr: {cfg.zarr_path}")
    st.caption(f"Stats: {cfg.stats_path}")
    st.caption(f"Model: {Path(cfg.model_path) / cfg.model_file}")
    st.caption(f"Device: {cfg.torch_device}")

    show_debug = st.checkbox("Show debug", value=True)
    view = st.radio("View", ["prediction", "label", "both"], index=0)
    run = st.button("Render")

with st.spinner("Loading dataset + model (cached)..."):
    ds = load_dataset(cfg)
    model = load_model(cfg)
    bounds = compute_bounds(ds)

dates = list_available_dates(ds)
date = st.selectbox("Date", options=dates, index=0)

if run:
    idx = find_index_by_date(ds, date)

    X, y = ds[idx]  # X: [C,H,W], y: [1,H,W]
    device = torch.device(cfg.torch_device)

    with torch.no_grad():
        out = model(X.unsqueeze(0).to(device).float())
        probs = torch.sigmoid(out).detach().cpu().numpy()

    # Normalize shape to [H,W]
    if probs.ndim == 4:
        p2d = probs[0, 0]
    elif probs.ndim == 3:
        p2d = probs[0]
    else:
        p2d = probs

    rgba = np.zeros((p2d.shape[0], p2d.shape[1], 4), dtype=np.uint8)

    if view in ("prediction", "both"):
        rgba = alpha_over(rgba, probs_to_rgba(p2d))

    if view in ("label", "both"):
        gt = y.squeeze().numpy()  # [H,W]
        rgba = alpha_over(rgba, mask_to_rgba(gt, color="blue"))

    png = rgba_to_png_bytes(rgba)

    # Debug: show raw PNG and stats
    if show_debug:
        st.subheader("Debug")
        st.image(png, caption="Raw overlay PNG (what Folium receives)")
        st.json({
            "idx": idx,
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "out_shape": list(out.shape) if hasattr(out, "shape") else "unknown",
            "pred_min": float(np.min(p2d)),
            "pred_max": float(np.max(p2d)),
            "pred_mean": float(np.mean(p2d)),
            "p99": float(np.percentile(p2d, 99)),
            "bounds": bounds,
        })

    st.subheader(f"Map for {date} ({view})")
    m = render_folium(png, bounds)
    st.components.v1.html(m.get_root().render(), height=650, scrolling=True)
else:
    st.info("Pick a date + view, then click **Render**.")