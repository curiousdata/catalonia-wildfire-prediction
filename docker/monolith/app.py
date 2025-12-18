from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import torch
from PIL import Image
from pyproj import Transformer
import segmentation_models_pytorch as smp

from src.data.datasets import SimpleIberFireSegmentationDataset


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
    source_epsg: int


def get_cfg() -> Cfg:
    return Cfg(
        zarr_path=os.getenv("IBERFIRE_ZARR_PATH", "/app/data/gold/IberFire_coarse8_time1.zarr"),
        stats_path=os.getenv("NORM_STATS_PATH", "/app/stats/simple_iberfire_stats_train.json"),
        model_path=os.getenv("MODEL_PATH", "/app/models"),
        model_file=os.getenv("MODEL_FILE", "resnet34_v9.pth"),
        label_var=os.getenv("LABEL_VAR", "is_fire"),
        lead_time=int(os.getenv("LEAD_TIME", "1")),
        time_start=os.getenv("TIME_START", "2008-01-01"),
        time_end=os.getenv("TIME_END", "2024-12-31"),
        torch_device=os.getenv("TORCH_DEVICE", "cpu"),
        source_epsg=int(os.getenv("SOURCE_EPSG", "3035")),
    )


# IMPORTANT: Must match scripts/train.py feature_vars for the served model.
FEATURE_VARS: List[str] = [
    "FAPAR",
    "FWI",
    "LAI",
    "LST",
    "NDVI",
    "RH_max",
    "RH_mean",
    "RH_min",
    "RH_range",
    "SWI_001",
    "SWI_005",
    "SWI_010",
    "SWI_020",
    "is_holiday",
    "surface_pressure_max",
    "surface_pressure_mean",
    "surface_pressure_min",
    "surface_pressure_range",
    "t2m_max",
    "t2m_mean",
    "t2m_min",
    "t2m_range",
    "total_precipitation_mean",
    "wind_direction_at_max_speed",
    "wind_direction_mean",
    "wind_speed_max",
    "wind_speed_mean",
    *[f"CLC_{i}" for i in range(1, 45)],
    "CLC_agricultural_proportion",
    "CLC_arable_land_proportion",
    "CLC_artificial_proportion",
    "CLC_artificial_vegetation_proportion",
    "CLC_forest_and_semi_natural_proportion",
    "CLC_forest_proportion",
    "CLC_heterogeneous_agriculture_proportion",
    "CLC_industrial_proportion",
    "CLC_inland_waters_proportion",
    "CLC_inland_wetlands_proportion",
    "CLC_marine_waters_proportion",
    "CLC_maritime_wetlands_proportion",
    "CLC_mine_proportion",
    "CLC_open_space_proportion",
    "CLC_permanent_crops_proportion",
    "CLC_scrub_proportion",
    "CLC_urban_fabric_proportion",
    "CLC_waterbody_proportion",
    "CLC_wetlands_proportion",
    "aspect_1",
    "aspect_2",
    "aspect_3",
    "aspect_4",
    "aspect_5",
    "aspect_6",
    "aspect_7",
    "aspect_8",
    "aspect_NODATA",
    "dist_to_railways_mean",
    "dist_to_railways_stdev",
    "dist_to_roads_mean",
    "dist_to_roads_stdev",
    "dist_to_waterways_mean",
    "dist_to_waterways_stdev",
    "elevation_mean",
    "elevation_stdev",
    "is_natura2000",
    "is_sea",
    "is_spain",
    "is_waterbody",
    "roughness_mean",
    "roughness_stdev",
    "slope_mean",
    "slope_stdev",
    "popdens",
]


@st.cache_resource(show_spinner=False)
def load_dataset(cfg: Cfg) -> SimpleIberFireSegmentationDataset:
    return SimpleIberFireSegmentationDataset(
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


@st.cache_resource(show_spinner=False)
def load_model(cfg: Cfg) -> torch.nn.Module:
    model_file = Path(cfg.model_path) / cfg.model_file
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    device = torch.device(cfg.torch_device)

    # Match the notebook/training architecture exactly
    in_channels = len(FEATURE_VARS)
    m = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None,
    ).to(device)

    # Load checkpoint. In PyTorch 2.6, torch.load defaults to weights_only=True.
    # We want a pure state_dict for .pth checkpoints, but users may accidentally point to a TorchScript archive.
    try:
        state = torch.load(str(model_file), map_location=device, weights_only=True)
    except RuntimeError as e:
        msg = str(e)
        if "TorchScript archives" in msg or "weights_only=True" in msg and "TorchScript" in msg:
            # The file is actually a TorchScript archive; fall back gracefully.
            st.warning(
                f"Model file '{model_file.name}' appears to be a TorchScript archive. "
                "Loading with torch.jit.load(). If you intended to use a .pth state_dict, set MODEL_FILE to the correct .pth checkpoint."
            )
            scripted = torch.jit.load(str(model_file), map_location=device)
            scripted.eval()
            return scripted

        # Non-TorchScript pickle: retry with weights_only=False (trusted local artifact)
        state = torch.load(str(model_file), map_location=device, weights_only=False)

    # Be tolerant if the checkpoint is wrapped
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Expected a state_dict dict in {model_file}, got {type(state)}")

    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def list_available_dates(ds: SimpleIberFireSegmentationDataset) -> List[str]:
    return [str(ds.get_time_value(i))[:10] for i in range(len(ds))]


def find_index_by_date(ds: SimpleIberFireSegmentationDataset, date_str: str) -> int:
    for i in range(len(ds)):
        if str(ds.get_time_value(i))[:10] == date_str:
            return i
    raise ValueError(f"Date not found in dataset: {date_str}")


def probs_to_rgba(
    prob: np.ndarray,
    *,
    alpha_fixed: int | None = 255,
    min_visible_prob: float = 1e-4,
) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    # Pure red overlay; intensity is conveyed via alpha (fade), not by darkening RGB.
    red = np.where(p > 0, 255, 0).astype(np.uint8)

    # Treat (near) zero probabilities as fully transparent.
    # This avoids faint red haze from tiny positive values.
    eps = float(min_visible_prob)

    if alpha_fixed is None:
        # Alpha proportional to probability
        alpha = (p * 255.0).astype(np.uint8)
        alpha = np.where(p > eps, alpha, 0).astype(np.uint8)
    else:
        # Fade with probability up to a maximum alpha (semi-transparent cap)
        a = float(np.clip(alpha_fixed, 0, 255))
        alpha = np.where(p > eps, np.clip(p * a, 0.0, a), 0.0).astype(np.uint8)

    rgba = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = red
    rgba[..., 3] = alpha
    return rgba


def mask_to_rgba(mask01: np.ndarray, color: str) -> np.ndarray:
    m = (np.asarray(mask01) > 0.5).astype(np.uint8)
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)

    if color == "blue":
        rgba[..., 2] = m * 255
    elif color == "green":
        rgba[..., 1] = m * 255
    else:
        rgba[..., 0] = m * 255

    rgba[..., 3] = m * 255
    return rgba


def alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    b = bottom.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0

    ta = t[..., 3:4]
    ba = b[..., 3:4]
    out_a = ta + ba * (1 - ta)
    out_rgb = (t[..., :3] * ta + b[..., :3] * ba * (1 - ta)) / np.clip(out_a, 1e-8, 1.0)

    out = np.zeros_like(b)
    out[..., :3] = out_rgb
    out[..., 3:4] = out_a
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    img = Image.fromarray(rgba, mode="RGBA")
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_bounds(ds: SimpleIberFireSegmentationDataset, cfg: Cfg) -> List[List[float]]:
    """Return Folium-compatible bounds [[lat_min, lon_min], [lat_max, lon_max]]."""
    xr_ds = ds.ds
    y = np.asarray(xr_ds["y"].values)
    x = np.asarray(xr_ds["x"].values)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    looks_projected = abs(xmin) > 360 or abs(xmax) > 360 or abs(ymin) > 180 or abs(ymax) > 180

    if not looks_projected:
        return [[ymin, xmin], [ymax, xmax]]

    transformer = Transformer.from_crs(f"EPSG:{cfg.source_epsg}", "EPSG:4326", always_xy=True)
    corners_xy = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

    lons: List[float] = []
    lats: List[float] = []
    for xx, yy in corners_xy:
        lon, lat = transformer.transform(xx, yy)
        lons.append(float(lon))
        lats.append(float(lat))

    return [[float(np.min(lats)), float(np.min(lons))], [float(np.max(lats)), float(np.max(lons))]]


def render_folium(png_bytes: bytes, bounds: List[List[float]]):
    import folium

    (lat_min, lon_min), (lat_max, lon_max) = bounds
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

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


st.set_page_config(page_title="Catalonia Wildfire Monolith MVP", layout="wide")
st.title("Catalonia Wildfire Prediction (Monolith MVP)")

cfg = get_cfg()

with st.sidebar:
    st.subheader("Runtime")
    st.caption(f"Zarr: {cfg.zarr_path}")
    st.caption(f"Stats: {cfg.stats_path}")
    st.caption(f"Model: {Path(cfg.model_path) / cfg.model_file}")
    st.caption(f"Device: {cfg.torch_device}")
    st.caption(f"SOURCE_EPSG: {cfg.source_epsg}")

    show_debug = st.checkbox("Show debug", value=True)
    show_raw_matrix = st.checkbox("Show raw probability heatmap (no map)", value=False)
    st.markdown("---")
    st.subheader("Overlay visualization")
    viz_mode = st.radio("Scaling", ["raw", "p99_stretch"], index=1)
    alpha_mode = st.radio("Alpha", ["fixed_semi", "scaled"], index=0)

    st.markdown("**Transparency**")
    # Log-ish slider: choose an exponent, then compute threshold = 10**exp
    exp = st.slider(
        "Hide probabilities below (log10)",
        min_value=-3.0,
        max_value=-1.0,
        value=-2.0,
        step=0.05,
        format="%.2f",
        help="Threshold = 10^exp. Pixels with probability <= threshold are fully transparent.",
    )
    min_visible_prob = float(10 ** exp)
    st.caption(f"Threshold: {min_visible_prob:.6f}")

    mask_outside_spain = st.checkbox(
        "Mask outside Spain (is_spain==0)",
        value=True,
        help="Sets pixels to fully transparent where the is_spain feature is 0.",
    )

    view = st.radio("View", ["prediction", "label", "both"], index=0)
    run = st.button("Render")

    st.markdown("---")
    st.subheader("Bounds tweak")
    st.caption("Manual alignment (degrees). Positive values: N/S move inward; W/E move inward.")

    nudge_north_deg = st.slider(
        "Nudge north (top)  Δlat",  # applied to lat_max
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.05,
        format="%.2f",
        help="Adds to lat_max. Use negative to move top edge DOWN.",
    )
    nudge_south_deg = st.slider(
        "Nudge south (bottom) Δlat",  # applied to lat_min
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.05,
        format="%.2f",
        help="Adds to lat_min. Use positive to move bottom edge UP.",
    )
    nudge_west_deg = st.slider(
        "Nudge west (left)  Δlon",  # applied to lon_min
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.05,
        format="%.2f",
        help="Adds to lon_min. Use positive to move left edge EAST.",
    )
    nudge_east_deg = st.slider(
        "Nudge east (right) Δlon",  # applied to lon_max
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.05,
        format="%.2f",
        help="Adds to lon_max. Use negative to move right edge WEST.",
    )

    # Safety: prevent inverted bounds
    if nudge_south_deg + 1e-9 >= (nudge_north_deg + (0.0)):
        # This check is only meaningful relative to real bounds; we'll clamp after computing adj_bounds too.
        pass

with st.spinner("Loading dataset + model (cached)..."):
    ds = load_dataset(cfg)
    model = load_model(cfg)
    bounds = compute_bounds(ds, cfg)

dates = list_available_dates(ds)
date = st.selectbox("Date", options=dates, index=0)

if run:
    idx = find_index_by_date(ds, date)

    X, y = ds[idx]
    device = torch.device(cfg.torch_device)

    with torch.no_grad():
        out = model(X.unsqueeze(0).to(device).float())
        probs = torch.sigmoid(out).detach().cpu().numpy()

    if probs.ndim == 4:
        p2d = probs[0, 0]
    elif probs.ndim == 3:
        p2d = probs[0]
    else:
        p2d = probs

    # Visualization copy (do NOT change the underlying probabilities)
    p_vis = np.asarray(p2d, dtype=np.float32)
    p_vis = np.nan_to_num(p_vis, nan=0.0, posinf=1.0, neginf=0.0)
    p_vis = np.clip(p_vis, 0.0, 1.0)

    if viz_mode == "p99_stretch":
        denom = float(np.percentile(p_vis, 99))
        if denom > 0:
            p_vis = np.clip(p_vis / denom, 0.0, 1.0)

    # Semi-transparent red overlay by default
    alpha_fixed = 160 if alpha_mode == "fixed_semi" else None

    rgba = np.zeros((p2d.shape[0], p2d.shape[1], 4), dtype=np.uint8)

    if view in ("prediction", "both"):
        rgba = alpha_over(
            rgba,
            probs_to_rgba(p_vis, alpha_fixed=alpha_fixed, min_visible_prob=min_visible_prob),
        )

    if view in ("label", "both"):
        gt = y.squeeze().numpy()
        rgba = alpha_over(rgba, mask_to_rgba(gt, color="blue"))

    # Optional: fully hide anything outside Spain (based on the is_spain feature channel)
    if mask_outside_spain:
        try:
            spain_idx = FEATURE_VARS.index("is_spain")
            spain_mask = (X[spain_idx].cpu().numpy() > 0.5)
            rgba[..., 3] = np.where(spain_mask, rgba[..., 3], 0).astype(np.uint8)
        except ValueError:
            # FEATURE_VARS may not contain is_spain
            pass

    png = rgba_to_png_bytes(rgba)

    # --- Bounds tweak logic ---
    (lat_min, lon_min), (lat_max, lon_max) = bounds

    # Independent nudges for each side
    adj_lat_min = lat_min + float(nudge_south_deg)
    adj_lat_max = lat_max + float(nudge_north_deg)
    adj_lon_min = lon_min + float(nudge_west_deg)
    adj_lon_max = lon_max + float(nudge_east_deg)

    # Safety: avoid inverted/degenerate bounds
    if adj_lat_max <= adj_lat_min:
        mid = 0.5 * (adj_lat_min + adj_lat_max)
        adj_lat_min, adj_lat_max = mid - 0.1, mid + 0.1
    if adj_lon_max <= adj_lon_min:
        mid = 0.5 * (adj_lon_min + adj_lon_max)
        adj_lon_min, adj_lon_max = mid - 0.1, mid + 0.1

    adj_bounds = [[adj_lat_min, adj_lon_min], [adj_lat_max, adj_lon_max]]

    if show_debug:
        st.subheader("Debug")
        st.image(png, caption="Raw overlay PNG (what Folium receives)")

        if show_raw_matrix:
            st.subheader("Raw probability heatmap (no map)")

            # Raw as produced by inference: unscaled, unflipped p2d
            p_raw = np.asarray(p2d, dtype=np.float32)
            p_raw = np.nan_to_num(p_raw, nan=0.0, posinf=1.0, neginf=0.0)
            p_raw = np.clip(p_raw, 0.0, 1.0)

            st.caption(
                f"raw shape={p_raw.shape} | min={float(np.min(p_raw)):.6g} | "
                f"max={float(np.max(p_raw)):.6g} | mean={float(np.mean(p_raw)):.6g}"
            )

            # Controls specific to raw heatmap rendering
            raw_origin = st.radio("Raw heatmap origin", ["lower", "upper"], index=0, horizontal=True)
            raw_scale = st.radio("Raw heatmap scaling", ["linear_0_1", "p99_stretch"], index=1, horizontal=True)

            p_show = p_raw.copy()
            if raw_scale == "p99_stretch":
                denom = float(np.percentile(p_show, 99))
                if denom > 0:
                    p_show = np.clip(p_show / denom, 0.0, 1.0)

            if raw_origin == "lower":
                # Streamlit images treat first row as top; matplotlib origin='lower' would flip vertically.
                p_show = np.flipud(p_show)

            # Render a red heatmap (R channel), alpha fixed for visibility
            rgba_raw = probs_to_rgba(p_show, alpha_fixed=255, min_visible_prob=min_visible_prob)
            st.image(rgba_to_png_bytes(rgba_raw), caption="Raw probs rendered as red heatmap")

        max_pos = np.unravel_index(int(np.argmax(p2d)), p2d.shape)
        r0, c0 = int(max_pos[0]), int(max_pos[1])
        st.caption(f"Max prob at (row={r0}, col={c0}) = {float(p2d[r0, c0]):.6f}")

        half = 12
        r1, r2 = max(0, r0 - half), min(p2d.shape[0], r0 + half + 1)
        c1, c2 = max(0, c0 - half), min(p2d.shape[1], c0 + half + 1)
        crop = p2d[r1:r2, c1:c2]
        crop_vis = np.asarray(crop, dtype=np.float32)
        crop_vis = np.nan_to_num(crop_vis, nan=0.0, posinf=1.0, neginf=0.0)
        crop_vis = np.clip(crop_vis, 0.0, 1.0)
        if viz_mode == "p99_stretch":
            denom_c = float(np.percentile(crop_vis, 99))
            if denom_c > 0:
                crop_vis = np.clip(crop_vis / denom_c, 0.0, 1.0)
        st.image(
            rgba_to_png_bytes(
                probs_to_rgba(crop_vis, alpha_fixed=alpha_fixed, min_visible_prob=min_visible_prob)
            ),
            caption="Zoomed crop around max",
        )

        st.json(
            {
                "idx": idx,
                "X_shape": list(X.shape),
                "y_shape": list(y.shape),
                "out_shape": list(out.shape) if hasattr(out, "shape") else "unknown",
                "pred_min": float(np.min(p2d)),
                "pred_max": float(np.max(p2d)),
                "pred_mean": float(np.mean(p2d)),
                "p99": float(np.percentile(p2d, 99)),
                "viz_mode": viz_mode,
                "alpha_mode": alpha_mode,
                "count_gt_1e-6": int(np.sum(p2d > 1e-6)),
                "count_gt_1e-4": int(np.sum(p2d > 1e-4)),
                "count_gt_1e-2": int(np.sum(p2d > 1e-2)),
                "bounds": {"raw": bounds, "adjusted": adj_bounds},
                "source_epsg": cfg.source_epsg,
            }
        )

    st.subheader(f"Map for {date} ({view})")
    m = render_folium(png, adj_bounds)
    st.components.v1.html(m.get_root().render(), height=650, scrolling=True)
else:
    st.info("Pick a date + view, then click **Render**.")