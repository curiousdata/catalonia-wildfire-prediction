from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

# NOTE:
# - This module MUST NOT import FastAPI.
# - It MUST be pure inference/data logic used by the API layer.


ViewStr = Literal["prediction", "label", "both"]


@dataclass(frozen=True)
class OverlayResult:
    image_b64: str
    bounds: List[List[float]]  # [[lat_min, lon_min], [lat_max, lon_max]] in EPSG:4326


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


# Default aligns with your current project memory: coarsened zarr with time=1 chunks.
DEFAULT_ZARR_PATH = "/app/data/IberFire_coarse8_time1.zarr"


def _parse_feature_vars(s: str) -> List[str]:
    # Comma-separated list
    items = [x.strip() for x in (s or "").split(",")]
    return [x for x in items if x]


@lru_cache(maxsize=1)
def _open_dataset():
    """Open the Zarr dataset once per process."""
    import xarray as xr

    zarr_path = _env("IBERFIRE_ZARR_PATH", DEFAULT_ZARR_PATH)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"Zarr dataset not found at {zarr_path}. "
            f"Set IBERFIRE_ZARR_PATH env var or mount data into /app/data."
        )

    # Consolidated metadata is ideal; if not consolidated, xarray will still usually open.
    return xr.open_zarr(zarr_path, consolidated=False)


def list_available_dates() -> List[str]:
    """Return available dates as ISO strings (YYYY-MM-DD) for the Streamlit date picker."""
    ds = _open_dataset()
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate.")

    times = ds["time"].values
    # Convert to YYYY-MM-DD; handle numpy datetime64, pandas timestamps, etc.
    out: List[str] = []
    for t in times:
        # numpy.datetime64 -> 'YYYY-MM-DDTHH:MM:SS' like string sometimes
        s = np.datetime_as_string(np.datetime64(t), unit="D")
        out.append(s)
    return out


def build_map_overlay(*, date: str, view: ViewStr, model: Any = None) -> Dict[str, Any]:
    """Build a base64 PNG overlay + bounds for Folium.

    Parameters
    ----------
    date:
        ISO date string (YYYY-MM-DD)
    view:
        'prediction' | 'label' | 'both'
    model:
        Torch model (already loaded) for prediction views.

    Returns
    -------
    dict with keys:
      - image_b64: base64-encoded PNG (RGBA)
      - bounds: [[lat_min, lon_min], [lat_max, lon_max]]
    """

    if view not in ("prediction", "label", "both"):
        raise ValueError(f"Invalid view={view}")

    ds = _open_dataset()

    # ---- Resolve coordinates ----
    # The paper + your notes: x_coords/y_coords represent EPSG:3035 grid coordinates.
    # We accept either coords or data variables with these names.
    x_name = _env("X_COORD_NAME", "x_coords")
    y_name = _env("Y_COORD_NAME", "y_coords")

    if x_name in ds.coords:
        x = ds.coords[x_name].values
    elif x_name in ds:
        x = ds[x_name].values
    else:
        # Fallback common names
        if "x" in ds.coords:
            x = ds.coords["x"].values
        else:
            raise ValueError(f"Could not find x coordinate '{x_name}' (nor fallback 'x') in dataset")

    if y_name in ds.coords:
        y = ds.coords[y_name].values
    elif y_name in ds:
        y = ds[y_name].values
    else:
        if "y" in ds.coords:
            y = ds.coords["y"].values
        else:
            raise ValueError(f"Could not find y coordinate '{y_name}' (nor fallback 'y') in dataset")

    bounds = _bounds_epsg3035_to_wgs84(x, y)

    # ---- Select time slice ----
    # We match by day. Dataset might store datetime64 with time-of-day; select nearest day.
    t = np.datetime64(date)
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate.")

    # Try exact day match
    try:
        day_index = int(np.where(ds["time"].values.astype("datetime64[D]") == t.astype("datetime64[D]"))[0][0])
    except Exception:
        # Fallback: nearest (safe for MVP)
        # Convert to integer days
        time_days = ds["time"].values.astype("datetime64[D]")
        day_deltas = np.abs(time_days.astype("int64") - t.astype("datetime64[D]").astype("int64"))
        day_index = int(day_deltas.argmin())

    # ---- Build arrays ----
    target_var = _env("TARGET_VAR", "is_fire")

    label_arr: Optional[np.ndarray] = None
    if view in ("label", "both"):
        if target_var not in ds:
            raise ValueError(
                f"TARGET_VAR='{target_var}' not found in dataset. "
                "Set TARGET_VAR env var to the correct target variable name."
            )
        # Expect dims include time,y,x (or time, y, x). Use squeeze to get [H,W]
        label_arr = ds[target_var].isel(time=day_index).values
        label_arr = np.asarray(label_arr)
        label_arr = np.squeeze(label_arr)

    pred_arr: Optional[np.ndarray] = None
    if view in ("prediction", "both"):
        if model is None:
            raise ValueError("Model is required for view='prediction' or 'both'.")
        pred_arr = _predict_for_day(ds=ds, day_index=day_index, model=model)

    # ---- Compose overlay (RGBA PNG) ----
    rgba = _compose_rgba(pred=pred_arr, label=label_arr, mode=view)
    image_b64 = _rgba_to_png_b64(rgba)

    return {"image_b64": image_b64, "bounds": bounds}


def _predict_for_day(*, ds: Any, day_index: int, model: Any) -> np.ndarray:
    """Run model inference for one day.

    MVP contract:
    - FEATURE_VARS env var must be a comma-separated list of variables to stack as channels.
    - Returns a 2D array [H,W] of probabilities in [0,1].

    Notes:
    - This function intentionally keeps assumptions minimal.
    - If your training pipeline uses a dataset class for normalization / feature construction,
      you can later swap this implementation to reuse that logic.
    """
    import torch

    feature_vars = _parse_feature_vars(os.getenv("FEATURE_VARS", ""))
    if not feature_vars:
        raise ValueError(
            "FEATURE_VARS env var is not set. Provide comma-separated feature variables used by the trained model."
        )

    # Stack [C,H,W]
    chans: List[np.ndarray] = []
    for v in feature_vars:
        if v not in ds:
            raise ValueError(f"Feature var '{v}' not found in dataset")
        a = ds[v].isel(time=day_index).values
        a = np.asarray(a)
        a = np.squeeze(a)
        chans.append(a)

    x = np.stack(chans, axis=0).astype(np.float32)

    # Optional: apply normalization stats if provided
    # Expect JSON stored and mounted, or env paths. Keep this minimal for MVP.
    # If you already save stats in JSON (per project memory), you can wire it here.

    xt = torch.from_numpy(x).unsqueeze(0)  # [1,C,H,W]

    # Device selection
    device = os.getenv("TORCH_DEVICE", "cpu")
    if device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    model = model.to(dev)
    model.eval()
    xt = xt.to(dev)

    with torch.no_grad():
        logits = model(xt)

    # Accept common segmentation outputs:
    # - [1,1,H,W]
    # - [1,H,W]
    # - [H,W]
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    if logits.ndim == 4:
        logits = logits[:, 0, :, :]
    if logits.ndim == 3:
        logits = logits[0]

    # Sigmoid to probability
    probs = torch.sigmoid(logits).detach().float().cpu().numpy()
    probs = np.asarray(probs)
    probs = np.squeeze(probs)

    return probs


def _bounds_epsg3035_to_wgs84(x: np.ndarray, y: np.ndarray) -> List[List[float]]:
    """Compute map bounds in lat/lon from EPSG:3035 coordinates."""
    # x/y are 1D coordinate axes (preferred). If they are 2D, flatten.
    x1 = np.asarray(x).reshape(-1)
    y1 = np.asarray(y).reshape(-1)

    xmin, xmax = float(np.nanmin(x1)), float(np.nanmax(x1))
    ymin, ymax = float(np.nanmin(y1)), float(np.nanmax(y1))

    try:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
        # corners: (xmin,ymin) and (xmax,ymax)
        lon_min, lat_min = tf.transform(xmin, ymin)
        lon_max, lat_max = tf.transform(xmax, ymax)
        return [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]
    except Exception as e:
        raise RuntimeError(
            "Failed to reproject EPSG:3035 -> EPSG:4326. Ensure 'pyproj' is installed in backend requirements."
        ) from e


def _compose_rgba(*, pred: Optional[np.ndarray], label: Optional[np.ndarray], mode: ViewStr) -> np.ndarray:
    """Create an RGBA overlay image.

    - prediction: grayscale heatmap (white = high probability)
    - label: red mask
    - both: prediction grayscale + label red on top

    Returns RGBA uint8 image [H,W,4].
    """
    # Determine shape
    base = pred if pred is not None else label
    if base is None:
        raise ValueError("Nothing to render")

    base = np.asarray(base)
    base = np.squeeze(base)
    if base.ndim != 2:
        raise ValueError(f"Expected 2D array for overlay, got shape {base.shape}")

    h, w = base.shape

    # Start transparent
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if pred is not None and mode in ("prediction", "both"):
        p = np.asarray(pred, dtype=np.float32)
        p = np.clip(p, 0.0, 1.0)
        g = (p * 255.0).astype(np.uint8)
        # grayscale to RGB
        rgba[..., 0] = g
        rgba[..., 1] = g
        rgba[..., 2] = g
        # alpha: slightly transparent so base map is visible
        rgba[..., 3] = 140

    if label is not None and mode in ("label", "both"):
        m = np.asarray(label)
        # Interpret anything >0 as fire label
        mask = m > 0
        # Paint red where label is true
        rgba[mask, 0] = 255
        rgba[mask, 1] = 0
        rgba[mask, 2] = 0
        rgba[mask, 3] = 180

    return rgba


def _rgba_to_png_b64(rgba: np.ndarray) -> str:
    """Encode RGBA array to base64 PNG."""
    from PIL import Image

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")