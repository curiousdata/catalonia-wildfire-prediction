from __future__ import annotations

import base64
import io
import json
import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

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

logger = logging.getLogger("iberfire.inference")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _parse_feature_vars(s: str) -> List[str]:
    # Comma-separated list
    items = [x.strip() for x in (s or "").split(",")]
    return [x for x in items if x]


@lru_cache(maxsize=1)
def _get_segmentation_dataset() -> Any:
    """Create the training-time dataset object inside the backend.

    We rely on SimpleIberFireSegmentationDataset to reproduce *exactly* the
    feature resolution + normalization + NaN handling + lead_time shifting logic
    used during training.

    Requirements:
    - docker-compose mounts `../src` to `/workspace/train_src:ro`
    - PYTHONPATH includes `/workspace/train_src`
    """
    zarr_path = _env("IBERFIRE_ZARR_PATH", DEFAULT_ZARR_PATH)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"Zarr dataset not found at {zarr_path}. "
            f"Set IBERFIRE_ZARR_PATH env var or mount data into /app/data."
        )

    # Import training dataset implementation (mounted into the container)
    try:
        from data.datasets import SimpleIberFireSegmentationDataset  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import training dataset code. Ensure docker-compose mounts ../src as "
            "`/workspace/train_src:ro` and sets PYTHONPATH to include `/workspace/train_src`."
        ) from e

    feature_vars = _parse_feature_vars(os.getenv("FEATURE_VARS", ""))
    if not feature_vars:
        raise ValueError(
            "FEATURE_VARS env var is not set. Provide comma-separated feature variables used by the trained model."
        )

    # Time range / target / lead_time mirror train.py defaults, but are configurable.
    time_start = _env("TIME_START", "2001-01-01")
    time_end = _env("TIME_END", "2030-12-31")
    label_var = _env("TARGET_VAR", "is_fire")
    lead_time = int(_env("LEAD_TIME", "1"))

    # Optional: use the same stats json as training (recommended). If empty, dataset may compute or skip.
    # You already mount ../stats -> /app/stats.
    stats_path = os.getenv("NORM_STATS_PATH") or os.getenv("STATS_PATH")

    # Optional: day selection / balancing controls (safe defaults: use all days)
    mode = _env("DATA_MODE", "all")
    day_indices_path = os.getenv("DAY_INDICES_PATH")

    # Many dataset implementations accept additional knobs; keep a conservative set.
    # If your datasets.py signature differs, this will raise a helpful error at startup.
    try:
        ds = SimpleIberFireSegmentationDataset(
            zarr_path=zarr_path,
            time_start=time_start,
            time_end=time_end,
            feature_vars=feature_vars,
            label_var=label_var,
            lead_time=lead_time,
            stats_path=stats_path,
            mode=mode,
            day_indices_path=day_indices_path,
        )
    except TypeError:
        # Fallback for older signatures (minimal required args)
        ds = SimpleIberFireSegmentationDataset(
            zarr_path=zarr_path,
            time_start=time_start,
            time_end=time_end,
            feature_vars=feature_vars,
            label_var=label_var,
            lead_time=lead_time,
        )

    return ds


def list_available_dates() -> List[str]:
    """Return available dates as ISO strings (YYYY-MM-DD) for the Streamlit date picker."""
    ds = _get_segmentation_dataset()

    out: List[str] = []
    for i in range(len(ds)):
        t = ds.get_time_value(i)
        out.append(str(t)[:10])

    # Deduplicate while preserving order (some dataset modes can repeat)
    seen = set()
    uniq: List[str] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


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

    debug_info: Dict[str, Any] = {}
    debug_enabled = os.getenv("DEBUG_PRED", "0") == "1"

    dataset = _get_segmentation_dataset()
    # underlying xarray dataset for coords/bounds
    xr_ds = getattr(dataset, "ds", None)
    if xr_ds is None:
        raise ValueError("Training dataset object has no `.ds` attribute (xarray dataset).")

    # ---- Resolve coordinates ----
    # The paper + your notes: x_coords/y_coords represent EPSG:3035 grid coordinates.
    # We accept either coords or data variables with these names.
    x_name = _env("X_COORD_NAME", "x_coords")
    y_name = _env("Y_COORD_NAME", "y_coords")

    if x_name in xr_ds.coords:
        x = xr_ds.coords[x_name].values
    elif x_name in xr_ds:
        x = xr_ds[x_name].values
    else:
        # Fallback common names
        if "x" in xr_ds.coords:
            x = xr_ds.coords["x"].values
        else:
            raise ValueError(f"Could not find x coordinate '{x_name}' (nor fallback 'x') in dataset")

    if y_name in xr_ds.coords:
        y = xr_ds.coords[y_name].values
    elif y_name in xr_ds:
        y = xr_ds[y_name].values
    else:
        if "y" in xr_ds.coords:
            y = xr_ds.coords["y"].values
        else:
            raise ValueError(f"Could not find y coordinate '{y_name}' (nor fallback 'y') in dataset")

    bounds = _bounds_epsg3035_to_wgs84(x, y)

    # ---- Resolve dataset sample index for the requested date ----
    # We match by YYYY-MM-DD on the dataset-provided time value.
    date_str = str(date)[:10]
    idxs = [i for i in range(len(dataset)) if str(dataset.get_time_value(i))[:10] == date_str]
    if not idxs:
        raise ValueError(f"No sample found for date {date_str} in the mounted dataset")
    sample_index = idxs[0]
    if debug_enabled:
        debug_info["sample_index"] = sample_index
        debug_info["date"] = date_str

    label_arr: Optional[np.ndarray] = None
    if view in ("label", "both"):
        _x, y = dataset[sample_index]
        # y is typically [1,H,W]
        label_arr = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
        label_arr = np.squeeze(label_arr)

    pred_arr: Optional[np.ndarray] = None
    if view in ("prediction", "both"):
        if model is None:
            raise ValueError("Model is required for view='prediction' or 'both'.")
        pred_arr = _predict_for_sample(
            dataset=dataset,
            sample_index=sample_index,
            model=model,
            debug_info=(debug_info if debug_enabled else None),
        )

    # ---- Compose overlay (RGBA PNG) ----
    rgba = _compose_rgba(pred=pred_arr, label=label_arr, mode=view)
    image_b64 = _rgba_to_png_b64(rgba)

    out: Dict[str, Any] = {"image_b64": image_b64, "bounds": bounds}
    if debug_enabled:
        out["debug"] = debug_info
    return out


def _predict_for_sample(*, dataset: Any, sample_index: int, model: Any, debug_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Run model inference for one dataset sample.

    The dataset returns X already normalized and with correct feature construction.
    Returns a 2D array [H,W] of probabilities in [0,1].
    """
    import torch

    X, _y = dataset[sample_index]  # X: [C,H,W]

    # Ensure torch tensor
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)

    xt = X.unsqueeze(0).float()  # [1,C,H,W]

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
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    if hasattr(logits, "ndim") and logits.ndim == 4:
        logits = logits[:, 0, :, :]
    if hasattr(logits, "ndim") and logits.ndim == 3:
        logits = logits[0]

    probs = torch.sigmoid(logits).detach().float().cpu().numpy()
    probs = np.asarray(probs)
    probs = np.squeeze(probs)
    # Optional debug: log summary stats so we can see if the overlay is invisible due to near-zero probs.
    if debug_info is not None:
        p = np.asarray(probs, dtype=np.float32)
        p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        debug_info["pred_min"] = float(np.min(p))
        debug_info["pred_max"] = float(np.max(p))
        debug_info["pred_mean"] = float(np.mean(p))
        for q in (50, 90, 95, 99, 99.5, 99.9):
            debug_info[f"pred_p{str(q).replace('.', '_')}"] = float(np.percentile(p, q))

        logger.info(
            "Pred stats idx=%s min=%.3e max=%.3e mean=%.3e p99=%.3e p99.9=%.3e",
            sample_index,
            debug_info["pred_min"],
            debug_info["pred_max"],
            debug_info["pred_mean"],
            debug_info["pred_p99"],
            debug_info["pred_p99_9"],
        )
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
        # Direct visualization: probability in [0,1] -> red intensity
        p = np.asarray(pred, dtype=np.float32)
        p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        p = np.clip(p, 0.0, 1.0)

        red = (p * 255.0).astype(np.uint8)
        alpha = (p * 255.0).astype(np.uint8)

        # Keep exact zeros fully transparent
        alpha = np.where(p > 0, alpha, 0).astype(np.uint8)

        rgba[..., 0] = red
        rgba[..., 1] = 0
        rgba[..., 2] = 0
        rgba[..., 3] = alpha

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


def _year_for_index(ds: Any, day_index: int) -> int:
    """Return calendar year for a given time index."""
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate.")
    t = ds["time"].values[day_index]
    try:
        return int(str(np.datetime64(t, "D"))[:4])
    except Exception:
        return int(str(t)[:4])


def _resolve_clc_var(ds: Any, base: str, year: int) -> str:
    """Resolve base CLC feature (e.g., 'CLC_1' or 'CLC_forest_proportion') to a year-specific variable name."""
    suffix = base[len("CLC_") :]
    candidates = {
        2006: f"CLC_2006_{suffix}",
        2012: f"CLC_2012_{suffix}",
        2018: f"CLC_2018_{suffix}",
    }

    if year <= 2011:
        preferred_year = 2006
    elif year <= 2017:
        preferred_year = 2012
    else:
        preferred_year = 2018

    preferred = candidates[preferred_year]
    if preferred in ds:
        return preferred

    available_years = [yy for yy, name in candidates.items() if name in ds]
    if not available_years:
        raise KeyError(
            f"CLC base feature '{base}' could not be resolved. Tried: {list(candidates.values())}"
        )

    nearest = min(available_years, key=lambda yy: abs(yy - year))
    return candidates[nearest]

