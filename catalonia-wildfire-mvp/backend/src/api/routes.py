from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

# IMPORTANT:
# - API layer must not hardcode local laptop paths.
# - API layer should only orchestrate: validate request, call inference functions, shape response.
# - Pydantic schemas live in backend/src/types/schema.py.

router = APIRouter()


# ---- Schemas (canonical: types/schema.py; dev fallback keeps server runnable) ----
try:
    from ..types.schema import DatesResponse, MapResponse, ViewMode  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    from enum import Enum
    from pydantic import BaseModel
    from typing import List, Optional

    class ViewMode(str, Enum):
        prediction = "prediction"
        label = "label"
        both = "both"

    class DatesResponse(BaseModel):
        dates: List[str]

    class MapResponse(BaseModel):
        image_b64: str
        bounds: List[List[float]]
        date: Optional[str] = None
        view: Optional[str] = None


# ---- Model loader (singleton accessor) ----
try:
    from ..models.loader import get_model  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    get_model = None  # type: ignore


# ---- Inference functions ----
try:
    from ..inference.predict import list_available_dates, build_map_overlay  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    list_available_dates = None  # type: ignore
    build_map_overlay = None  # type: ignore


@router.get("/health")
def health() -> dict:
    """Lightweight liveness endpoint."""
    return {"status": "ok"}


@router.get("/dates", response_model=DatesResponse)
def dates() -> DatesResponse:
    """Return available dates the system can serve (for the Streamlit date picker)."""
    if list_available_dates is None or not callable(list_available_dates):
        return DatesResponse(dates=[])

    try:
        dts = list_available_dates()
        dts_str = [str(d) for d in dts]
        return DatesResponse(dates=dts_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list dates: {e}")


@router.get("/map", response_model=MapResponse)
def map_overlay(
    date: str = Query(..., description="Date to predict/visualize in YYYY-MM-DD."),
    view: ViewMode = Query(ViewMode.prediction, description="prediction | label | both"),
) -> MapResponse:
    """Return a map-ready overlay (base64 PNG + lat/lon bounds) for Folium."""
    if build_map_overlay is None or not callable(build_map_overlay):
        raise HTTPException(
            status_code=501,
            detail="Map overlay inference is not wired yet. Implement build_map_overlay() in inference/predict.py.",
        )

    try:
        model = get_model() if callable(get_model) else None
        payload = build_map_overlay(date=date, view=view.value, model=model)

        if not isinstance(payload, dict):
            raise ValueError("build_map_overlay must return a dict")

        if "image_b64" not in payload or "bounds" not in payload:
            raise ValueError("build_map_overlay must return keys: image_b64, bounds")

        return MapResponse(
            image_b64=payload["image_b64"],
            bounds=payload["bounds"],
            date=date,
            view=view,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build overlay: {e}")