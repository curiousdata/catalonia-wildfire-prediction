from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# -----------------------------
# Enums
# -----------------------------

class ViewMode(str, Enum):
    """Which layer(s) to render on the map overlay."""

    prediction = "prediction"
    label = "label"
    both = "both"


# -----------------------------
# /dates endpoint
# -----------------------------

class DatesResponse(BaseModel):
    """Response model for GET /dates."""

    dates: List[str] = Field(
        ..., description="Available dates in YYYY-MM-DD format that can be visualized"
    )


# -----------------------------
# /map endpoint
# -----------------------------

class MapResponse(BaseModel):
    """Response model for GET /map.

    image_b64:
        Base64-encoded PNG (RGBA) suitable for Folium ImageOverlay.

    bounds:
        Geographic bounds in WGS84 (EPSG:4326), formatted as:
        [[lat_min, lon_min], [lat_max, lon_max]]
    """

    image_b64: str = Field(
        ..., description="Base64-encoded RGBA PNG overlay"
    )
    bounds: List[List[float]] = Field(
        ..., description="[[lat_min, lon_min], [lat_max, lon_max]] in EPSG:4326"
    )

    # Optional echo/debug fields (useful for frontend state & debugging)
    date: Optional[str] = Field(
        None, description="Date used to generate this overlay (YYYY-MM-DD)"
    )
    view: Optional[ViewMode] = Field(
        None, description="View mode used to generate this overlay"
    )