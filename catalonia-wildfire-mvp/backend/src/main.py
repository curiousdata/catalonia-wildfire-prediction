from __future__ import annotations

from fastapi import FastAPI

# Canonical router import per enforced structure:
# backend/src/api/routes.py defines `router = APIRouter()` and endpoints: /health, /dates, /map
from .api.routes import router as api_router


app = FastAPI(
    title="Catalonia Wildfire Prediction API",
    version="0.1.0",
)

# Include API routes
app.include_router(api_router)


@app.get("/")
def root() -> dict:
    """Simple landing endpoint for quick checks in the browser."""
    return {
        "service": "catalonia-wildfire-mvp",
        "status": "ok",
        "endpoints": ["/health", "/dates", "/map"],
    }


if __name__ == "__main__":
    import uvicorn

    # For local debugging only; in Docker you typically run uvicorn via CMD.
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)