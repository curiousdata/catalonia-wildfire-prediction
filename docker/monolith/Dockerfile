FROM python:3.11-slim

# Basic OS deps (folium uses leaflet in HTML; pillow for PNG; numpy/scipy for ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Keep python logs unbuffered
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install runtime dependencies (pin later)
RUN pip install --no-cache-dir --upgrade pip

# Minimal set for your monolith app
RUN pip install --no-cache-dir \
    streamlit \
    requests \
    numpy \
    pillow \
    folium \
    torch \
    segmentation-models-pytorch \
    zarr \
    xarray \
    numcodecs \
    pyproj \
    tqdm

# NOTE:
# If your src/data/datasets.py needs additional packages (pandas, scipy, etc.),
# add them here. Keep it minimal for MVP.