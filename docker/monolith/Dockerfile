FROM python:3.11-slim

# Basic OS deps (folium uses leaflet in HTML; pillow for PNG; numpy/scipy for ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    proj-data \
    libproj-dev \
    libgeos-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep python logs unbuffered
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy monolith requirements (for reproducible builds)
COPY docker/monolith/requirements.txt /tmp/requirements.txt

# Install runtime dependencies (prefer wheels)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install python deps for the monolith app
RUN pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt

# Fail fast if native wheels/libs are missing
RUN python -c "import rasterio; import pyproj; print('rasterio', rasterio.__version__, 'pyproj', pyproj.__version__)"