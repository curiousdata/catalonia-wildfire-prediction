FROM python:alpine

# Basic OS deps (folium uses leaflet in HTML; pillow for PNG; numpy/scipy for ML)
RUN apk add --no-cache \
    build-base \
    gcc \
    g++ \
    gdal \
    gdal-dev \
    proj \
    proj-dev \
    geos \
    geos-dev \
    ca-certificates \
    # Add these for numpy/scipy performance:
    openblas-dev \
    lapack-dev \
    # For Pillow:
    jpeg-dev \
    zlib-dev \
    # Cleanup build deps later if needed
    && rm -rf /var/cache/apk/*

# Keep python logs unbuffered
ENV PYTHONUNBUFFERED=1

# Create a non-root user for improved security (Alpine uses addgroup/adduser)
RUN addgroup -S appuser && adduser -S appuser -G appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python packages (--no-cache-dir saves space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Switch to non-root user
USER appuser

# Expose port if running a web service
EXPOSE 8000

CMD ["python", "app.py"]
