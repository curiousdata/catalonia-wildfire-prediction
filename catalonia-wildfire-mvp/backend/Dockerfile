FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Create a non-root user for safer containers
RUN addgroup --system app && adduser --system --ingroup app app

# Install Python deps (cached layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure permissions for the non-root user
RUN chown -R app:app /app
USER app

EXPOSE 8000

# FastAPI entrypoint (matches backend/src/main.py)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]