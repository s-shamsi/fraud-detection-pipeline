# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for Cloud Run security best practices
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (only what the inference service needs)
COPY inference/       ./inference/
COPY data/__init__.py ./data/__init__.py
COPY data/generate_data.py ./data/generate_data.py

# Artefacts are mounted at runtime via Cloud Run volume or GCS FUSE.
# For local Docker usage, mount with: -v ./artifacts:/app/artifacts
RUN mkdir -p artifacts data/predictions data/processed

# Cloud Run listens on PORT env variable (default 8080)
ENV PORT=8080 \
    ARTIFACT_DIR=/app/artifacts \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORT_THREADS=2

USER appuser

EXPOSE 8080

# Use gunicorn + uvicorn workers for production throughput.
# Cloud Run scales horizontally; keep workers=1 and scale via instances.
CMD uvicorn inference.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --loop uvloop \
    --access-log \
    --log-level info
