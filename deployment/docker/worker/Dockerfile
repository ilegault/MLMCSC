# Celery Worker Dockerfile for MLMCSC
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    C_FORCE_ROOT=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r celery && useradd -r -g celery celery

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt && \
    pip install celery[redis]==5.3.4

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/celery && \
    chown -R celery:celery /app

# Switch to non-root user
USER celery

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD celery -A src.workers.celery_app inspect ping || exit 1

# Start Celery worker
CMD ["celery", "-A", "src.workers.celery_app", "worker", "--loglevel=info", "--concurrency=2"]