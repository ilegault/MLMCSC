# MLMCSC Single Container Deployment
# This Dockerfile creates a complete MLMCSC environment in a single container

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    redis-tools \
    curl \
    wget \
    git \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR $APP_HOME

# Create non-root user
RUN groupadd -r mlmcsc && useradd -r -g mlmcsc mlmcsc

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/data/images \
    /app/data/models \
    /app/data/temp \
    /app/logs \
    /app/models/cache \
    /app/models/versions \
    /app/models/configs \
    /app/models/metadata \
    /var/log/supervisor \
    /run/supervisor

# Copy configuration files
COPY mlmcsc-deployment/config/ /app/config/
COPY mlmcsc-deployment/docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY mlmcsc-deployment/docker/nginx.conf /etc/nginx/sites-available/default

# Set permissions
RUN chown -R mlmcsc:mlmcsc /app && \
    chmod +x /app/mlmcsc-deployment/scripts/*.py

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting MLMCSC application..."

# Wait for external services if URLs are provided
if [ ! -z "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    python -c "
import time
import psycopg2
import os
url = os.getenv('DATABASE_URL')
for i in range(30):
    try:
        conn = psycopg2.connect(url)
        conn.close()
        print('Database is ready!')
        break
    except:
        print(f'Waiting for database... ({i+1}/30)')
        time.sleep(2)
else:
    print('Database connection timeout')
    exit(1)
"
fi

if [ ! -z "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    python -c "
import time
import redis
import os
url = os.getenv('REDIS_URL')
for i in range(30):
    try:
        r = redis.from_url(url)
        r.ping()
        print('Redis is ready!')
        break
    except:
        print(f'Waiting for Redis... ({i+1}/30)')
        time.sleep(2)
else:
    print('Redis connection timeout')
    exit(1)
"
fi

# Run database migrations
if [ -f "alembic.ini" ]; then
    echo "Running database migrations..."
    alembic upgrade head || echo "Migration failed or not needed"
fi

# Start supervisor
echo "Starting services with supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
EOF

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/mlmcsc-deployment/scripts/health_check.py || exit 1

# Expose ports
EXPOSE 8000 80

# Switch to non-root user
USER mlmcsc

# Set default environment variables
ENV DATABASE_URL=postgresql://mlmcsc:mlmcsc123@localhost:5432/mlmcsc \
    REDIS_URL=redis://localhost:6379/0 \
    CELERY_BROKER_URL=redis://localhost:6379/1 \
    CELERY_RESULT_BACKEND=redis://localhost:6379/2 \
    DEBUG=false \
    ENVIRONMENT=production

# Start the application
CMD ["/app/start.sh"]