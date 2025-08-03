#!/bin/bash
# MLMCSC Environment Setup Script for Linux
# This script sets up the complete MLMCSC environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.9"
VENV_NAME="mlmcsc-env"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS_FILE="$PROJECT_DIR/../requirements.txt"

echo -e "${BLUE}MLMCSC Environment Setup for Linux${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check Python version
print_status "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python is not installed. Please install Python $PYTHON_VERSION or higher."
    exit 1
fi

PYTHON_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_status "Found Python $PYTHON_VER"

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip is not available. Please install pip."
    exit 1
fi

# Install system dependencies
print_status "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y \
        python3-venv \
        python3-dev \
        build-essential \
        libpq-dev \
        redis-server \
        postgresql \
        postgresql-contrib \
        nginx \
        git \
        curl \
        wget
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    sudo yum update -y
    sudo yum install -y \
        python3-venv \
        python3-devel \
        gcc \
        gcc-c++ \
        postgresql-devel \
        redis \
        postgresql-server \
        postgresql-contrib \
        nginx \
        git \
        curl \
        wget
elif command -v dnf &> /dev/null; then
    # Fedora
    sudo dnf update -y
    sudo dnf install -y \
        python3-venv \
        python3-devel \
        gcc \
        gcc-c++ \
        postgresql-devel \
        redis \
        postgresql-server \
        postgresql-contrib \
        nginx \
        git \
        curl \
        wget
else
    print_warning "Unknown package manager. Please install dependencies manually."
fi

# Create virtual environment
print_status "Creating virtual environment..."
cd "$PROJECT_DIR/.."
if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment already exists. Removing..."
    rm -rf "$VENV_NAME"
fi

$PYTHON_CMD -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
else
    print_warning "Requirements file not found. Installing basic dependencies..."
    pip install \
        fastapi \
        uvicorn \
        celery \
        redis \
        psycopg2-binary \
        sqlalchemy \
        alembic \
        pydantic \
        python-multipart \
        jinja2 \
        aiofiles \
        python-jose \
        passlib \
        bcrypt \
        pillow \
        numpy \
        pandas \
        scikit-learn \
        torch \
        torchvision \
        opencv-python \
        matplotlib \
        seaborn \
        pytest \
        pytest-asyncio
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p "$PROJECT_DIR/../data/images"
mkdir -p "$PROJECT_DIR/../data/models"
mkdir -p "$PROJECT_DIR/../data/temp"
mkdir -p "$PROJECT_DIR/../logs"
mkdir -p "$PROJECT_DIR/../models/cache"
mkdir -p "$PROJECT_DIR/../models/versions"
mkdir -p "$PROJECT_DIR/../models/configs"
mkdir -p "$PROJECT_DIR/../models/metadata"

# Set up PostgreSQL
print_status "Setting up PostgreSQL..."
if command -v systemctl &> /dev/null; then
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
fi

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE mlmcsc;" 2>/dev/null || print_warning "Database might already exist"
sudo -u postgres psql -c "CREATE USER mlmcsc WITH PASSWORD 'mlmcsc123';" 2>/dev/null || print_warning "User might already exist"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mlmcsc TO mlmcsc;" 2>/dev/null || true

# Set up Redis
print_status "Setting up Redis..."
if command -v systemctl &> /dev/null; then
    sudo systemctl start redis
    sudo systemctl enable redis
fi

# Create systemd service files
print_status "Creating systemd service files..."

# MLMCSC API service
sudo tee /etc/systemd/system/mlmcsc-api.service > /dev/null <<EOF
[Unit]
Description=MLMCSC API Server
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR/..
Environment=PATH=$PROJECT_DIR/../$VENV_NAME/bin
ExecStart=$PROJECT_DIR/../$VENV_NAME/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# MLMCSC Worker service
sudo tee /etc/systemd/system/mlmcsc-worker.service > /dev/null <<EOF
[Unit]
Description=MLMCSC Celery Worker
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR/..
Environment=PATH=$PROJECT_DIR/../$VENV_NAME/bin
ExecStart=$PROJECT_DIR/../$VENV_NAME/bin/celery -A src.workers.celery_app worker --loglevel=info
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Set permissions
print_status "Setting permissions..."
chmod +x "$PROJECT_DIR/scripts/"*.sh
chown -R $USER:$USER "$PROJECT_DIR/../data"
chown -R $USER:$USER "$PROJECT_DIR/../logs"

# Create environment file
print_status "Creating environment file..."
cat > "$PROJECT_DIR/../.env" <<EOF
# MLMCSC Environment Variables
DATABASE_URL=postgresql://mlmcsc:mlmcsc123@localhost:5432/mlmcsc
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
SECRET_KEY=$(openssl rand -hex 32)
DEBUG=false
ENVIRONMENT=production
EOF

print_status "Environment setup completed successfully!"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Activate the virtual environment: source $VENV_NAME/bin/activate"
echo "2. Run the health check: python mlmcsc-deployment/scripts/health_check.py"
echo "3. Start the application: ./mlmcsc-deployment/scripts/start_app.sh"
echo ""
echo -e "${YELLOW}Services created:${NC}"
echo "- mlmcsc-api.service (API server)"
echo "- mlmcsc-worker.service (Celery worker)"
echo ""
echo -e "${YELLOW}To start services:${NC}"
echo "sudo systemctl start mlmcsc-api"
echo "sudo systemctl start mlmcsc-worker"
echo ""
echo -e "${YELLOW}To enable auto-start:${NC}"
echo "sudo systemctl enable mlmcsc-api"
echo "sudo systemctl enable mlmcsc-worker"