#!/bin/bash
# MLMCSC Application Launcher for Linux
# This script starts all MLMCSC services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME="mlmcsc-env"
VENV_PATH="$PROJECT_DIR/../$VENV_NAME"
LOG_DIR="$PROJECT_DIR/../logs"
PID_DIR="$PROJECT_DIR/../pids"

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

# Create PID directory
mkdir -p "$PID_DIR"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}MLMCSC Application Launcher${NC}"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    print_error "Virtual environment not found at $VENV_PATH"
    print_error "Please run setup_environment.sh first"
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Change to project directory
cd "$PROJECT_DIR/.."

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating default..."
    cat > .env <<EOF
DATABASE_URL=postgresql://mlmcsc:mlmcsc123@localhost:5432/mlmcsc
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
SECRET_KEY=$(openssl rand -hex 32)
DEBUG=false
ENVIRONMENT=production
EOF
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Function to check if service is running
is_service_running() {
    local service_name=$1
    if command -v systemctl &> /dev/null; then
        systemctl is-active --quiet "$service_name" 2>/dev/null
    else
        return 1
    fi
}

# Function to start service
start_service() {
    local service_name=$1
    local description=$2
    
    if is_service_running "$service_name"; then
        print_status "$description is already running"
    else
        print_status "Starting $description..."
        if command -v systemctl &> /dev/null; then
            sudo systemctl start "$service_name"
            if is_service_running "$service_name"; then
                print_status "$description started successfully"
            else
                print_error "Failed to start $description"
                return 1
            fi
        else
            print_warning "systemctl not available. Starting manually..."
            case "$service_name" in
                "postgresql")
                    sudo service postgresql start 2>/dev/null || sudo /etc/init.d/postgresql start 2>/dev/null || print_warning "Could not start PostgreSQL"
                    ;;
                "redis")
                    sudo service redis start 2>/dev/null || sudo /etc/init.d/redis-server start 2>/dev/null || print_warning "Could not start Redis"
                    ;;
            esac
        fi
    fi
}

# Function to start application component
start_component() {
    local component=$1
    local command=$2
    local log_file=$3
    local pid_file=$4
    
    if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
        print_status "$component is already running (PID: $(cat $pid_file))"
        return 0
    fi
    
    print_status "Starting $component..."
    nohup $command > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    # Wait a moment and check if process is still running
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        print_status "$component started successfully (PID: $pid)"
    else
        print_error "Failed to start $component"
        rm -f "$pid_file"
        return 1
    fi
}

# Check dependencies
print_status "Checking system dependencies..."

# Check PostgreSQL
if command -v psql &> /dev/null; then
    start_service "postgresql" "PostgreSQL"
else
    print_warning "PostgreSQL not found. Please install PostgreSQL."
fi

# Check Redis
if command -v redis-cli &> /dev/null; then
    start_service "redis" "Redis"
else
    print_warning "Redis not found. Please install Redis."
fi

# Test database connection
print_status "Testing database connection..."
python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" || {
    print_error "Database connection failed. Please check your database setup."
    exit 1
}

# Test Redis connection
print_status "Testing Redis connection..."
python -c "
import redis
import os
try:
    r = redis.from_url(os.getenv('REDIS_URL'))
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
" || {
    print_error "Redis connection failed. Please check your Redis setup."
    exit 1
}

# Run database migrations
print_status "Running database migrations..."
if [ -f "alembic.ini" ]; then
    alembic upgrade head || print_warning "Database migration failed or not needed"
else
    print_warning "Alembic configuration not found. Skipping migrations."
fi

# Start application components
print_status "Starting MLMCSC components..."

# Start Celery worker
start_component "Celery Worker" \
    "celery -A src.workers.celery_app worker --loglevel=info" \
    "$LOG_DIR/celery_worker.log" \
    "$PID_DIR/celery_worker.pid"

# Wait a moment for worker to initialize
sleep 3

# Start API server
start_component "API Server" \
    "python -m uvicorn main:app --host 0.0.0.0 --port 8000" \
    "$LOG_DIR/api_server.log" \
    "$PID_DIR/api_server.pid"

# Wait for API server to start
print_status "Waiting for API server to start..."
sleep 5

# Health check
print_status "Performing health check..."
python "$PROJECT_DIR/scripts/health_check.py" || {
    print_warning "Health check failed. Check logs for details."
}

echo ""
print_status "MLMCSC application started successfully!"
echo ""
echo -e "${GREEN}Services Status:${NC}"
echo "- API Server: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo "- Celery Worker: Running"
echo ""
echo -e "${GREEN}Log Files:${NC}"
echo "- API Server: $LOG_DIR/api_server.log"
echo "- Celery Worker: $LOG_DIR/celery_worker.log"
echo ""
echo -e "${GREEN}PID Files:${NC}"
echo "- API Server: $PID_DIR/api_server.pid"
echo "- Celery Worker: $PID_DIR/celery_worker.pid"
echo ""
echo -e "${YELLOW}To stop the application:${NC}"
echo "kill \$(cat $PID_DIR/api_server.pid)"
echo "kill \$(cat $PID_DIR/celery_worker.pid)"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo "tail -f $LOG_DIR/api_server.log"
echo "tail -f $LOG_DIR/celery_worker.log"