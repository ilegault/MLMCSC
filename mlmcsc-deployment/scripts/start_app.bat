@echo off
REM MLMCSC Application Launcher for Windows
REM This script starts all MLMCSC services

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_DIR=%~dp0..
set VENV_NAME=mlmcsc-env
set VENV_PATH=%PROJECT_DIR%\..\%VENV_NAME%
set LOG_DIR=%PROJECT_DIR%\..\logs
set PID_DIR=%PROJECT_DIR%\..\pids

echo MLMCSC Application Launcher
echo ==================================

REM Create directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PID_DIR%" mkdir "%PID_DIR%"

REM Check if virtual environment exists
if not exist "%VENV_PATH%" (
    echo [ERROR] Virtual environment not found at %VENV_PATH%
    echo [ERROR] Please run setup_environment.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

REM Change to project directory
cd /d "%PROJECT_DIR%\.."

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found. Creating default...
    (
    echo DATABASE_URL=postgresql://mlmcsc:mlmcsc123@localhost:5432/mlmcsc
    echo REDIS_URL=redis://localhost:6379/0
    echo CELERY_BROKER_URL=redis://localhost:6379/1
    echo CELERY_RESULT_BACKEND=redis://localhost:6379/2
    echo SECRET_KEY=%RANDOM%%RANDOM%%RANDOM%%RANDOM%
    echo DEBUG=false
    echo ENVIRONMENT=production
    ) > .env
)

REM Load environment variables
for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
)

REM Function to check if process is running
:check_process
set "process_name=%~1"
set "pid_file=%~2"
if exist "%pid_file%" (
    set /p pid=<"%pid_file%"
    tasklist /FI "PID eq !pid!" 2>nul | find "!pid!" >nul
    if !errorlevel! == 0 (
        echo [INFO] %process_name% is already running (PID: !pid!)
        goto :eof
    ) else (
        del "%pid_file%" 2>nul
    )
)
exit /b 1

REM Function to start service
:start_service
set "service_name=%~1"
set "description=%~2"

echo [INFO] Checking %description%...
net start | find "%service_name%" >nul
if %errorlevel% == 0 (
    echo [INFO] %description% is already running
) else (
    echo [INFO] Starting %description%...
    net start "%service_name%" >nul 2>&1
    if %errorlevel% == 0 (
        echo [INFO] %description% started successfully
    ) else (
        echo [WARNING] Could not start %description% as Windows service
        echo [WARNING] Please start %description% manually
    )
)
goto :eof

REM Check system dependencies
echo [INFO] Checking system dependencies...

REM Check PostgreSQL
where psql >nul 2>&1
if %errorLevel% == 0 (
    call :start_service "postgresql-x64-15" "PostgreSQL"
) else (
    echo [WARNING] PostgreSQL not found in PATH. Please ensure PostgreSQL is running.
)

REM Check Redis
where redis-cli >nul 2>&1
if %errorLevel% == 0 (
    call :start_service "Redis" "Redis"
) else (
    echo [WARNING] Redis not found in PATH. Please ensure Redis is running.
)

REM Test database connection
echo [INFO] Testing database connection...
python -c "import psycopg2; import os; conn = psycopg2.connect(os.getenv('DATABASE_URL')); conn.close(); print('Database connection successful')" 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] Database connection failed. Please check your database setup.
    pause
    exit /b 1
)

REM Test Redis connection
echo [INFO] Testing Redis connection...
python -c "import redis; import os; r = redis.from_url(os.getenv('REDIS_URL')); r.ping(); print('Redis connection successful')" 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] Redis connection failed. Please check your Redis setup.
    pause
    exit /b 1
)

REM Run database migrations
echo [INFO] Running database migrations...
if exist "alembic.ini" (
    alembic upgrade head 2>nul
    if %errorLevel% neq 0 (
        echo [WARNING] Database migration failed or not needed
    )
) else (
    echo [WARNING] Alembic configuration not found. Skipping migrations.
)

REM Start application components
echo [INFO] Starting MLMCSC components...

REM Check if Celery worker is already running
call :check_process "Celery Worker" "%PID_DIR%\celery_worker.pid"
if %errorLevel% neq 0 (
    echo [INFO] Starting Celery Worker...
    start "MLMCSC Celery Worker" /MIN cmd /c "celery -A src.workers.celery_app worker --loglevel=info > %LOG_DIR%\celery_worker.log 2>&1"
    
    REM Get the PID of the started process (approximate)
    timeout /t 2 >nul
    for /f "tokens=2" %%i in ('tasklist /FI "WINDOWTITLE eq MLMCSC Celery Worker" /FO CSV ^| find "cmd.exe"') do (
        echo %%i > "%PID_DIR%\celery_worker.pid"
        echo [INFO] Celery Worker started successfully (PID: %%i)
        goto worker_started
    )
    echo [WARNING] Could not determine Celery Worker PID
    :worker_started
)

REM Wait for worker to initialize
timeout /t 3 >nul

REM Check if API server is already running
call :check_process "API Server" "%PID_DIR%\api_server.pid"
if %errorLevel% neq 0 (
    echo [INFO] Starting API Server...
    start "MLMCSC API Server" /MIN cmd /c "python -m uvicorn main:app --host 0.0.0.0 --port 8000 > %LOG_DIR%\api_server.log 2>&1"
    
    REM Get the PID of the started process (approximate)
    timeout /t 2 >nul
    for /f "tokens=2" %%i in ('tasklist /FI "WINDOWTITLE eq MLMCSC API Server" /FO CSV ^| find "cmd.exe"') do (
        echo %%i > "%PID_DIR%\api_server.pid"
        echo [INFO] API Server started successfully (PID: %%i)
        goto api_started
    )
    echo [WARNING] Could not determine API Server PID
    :api_started
)

REM Wait for API server to start
echo [INFO] Waiting for API server to start...
timeout /t 5 >nul

REM Health check
echo [INFO] Performing health check...
python "%PROJECT_DIR%\scripts\health_check.py"
if %errorLevel% neq 0 (
    echo [WARNING] Health check failed. Check logs for details.
)

echo.
echo [INFO] MLMCSC application started successfully!
echo.
echo Services Status:
echo - API Server: http://localhost:8000
echo - API Documentation: http://localhost:8000/docs
echo - Celery Worker: Running
echo.
echo Log Files:
echo - API Server: %LOG_DIR%\api_server.log
echo - Celery Worker: %LOG_DIR%\celery_worker.log
echo.
echo PID Files:
echo - API Server: %PID_DIR%\api_server.pid
echo - Celery Worker: %PID_DIR%\celery_worker.pid
echo.
echo To stop the application:
echo - Close the command windows or use Task Manager
echo - Or run: taskkill /PID [PID_NUMBER] /F
echo.
echo To view logs:
echo - type %LOG_DIR%\api_server.log
echo - type %LOG_DIR%\celery_worker.log
echo.
echo Press any key to open the web interface...
pause >nul
start http://localhost:8000