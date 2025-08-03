@echo off
REM MLMCSC Environment Setup Script for Windows
REM This script sets up the complete MLMCSC environment

setlocal enabledelayedexpansion

echo MLMCSC Environment Setup for Windows
echo ==================================================

REM Configuration
set PYTHON_VERSION=3.9
set VENV_NAME=mlmcsc-env
set PROJECT_DIR=%~dp0..
set REQUIREMENTS_FILE=%PROJECT_DIR%\..\requirements.txt

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Running with administrator privileges
) else (
    echo [WARNING] Not running as administrator. Some features may not work.
    echo [WARNING] Consider running as administrator for full setup.
)

REM Check Python installation
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo [ERROR] Please install Python %PYTHON_VERSION% or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [INFO] Found Python %PYTHON_VER%

REM Check pip
echo [INFO] Checking pip installation...
python -m pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] pip is not available. Please install pip.
    pause
    exit /b 1
)

REM Install Chocolatey if not present (for system dependencies)
where choco >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Installing Chocolatey package manager...
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    if %errorLevel% neq 0 (
        echo [WARNING] Failed to install Chocolatey. Please install dependencies manually.
    )
)

REM Install system dependencies
echo [INFO] Installing system dependencies...
where choco >nul 2>&1
if %errorLevel% == 0 (
    choco install -y postgresql redis git curl wget
    if %errorLevel% neq 0 (
        echo [WARNING] Some dependencies may not have been installed correctly.
    )
) else (
    echo [WARNING] Chocolatey not available. Please install the following manually:
    echo - PostgreSQL
    echo - Redis
    echo - Git
)

REM Create virtual environment
echo [INFO] Creating virtual environment...
cd /d "%PROJECT_DIR%\.."
if exist "%VENV_NAME%" (
    echo [WARNING] Virtual environment already exists. Removing...
    rmdir /s /q "%VENV_NAME%"
)

python -m venv "%VENV_NAME%"
if %errorLevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

call "%VENV_NAME%\Scripts\activate.bat"

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install Python dependencies
echo [INFO] Installing Python dependencies...
if exist "%REQUIREMENTS_FILE%" (
    pip install -r "%REQUIREMENTS_FILE%"
) else (
    echo [WARNING] Requirements file not found. Installing basic dependencies...
    pip install fastapi uvicorn celery redis psycopg2-binary sqlalchemy alembic pydantic python-multipart jinja2 aiofiles python-jose passlib bcrypt pillow numpy pandas scikit-learn torch torchvision opencv-python matplotlib seaborn pytest pytest-asyncio
)

if %errorLevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)

REM Create necessary directories
echo [INFO] Creating directory structure...
mkdir "%PROJECT_DIR%\..\data\images" 2>nul
mkdir "%PROJECT_DIR%\..\data\models" 2>nul
mkdir "%PROJECT_DIR%\..\data\temp" 2>nul
mkdir "%PROJECT_DIR%\..\logs" 2>nul
mkdir "%PROJECT_DIR%\..\models\cache" 2>nul
mkdir "%PROJECT_DIR%\..\models\versions" 2>nul
mkdir "%PROJECT_DIR%\..\models\configs" 2>nul
mkdir "%PROJECT_DIR%\..\models\metadata" 2>nul

REM Create environment file
echo [INFO] Creating environment file...
(
echo # MLMCSC Environment Variables
echo DATABASE_URL=postgresql://mlmcsc:mlmcsc123@localhost:5432/mlmcsc
echo REDIS_URL=redis://localhost:6379/0
echo CELERY_BROKER_URL=redis://localhost:6379/1
echo CELERY_RESULT_BACKEND=redis://localhost:6379/2
echo SECRET_KEY=%RANDOM%%RANDOM%%RANDOM%%RANDOM%
echo DEBUG=false
echo ENVIRONMENT=production
) > "%PROJECT_DIR%\..\.env"

REM Create Windows service batch files
echo [INFO] Creating service scripts...

REM API service script
(
echo @echo off
echo cd /d "%PROJECT_DIR%\.."
echo call "%VENV_NAME%\Scripts\activate.bat"
echo python -m uvicorn main:app --host 0.0.0.0 --port 8000
) > "%PROJECT_DIR%\..\start_api_service.bat"

REM Worker service script
(
echo @echo off
echo cd /d "%PROJECT_DIR%\.."
echo call "%VENV_NAME%\Scripts\activate.bat"
echo celery -A src.workers.celery_app worker --loglevel=info
) > "%PROJECT_DIR%\..\start_worker_service.bat"

REM Create NSSM service installer (if NSSM is available)
where nssm >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Creating Windows services with NSSM...
    nssm install MLMCSC-API "%PROJECT_DIR%\..\start_api_service.bat"
    nssm set MLMCSC-API Description "MLMCSC API Server"
    nssm set MLMCSC-API Start SERVICE_AUTO_START
    
    nssm install MLMCSC-Worker "%PROJECT_DIR%\..\start_worker_service.bat"
    nssm set MLMCSC-Worker Description "MLMCSC Celery Worker"
    nssm set MLMCSC-Worker Start SERVICE_AUTO_START
    
    echo [INFO] Windows services created successfully.
) else (
    echo [WARNING] NSSM not found. Services not created automatically.
    echo [WARNING] Install NSSM to create Windows services: choco install nssm
)

REM Setup PostgreSQL database
echo [INFO] Setting up PostgreSQL database...
where psql >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Creating database and user...
    psql -U postgres -c "CREATE DATABASE mlmcsc;" 2>nul
    psql -U postgres -c "CREATE USER mlmcsc WITH PASSWORD 'mlmcsc123';" 2>nul
    psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE mlmcsc TO mlmcsc;" 2>nul
    echo [INFO] Database setup completed.
) else (
    echo [WARNING] PostgreSQL command line tools not found.
    echo [WARNING] Please create database manually:
    echo   - Database: mlmcsc
    echo   - User: mlmcsc
    echo   - Password: mlmcsc123
)

REM Create desktop shortcuts
echo [INFO] Creating desktop shortcuts...
set DESKTOP=%USERPROFILE%\Desktop

REM API shortcut
(
echo @echo off
echo cd /d "%PROJECT_DIR%\.."
echo call "%VENV_NAME%\Scripts\activate.bat"
echo start "MLMCSC API" python -m uvicorn main:app --host 0.0.0.0 --port 8000
echo echo MLMCSC API started at http://localhost:8000
echo pause
) > "%DESKTOP%\Start MLMCSC API.bat"

REM Worker shortcut
(
echo @echo off
echo cd /d "%PROJECT_DIR%\.."
echo call "%VENV_NAME%\Scripts\activate.bat"
echo start "MLMCSC Worker" celery -A src.workers.celery_app worker --loglevel=info
echo echo MLMCSC Worker started
echo pause
) > "%DESKTOP%\Start MLMCSC Worker.bat"

echo [INFO] Environment setup completed successfully!
echo.
echo Next steps:
echo 1. Activate the virtual environment: %VENV_NAME%\Scripts\activate.bat
echo 2. Run the health check: python mlmcsc-deployment\scripts\health_check.py
echo 3. Start the application: mlmcsc-deployment\scripts\start_app.bat
echo.
echo Desktop shortcuts created:
echo - Start MLMCSC API.bat
echo - Start MLMCSC Worker.bat
echo.
echo Services (if NSSM installed):
echo - MLMCSC-API
echo - MLMCSC-Worker
echo.
echo To start services manually:
echo net start MLMCSC-API
echo net start MLMCSC-Worker

pause