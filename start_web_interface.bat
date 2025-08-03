@echo off
REM MLMCSC Human-in-the-Loop Web Interface Startup Script
REM This script starts the web interface server

echo ========================================
echo MLMCSC Human-in-the-Loop Web Interface
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Change to the web directory
cd /d "%~dp0src\web"

REM Check if required files exist
if not exist "api.py" (
    echo ERROR: Web interface files not found
    echo Please ensure you're running this from the MLMCSC root directory
    pause
    exit /b 1
)

echo Starting web server...
echo.
echo The interface will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python run_server.py

REM If we get here, the server has stopped
echo.
echo Server stopped.
pause