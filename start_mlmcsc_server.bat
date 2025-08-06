@echo off
echo ========================================
echo MLMCSC Server Launcher
echo ========================================
echo.
echo Starting MLMCSC server...
echo.

cd /d "%~dp0"
python app.py

echo.
echo Server stopped. Press any key to exit...
pause >nul