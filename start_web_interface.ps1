# MLMCSC Human-in-the-Loop Web Interface Startup Script
# PowerShell version for better Windows compatibility

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MLMCSC Human-in-the-Loop Web Interface" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Get script directory and change to web directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$webDir = Join-Path $scriptDir "src\web"

if (-not (Test-Path $webDir)) {
    Write-Host "✗ ERROR: Web directory not found at $webDir" -ForegroundColor Red
    Write-Host "Please ensure you're running this from the MLMCSC root directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Set-Location $webDir

# Check if required files exist
if (-not (Test-Path "api.py")) {
    Write-Host "✗ ERROR: Web interface files not found" -ForegroundColor Red
    Write-Host "Please ensure the web interface is properly installed" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import fastapi, uvicorn, cv2, numpy, sklearn" 2>$null
    Write-Host "✓ Dependencies check passed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Some dependencies may be missing" -ForegroundColor Yellow
    Write-Host "If the server fails to start, run: pip install -r ../../requirements.txt" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting web server..." -ForegroundColor Green
Write-Host ""
Write-Host "The interface will be available at: " -NoNewline
Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
try {
    python run_server.py
} catch {
    Write-Host ""
    Write-Host "✗ Server failed to start" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Yellow
} finally {
    Write-Host ""
    Write-Host "Server stopped." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}