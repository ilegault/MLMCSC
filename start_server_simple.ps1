# MLMCSC Server Launcher - Simple Version
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "MLMCSC Server Launcher" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Working directory: $scriptDir" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found!" -ForegroundColor Red
    Write-Host "Please install Python and add it to PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check app.py
if (-not (Test-Path "app.py")) {
    Write-Host "app.py not found!" -ForegroundColor Red
    Write-Host "Make sure you're running from the project root" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "All checks passed!" -ForegroundColor Green
Write-Host ""

# Start server
Write-Host "Starting MLMCSC server..." -ForegroundColor Yellow
Write-Host "Server output will appear below:" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Gray
Write-Host ""

try {
    # Run the server directly in this PowerShell window
    python "app.py" --server-only
} catch {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Gray
    Write-Host "Server encountered an error!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Gray
    Write-Host "Server stopped." -ForegroundColor Yellow
    Write-Host "Press any key to close this window..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}