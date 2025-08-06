# MLMCSC Human-in-the-Loop Interface
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "MLMCSC Human-in-the-Loop Interface" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""
Write-Host "Features included:" -ForegroundColor Yellow
Write-Host "✓ Image upload and analysis" -ForegroundColor Green
Write-Host "✓ Technician labeling interface" -ForegroundColor Green
Write-Host "✓ Model performance metrics" -ForegroundColor Green
Write-Host "✓ Labeling history and export" -ForegroundColor Green
Write-Host "✓ Online learning system" -ForegroundColor Green
Write-Host ""
Write-Host "Server Information:" -ForegroundColor Yellow
Write-Host "• Web interface: http://localhost:8000" -ForegroundColor Cyan
Write-Host "• API documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Main Endpoints:" -ForegroundColor Yellow
Write-Host "• GET  / - Main web interface" -ForegroundColor Cyan
Write-Host "• POST /predict - Image prediction" -ForegroundColor Cyan
Write-Host "• POST /submit_label - Submit technician labels" -ForegroundColor Cyan
Write-Host "• GET  /get_metrics - Model performance metrics" -ForegroundColor Cyan
Write-Host "• GET  /get_history - Labeling history" -ForegroundColor Cyan
Write-Host ""
Write-Host "Controls:" -ForegroundColor Yellow
Write-Host "• Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host "• Type 'browser' and press Enter to open web interface" -ForegroundColor Cyan
Write-Host "• Type 'status' and press Enter to check server status" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Change to project directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Function to handle user commands
function Handle-Command {
    param($command)
    
    switch ($command.ToLower()) {
        "status" {
            Write-Host "Checking server status..." -ForegroundColor Yellow
            try {
                $result = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
                Write-Host "Server Status:" -ForegroundColor Green
                Write-Host "  Status: $($result.status)" -ForegroundColor Cyan
                Write-Host "  Models Loaded:" -ForegroundColor Cyan
                Write-Host "    Detector: $($result.models_loaded.detector)" -ForegroundColor Cyan
                Write-Host "    Classifier: $($result.models_loaded.classifier)" -ForegroundColor Cyan
                Write-Host "    Regression: $($result.models_loaded.regression)" -ForegroundColor Cyan
                Write-Host "  Timestamp: $($result.timestamp)" -ForegroundColor Cyan
            } catch {
                Write-Host "Error: Server not responding" -ForegroundColor Red
            }
        }
        "browser" {
            Write-Host "Opening web interface..." -ForegroundColor Yellow
            Start-Process "http://localhost:8000"
        }
        "help" {
            Write-Host "Available commands:" -ForegroundColor Yellow
            Write-Host "  status  - Check server status" -ForegroundColor Cyan
            Write-Host "  browser - Open web interface" -ForegroundColor Cyan
            Write-Host "  help    - Show this help" -ForegroundColor Cyan
        }
        default {
            if ($command -ne "") {
                Write-Host "Unknown command: $command" -ForegroundColor Red
                Write-Host "Type 'help' for available commands" -ForegroundColor Yellow
            }
        }
    }
}

# Start the Python server in background
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host "Working directory: $scriptDir" -ForegroundColor Cyan
Write-Host "Python command: python app.py --server-only" -ForegroundColor Cyan

# Test if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found in PATH!" -ForegroundColor Red
    Write-Host "Please make sure Python is installed and added to PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Test if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "❌ app.py not found in current directory!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Start the server directly in this PowerShell session
Write-Host "Launching Python server..." -ForegroundColor Yellow

# Start server in background process
$serverProcess = Start-Process -FilePath "python" -ArgumentList "app.py --server-only" -WorkingDirectory $scriptDir -PassThru -WindowStyle Hidden

# Wait for server to be ready with proper health checking
Write-Host "Waiting for server to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$serverReady = $false

while ($attempt -lt $maxAttempts -and -not $serverReady) {
    $attempt++
    Start-Sleep -Seconds 1
    
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 2
        if ($health.status -eq "healthy") {
            $serverReady = $true
            Write-Host "✓ Server started successfully!" -ForegroundColor Green
            Write-Host "✓ Health check passed" -ForegroundColor Green
            Write-Host "✓ Models loaded: Detector=$($health.models_loaded.detector), Classifier=$($health.models_loaded.classifier), Regression=$($health.models_loaded.regression)" -ForegroundColor Green
        }
    } catch {
        Write-Host "." -NoNewline -ForegroundColor Yellow
    }
}

if ($serverReady) {
    # Auto-open browser now that server is confirmed ready
    Write-Host ""
    Write-Host "Opening web interface..." -ForegroundColor Yellow
    Start-Process "http://localhost:8000"
    Write-Host "✓ Browser opened successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "✗ Server failed to start or health check timed out" -ForegroundColor Red
    Write-Host "Check the server output above for errors" -ForegroundColor Yellow
    Write-Host "You can still try to access http://localhost:8000 manually" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Server is running! You can now:" -ForegroundColor Green
Write-Host "• Use the web interface that just opened" -ForegroundColor Cyan
Write-Host "• Type commands below for camera management" -ForegroundColor Cyan
Write-Host "• Press Ctrl+C to stop everything" -ForegroundColor Red
Write-Host ""

# Interactive command loop
try {
    while ($true) {
        Write-Host "MLMCSC> " -NoNewline -ForegroundColor Green
        $command = Read-Host
        
        if ($command -eq "exit" -or $command -eq "quit") {
            break
        }
        
        Handle-Command $command
        Write-Host ""
    }
} catch {
    # User pressed Ctrl+C
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Yellow
} finally {
    # Clean up
    Write-Host "Stopping server..." -ForegroundColor Yellow
    if ($serverProcess -and -not $serverProcess.HasExited) {
        $serverProcess.Kill()
        Write-Host "Server process terminated." -ForegroundColor Yellow
    }
    Write-Host "Server stopped. Goodbye!" -ForegroundColor Green
}