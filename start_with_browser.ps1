# MLMCSC Server with Auto Browser Opening
param(
    [int]$DelaySeconds = 10
)

Write-Host "🚀 MLMCSC Server Starting..." -ForegroundColor Green
Write-Host "🌐 Browser will open in $DelaySeconds seconds" -ForegroundColor Yellow
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Start server and browser opener in parallel
$serverJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    python "app.py" --server-only
} -ArgumentList $scriptDir

$browserJob = Start-Job -ScriptBlock {
    param($delay)
    Start-Sleep -Seconds $delay
    
    # Wait for server to be ready
    $maxAttempts = 30
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 2
            if ($response.status -eq "healthy") {
                Start-Process "http://localhost:8000"
                Write-Host "✅ Browser opened!" -ForegroundColor Green
                break
            }
        } catch {
            # Server not ready yet
        }
        Start-Sleep -Seconds 1
        $attempt++
    }
} -ArgumentList $DelaySeconds

Write-Host "📝 Server output:" -ForegroundColor Cyan
Write-Host ("=" * 50) -ForegroundColor Gray

try {
    # Wait for jobs and show output
    while ($serverJob.State -eq "Running") {
        $output = Receive-Job $serverJob
        if ($output) {
            Write-Host $output
        }
        Start-Sleep -Milliseconds 100
    }
} finally {
    # Cleanup
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Stop-Job $browserJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $browserJob -ErrorAction SilentlyContinue
    
    Write-Host ""
    Write-Host "🛑 Server stopped. Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}