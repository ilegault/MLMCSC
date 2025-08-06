# Integrated Live Microscope System Guide

## Overview

Your MLMCSC system now has **integrated live microscope functionality** built directly into the main `app.py`. You no longer need to run separate servers - everything is unified in one application.

## What's Integrated

✅ **Web Interface** - Technician labeling and model feedback  
✅ **Live Microscope** - Real-time video streaming from microscope camera  
✅ **Camera Control** - Start/stop camera, capture frames, detect cameras  
✅ **Live Predictions** - Real-time analysis of microscope feed  
✅ **Online Learning** - Continuous model improvement  
✅ **API Endpoints** - All functionality accessible via REST API  

## How to Start the Server

### Option 1: Use the main app.py (Recommended)
```bash
python app.py
```
This will give you options for how to run the server (PowerShell window, current terminal, etc.)

### Option 2: Direct server start
```bash
python app.py --server-only
```
This starts the server directly in the current terminal.


This is a simplified startup script with clear messaging.

## Server Access

Once started, access your integrated system at:

- **Web Interface**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **Live Video Feed**: http://127.0.0.1:8000/video_feed
- **Health Check**: http://127.0.0.1:8000/health

## Live Microscope Endpoints

The following camera endpoints are now integrated:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video_feed` | GET | Live video stream from microscope |
| `/camera/start` | POST | Start the microscope camera |
| `/camera/stop` | POST | Stop the microscope camera |
| `/camera/status` | GET | Get current camera status |
| `/camera/capture` | POST | Capture a single frame |
| `/camera/predict_live` | POST | Capture frame and run prediction |
| `/camera/detect` | GET | Detect available cameras |

## Testing the Integration

Run the integration test to verify everything works:

```bash
python test_integration.py
```

This will test all endpoints and confirm the integration is working properly.

## Demo Script

You can still use the demo script to test the live microscope functionality:



Make sure the integrated server is running first.

## What Changed

### Before (Separate Servers)
- `app.py` - Main web interface
- `start_live_microscope_server.py` - Separate microscope server
- Had to run two different servers
- Different ports and configurations

### After (Integrated)
- `app.py` - **Everything in one server**
- All microscope functionality built-in
- Single server, single configuration
- Unified startup and management

## Configuration

The integrated server uses your existing `config/app_config.yaml` configuration. No additional camera configuration needed - the camera endpoints are automatically available.

## Troubleshooting

### Server Won't Start
- Check if port 8000 is already in use
- Kill any existing Python processes: `Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force`

### Camera Not Working
- Use `/camera/detect` endpoint to find available cameras
- Try different camera IDs (0, 1, 2, etc.) with `/camera/start`
- Check camera connections and drivers

### Can't Access Web Interface
- Verify server is running: check for "Uvicorn running on..." message
- Try accessing http://127.0.0.1:8000 instead of localhost:8000
- Check firewall settings

## Benefits of Integration

1. **Simplified Deployment** - One server to manage
2. **Unified Configuration** - Single config file
3. **Better Resource Management** - Shared models and database connections
4. **Easier Development** - All functionality in one place
5. **Consistent API** - All endpoints follow the same patterns

## Next Steps

1. **Start the integrated server**: `python app.py`
2. **Test the integration**: `python test_integration.py`
3. **Try the demo**: `python demo_live_microscope.py`
4. **Access the web interface**: http://127.0.0.1:8000
5. **Explore the API**: http://127.0.0.1:8000/docs

Your live microscope system is now fully integrated and ready to use! 🎉