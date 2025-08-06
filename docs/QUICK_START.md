# MLMCSC Quick Start Guide

## Running the Application

### Option 1: PowerShell (Recommended)
Simply run the main app file:
```bash
python app.py
```

This will:
- ✅ Launch the server in a new PowerShell window
- ✅ Wait for the server to be fully ready
- ✅ Automatically open your browser when ready
- ✅ Provide interactive controls in the PowerShell window

### Option 2: Batch File (Windows)
Double-click `start_mlmcsc_server.bat` or run:
```cmd
start_mlmcsc_server.bat
```

### Option 3: Direct Server Mode
For advanced users or debugging:
```bash
python app.py --server-only
```

## Features Available

- 🔬 **Image Upload & Analysis**: Upload Charpy specimen images for analysis
- 👨‍🔬 **Technician Labeling**: Human-in-the-loop labeling interface
- 📊 **Performance Metrics**: Real-time model performance tracking
- 📝 **Labeling History**: Complete history of all labels and predictions
- 🧠 **Online Learning**: Continuous model improvement from new labels

## Web Interface

Once the server starts, access the web interface at:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## PowerShell Controls

When running in PowerShell mode, you can use these commands:
- `status` - Check server and model status
- `browser` - Open web interface
- `help` - Show available commands
- `Ctrl+C` - Stop the server

## Troubleshooting

### Server Already Running
If you see "Server is already running!", you can:
1. Open browser to existing server
2. Stop and restart server
3. Exit

### Import Errors
Make sure you're in the project root directory and have installed dependencies:
```bash
pip install -r requirements.txt
```

### Browser Doesn't Open
The system waits for the server to be fully ready before opening the browser. If it doesn't open automatically:
1. Check the PowerShell window for any errors
2. Manually navigate to http://localhost:8000
3. Use the `browser` command in PowerShell

## What's Changed

- ❌ **Removed**: Live camera functionality (was causing glitches)
- ✅ **Improved**: Server startup timing and browser opening
- ✅ **Enhanced**: PowerShell management interface
- ✅ **Fixed**: Browser opening before server is ready

The application now focuses on the core image analysis and labeling functionality without the problematic camera server components.