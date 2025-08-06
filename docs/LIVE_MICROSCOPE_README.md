# Live Microscope Feature

This document describes the implementation of the Live Microscope feature for the MLMCSC Human-in-the-Loop Interface.

## Overview

The Live Microscope feature allows technicians to:
- View live video feed from a connected microscope camera
- Capture individual frames for analysis
- Run real-time predictions on live video
- Enable automatic prediction at configurable intervals
- Switch between different camera sources

## Implementation Details

### Backend API Endpoints

#### Camera Control
- `GET /camera/status` - Get current camera status
- `POST /camera/start` - Start the microscope camera
- `POST /camera/stop` - Stop the microscope camera

#### Video Streaming
- `GET /video_feed` - Live video stream endpoint (multipart/x-mixed-replace)

#### Frame Operations
- `POST /camera/capture` - Capture a single frame
- `POST /camera/predict_live` - Capture frame and run prediction

### Frontend Features

#### Live Video Display
- Real-time video streaming with timestamp overlay
- Camera status indicator
- Responsive video container with controls

#### Camera Controls
- Start/Stop camera buttons
- Camera selection dropdown (Camera 0, 1, 2)
- Frame capture button
- Live prediction button

#### Auto Prediction
- Toggle switch to enable/disable automatic predictions
- Configurable prediction interval (1-10 seconds)
- Real-time prediction results display

#### Live Prediction Display
- Current shear percentage prediction
- Confidence level with color coding
- Specimen ID
- Last prediction timestamp

## Usage Instructions

### Starting the Camera

1. **Select Camera**: Choose the appropriate camera ID from the dropdown (usually Camera 1 for microscope)
2. **Start Camera**: Click the "Start Camera" button
3. **Verify Feed**: Confirm that the live video feed appears

### Capturing Frames

1. **Ensure Camera is Active**: The camera must be running
2. **Capture Frame**: Click "Capture Frame" button
3. **View Result**: A modal will display the captured frame
4. **Use for Analysis**: Optionally use the captured frame for detailed analysis

### Live Predictions

#### Manual Prediction
1. **Click "Predict Live"**: Runs prediction on current frame
2. **View Results**: Results appear in the Live Prediction panel

#### Automatic Prediction
1. **Enable Auto Prediction**: Toggle the switch in Camera Settings
2. **Set Interval**: Adjust prediction interval (1-10 seconds)
3. **Monitor Results**: Predictions update automatically

### Camera Settings

- **Camera ID**: Select which camera to use (0, 1, or 2)
- **Auto Prediction**: Enable/disable automatic predictions
- **Prediction Interval**: Set how often predictions run (1-10 seconds)

## Technical Implementation

### Backend Components

#### Camera Management
```python
# Global camera instance
camera: Optional[cv2.VideoCapture] = None
camera_active: bool = False

def initialize_camera(camera_id: int = 1) -> bool:
    """Initialize the microscope camera."""
    # Implementation details...

def release_camera():
    """Release the camera resource."""
    # Implementation details...
```

#### Video Streaming
```python
async def generate_frames():
    """Generate frames from the microscope camera."""
    # Yields frames in multipart format for streaming

@app.get("/video_feed")
async def video_feed():
    """Video streaming route for live microscope feed."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

### Frontend Components

#### Video Display
```html
<div class="video-container">
    <img id="videoFeed" class="video-feed" src="/video_feed">
    <div class="video-overlay">LIVE</div>
</div>
```

#### JavaScript Integration
```javascript
class MLMCSCInterface {
    // Camera control methods
    async startCamera() { /* ... */ }
    async stopCamera() { /* ... */ }
    async captureFrame() { /* ... */ }
    async predictLiveFrame() { /* ... */ }
    
    // Auto prediction methods
    toggleAutoPrediction(enabled) { /* ... */ }
    startAutoPrediction() { /* ... */ }
    stopAutoPrediction() { /* ... */ }
}
```

## Configuration

### Camera Settings
- **Default Camera ID**: 1 (configurable)
- **Video Resolution**: 1280x720 (configurable)
- **Frame Rate**: 30 FPS (configurable)
- **JPEG Quality**: 85% (configurable)

### Auto Prediction
- **Default Interval**: 3 seconds
- **Min Interval**: 1 second
- **Max Interval**: 10 seconds

## Error Handling

### Camera Errors
- Camera not found or unavailable
- Camera initialization failure
- Frame capture errors
- Video streaming interruptions

### Network Errors
- API endpoint failures
- Streaming connection issues
- Prediction request timeouts

### User Feedback
- Status indicators for camera state
- Error messages for failed operations
- Loading indicators for long operations

## Testing

### Test Script
Run the test script to verify functionality:
```bash
python test_live_microscope.py
```

### Manual Testing
1. Start the web server
2. Navigate to the web interface
3. Test each camera control button
4. Verify video streaming works
5. Test frame capture and predictions
6. Test auto prediction feature

## Troubleshooting

### Common Issues

#### Camera Not Starting
- **Check Camera Connection**: Ensure microscope camera is connected
- **Verify Camera ID**: Try different camera IDs (0, 1, 2)
- **Check Permissions**: Ensure application has camera access
- **Restart Application**: Sometimes helps with camera initialization

#### Video Feed Not Displaying
- **Check Network Connection**: Ensure stable connection to server
- **Browser Compatibility**: Test with different browsers
- **Clear Browser Cache**: May resolve display issues

#### Predictions Not Working
- **Verify Models Loaded**: Check health endpoint for model status
- **Check Camera Feed**: Ensure camera is capturing valid frames
- **Review Server Logs**: Look for error messages in console

### Debug Information

#### Health Endpoint
Check `/health` endpoint for system status:
```json
{
    "status": "healthy",
    "models_loaded": {
        "detector": true,
        "classifier": true,
        "regression": true
    },
    "camera_status": {
        "active": true,
        "initialized": true
    }
}
```

#### Camera Status Endpoint
Check `/camera/status` for camera-specific information:
```json
{
    "camera_active": true,
    "camera_initialized": true,
    "timestamp": "2024-01-15T10:30:00"
}
```

## Future Enhancements

### Planned Features
- Multiple camera support with switching
- Video recording capability
- Advanced camera controls (zoom, focus, exposure)
- Real-time image enhancement filters
- Batch prediction on recorded video
- Integration with automated specimen handling

### Performance Optimizations
- Frame rate optimization based on network conditions
- Adaptive video quality
- Efficient memory management for long sessions
- Background processing for predictions

## Security Considerations

### Camera Access
- Ensure proper camera permissions
- Implement access controls for camera endpoints
- Monitor camera usage and sessions

### Network Security
- Use HTTPS for production deployments
- Implement proper authentication
- Rate limiting for prediction endpoints

## Dependencies

### Python Packages
- `opencv-python` (cv2) - Camera and video processing
- `fastapi` - Web framework
- `numpy` - Image processing
- `uvicorn` - ASGI server

### Frontend Libraries
- Bootstrap 5 - UI components
- Font Awesome - Icons
- Native JavaScript - No additional frameworks required

## Conclusion

The Live Microscope feature provides a comprehensive solution for real-time microscope integration with the MLMCSC system. It enables technicians to work more efficiently by providing immediate feedback and reducing the need for manual image capture and upload processes.

For support or questions, please refer to the main project documentation or contact the development team.