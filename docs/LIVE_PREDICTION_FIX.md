# Live Prediction Error Fix

## Problem
When using the live microscope function, the system was throwing a 500 Internal Server Error when no specimens were detected in the camera feed:

```
ERROR:web.api:Error in live prediction: 400: No specimens detected in image
INFO:     127.0.0.1:56992 - "POST /camera/predict_live HTTP/1.1" 500 Internal Server Error
```

## Root Cause
1. The `SpecimenDetector` was using a high confidence threshold (0.8) with a pre-trained YOLOv8 model
2. The pre-trained model wasn't specifically trained on microscope specimens
3. When no specimens were detected, the system raised an HTTPException with status 400
4. This exception was being caught and re-raised as a 500 error in the live prediction endpoint

## Solution Implemented

### 1. Modified `predict_image` function
- Added an optional `allow_no_specimens` parameter
- When `allow_no_specimens=True`, returns a structured response instead of raising an exception
- Removed the strict response model to allow flexible return types

### 2. Enhanced `predict_live_frame` function
- Temporarily lowers the confidence threshold to 0.3 for live predictions (more sensitive detection)
- Calls `predict_image` with `allow_no_specimens=True`
- Handles the "no specimens" case gracefully by returning a status response
- Added better logging for debugging

### 3. Added diagnostic endpoint
- New `/camera/detector_info` endpoint to check detector configuration
- Helps with debugging detection issues

### 4. Improved error handling
- Live predictions no longer throw 500 errors when no specimens are detected
- Returns structured JSON responses with appropriate status messages
- Better separation of HTTP exceptions vs general exceptions

## Changes Made

### File: `src/web/api.py`

1. **Modified `predict_image` function** (lines ~436-461):
   ```python
   @app.post("/predict")
   async def predict_image(request: PredictionRequest, allow_no_specimens: bool = False):
       # ... existing code ...
       if not detections:
           if allow_no_specimens:
               return {
                   "status": "no_specimens",
                   "message": "No specimens detected in image",
                   "processing_time": (datetime.now() - start_time).total_seconds(),
                   "timestamp": datetime.now().isoformat()
               }
           else:
               raise HTTPException(status_code=400, detail="No specimens detected in image")
   ```

2. **Enhanced `predict_live_frame` function** (lines ~1291-1350):
   ```python
   @app.post("/camera/predict_live")
   async def predict_live_frame():
       # Temporarily lower confidence threshold for live predictions
       original_confidence = None
       if detector:
           original_confidence = detector.confidence_threshold
           detector.confidence_threshold = 0.3  # Lower threshold for live detection
       
       try:
           # ... prediction logic ...
           result = await predict_image(prediction_request, allow_no_specimens=True)
       finally:
           # Restore original confidence threshold
           if detector and original_confidence is not None:
               detector.confidence_threshold = original_confidence
       
       # Handle no specimens case gracefully
       if isinstance(result, dict) and result.get("status") == "no_specimens":
           return {
               "status": "no_specimens",
               "message": "No specimens detected in current frame",
               "capture_timestamp": datetime.now().isoformat(),
               "processing_time": result.get("processing_time", 0)
           }
   ```

3. **Added diagnostic endpoint** (lines ~1352-1370):
   ```python
   @app.get("/camera/detector_info")
   async def get_detector_info():
       # Returns detector configuration information
   ```

## Testing

A test script `test_live_prediction_fix.py` has been created to verify the fix works correctly.

Run the test with:
```bash
python test_live_prediction_fix.py
```

## Expected Behavior After Fix

1. **No more 500 errors** when no specimens are detected
2. **Graceful handling** with structured JSON responses:
   ```json
   {
     "status": "no_specimens",
     "message": "No specimens detected in current frame",
     "capture_timestamp": "2024-01-XX...",
     "processing_time": 0.123
   }
   ```
3. **More sensitive detection** during live predictions (confidence threshold lowered to 0.3)
4. **Better logging** for debugging detection issues
5. **Diagnostic information** available via `/camera/detector_info` endpoint

## Benefits

- **Improved user experience**: No more error messages when camera doesn't see specimens
- **Better debugging**: Clear status messages and diagnostic endpoints
- **More robust detection**: Lower confidence threshold for live predictions
- **Backward compatibility**: Regular prediction endpoint behavior unchanged
- **Better error separation**: Distinguishes between actual errors and "no detection" cases