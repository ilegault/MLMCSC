# MLMCSC Human-in-the-Loop Web Interface

A comprehensive web interface for technicians to interact with the MLMCSC system, providing image analysis, labeling capabilities, and performance monitoring.

## Features

### 🖼️ Image Analysis
- **Drag & Drop Upload**: Easy image upload with drag-and-drop support
- **YOLO Detection Overlay**: Real-time specimen detection with bounding boxes
- **Automatic Processing**: Instant analysis upon image upload
- **Multiple Format Support**: JPG, PNG, BMP images up to 10MB

### 🏷️ Labeling Interface
- **Interactive Slider**: Intuitive shear percentage input (0-100%)
- **Model Predictions**: Display of AI model predictions with confidence scores
- **Confidence Indicators**: Color-coded confidence levels (High/Medium/Low)
- **Specimen Tracking**: Unique specimen ID assignment and tracking

### 📊 Performance Monitoring
- **Real-time Metrics**: Live model performance statistics
- **Accuracy Tracking**: MAE, RMSE, and R² score monitoring
- **Version Control**: Model version tracking and update history
- **Trend Analysis**: Performance trends over time

### 📋 History Management
- **Complete History**: All labeling sessions with timestamps
- **Pagination**: Efficient browsing of large datasets
- **Export Functionality**: CSV export for external analysis
- **Search & Filter**: Easy data retrieval and analysis

### 🔄 Online Learning
- **Continuous Improvement**: Automatic model updates based on technician feedback
- **Threshold-based Updates**: Configurable update triggers
- **Performance Tracking**: Monitor improvement over time
- **Version Management**: Automatic model versioning

## API Endpoints

### Core Endpoints

#### `POST /predict`
Get model prediction for uploaded image.

**Request:**
```json
{
    "image_data": "base64_encoded_image",
    "image_format": "jpg"
}
```

**Response:**
```json
{
    "specimen_id": 1,
    "shear_percentage": 75.3,
    "confidence": 0.85,
    "detection_bbox": [100, 150, 200, 180],
    "detection_confidence": 0.92,
    "processing_time": 0.45,
    "timestamp": "2024-01-15T10:30:00",
    "overlay_image": "base64_encoded_overlay"
}
```

#### `POST /submit_label`
Submit technician's label for training.

**Request:**
```json
{
    "specimen_id": 1,
    "image_data": "base64_encoded_image",
    "technician_label": 78.5,
    "model_prediction": 75.3,
    "model_confidence": 0.85,
    "technician_id": "tech001",
    "notes": "Optional notes"
}
```

#### `GET /get_metrics`
View model performance metrics.

**Response:**
```json
{
    "total_predictions": 1250,
    "total_labels": 890,
    "accuracy_metrics": {
        "mae": 3.2,
        "rmse": 4.8,
        "sample_count": 890
    },
    "recent_performance": {
        "daily_mae": [
            {"date": "2024-01-15", "mae": 3.1}
        ]
    },
    "model_version": "1.2.3",
    "last_updated": "2024-01-15T10:30:00"
}
```

#### `GET /get_history`
View labeling history with pagination.

**Parameters:**
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 20)

**Response:**
```json
{
    "items": [
        {
            "id": 1,
            "timestamp": "2024-01-15T10:30:00",
            "specimen_id": 1,
            "technician_label": 78.5,
            "model_prediction": 75.3,
            "difference": 3.2,
            "technician_id": "tech001",
            "notes": "Optional notes"
        }
    ],
    "total_count": 890,
    "page": 1,
    "page_size": 20
}
```

### Utility Endpoints

#### `GET /export_history`
Export complete labeling history as CSV file.

#### `GET /health`
Health check endpoint for monitoring.

#### `GET /`
Serve the main web interface.

## Installation & Setup

### Prerequisites
- Python 3.8+
- All MLMCSC dependencies (see main requirements.txt)
- Web browser with JavaScript enabled

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**
   ```bash
   cd src/web
   python run_server.py
   ```

3. **Access Interface**
   Open http://localhost:8000 in your web browser

### Configuration

Create a `config.json` file in the web directory:

```json
{
    "host": "0.0.0.0",
    "port": 8000,
    "debug": false,
    "detection_model_path": "path/to/detection/model.pt",
    "classification_model_path": "path/to/classification/model.pkl",
    "online_learning_enabled": true,
    "update_threshold": 10,
    "update_interval": 300
}
```

### Environment Variables

- `MLMCSC_WEB_HOST`: Server host (default: 0.0.0.0)
- `MLMCSC_WEB_PORT`: Server port (default: 8000)
- `MLMCSC_WEB_DEBUG`: Debug mode (default: true)
- `MLMCSC_DATABASE_PATH`: Database file path
- `MLMCSC_DETECTION_MODEL`: Detection model path
- `MLMCSC_CLASSIFICATION_MODEL`: Classification model path
- `MLMCSC_ONLINE_LEARNING`: Enable online learning (default: true)

## Usage Guide

### For Technicians

1. **Upload Image**
   - Drag and drop an image or click to select
   - Wait for automatic analysis to complete

2. **Review Prediction**
   - Check the model's prediction and confidence
   - Examine the detection overlay on the image

3. **Provide Label**
   - Adjust the slider to your assessment
   - Consider the model's prediction as reference

4. **Submit**
   - Click "Submit Label" to save your assessment
   - The system will use this for model improvement

### For Administrators

1. **Monitor Performance**
   - Check the metrics dashboard regularly
   - Watch for accuracy trends and model updates

2. **Export Data**
   - Use the export function for external analysis
   - Regular backups of labeling history

3. **System Health**
   - Monitor the health endpoint
   - Check logs for any issues

## Architecture

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **Bootstrap 5**: UI framework for consistent styling
- **Vanilla JavaScript**: No external dependencies
- **Font Awesome**: Icons and visual elements

### Backend
- **FastAPI**: High-performance async web framework
- **SQLite**: Lightweight database for data storage
- **OpenCV**: Image processing and analysis
- **scikit-learn**: Machine learning utilities

### Data Flow
1. Image upload → Base64 encoding → API
2. YOLO detection → Feature extraction → Prediction
3. Overlay generation → Display to user
4. User labeling → Database storage → Online learning

## Database Schema

### Predictions Table
- `id`: Primary key
- `timestamp`: Prediction timestamp
- `specimen_id`: Unique specimen identifier
- `shear_percentage`: Model prediction
- `confidence`: Prediction confidence
- `detection_bbox`: Bounding box coordinates
- `detection_confidence`: Detection confidence
- `processing_time`: Analysis duration
- `image_data`: Base64 encoded image

### Labels Table
- `id`: Primary key
- `timestamp`: Label timestamp
- `specimen_id`: Specimen identifier
- `technician_label`: Human assessment
- `model_prediction`: Model's prediction
- `model_confidence`: Model confidence
- `technician_id`: Technician identifier
- `notes`: Optional notes
- `image_data`: Base64 encoded image

### Model Metrics Table
- `id`: Primary key
- `timestamp`: Metric timestamp
- `metric_name`: Metric identifier
- `metric_value`: Metric value
- `model_version`: Model version

## Security Considerations

### Data Protection
- Images stored as base64 in database
- No external image storage required
- Local database file protection

### Access Control
- Technician ID tracking
- Optional API key authentication
- CORS configuration for production

### Privacy
- No external data transmission
- Local processing only
- Configurable data retention

## Performance Optimization

### Frontend
- Lazy loading of history data
- Efficient image encoding/decoding
- Responsive design for all devices

### Backend
- Async request handling
- Database connection pooling
- Efficient image processing

### Database
- Indexed queries for performance
- Automatic cleanup of old data
- Optimized storage format

## Troubleshooting

### Common Issues

1. **Model Not Loading**
   - Check model file paths in configuration
   - Verify model file permissions
   - Check logs for specific errors

2. **Image Upload Fails**
   - Verify image format and size limits
   - Check browser JavaScript console
   - Ensure sufficient disk space

3. **Predictions Not Working**
   - Verify all models are loaded correctly
   - Check detection model compatibility
   - Review processing logs

4. **Database Errors**
   - Check database file permissions
   - Verify disk space availability
   - Review database initialization logs

### Log Files
- Web interface logs: `data/web_interface.log`
- Database location: `data/labeling_history.db`
- Model storage: `data/online_models/`

## Development

### Adding New Features

1. **Backend Endpoints**
   - Add new routes in `api.py`
   - Update database schema if needed
   - Add appropriate error handling

2. **Frontend Components**
   - Extend JavaScript classes
   - Update HTML templates
   - Add CSS styling

3. **Database Changes**
   - Update schema in `database.py`
   - Add migration scripts if needed
   - Update API models

### Testing

```bash
# Run the development server
python run_server.py --debug

# Test API endpoints
curl -X GET http://localhost:8000/health

# Check database
sqlite3 data/labeling_history.db ".tables"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the MLMCSC system. See the main project license for details.

## Support

For technical support or questions:
- Check the troubleshooting section
- Review log files for errors
- Contact the development team

---

**Version**: 1.0.0  
**Last Updated**: January 2024  
**Compatibility**: MLMCSC v2.0.0+