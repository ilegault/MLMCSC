#!/usr/bin/env python3
"""
FastAPI Backend for Human-in-the-Loop Interface

This module provides REST API endpoints for the technician web interface,
including image prediction, label submission, metrics viewing, and history tracking.
"""

import os
import io
import base64
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import MLMCSC modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mlmcsc.detection.object_detector import SpecimenDetector, DetectionResult
from src.mlmcsc.classification.shiny_region_classifier import ShinyRegionBasedClassifier
from src.mlmcsc.regression.regression_model import FractureRegressionModel
from src.mlmcsc.regression.online_learning import OnlineLearningSystem
from src.mlmcsc.feature_extraction import FractureFeatureExtractor
from src.web.config import get_config
from src.web.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting MLMCSC Human-in-the-Loop API...")
    init_database()
    load_models()
    logger.info("API startup complete")
    yield
    # Shutdown
    logger.info("Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="MLMCSC Human-in-the-Loop Interface",
    description="Web interface for technician labeling and model feedback",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class PredictionRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    image_format: str = Field(default="jpg", description="Image format (jpg, png)")

class PredictionResponse(BaseModel):
    specimen_id: int
    shear_percentage: float
    confidence: float
    detection_bbox: List[float]
    detection_confidence: float
    processing_time: float
    timestamp: str
    overlay_image: str  # Base64 encoded image with overlay

class LabelSubmission(BaseModel):
    specimen_id: int
    image_data: str
    technician_label: float = Field(..., ge=0, le=100, description="Shear percentage (0-100)")
    model_prediction: float
    model_confidence: float
    technician_id: str
    notes: Optional[str] = None

class MetricsResponse(BaseModel):
    total_predictions: int
    total_labels: int
    accuracy_metrics: Dict[str, float]
    recent_performance: Dict[str, Any]
    model_version: str
    last_updated: str

class HistoryItem(BaseModel):
    id: int
    timestamp: str
    specimen_id: int
    technician_label: float
    model_prediction: float
    difference: float
    technician_id: str
    notes: Optional[str]

class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total_count: int
    page: int
    page_size: int

# Global model instances
detector: Optional[SpecimenDetector] = None
classifier: Optional[ShinyRegionBasedClassifier] = None
regression_model: Optional[FractureRegressionModel] = None
online_learner: Optional[OnlineLearningSystem] = None
feature_extractor: Optional[FractureFeatureExtractor] = None
db_manager: Optional[DatabaseManager] = None

# Database setup
DB_PATH = Path(__file__).parent / "data" / "labeling_history.db"

def init_database():
    """Initialize SQLite database for storing labeling history."""
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            specimen_id INTEGER NOT NULL,
            shear_percentage REAL NOT NULL,
            confidence REAL NOT NULL,
            detection_bbox TEXT NOT NULL,
            detection_confidence REAL NOT NULL,
            processing_time REAL NOT NULL,
            image_data TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            specimen_id INTEGER NOT NULL,
            technician_label REAL NOT NULL,
            model_prediction REAL NOT NULL,
            model_confidence REAL NOT NULL,
            technician_id TEXT NOT NULL,
            notes TEXT,
            image_data TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            model_version TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

def load_models():
    """Load ML models."""
    global detector, classifier, regression_model, online_learner, feature_extractor, db_manager
    
    try:
        config = get_config()
        
        # Initialize database manager
        db_manager = DatabaseManager()
        logger.info("Database manager initialized")
        
        # Initialize feature extractor
        feature_extractor = FractureFeatureExtractor()
        logger.info("Feature extractor initialized")
        
        # Load detection model
        if config.detection_model_path and Path(config.detection_model_path).exists():
            detector = SpecimenDetector(model_path=str(config.detection_model_path))
            logger.info(f"Detection model loaded from {config.detection_model_path}")
        else:
            detector = SpecimenDetector()  # Initialize with default
            logger.warning("Detection model path not found, using default")
        
        # Load classification model
        classifier = ShinyRegionBasedClassifier()
        if config.classification_model_path and Path(config.classification_model_path).exists():
            classifier.load_model(str(config.classification_model_path))
            logger.info(f"Classification model loaded from {config.classification_model_path}")
        else:
            logger.warning("Classification model path not found, using default")
        
        # Load regression model
        regression_model = FractureRegressionModel()
        if config.regression_model_path and Path(config.regression_model_path).exists():
            regression_model.load_model(str(config.regression_model_path))
            logger.info(f"Regression model loaded from {config.regression_model_path}")
        else:
            logger.warning("Regression model path not found, using default")
        
        # Initialize online learning system
        online_learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='batch',  # Default to batch updates
            batch_size=10,
            confidence_threshold=0.7
        )
        
        # Try to load existing online model
        online_model_path = Path(__file__).parent / "models" / "online_model.joblib"
        if online_model_path.exists():
            try:
                online_learner.load_model(online_model_path)
                logger.info("Online learning model loaded from saved state")
            except Exception as e:
                logger.warning(f"Could not load online model: {e}")
                # Initialize with some dummy data if available
                _initialize_online_model()
        else:
            logger.info("No existing online model found, will initialize on first use")
            _initialize_online_model()
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Initialize with default models for development
        detector = SpecimenDetector()
        classifier = ShinyRegionBasedClassifier()
        regression_model = FractureRegressionModel()
        logger.info("Initialized with default models")

def _initialize_online_model():
    """Initialize online learning model with existing labels if available."""
    global online_learner, db_manager
    
    if not online_learner or not db_manager:
        return
    
    try:
        # Get existing labels from database
        labels = db_manager.get_labels(limit=1000)  # Get up to 1000 labels
        
        if len(labels) >= 10:  # Need minimum samples for initialization
            logger.info(f"Initializing online model with {len(labels)} existing labels")
            
            # Extract features from stored labels
            feature_data = []
            target_values = []
            
            for label in labels[:100]:  # Use first 100 for initialization
                try:
                    # Decode image
                    image = decode_image(label.image_data)
                    
                    # Extract features (assuming we have bbox info stored)
                    # For now, use a default bbox - in production, store bbox with labels
                    h, w = image.shape[:2]
                    bbox = [w//4, h//4, w//2, h//2]  # Center region
                    
                    result = feature_extractor.extract_features(
                        image=image,
                        specimen_id=label.specimen_id,
                        bbox=bbox
                    )
                    
                    feature_data.append({
                        'feature_vector': result.feature_vector,
                        'feature_names': result.features.keys()
                    })
                    target_values.append(label.technician_label)
                    
                except Exception as e:
                    logger.warning(f"Failed to process label {label.id}: {e}")
                    continue
            
            if len(feature_data) >= 10:
                # Initialize the online model
                performance = online_learner.initialize_model(feature_data, target_values)
                logger.info(f"Online model initialized with R²: {performance.get('r2', 0):.3f}")
                
                # Add remaining labels to holdout set
                for label in labels[100:]:
                    try:
                        image = decode_image(label.image_data)
                        h, w = image.shape[:2]
                        bbox = [w//4, h//4, w//2, h//2]
                        
                        result = feature_extractor.extract_features(
                            image=image,
                            specimen_id=label.specimen_id,
                            bbox=bbox
                        )
                        
                        online_learner.add_to_holdout(
                            {'feature_vector': result.feature_vector},
                            label.technician_label
                        )
                    except Exception as e:
                        continue
                
                logger.info(f"Added {len(labels[100:])} samples to holdout set")
            else:
                logger.warning("Not enough valid features extracted for initialization")
        else:
            logger.info("Not enough existing labels for online model initialization")
            
    except Exception as e:
        logger.error(f"Error initializing online model: {e}")

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def encode_image(image: np.ndarray, format: str = "jpg") -> str:
    """Encode numpy array to base64 image data."""
    try:
        # Encode image
        if format.lower() == "png":
            _, buffer = cv2.imencode('.png', image)
        else:
            _, buffer = cv2.imencode('.jpg', image)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{format};base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise HTTPException(status_code=500, detail="Error encoding image")

def draw_detection_overlay(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Draw YOLO detection overlay on image."""
    overlay = image.copy()
    
    for detection in detections:
        # Extract bbox coordinates
        x, y, w, h = detection.bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Draw bounding box
        color = (0, 255, 0) if detection.is_stable else (0, 255, 255)  # Green if stable, yellow if not
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Draw specimen ID and confidence
        label = f"ID: {detection.specimen_id} ({detection.confidence:.2f})"
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center point
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        cv2.circle(overlay, (center_x, center_y), 5, color, -1)
        
        # Draw stability indicator
        if detection.is_stable:
            cv2.putText(overlay, "STABLE", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return overlay

# Startup event moved to lifespan context manager above

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        # Fallback to basic HTML if template not found
        return HTMLResponse(content="""
        <html>
        <head><title>MLMCSC Interface</title></head>
        <body>
            <h1>MLMCSC Human-in-the-Loop Interface</h1>
            <p>Template file not found. Please check the installation.</p>
        </body>
        </html>
        """)

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(request: PredictionRequest):
    """Get model prediction for uploaded image."""
    try:
        start_time = datetime.now()
        
        # Decode image
        image = decode_image(request.image_data)
        
        # Run detection
        if detector is None:
            raise HTTPException(status_code=500, detail="Detection model not loaded")
        
        detections = detector.detect_specimen(image)
        
        if not detections:
            raise HTTPException(status_code=400, detail="No specimens detected in image")
        
        # Use the first detection (most confident)
        detection = detections[0]
        
        # Extract ROI for classification
        x, y, w, h = detection.bbox
        roi = image[int(y):int(y+h), int(x):int(x+w)]
        
        # Predict shear percentage
        shear_prediction = 50.0  # Default prediction
        confidence = 0.5
        
        if classifier and classifier.is_trained:
            try:
                features = classifier.process_image(roi)
                if features:
                    prediction_result = classifier.predict(features.to_array().reshape(1, -1))
                    shear_prediction = float(prediction_result[0])
                    confidence = 0.8  # Placeholder confidence
            except Exception as e:
                logger.warning(f"Classification failed, using default: {e}")
        
        # Create overlay image
        overlay_image = draw_detection_overlay(image, detections)
        overlay_base64 = encode_image(overlay_image)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store prediction in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, specimen_id, shear_percentage, confidence, detection_bbox, 
             detection_confidence, processing_time, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            detection.specimen_id,
            shear_prediction,
            confidence,
            json.dumps(detection.bbox),
            detection.confidence,
            processing_time,
            request.image_data
        ))
        conn.commit()
        conn.close()
        
        return PredictionResponse(
            specimen_id=detection.specimen_id,
            shear_percentage=shear_prediction,
            confidence=confidence,
            detection_bbox=detection.bbox,
            detection_confidence=detection.confidence,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            overlay_image=overlay_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit_label")
async def submit_label(submission: LabelSubmission):
    """Submit technician's label for training."""
    try:
        # Store label in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO labels 
            (timestamp, specimen_id, technician_label, model_prediction, 
             model_confidence, technician_id, notes, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            submission.specimen_id,
            submission.technician_label,
            submission.model_prediction,
            submission.model_confidence,
            submission.technician_id,
            submission.notes,
            submission.image_data
        ))
        conn.commit()
        conn.close()
        
        # CORE ONLINE LEARNING PIPELINE
        online_result = await _process_online_learning(submission)
        
        return {
            "status": "success", 
            "message": "Label submitted successfully",
            "online_learning": online_result
        }
        
    except Exception as e:
        logger.error(f"Error submitting label: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_online_learning(submission: LabelSubmission) -> Dict[str, Any]:
    """
    Core online learning pipeline implementation.
    
    When technician submits label:
    1. Extract features from image
    2. Store (features, label, timestamp, technician_id) - already done above
    3. Update model with new data point
    4. Validate on recent holdout set
    5. Log performance metrics
    """
    global online_learner, feature_extractor
    
    try:
        if not online_learner or not feature_extractor:
            return {"status": "error", "message": "Online learning system not initialized"}
        
        # 1. Extract features from image
        image = decode_image(submission.image_data)
        
        # For feature extraction, we need a bounding box
        # In a real system, this would come from the detection step
        # For now, use center region as default
        h, w = image.shape[:2]
        bbox = [w//4, h//4, w//2, h//2]  # Center region
        
        feature_result = feature_extractor.extract_features(
            image=image,
            specimen_id=submission.specimen_id,
            bbox=bbox
        )
        
        # 2. Prepare feature data for online learning
        feature_data = {
            'feature_vector': feature_result.feature_vector,
            'feature_names': list(feature_result.features.keys()),
            'specimen_id': submission.specimen_id
        }
        
        # 3. Process through online learning system
        if not online_learner.is_initialized:
            # Initialize with this first sample (need more samples in practice)
            logger.info("Initializing online learner with first submission")
            try:
                performance = online_learner.initialize_model(
                    feature_data=[feature_data],
                    target_values=[submission.technician_label]
                )
                return {
                    "status": "initialized",
                    "message": "Online learning system initialized",
                    "initial_performance": performance
                }
            except Exception as e:
                logger.error(f"Failed to initialize online learner: {e}")
                return {"status": "error", "message": f"Initialization failed: {e}"}
        
        # 4. Process technician submission through online learning pipeline
        result = online_learner.process_technician_submission(
            feature_data=feature_data,
            label=submission.technician_label,
            timestamp=datetime.now().isoformat(),
            technician_id=submission.technician_id,
            confidence=submission.model_confidence
        )
        
        # 5. Save updated model if an update was applied
        if result.get('update_applied', False):
            try:
                model_save_path = Path(__file__).parent / "models"
                model_save_path.mkdir(exist_ok=True)
                online_learner.save_model(model_save_path / "online_model.joblib")
                logger.info("Online model saved after update")
            except Exception as e:
                logger.warning(f"Failed to save online model: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in online learning pipeline: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/get_metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total counts
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM labels")
        total_labels = cursor.fetchone()[0]
        
        # Calculate accuracy metrics
        cursor.execute("""
            SELECT technician_label, model_prediction 
            FROM labels 
            WHERE timestamp > datetime('now', '-30 days')
        """)
        recent_data = cursor.fetchall()
        
        accuracy_metrics = {}
        if recent_data:
            technician_labels = [row[0] for row in recent_data]
            model_predictions = [row[1] for row in recent_data]
            
            # Calculate MAE and RMSE
            differences = [abs(t - m) for t, m in zip(technician_labels, model_predictions)]
            mae = sum(differences) / len(differences)
            rmse = (sum(d**2 for d in differences) / len(differences)) ** 0.5
            
            accuracy_metrics = {
                "mae": mae,
                "rmse": rmse,
                "sample_count": len(recent_data)
            }
        
        # Get recent performance trends
        cursor.execute("""
            SELECT DATE(timestamp) as date, 
                   AVG(ABS(technician_label - model_prediction)) as daily_mae
            FROM labels 
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """)
        daily_performance = cursor.fetchall()
        
        recent_performance = {
            "daily_mae": [{"date": row[0], "mae": row[1]} for row in daily_performance]
        }
        
        conn.close()
        
        return MetricsResponse(
            total_predictions=total_predictions,
            total_labels=total_labels,
            accuracy_metrics=accuracy_metrics,
            recent_performance=recent_performance,
            model_version="1.0.0",
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_history", response_model=HistoryResponse)
async def get_history(page: int = 1, page_size: int = 20):
    """Get labeling history with pagination."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM labels")
        total_count = cursor.fetchone()[0]
        
        # Get paginated results
        offset = (page - 1) * page_size
        cursor.execute("""
            SELECT id, timestamp, specimen_id, technician_label, 
                   model_prediction, technician_id, notes
            FROM labels 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """, (page_size, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            difference = abs(row[3] - row[4])  # |technician_label - model_prediction|
            items.append(HistoryItem(
                id=row[0],
                timestamp=row[1],
                specimen_id=row[2],
                technician_label=row[3],
                model_prediction=row[4],
                difference=difference,
                technician_id=row[5],
                notes=row[6]
            ))
        
        return HistoryResponse(
            items=items,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export_history")
async def export_history():
    """Export labeling history as CSV."""
    try:
        import io
        import csv
        from fastapi.responses import StreamingResponse
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, specimen_id, technician_label, model_prediction, 
                   model_confidence, technician_id, notes
            FROM labels 
            ORDER BY timestamp DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Timestamp', 'Specimen ID', 'Technician Label (%)', 
            'Model Prediction (%)', 'Model Confidence (%)', 
            'Technician ID', 'Notes', 'Difference (%)'
        ])
        
        # Write data
        for row in rows:
            difference = abs(row[2] - row[3])  # |technician_label - model_prediction|
            writer.writerow([
                row[0], row[1], row[2], row[3], 
                row[4] * 100, row[5], row[6] or '', difference
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=labeling_history.csv"}
        )
        
    except Exception as e:
        logger.error(f"Error exporting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "detector": detector is not None,
            "classifier": classifier is not None and getattr(classifier, 'is_trained', False),
            "regression": regression_model is not None
        }
    }

# Online Learning Configuration Endpoints

class OnlineLearningConfig(BaseModel):
    update_strategy: str = Field(..., description="Update strategy: immediate, batch, weighted, confidence")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for batch updates")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")
    weight_decay: float = Field(default=0.95, ge=0.1, le=1.0, description="Weight decay for weighted updates")

@app.get("/online_learning/status")
async def get_online_learning_status():
    """Get online learning system status and configuration."""
    global online_learner
    
    if not online_learner:
        return {"status": "not_initialized", "message": "Online learning system not available"}
    
    try:
        model_info = online_learner.get_model_info()
        update_stats = online_learner.get_update_statistics()
        
        return {
            "status": "active" if online_learner.is_initialized else "not_initialized",
            "configuration": {
                "update_strategy": online_learner.update_strategy,
                "batch_size": online_learner.batch_size,
                "confidence_threshold": online_learner.confidence_threshold,
                "weight_decay": online_learner.weight_decay,
                "model_type": online_learner.model_type
            },
            "model_info": model_info,
            "update_statistics": update_stats,
            "pending_samples": len(online_learner.pending_samples),
            "holdout_samples": len(online_learner.holdout_data)
        }
    except Exception as e:
        logger.error(f"Error getting online learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/online_learning/configure")
async def configure_online_learning(config: OnlineLearningConfig):
    """Configure online learning system parameters."""
    global online_learner
    
    if not online_learner:
        raise HTTPException(status_code=400, detail="Online learning system not available")
    
    try:
        # Validate update strategy
        valid_strategies = ['immediate', 'batch', 'weighted', 'confidence']
        if config.update_strategy not in valid_strategies:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid update strategy. Must be one of: {valid_strategies}"
            )
        
        # Update configuration
        online_learner.update_strategy = config.update_strategy
        online_learner.batch_size = config.batch_size
        online_learner.confidence_threshold = config.confidence_threshold
        online_learner.weight_decay = config.weight_decay
        
        logger.info(f"Online learning configured: {config.update_strategy}, "
                   f"batch_size={config.batch_size}, threshold={config.confidence_threshold}")
        
        return {
            "status": "success",
            "message": "Online learning configuration updated",
            "configuration": {
                "update_strategy": config.update_strategy,
                "batch_size": config.batch_size,
                "confidence_threshold": config.confidence_threshold,
                "weight_decay": config.weight_decay
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring online learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/online_learning/force_update")
async def force_online_update():
    """Force an immediate online learning update with pending samples."""
    global online_learner
    
    if not online_learner or not online_learner.is_initialized:
        raise HTTPException(status_code=400, detail="Online learning system not initialized")
    
    try:
        if not online_learner.pending_samples:
            return {
                "status": "no_action",
                "message": "No pending samples to process"
            }
        
        # Extract features and labels from pending samples
        feature_data = [s['feature_data'] for s in online_learner.pending_samples]
        target_values = [s['label'] for s in online_learner.pending_samples]
        
        # Force update
        result = online_learner.update_model(feature_data, target_values)
        
        # Clear pending samples
        online_learner.pending_samples.clear()
        
        # Save model
        try:
            model_save_path = Path(__file__).parent / "models"
            model_save_path.mkdir(exist_ok=True)
            online_learner.save_model(model_save_path / "online_model.joblib")
        except Exception as e:
            logger.warning(f"Failed to save model after force update: {e}")
        
        return {
            "status": "success",
            "message": f"Forced update completed with {result.samples_added} samples",
            "update_result": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error in force update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/online_learning/performance")
async def get_online_learning_performance():
    """Get detailed online learning performance metrics."""
    global online_learner
    
    if not online_learner or not online_learner.is_initialized:
        raise HTTPException(status_code=400, detail="Online learning system not initialized")
    
    try:
        # Get learning curve data
        learning_curve = online_learner.get_learning_curve()
        
        # Get update statistics
        update_stats = online_learner.get_update_statistics()
        
        # Get recent performance history
        recent_performance = online_learner.performance_history[-10:] if online_learner.performance_history else []
        
        return {
            "learning_curve": learning_curve.to_dict('records') if not learning_curve.empty else [],
            "update_statistics": update_stats,
            "recent_performance": recent_performance,
            "model_info": online_learner.get_model_info()
        }
        
    except Exception as e:
        logger.error(f"Error getting online learning performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )