#!/usr/bin/env python3
"""
Database Models and Management for Human-in-the-Loop Interface

This module provides database models and utilities for storing predictions,
labels, and metrics in the human-in-the-loop system.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Database record for model predictions."""
    id: Optional[int]
    timestamp: str
    specimen_id: int
    shear_percentage: float
    confidence: float
    detection_bbox: List[float]
    detection_confidence: float
    processing_time: float
    image_data: str


@dataclass
class LabelRecord:
    """Database record for technician labels."""
    id: Optional[int]
    timestamp: str
    specimen_id: int
    technician_label: float
    model_prediction: float
    model_confidence: float
    technician_id: str
    notes: Optional[str]
    image_data: str


@dataclass
class MetricRecord:
    """Database record for model metrics."""
    id: Optional[int]
    timestamp: str
    metric_name: str
    metric_value: float
    model_version: str


class DatabaseManager:
    """Manages database operations for the human-in-the-loop system."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager."""
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "labeling_history.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
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
                image_data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create labels table
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
                image_data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                model_version TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_timestamp ON labels(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_technician ON labels(technician_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON model_metrics(timestamp)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def store_prediction(self, prediction: PredictionRecord) -> int:
        """Store a prediction record and return the ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, specimen_id, shear_percentage, confidence, detection_bbox, 
             detection_confidence, processing_time, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.timestamp,
            prediction.specimen_id,
            prediction.shear_percentage,
            prediction.confidence,
            json.dumps(prediction.detection_bbox),
            prediction.detection_confidence,
            prediction.processing_time,
            prediction.image_data
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Stored prediction {prediction_id} for specimen {prediction.specimen_id}")
        return prediction_id
    
    def store_label(self, label: LabelRecord) -> int:
        """Store a label record and return the ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO labels 
            (timestamp, specimen_id, technician_label, model_prediction, 
             model_confidence, technician_id, notes, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            label.timestamp,
            label.specimen_id,
            label.technician_label,
            label.model_prediction,
            label.model_confidence,
            label.technician_id,
            label.notes,
            label.image_data
        ))
        
        label_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Stored label {label_id} by {label.technician_id} for specimen {label.specimen_id}")
        return label_id
    
    def store_metric(self, metric: MetricRecord) -> int:
        """Store a metric record and return the ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_metrics 
            (timestamp, metric_name, metric_value, model_version)
            VALUES (?, ?, ?, ?)
        """, (
            metric.timestamp,
            metric.metric_name,
            metric.metric_value,
            metric.model_version
        ))
        
        metric_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return metric_id
    
    def get_predictions(self, limit: int = 100, offset: int = 0) -> List[PredictionRecord]:
        """Get prediction records with pagination."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, specimen_id, shear_percentage, confidence, 
                   detection_bbox, detection_confidence, processing_time, image_data
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append(PredictionRecord(
                id=row[0],
                timestamp=row[1],
                specimen_id=row[2],
                shear_percentage=row[3],
                confidence=row[4],
                detection_bbox=json.loads(row[5]),
                detection_confidence=row[6],
                processing_time=row[7],
                image_data=row[8]
            ))
        
        return predictions
    
    def get_labels(self, limit: int = 100, offset: int = 0) -> List[LabelRecord]:
        """Get label records with pagination."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, specimen_id, technician_label, model_prediction, 
                   model_confidence, technician_id, notes, image_data
            FROM labels 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        labels = []
        for row in rows:
            labels.append(LabelRecord(
                id=row[0],
                timestamp=row[1],
                specimen_id=row[2],
                technician_label=row[3],
                model_prediction=row[4],
                model_confidence=row[5],
                technician_id=row[6],
                notes=row[7],
                image_data=row[8]
            ))
        
        return labels
    
    def get_labels_count(self) -> int:
        """Get total count of labels."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM labels")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_predictions_count(self) -> int:
        """Get total count of predictions."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_recent_labels(self, days: int = 30) -> List[LabelRecord]:
        """Get labels from the last N days."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT id, timestamp, specimen_id, technician_label, model_prediction, 
                   model_confidence, technician_id, notes, image_data
            FROM labels 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        labels = []
        for row in rows:
            labels.append(LabelRecord(
                id=row[0],
                timestamp=row[1],
                specimen_id=row[2],
                technician_label=row[3],
                model_prediction=row[4],
                model_confidence=row[5],
                technician_id=row[6],
                notes=row[7],
                image_data=row[8]
            ))
        
        return labels
    
    def calculate_accuracy_metrics(self, days: int = 30) -> Dict[str, float]:
        """Calculate accuracy metrics for recent labels."""
        labels = self.get_recent_labels(days)
        
        if not labels:
            return {}
        
        technician_labels = [label.technician_label for label in labels]
        model_predictions = [label.model_prediction for label in labels]
        
        # Calculate metrics
        differences = [abs(t - m) for t, m in zip(technician_labels, model_predictions)]
        mae = sum(differences) / len(differences)
        rmse = (sum(d**2 for d in differences) / len(differences)) ** 0.5
        
        # Calculate R² score
        mean_technician = sum(technician_labels) / len(technician_labels)
        ss_res = sum((t - m)**2 for t, m in zip(technician_labels, model_predictions))
        ss_tot = sum((t - mean_technician)**2 for t in technician_labels)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "sample_count": len(labels)
        }
    
    def get_daily_performance(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily performance metrics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT DATE(timestamp) as date, 
                   AVG(ABS(technician_label - model_prediction)) as daily_mae,
                   COUNT(*) as daily_count
            FROM labels 
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{"date": row[0], "mae": row[1], "count": row[2]} for row in rows]
    
    def get_technician_performance(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get performance metrics by technician."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT technician_id,
                   COUNT(*) as label_count,
                   AVG(ABS(technician_label - model_prediction)) as avg_difference,
                   AVG(model_confidence) as avg_model_confidence
            FROM labels 
            WHERE timestamp > ?
            GROUP BY technician_id
            ORDER BY label_count DESC
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            "technician_id": row[0],
            "label_count": row[1],
            "avg_difference": row[2],
            "avg_model_confidence": row[3]
        } for row in rows]
    
    def export_training_data(self, output_path: Path) -> None:
        """Export labels as training data."""
        labels = self.get_labels(limit=10000)  # Get all labels
        
        # Convert to DataFrame
        data = []
        for label in labels:
            data.append({
                'timestamp': label.timestamp,
                'specimen_id': label.specimen_id,
                'technician_label': label.technician_label,
                'model_prediction': label.model_prediction,
                'model_confidence': label.model_confidence,
                'technician_id': label.technician_id,
                'notes': label.notes
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(data)} training records to {output_path}")
    
    def cleanup_old_data(self, days: int = 90) -> int:
        """Clean up old prediction data (keep labels forever)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff_date,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_count} old prediction records")
        return deleted_count