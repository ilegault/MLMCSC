#!/usr/bin/env python3
"""
Online Learning Module for Human-in-the-Loop System

This module implements online learning capabilities that continuously improve
the model based on technician feedback and labels.
"""

import cv2
import numpy as np
import logging
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
import time
from collections import deque

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Import MLMCSC modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mlmcsc.classification.shiny_region_classifier import ShinyRegionBasedClassifier
from src.web.database import DatabaseManager, LabelRecord

logger = logging.getLogger(__name__)


class OnlineLearningManager:
    """
    Manages online learning for the human-in-the-loop system.
    
    This class continuously monitors for new labels and updates the model
    when sufficient new data is available.
    """
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 model_save_path: Path,
                 update_threshold: int = 10,
                 update_interval: int = 300):  # 5 minutes
        """
        Initialize online learning manager.
        
        Args:
            db_manager: Database manager instance
            model_save_path: Path to save updated models
            update_threshold: Minimum new labels before updating
            update_interval: Seconds between update checks
        """
        self.db_manager = db_manager
        self.model_save_path = model_save_path
        self.update_threshold = update_threshold
        self.update_interval = update_interval
        
        # Model components
        self.classifier = ShinyRegionBasedClassifier()
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Tracking
        self.last_update_time = datetime.now()
        self.last_processed_label_id = 0
        self.model_version = "1.0.0"
        self.is_running = False
        self.update_thread = None
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        
        # Load existing model if available
        self.load_model()
        
        logger.info("Online learning manager initialized")
    
    def load_model(self) -> bool:
        """Load existing model if available."""
        try:
            model_file = self.model_save_path / "online_model.joblib"
            scaler_file = self.model_save_path / "online_scaler.joblib"
            metadata_file = self.model_save_path / "online_metadata.json"
            
            if all(f.exists() for f in [model_file, scaler_file, metadata_file]):
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.model_version = metadata.get('version', '1.0.0')
                    self.last_processed_label_id = metadata.get('last_label_id', 0)
                
                logger.info(f"Loaded online model version {self.model_version}")
                return True
            else:
                logger.info("No existing online model found, will train from scratch")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self) -> None:
        """Save current model state."""
        try:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            
            model_file = self.model_save_path / "online_model.joblib"
            scaler_file = self.model_save_path / "online_scaler.joblib"
            metadata_file = self.model_save_path / "online_metadata.json"
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            metadata = {
                'version': self.model_version,
                'last_update': datetime.now().isoformat(),
                'last_label_id': self.last_processed_label_id,
                'performance_history': list(self.performance_history)
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved online model version {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def start_online_learning(self) -> None:
        """Start the online learning process in a background thread."""
        if self.is_running:
            logger.warning("Online learning is already running")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Started online learning process")
    
    def stop_online_learning(self) -> None:
        """Stop the online learning process."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=10)
        
        logger.info("Stopped online learning process")
    
    def _update_loop(self) -> None:
        """Main update loop running in background thread."""
        while self.is_running:
            try:
                self.check_for_updates()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def check_for_updates(self) -> None:
        """Check if model should be updated based on new labels."""
        try:
            # Get count of new labels since last update
            total_labels = self.db_manager.get_labels_count()
            new_labels_count = total_labels - self.last_processed_label_id
            
            if new_labels_count >= self.update_threshold:
                logger.info(f"Found {new_labels_count} new labels, triggering model update")
                self.update_model()
            else:
                logger.debug(f"Only {new_labels_count} new labels, waiting for more")
                
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
    
    def update_model(self) -> bool:
        """Update the model with new training data."""
        try:
            logger.info("Starting model update...")
            
            # Get all labels for training
            labels = self.db_manager.get_labels(limit=10000)  # Get all labels
            
            if len(labels) < 10:
                logger.warning("Not enough labels for training")
                return False
            
            # Extract features and prepare training data
            X, y = self._prepare_training_data(labels)
            
            if X is None or len(X) == 0:
                logger.warning("No valid training data extracted")
                return False
            
            # Split into train/validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            y_pred = self.model.predict(X_val_scaled)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            # Update tracking
            self.last_processed_label_id = max(label.id for label in labels if label.id)
            self.last_update_time = datetime.now()
            
            # Increment version
            version_parts = self.model_version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            self.model_version = '.'.join(version_parts)
            
            # Store performance
            performance = {
                'timestamp': self.last_update_time.isoformat(),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'version': self.model_version
            }
            self.performance_history.append(performance)
            
            # Save updated model
            self.save_model()
            
            logger.info(f"Model updated to version {self.model_version}")
            logger.info(f"Performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
    
    def _prepare_training_data(self, labels: List[LabelRecord]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from label records."""
        features_list = []
        targets_list = []
        
        logger.info(f"Preparing training data from {len(labels)} labels...")
        
        for i, label in enumerate(labels):
            try:
                # Decode image
                image = self._decode_image(label.image_data)
                if image is None:
                    continue
                
                # Extract features using the classifier
                features = self.classifier.process_image(image)
                if features is None:
                    continue
                
                features_list.append(features.to_array())
                targets_list.append(label.technician_label)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(labels)} labels")
                
            except Exception as e:
                logger.warning(f"Failed to process label {label.id}: {e}")
                continue
        
        if not features_list:
            logger.error("No valid features extracted from labels")
            return None, None
        
        X = np.array(features_list)
        y = np.array(targets_list)
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image data."""
        try:
            import base64
            
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction using the online model.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate confidence based on model uncertainty
            # For Random Forest, we can use the variance of tree predictions
            if hasattr(self.model, 'estimators_'):
                tree_predictions = [tree.predict(features_scaled)[0] for tree in self.model.estimators_]
                confidence = 1.0 / (1.0 + np.var(tree_predictions))
            else:
                confidence = 0.8  # Default confidence
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 50.0, 0.5  # Default prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'version': self.model_version,
            'last_update': self.last_update_time.isoformat(),
            'last_processed_label_id': self.last_processed_label_id,
            'is_running': self.is_running,
            'performance_history': list(self.performance_history)[-10:],  # Last 10 updates
            'model_type': type(self.model).__name__,
            'feature_count': getattr(self.scaler, 'n_features_in_', 0)
        }
    
    def force_update(self) -> bool:
        """Force an immediate model update regardless of threshold."""
        logger.info("Forcing model update...")
        return self.update_model()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_history:
            return {}
        
        latest = self.performance_history[-1]
        
        # Calculate trends if we have multiple updates
        if len(self.performance_history) > 1:
            prev = self.performance_history[-2]
            mae_trend = latest['mae'] - prev['mae']
            rmse_trend = latest['rmse'] - prev['rmse']
            r2_trend = latest['r2'] - prev['r2']
        else:
            mae_trend = rmse_trend = r2_trend = 0.0
        
        return {
            'current_performance': latest,
            'trends': {
                'mae_trend': mae_trend,
                'rmse_trend': rmse_trend,
                'r2_trend': r2_trend
            },
            'update_count': len(self.performance_history)
        }