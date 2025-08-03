#!/usr/bin/env python3
"""
Online Learning System for Fracture Surface Analysis

This module implements incremental learning capabilities for continuously
improving the regression model as new labeled data becomes available.
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import time
from datetime import datetime

from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

# Import new systems
try:
    from ..regression.model_versioning import EnhancedModelVersionManager
    from ..quality_control import QualityControlSystem
    from ..monitoring import PerformanceMonitor
    from ..active_learning import ActiveLearningSystem
except ImportError:
    # Fallback imports for testing
    EnhancedModelVersionManager = None
    QualityControlSystem = None
    PerformanceMonitor = None
    ActiveLearningSystem = None

logger = logging.getLogger(__name__)


@dataclass
class OnlineUpdateResult:
    """Container for online update results."""
    samples_added: int
    update_time: float
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    improvement: Dict[str, float]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'samples_added': self.samples_added,
            'update_time': self.update_time,
            'performance_before': self.performance_before,
            'performance_after': self.performance_after,
            'improvement': self.improvement,
            'timestamp': self.timestamp
        }


class OnlineLearningSystem:
    """
    Online learning system for incremental model updates.
    
    Supports multiple online learning algorithms and provides mechanisms
    for continuous model improvement as new labeled data becomes available.
    
    Update Strategies:
    - Immediate: Update after each submission
    - Batch: Collect N submissions, then update
    - Weighted: Weight recent samples more heavily
    - Confidence-based: Update more when model is uncertain
    """
    
    def __init__(self, 
                 model_type: str = 'sgd',
                 scaler_type: str = 'standard',
                 target_property: str = 'shear_percentage',
                 learning_rate: str = 'constant',
                 alpha: float = 0.0001,
                 random_state: int = 42,
                 update_strategy: str = 'batch',
                 batch_size: int = 10,
                 confidence_threshold: float = 0.7,
                 weight_decay: float = 0.95,
                 enable_versioning: bool = True,
                 enable_quality_control: bool = True,
                 enable_monitoring: bool = True,
                 enable_active_learning: bool = True):
        """
        Initialize the online learning system.
        
        Args:
            model_type: Type of online model ('sgd', 'passive_aggressive', 'mlp')
            scaler_type: Type of feature scaler ('standard', 'robust', 'none')
            target_property: Name of the target property to predict
            learning_rate: Learning rate schedule for SGD
            alpha: Regularization strength
            random_state: Random state for reproducibility
            update_strategy: Update strategy ('immediate', 'batch', 'weighted', 'confidence')
            batch_size: Number of samples to collect before batch update
            confidence_threshold: Confidence threshold for confidence-based updates
            weight_decay: Decay factor for weighted updates
            enable_versioning: Enable model versioning system
            enable_quality_control: Enable quality control system
            enable_monitoring: Enable performance monitoring
            enable_active_learning: Enable active learning system
        """
        self.model_type = model_type
        self.scaler_type = scaler_type
        self.target_property = target_property
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.random_state = random_state
        self.update_strategy = update_strategy
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.weight_decay = weight_decay
        
        # Initialize model and scaler
        self.model = self._create_online_model()
        self.scaler = self._create_scaler()
        
        # Training state
        self.feature_names = []
        self.is_initialized = False
        self.total_samples_seen = 0
        self.update_history = []
        self.performance_history = []
        
        # Update strategy state
        self.pending_samples = []  # For batch and confidence-based updates
        self.sample_weights = []   # For weighted updates
        self.holdout_data = []     # For validation
        
        # Model metadata
        self.creation_time = datetime.now()
        self.last_update_time = None
        
        # Initialize advanced systems
        self.version_manager = None
        self.quality_control = None
        self.performance_monitor = None
        self.active_learning = None
        
        if enable_versioning and EnhancedModelVersionManager:
            try:
                self.version_manager = EnhancedModelVersionManager(
                    models_directory=Path("models/versions"),
                    checkpoint_frequency=10,
                    performance_threshold=0.05
                )
                logger.info("Model versioning system enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize versioning system: {e}")
        
        if enable_quality_control and QualityControlSystem:
            try:
                self.quality_control = QualityControlSystem(
                    storage_path="quality_control",
                    outlier_threshold=2.5,
                    variance_threshold=0.3
                )
                logger.info("Quality control system enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize quality control system: {e}")
        
        if enable_monitoring and PerformanceMonitor:
            try:
                self.performance_monitor = PerformanceMonitor(
                    storage_path="monitoring",
                    alert_thresholds={
                        'r2_min': 0.7,
                        'rmse_max': 10.0,
                        'mae_max': 8.0,
                        'confidence_min': 0.5
                    }
                )
                logger.info("Performance monitoring system enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize monitoring system: {e}")
        
        if enable_active_learning and ActiveLearningSystem:
            try:
                self.active_learning = ActiveLearningSystem(
                    storage_path="active_learning",
                    default_strategy="hybrid",
                    max_queries_per_batch=10
                )
                logger.info("Active learning system enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize active learning system: {e}")
        
        logger.info(f"OnlineLearningSystem initialized: {model_type} with {scaler_type} scaling")
        logger.info(f"Update strategy: {update_strategy}, batch_size: {batch_size}")
        logger.info(f"Advanced features: versioning={self.version_manager is not None}, "
                   f"quality_control={self.quality_control is not None}, "
                   f"monitoring={self.performance_monitor is not None}, "
                   f"active_learning={self.active_learning is not None}")
    
    def _create_online_model(self):
        """Create the online learning model based on model_type."""
        models = {
            'sgd': SGDRegressor(
                loss='squared_error',
                learning_rate=self.learning_rate,
                alpha=self.alpha,
                random_state=self.random_state,
                warm_start=True  # Enable incremental learning
            ),
            'passive_aggressive': PassiveAggressiveRegressor(
                C=1.0,
                random_state=self.random_state,
                warm_start=True
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=self.alpha,
                learning_rate='adaptive',
                warm_start=True,  # Enable incremental learning
                max_iter=1,  # Single epoch for online learning
                random_state=self.random_state
            )
        }
        
        if self.model_type not in models:
            logger.warning(f"Unknown model type {self.model_type}, using sgd")
            self.model_type = 'sgd'
        
        return models[self.model_type]
    
    def _create_scaler(self):
        """Create the feature scaler based on scaler_type."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'none': None
        }
        
        if self.scaler_type not in scalers:
            logger.warning(f"Unknown scaler type {self.scaler_type}, using standard")
            self.scaler_type = 'standard'
        
        return scalers[self.scaler_type]
    
    def initialize_model(self, 
                        feature_data: List[Dict],
                        target_values: List[float],
                        feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Initialize the model with initial training data.
        
        Args:
            feature_data: List of feature dictionaries or feature vectors
            target_values: List of target values
            feature_names: Optional list of feature names
            
        Returns:
            Initial performance metrics
        """
        start_time = time.time()
        
        try:
            # Prepare training data
            X, y = self._prepare_data(feature_data, target_values, feature_names)
            
            # Initialize scaler if needed
            if self.scaler is not None:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Initial training
            self.model.fit(X_scaled, y)
            
            # Calculate initial performance
            y_pred = self.model.predict(X_scaled)
            performance = self._calculate_performance(y, y_pred)
            
            # Update state
            self.is_initialized = True
            self.total_samples_seen = len(X)
            self.last_update_time = datetime.now()
            
            # Record performance history
            self.performance_history.append({
                'timestamp': self.last_update_time,
                'samples_count': self.total_samples_seen,
                'performance': performance,
                'update_type': 'initialization'
            })
            
            initialization_time = time.time() - start_time
            logger.info(f"Model initialized with {len(X)} samples in {initialization_time:.2f}s")
            
            # Handle None values in performance metrics for logging
            r2_str = f"{performance['r2']:.3f}" if performance['r2'] is not None else "N/A (insufficient samples)"
            rmse_str = f"{performance['rmse']:.3f}" if performance['rmse'] is not None else "N/A"
            logger.info(f"Initial performance - R²: {r2_str}, RMSE: {rmse_str}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def update_model(self, 
                    new_feature_data: List[Dict],
                    new_target_values: List[float],
                    batch_size: Optional[int] = None,
                    evaluate_improvement: bool = True) -> OnlineUpdateResult:
        """
        Update the model with new training data.
        
        Args:
            new_feature_data: New feature data for training
            new_target_values: New target values
            batch_size: Optional batch size for mini-batch updates
            evaluate_improvement: Whether to evaluate performance improvement
            
        Returns:
            OnlineUpdateResult with update statistics
        """
        if not self.is_initialized:
            raise ValueError("Model must be initialized before updates")
        
        start_time = time.time()
        
        try:
            # Prepare new data
            X_new, y_new = self._prepare_data(new_feature_data, new_target_values)
            
            # Scale new data using existing scaler
            if self.scaler is not None:
                X_new_scaled = self.scaler.transform(X_new)
            else:
                X_new_scaled = X_new
            
            # Get performance before update
            performance_before = None
            if evaluate_improvement:
                performance_before = self._evaluate_current_performance(X_new_scaled, y_new)
            
            # Perform incremental learning
            if batch_size is None or batch_size >= len(X_new):
                # Single batch update
                self.model.partial_fit(X_new_scaled, y_new)
            else:
                # Mini-batch updates
                for i in range(0, len(X_new), batch_size):
                    batch_X = X_new_scaled[i:i+batch_size]
                    batch_y = y_new[i:i+batch_size]
                    self.model.partial_fit(batch_X, batch_y)
            
            # Get performance after update
            performance_after = None
            improvement = None
            if evaluate_improvement:
                performance_after = self._evaluate_current_performance(X_new_scaled, y_new)
                improvement = {
                    key: performance_after[key] - performance_before[key]
                    for key in performance_before.keys()
                }
            
            # Update state
            self.total_samples_seen += len(X_new)
            self.last_update_time = datetime.now()
            update_time = time.time() - start_time
            
            # Create update result
            update_result = OnlineUpdateResult(
                samples_added=len(X_new),
                update_time=update_time,
                performance_before=performance_before or {},
                performance_after=performance_after or {},
                improvement=improvement or {},
                timestamp=time.time()
            )
            
            # Record update history
            self.update_history.append(update_result.to_dict())
            
            # Record performance history
            if performance_after:
                self.performance_history.append({
                    'timestamp': self.last_update_time,
                    'samples_count': self.total_samples_seen,
                    'performance': performance_after,
                    'update_type': 'incremental'
                })
            
            logger.info(f"Model updated with {len(X_new)} samples in {update_time:.2f}s")
            if improvement:
                logger.info(f"Performance change - R²: {improvement['r2']:+.3f}, RMSE: {improvement['rmse']:+.3f}")
            
            return update_result
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            raise
    
    def predict(self, feature_data: Union[Dict, List[Dict], np.ndarray]) -> Union[float, List[float]]:
        """
        Make predictions using the current model.
        
        Args:
            feature_data: Feature data for prediction
            
        Returns:
            Prediction(s)
        """
        if not self.is_initialized:
            raise ValueError("Model must be initialized before making predictions")
        
        try:
            # Prepare input data
            X, _ = self._prepare_data(feature_data, None)
            
            # Scale data
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Return single value or list based on input
            if len(predictions) == 1:
                return float(predictions[0])
            else:
                return predictions.tolist()
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def process_technician_submission(self, 
                                    feature_data: Dict,
                                    label: float,
                                    timestamp: str,
                                    technician_id: str,
                                    confidence: Optional[float] = None,
                                    image_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Core online learning pipeline: Process technician submission.
        
        This is the main entry point for the online learning system when
        a technician submits a label. Now includes advanced quality control,
        monitoring, versioning, and active learning integration.
        
        Args:
            feature_data: Extracted features from the image
            label: Technician's label (shear percentage)
            timestamp: Submission timestamp
            technician_id: ID of the technician who submitted the label
            confidence: Model confidence for this prediction (optional)
            image_id: Image identifier (optional)
            
        Returns:
            Dictionary with update results and metrics
        """
        if not self.is_initialized:
            raise ValueError("Model must be initialized before processing submissions")
        
        try:
            # Store submission data
            submission = {
                'feature_data': feature_data,
                'label': label,
                'timestamp': timestamp,
                'technician_id': technician_id,
                'confidence': confidence or 0.5,
                'image_id': image_id or f"img_{timestamp}"
            }
            
            # Quality control checks
            quality_flags = []
            if self.quality_control:
                try:
                    # Check for label outliers
                    recent_labels = [s['label'] for s in self.pending_samples[-50:]] + [label]
                    recent_techs = [s['technician_id'] for s in self.pending_samples[-50:]] + [technician_id]
                    recent_images = [s.get('image_id', 'unknown') for s in self.pending_samples[-50:]] + [submission['image_id']]
                    recent_timestamps = [s['timestamp'] for s in self.pending_samples[-50:]] + [timestamp]
                    
                    if len(recent_labels) >= 5:
                        outlier_results = self.quality_control.detect_label_outliers(
                            recent_labels, recent_techs, recent_images, recent_timestamps
                        )
                        
                        # Check if current submission is flagged as outlier
                        current_outlier = next((r for r in outlier_results 
                                              if r.image_id == submission['image_id'] and r.technician_id == technician_id), None)
                        if current_outlier and current_outlier.is_outlier:
                            quality_flags.append({
                                'type': 'outlier',
                                'severity': 'medium',
                                'message': f"Label {label} flagged as outlier (score: {current_outlier.outlier_score:.3f})"
                            })
                    
                    # Check for feature outliers if features available
                    if 'feature_vector' in feature_data or isinstance(feature_data, dict):
                        feature_outliers = self.quality_control.detect_feature_outliers(
                            [feature_data], [submission['image_id']], [technician_id], [timestamp]
                        )
                        if feature_outliers and feature_outliers[0].is_outlier:
                            quality_flags.append({
                                'type': 'feature_outlier',
                                'severity': 'low',
                                'message': f"Features flagged as outliers"
                            })
                            
                except Exception as e:
                    logger.warning(f"Quality control check failed: {e}")
            
            # Record label for monitoring
            if self.performance_monitor:
                try:
                    self.performance_monitor.record_label(
                        label=label,
                        technician_id=technician_id,
                        image_id=submission['image_id'],
                        confidence=confidence
                    )
                except Exception as e:
                    logger.warning(f"Monitoring record failed: {e}")
            
            # Apply update strategy
            update_result = self._apply_update_strategy(submission)
            
            # Create model checkpoint if needed
            checkpoint_created = False
            if update_result and self.version_manager:
                try:
                    if self.version_manager.should_create_checkpoint():
                        # Calculate current performance metrics
                        current_metrics = self._get_current_performance_metrics()
                        
                        # Save current model temporarily
                        temp_model_path = Path("temp_model.joblib")
                        temp_scaler_path = Path("temp_scaler.joblib") if self.scaler else None
                        
                        joblib.dump(self.model, temp_model_path)
                        if self.scaler and temp_scaler_path:
                            joblib.dump(self.scaler, temp_scaler_path)
                        
                        # Create checkpoint
                        version_id = self.version_manager.create_checkpoint_version(
                            model_path=temp_model_path,
                            scaler_path=temp_scaler_path,
                            model_type=self.model_type,
                            performance_metrics=current_metrics,
                            training_samples_count=self.total_samples_seen,
                            feature_names=self.feature_names,
                            scaler_type=self.scaler_type,
                            reason="automatic_checkpoint",
                            notes=f"Checkpoint after {self.version_manager.update_count} updates"
                        )
                        
                        checkpoint_created = True
                        logger.info(f"Created automatic checkpoint: {version_id}")
                        
                        # Cleanup temp files
                        if temp_model_path.exists():
                            temp_model_path.unlink()
                        if temp_scaler_path and temp_scaler_path.exists():
                            temp_scaler_path.unlink()
                            
                        # Check for performance degradation
                        degradation_warning = self.version_manager.check_performance_degradation(current_metrics)
                        if degradation_warning:
                            quality_flags.append({
                                'type': 'performance_degradation',
                                'severity': 'high',
                                'message': degradation_warning
                            })
                            
                except Exception as e:
                    logger.warning(f"Model versioning failed: {e}")
            
            # Validate on holdout set if available
            validation_metrics = self._validate_on_holdout()
            
            # Record performance metrics
            if update_result and self.performance_monitor:
                try:
                    current_metrics = self._get_current_performance_metrics()
                    self.performance_monitor.record_performance_metrics(
                        mae=current_metrics.get('mae', 0),
                        rmse=current_metrics.get('rmse', 0),
                        r2=current_metrics.get('r2', 0),
                        mse=current_metrics.get('mse', 0),
                        sample_count=self.total_samples_seen,
                        prediction_count=1,  # This submission
                        avg_confidence=confidence or 0.5,
                        model_version=getattr(self.version_manager, 'active_version_id', None) if self.version_manager else None
                    )
                except Exception as e:
                    logger.warning(f"Performance monitoring failed: {e}")
            
            # Generate active learning queries if enabled
            active_learning_queries = []
            if self.active_learning and update_result:
                try:
                    # Use recent unlabeled samples as candidates (this would come from your image pipeline)
                    # For now, we'll skip this as it requires integration with the image processing pipeline
                    pass
                except Exception as e:
                    logger.warning(f"Active learning query generation failed: {e}")
            
            # Log performance metrics
            self._log_performance_metrics(update_result, validation_metrics)
            
            # Prepare response
            response = {
                'status': 'success',
                'update_applied': update_result is not None,
                'update_strategy': self.update_strategy,
                'samples_pending': len(self.pending_samples),
                'validation_metrics': validation_metrics,
                'update_result': update_result.to_dict() if update_result else None,
                'quality_flags': quality_flags,
                'checkpoint_created': checkpoint_created,
                'active_learning_queries': len(active_learning_queries),
                'systems_status': {
                    'versioning': self.version_manager is not None,
                    'quality_control': self.quality_control is not None,
                    'monitoring': self.performance_monitor is not None,
                    'active_learning': self.active_learning is not None
                }
            }
            
            # Add alerts if any
            if self.performance_monitor:
                try:
                    alerts = self.performance_monitor.get_alerts()
                    if alerts:
                        response['performance_alerts'] = alerts
                except Exception as e:
                    logger.warning(f"Failed to get performance alerts: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing technician submission: {e}")
            raise
    
    def _apply_update_strategy(self, submission: Dict[str, Any]) -> Optional[OnlineUpdateResult]:
        """Apply the configured update strategy."""
        if self.update_strategy == 'immediate':
            return self._immediate_update(submission)
        elif self.update_strategy == 'batch':
            return self._batch_update(submission)
        elif self.update_strategy == 'weighted':
            return self._weighted_update(submission)
        elif self.update_strategy == 'confidence':
            return self._confidence_based_update(submission)
        else:
            logger.warning(f"Unknown update strategy: {self.update_strategy}")
            return self._immediate_update(submission)
    
    def _immediate_update(self, submission: Dict[str, Any]) -> OnlineUpdateResult:
        """Immediate update: Update after each submission."""
        feature_data = [submission['feature_data']]
        target_values = [submission['label']]
        
        return self.update_model(feature_data, target_values, batch_size=1)
    
    def _batch_update(self, submission: Dict[str, Any]) -> Optional[OnlineUpdateResult]:
        """Batch update: Collect N submissions, then update."""
        self.pending_samples.append(submission)
        
        if len(self.pending_samples) >= self.batch_size:
            # Extract features and labels from pending samples
            feature_data = [s['feature_data'] for s in self.pending_samples]
            target_values = [s['label'] for s in self.pending_samples]
            
            # Update model
            result = self.update_model(feature_data, target_values)
            
            # Clear pending samples
            self.pending_samples.clear()
            
            return result
        
        return None  # No update applied yet
    
    def _weighted_update(self, submission: Dict[str, Any]) -> OnlineUpdateResult:
        """Weighted update: Weight recent samples more heavily."""
        # Add current submission
        self.pending_samples.append(submission)
        
        # Calculate weights (more recent = higher weight)
        weights = []
        for i, sample in enumerate(self.pending_samples):
            # Exponential decay: more recent samples get higher weights
            weight = self.weight_decay ** (len(self.pending_samples) - i - 1)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Extract features and labels
        feature_data = [s['feature_data'] for s in self.pending_samples]
        target_values = [s['label'] for s in self.pending_samples]
        
        # Update model with weighted samples
        # Note: sklearn doesn't directly support sample weights in partial_fit
        # So we simulate by repeating samples based on weights
        weighted_features = []
        weighted_targets = []
        
        for i, (features, target) in enumerate(zip(feature_data, target_values)):
            # Repeat sample based on weight (rounded to nearest integer)
            repeat_count = max(1, int(weights[i] * 10))  # Scale weights
            for _ in range(repeat_count):
                weighted_features.append(features)
                weighted_targets.append(target)
        
        result = self.update_model(weighted_features, weighted_targets)
        
        # Keep only recent samples for next update
        max_history = 50  # Keep last 50 samples
        if len(self.pending_samples) > max_history:
            self.pending_samples = self.pending_samples[-max_history:]
        
        return result
    
    def _confidence_based_update(self, submission: Dict[str, Any]) -> Optional[OnlineUpdateResult]:
        """Confidence-based: Update more when model is uncertain."""
        confidence = submission.get('confidence', 0.5)
        
        # Add to pending samples
        self.pending_samples.append(submission)
        
        # Decide whether to update based on confidence
        should_update = False
        
        if confidence < self.confidence_threshold:
            # Low confidence - update immediately
            should_update = True
            logger.info(f"Low confidence ({confidence:.3f}) - triggering immediate update")
        elif len(self.pending_samples) >= self.batch_size:
            # Batch size reached - update anyway
            should_update = True
            logger.info(f"Batch size reached ({len(self.pending_samples)}) - triggering update")
        
        if should_update:
            # Extract features and labels from pending samples
            feature_data = [s['feature_data'] for s in self.pending_samples]
            target_values = [s['label'] for s in self.pending_samples]
            
            # Update model
            result = self.update_model(feature_data, target_values)
            
            # Clear pending samples
            self.pending_samples.clear()
            
            return result
        
        return None  # No update applied yet
    
    def _validate_on_holdout(self) -> Dict[str, float]:
        """Validate model performance on recent holdout set."""
        if len(self.holdout_data) < 5:  # Need minimum samples for validation
            return {}
        
        try:
            # Use last 20% of holdout data for validation
            val_size = max(5, len(self.holdout_data) // 5)
            val_data = self.holdout_data[-val_size:]
            
            # Extract features and targets
            X_val = np.array([item['feature_vector'] for item in val_data])
            y_val = np.array([item['label'] for item in val_data])
            
            # Scale features
            if self.scaler is not None:
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_val_scaled = X_val
            
            # Make predictions
            y_pred = self.model.predict(X_val_scaled)
            
            # Calculate metrics
            return self._calculate_performance(y_val, y_pred)
            
        except Exception as e:
            logger.error(f"Holdout validation failed: {e}")
            return {}
    
    def _log_performance_metrics(self, 
                                update_result: Optional[OnlineUpdateResult],
                                validation_metrics: Dict[str, float]) -> None:
        """Log performance metrics for monitoring."""
        if update_result:
            logger.info(f"Model updated - samples: {update_result.samples_added}, "
                       f"time: {update_result.update_time:.2f}s")
            
            if update_result.improvement:
                r2_change = update_result.improvement.get('r2', 0)
                rmse_change = update_result.improvement.get('rmse', 0)
                logger.info(f"Performance change - R²: {r2_change:+.3f}, "
                           f"RMSE: {rmse_change:+.3f}")
        
        if validation_metrics:
            logger.info(f"Holdout validation - R²: {validation_metrics.get('r2', 0):.3f}, "
                       f"RMSE: {validation_metrics.get('rmse', 0):.3f}")
    
    def add_to_holdout(self, feature_data: Dict, label: float) -> None:
        """Add sample to holdout validation set."""
        self.holdout_data.append({
            'feature_vector': feature_data.get('feature_vector', []),
            'label': label,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep holdout set size manageable
        max_holdout_size = 200
        if len(self.holdout_data) > max_holdout_size:
            self.holdout_data = self.holdout_data[-max_holdout_size:]
    
    def _prepare_data(self, 
                     feature_data: Union[List[Dict], Dict, np.ndarray],
                     target_values: Optional[List[float]],
                     feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for training or prediction."""
        try:
            # Handle single sample
            if isinstance(feature_data, dict):
                feature_data = [feature_data]
            
            # Handle different input formats
            if isinstance(feature_data[0], dict):
                # Feature data is list of dictionaries
                if 'feature_vector' in feature_data[0]:
                    # Extract feature vectors from FeatureExtractionResult objects
                    X = np.array([item['feature_vector'] for item in feature_data])
                    if not self.feature_names and 'feature_names' in feature_data[0]:
                        self.feature_names = feature_data[0]['feature_names']
                else:
                    # Feature data is list of feature dictionaries
                    if not self.feature_names:
                        if feature_names is not None:
                            self.feature_names = feature_names
                        else:
                            self.feature_names = list(feature_data[0].keys())
                    
                    X = np.array([[item[name] for name in self.feature_names] for item in feature_data])
            else:
                # Feature data is already array-like
                X = np.array(feature_data)
                if not self.feature_names:
                    if feature_names is not None:
                        self.feature_names = feature_names
                    else:
                        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # Handle target values
            y = None
            if target_values is not None:
                y = np.array(target_values)
                
                # Validate data alignment
                if X.shape[0] != y.shape[0]:
                    raise ValueError(f"Feature data ({X.shape[0]}) and target values ({y.shape[0]}) have different lengths")
            
            # Handle NaN and infinite values
            if y is not None:
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
                if not np.all(valid_mask):
                    logger.warning(f"Removing {np.sum(~valid_mask)} samples with invalid values")
                    X = X[valid_mask]
                    y = y[valid_mask]
            else:
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
                if not np.all(valid_mask):
                    logger.warning(f"Removing {np.sum(~valid_mask)} samples with invalid values")
                    X = X[valid_mask]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def _calculate_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Handle NaN values that can occur with insufficient samples
            def safe_float(value):
                """Convert to float, replacing NaN with None for JSON compatibility."""
                if np.isnan(value) or np.isinf(value):
                    return None
                return float(value)
            
            return {
                'mse': safe_float(mse),
                'mae': safe_float(mae),
                'r2': safe_float(r2),
                'rmse': safe_float(rmse)
            }
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {'mse': 0.0, 'mae': 0.0, 'r2': None, 'rmse': 0.0}
    
    def _evaluate_current_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate current model performance on given data."""
        try:
            y_pred = self.model.predict(X)
            return self._calculate_performance(y, y_pred)
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {'mse': 0.0, 'mae': 0.0, 'r2': None, 'rmse': 0.0}
    
    def get_learning_curve(self) -> pd.DataFrame:
        """Get learning curve data showing performance over time."""
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Flatten performance metrics
        performance_df = pd.json_normalize(df['performance'])
        df = pd.concat([df.drop('performance', axis=1), performance_df], axis=1)
        
        return df
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """Get statistics about model updates."""
        if not self.update_history:
            return {}
        
        update_times = [update['update_time'] for update in self.update_history]
        samples_added = [update['samples_added'] for update in self.update_history]
        
        # Calculate R² improvements
        r2_improvements = []
        for update in self.update_history:
            if 'r2' in update['improvement']:
                r2_improvements.append(update['improvement']['r2'])
        
        return {
            'total_updates': len(self.update_history),
            'total_samples_seen': self.total_samples_seen,
            'avg_update_time': float(np.mean(update_times)),
            'avg_samples_per_update': float(np.mean(samples_added)),
            'avg_r2_improvement': float(np.mean(r2_improvements)) if r2_improvements else 0.0,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None
        }
    
    def save_model(self, filepath: Path) -> None:
        """Save the online learning model to file."""
        if not self.is_initialized:
            raise ValueError("Cannot save uninitialized model")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'scaler_type': self.scaler_type,
                'target_property': self.target_property,
                'learning_rate': self.learning_rate,
                'alpha': self.alpha,
                'random_state': self.random_state,
                'creation_time': self.creation_time,
                'last_update_time': self.last_update_time,
                'total_samples_seen': self.total_samples_seen,
                'update_history': self.update_history,
                'performance_history': self.performance_history
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Online learning model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: Path) -> None:
        """Load an online learning model from file."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.scaler_type = model_data['scaler_type']
            self.target_property = model_data['target_property']
            self.learning_rate = model_data['learning_rate']
            self.alpha = model_data['alpha']
            self.random_state = model_data['random_state']
            self.creation_time = model_data['creation_time']
            self.last_update_time = model_data['last_update_time']
            self.total_samples_seen = model_data['total_samples_seen']
            self.update_history = model_data['update_history']
            self.performance_history = model_data['performance_history']
            
            self.is_initialized = True
            logger.info(f"Online learning model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': self.model_type,
            'scaler_type': self.scaler_type,
            'target_property': self.target_property,
            'is_initialized': self.is_initialized,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'creation_time': self.creation_time.isoformat(),
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'total_samples_seen': self.total_samples_seen,
            'total_updates': len(self.update_history),
            'learning_rate': self.learning_rate,
            'alpha': self.alpha,
            'update_statistics': self.get_update_statistics()
        }
    
    def _get_current_performance_metrics(self) -> Dict[str, float]:
        """Get current model performance metrics."""
        try:
            if not self.holdout_data or len(self.holdout_data) < 5:
                # Return default metrics if no holdout data
                return {
                    'mae': 0.0,
                    'rmse': 0.0,
                    'r2': 0.0,
                    'mse': 0.0
                }
            
            # Use holdout data for evaluation
            X_holdout = np.array([item['feature_vector'] for item in self.holdout_data])
            y_holdout = np.array([item['label'] for item in self.holdout_data])
            
            # Scale features
            if self.scaler is not None:
                X_holdout_scaled = self.scaler.transform(X_holdout)
            else:
                X_holdout_scaled = X_holdout
            
            # Make predictions
            y_pred = self.model.predict(X_holdout_scaled)
            
            # Calculate metrics
            return self._calculate_performance(y_holdout, y_pred)
            
        except Exception as e:
            logger.error(f"Failed to get current performance metrics: {e}")
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'r2': 0.0,
                'mse': 0.0
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all subsystems."""
        try:
            status = {
                'online_learning': {
                    'initialized': self.is_initialized,
                    'model_type': self.model_type,
                    'update_strategy': self.update_strategy,
                    'total_samples_seen': self.total_samples_seen,
                    'pending_samples': len(self.pending_samples),
                    'last_update': self.last_update_time.isoformat() if self.last_update_time else None
                },
                'subsystems': {
                    'versioning': {
                        'enabled': self.version_manager is not None,
                        'active_version': getattr(self.version_manager, 'active_version_id', None) if self.version_manager else None,
                        'total_versions': len(getattr(self.version_manager, 'versions', {})) if self.version_manager else 0
                    },
                    'quality_control': {
                        'enabled': self.quality_control is not None,
                        'total_flags': len(getattr(self.quality_control, 'quality_flags', {})) if self.quality_control else 0
                    },
                    'monitoring': {
                        'enabled': self.performance_monitor is not None,
                        'metrics_recorded': len(getattr(self.performance_monitor, 'performance_history', [])) if self.performance_monitor else 0
                    },
                    'active_learning': {
                        'enabled': self.active_learning is not None,
                        'active_queries': len(getattr(self.active_learning, 'active_queries', {})) if self.active_learning else 0,
                        'completed_queries': len(getattr(self.active_learning, 'query_results', {})) if self.active_learning else 0
                    }
                }
            }
            
            # Add performance metrics if available
            try:
                current_metrics = self._get_current_performance_metrics()
                status['performance'] = current_metrics
            except Exception as e:
                logger.warning(f"Failed to get performance metrics for status: {e}")
                status['performance'] = {}
            
            # Add alerts if monitoring is enabled
            if self.performance_monitor:
                try:
                    alerts = self.performance_monitor.get_alerts()
                    status['alerts'] = alerts
                except Exception as e:
                    logger.warning(f"Failed to get alerts for status: {e}")
                    status['alerts'] = []
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise
    
    def generate_active_learning_queries(self, 
                                       candidate_features: List[Dict[str, Any]],
                                       n_queries: int = 5) -> List[Dict[str, Any]]:
        """
        Generate active learning queries for candidate samples.
        
        Args:
            candidate_features: List of candidate samples with features
            n_queries: Number of queries to generate
            
        Returns:
            List of active learning queries
        """
        try:
            if not self.active_learning or not self.is_initialized:
                return []
            
            # Generate queries using the active learning system
            queries = self.active_learning.generate_queries(
                candidates=candidate_features,
                model=self.model,
                n_queries=n_queries
            )
            
            # Convert to dictionary format for API response
            query_dicts = [query.to_dict() for query in queries]
            
            logger.info(f"Generated {len(query_dicts)} active learning queries")
            return query_dicts
            
        except Exception as e:
            logger.error(f"Failed to generate active learning queries: {e}")
            return []
    
    def submit_active_learning_result(self,
                                    query_id: str,
                                    technician_id: str,
                                    label: float,
                                    confidence: Optional[float] = None) -> bool:
        """
        Submit result for an active learning query.
        
        Args:
            query_id: ID of the active learning query
            technician_id: ID of the technician providing the label
            label: Provided label
            confidence: Confidence in the label
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.active_learning:
                logger.warning("Active learning system not enabled")
                return False
            
            success = self.active_learning.submit_query_result(
                query_id=query_id,
                technician_id=technician_id,
                label=label,
                confidence=confidence
            )
            
            if success:
                logger.info(f"Submitted active learning result for query {query_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to submit active learning result: {e}")
            return False