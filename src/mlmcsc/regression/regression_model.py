#!/usr/bin/env python3
"""
Regression Model for Fracture Surface Analysis

This module implements the initial regression model that learns from
extracted features to predict material properties like impact energy,
toughness, or other mechanical properties.
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    specimen_id: int
    predicted_value: float
    confidence_interval: Optional[Tuple[float, float]]
    feature_importance: Optional[Dict[str, float]]
    prediction_time: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'specimen_id': self.specimen_id,
            'predicted_value': self.predicted_value,
            'confidence_interval': self.confidence_interval,
            'feature_importance': self.feature_importance,
            'prediction_time': self.prediction_time,
            'timestamp': self.timestamp
        }


@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    mse: float
    mae: float
    r2: float
    rmse: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'rmse': self.rmse,
            'cv_scores': self.cv_scores,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std
        }


class FractureRegressionModel:
    """
    Main regression model for predicting material properties from fracture surface features.
    
    Supports multiple regression algorithms and provides comprehensive model evaluation,
    feature importance analysis, and prediction confidence estimation.
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 scaler_type: str = 'standard',
                 target_property: str = 'impact_energy',
                 random_state: int = 42):
        """
        Initialize the regression model.
        
        Args:
            model_type: Type of regression model ('random_forest', 'gradient_boosting', 
                       'linear', 'ridge', 'lasso', 'elastic_net', 'svr', 'mlp')
            scaler_type: Type of feature scaler ('standard', 'robust', 'minmax', 'none')
            target_property: Name of the target property to predict
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.scaler_type = scaler_type
        self.target_property = target_property
        self.random_state = random_state
        
        # Initialize model and scaler
        self.model = self._create_model()
        self.scaler = self._create_scaler()
        self.pipeline = None
        
        # Training data and metadata
        self.feature_names = []
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = None
        
        # Model metadata
        self.creation_time = datetime.now()
        self.last_training_time = None
        self.training_samples_count = 0
        
        logger.info(f"FractureRegressionModel initialized: {model_type} with {scaler_type} scaling")
    
    def _create_model(self):
        """Create the regression model based on model_type."""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            ),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=1.0, random_state=self.random_state),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.random_state),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=1000,
                random_state=self.random_state
            )
        }
        
        if self.model_type not in models:
            logger.warning(f"Unknown model type {self.model_type}, using random_forest")
            self.model_type = 'random_forest'
        
        return models[self.model_type]
    
    def _create_scaler(self):
        """Create the feature scaler based on scaler_type."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'none': None
        }
        
        if self.scaler_type not in scalers:
            logger.warning(f"Unknown scaler type {self.scaler_type}, using standard")
            self.scaler_type = 'standard'
        
        return scalers[self.scaler_type]
    
    def prepare_training_data(self, 
                            feature_data: List[Dict],
                            target_values: List[float],
                            feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from feature extraction results.
        
        Args:
            feature_data: List of feature dictionaries or feature vectors
            target_values: List of target values (e.g., impact energy)
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (X, y) arrays ready for training
        """
        try:
            # Handle different input formats
            if isinstance(feature_data[0], dict):
                # Feature data is list of dictionaries
                if 'feature_vector' in feature_data[0]:
                    # Extract feature vectors from FeatureExtractionResult objects
                    X = np.array([item['feature_vector'] for item in feature_data])
                    if feature_names is None and 'feature_names' in feature_data[0]:
                        self.feature_names = feature_data[0]['feature_names']
                else:
                    # Feature data is list of feature dictionaries
                    if feature_names is None:
                        self.feature_names = list(feature_data[0].keys())
                    else:
                        self.feature_names = feature_names
                    
                    X = np.array([[item[name] for name in self.feature_names] for item in feature_data])
            else:
                # Feature data is already array-like
                X = np.array(feature_data)
                if feature_names is not None:
                    self.feature_names = feature_names
                else:
                    self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            y = np.array(target_values)
            
            # Validate data
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Feature data ({X.shape[0]}) and target values ({y.shape[0]}) have different lengths")
            
            if X.shape[0] == 0:
                raise ValueError("No training data provided")
            
            # Handle NaN values
            nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
            if np.any(nan_mask):
                logger.warning(f"Removing {np.sum(nan_mask)} samples with NaN values")
                X = X[~nan_mask]
                y = y[~nan_mask]
            
            # Handle infinite values
            inf_mask = np.isinf(X).any(axis=1) | np.isinf(y)
            if np.any(inf_mask):
                logger.warning(f"Removing {np.sum(inf_mask)} samples with infinite values")
                X = X[~inf_mask]
                y = y[~inf_mask]
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def train(self, 
              feature_data: List[Dict],
              target_values: List[float],
              feature_names: Optional[List[str]] = None,
              test_size: float = 0.2,
              cv_folds: int = 5,
              hyperparameter_tuning: bool = False) -> ModelPerformance:
        """
        Train the regression model.
        
        Args:
            feature_data: List of feature dictionaries or feature vectors
            target_values: List of target values
            feature_names: Optional list of feature names
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            ModelPerformance object with training metrics
        """
        start_time = time.time()
        
        try:
            # Prepare training data
            X, y = self.prepare_training_data(feature_data, target_values, feature_names)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            # Create pipeline
            if self.scaler is not None:
                self.pipeline = Pipeline([
                    ('scaler', self.scaler),
                    ('regressor', self.model)
                ])
            else:
                self.pipeline = Pipeline([
                    ('regressor', self.model)
                ])
            
            # Hyperparameter tuning
            if hyperparameter_tuning:
                logger.info("Performing hyperparameter tuning...")
                self.pipeline = self._tune_hyperparameters(X_train, y_train, cv_folds)
            
            # Train the model
            logger.info("Training the model...")
            self.pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = self.pipeline.predict(X_train)
            y_pred_test = self.pipeline.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            
            # Cross-validation
            cv_scores = cross_val_score(self.pipeline, X, y, cv=cv_folds, scoring='r2')
            
            # Create performance object
            performance = ModelPerformance(
                mse=mse,
                mae=mae,
                r2=r2,
                rmse=rmse,
                cv_scores=cv_scores.tolist(),
                cv_mean=float(np.mean(cv_scores)),
                cv_std=float(np.std(cv_scores))
            )
            
            # Update model state
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_samples_count = len(X)
            self.performance_metrics = performance
            
            # Log training history
            training_time = time.time() - start_time
            self.training_history.append({
                'timestamp': self.last_training_time,
                'samples_count': self.training_samples_count,
                'training_time': training_time,
                'performance': performance.to_dict()
            })
            
            logger.info(f"Model training completed in {training_time:.2f}s")
            logger.info(f"Performance - R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
            logger.info(f"Cross-validation R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int):
        """Perform hyperparameter tuning using GridSearchCV."""
        param_grids = {
            'random_forest': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            },
            'ridge': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.1, 0.5, 0.9]
            },
            'svr': {
                'regressor__C': [0.1, 1.0, 10.0],
                'regressor__gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'mlp': {
                'regressor__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'regressor__alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        if self.model_type in param_grids:
            param_grid = param_grids[self.model_type]
            
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
        else:
            logger.warning(f"No hyperparameter grid defined for {self.model_type}")
            return self.pipeline
    
    def predict(self, 
                feature_data: Union[Dict, List[Dict], np.ndarray],
                return_confidence: bool = False,
                return_feature_importance: bool = False) -> Union[float, List[PredictionResult]]:
        """
        Make predictions using the trained model.
        
        Args:
            feature_data: Feature data for prediction
            return_confidence: Whether to return confidence intervals
            return_feature_importance: Whether to return feature importance
            
        Returns:
            Prediction result(s)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        start_time = time.time()
        
        try:
            # Prepare input data
            if isinstance(feature_data, dict):
                # Single prediction
                if 'feature_vector' in feature_data:
                    X = feature_data['feature_vector'].reshape(1, -1)
                    specimen_id = feature_data.get('specimen_id', 0)
                else:
                    X = np.array([[feature_data[name] for name in self.feature_names]]).reshape(1, -1)
                    specimen_id = 0
                single_prediction = True
            elif isinstance(feature_data, list):
                # Multiple predictions
                if isinstance(feature_data[0], dict):
                    if 'feature_vector' in feature_data[0]:
                        X = np.array([item['feature_vector'] for item in feature_data])
                        specimen_ids = [item.get('specimen_id', i) for i, item in enumerate(feature_data)]
                    else:
                        X = np.array([[item[name] for name in self.feature_names] for item in feature_data])
                        specimen_ids = list(range(len(feature_data)))
                else:
                    X = np.array(feature_data)
                    specimen_ids = list(range(len(feature_data)))
                single_prediction = False
            else:
                # Numpy array
                X = np.array(feature_data)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                    specimen_ids = [0]
                    single_prediction = True
                else:
                    specimen_ids = list(range(X.shape[0]))
                    single_prediction = False
            
            # Make predictions
            predictions = self.pipeline.predict(X)
            prediction_time = time.time() - start_time
            
            # Calculate confidence intervals if requested
            confidence_intervals = None
            if return_confidence and hasattr(self.pipeline.named_steps['regressor'], 'predict'):
                try:
                    # For tree-based models, use prediction variance
                    if self.model_type in ['random_forest', 'gradient_boosting']:
                        confidence_intervals = self._calculate_confidence_intervals(X)
                except Exception as e:
                    logger.warning(f"Could not calculate confidence intervals: {e}")
            
            # Calculate feature importance if requested
            feature_importance = None
            if return_feature_importance:
                feature_importance = self._get_feature_importance()
            
            # Create results
            if single_prediction:
                if return_confidence or return_feature_importance:
                    return PredictionResult(
                        specimen_id=specimen_ids[0],
                        predicted_value=float(predictions[0]),
                        confidence_interval=confidence_intervals[0] if confidence_intervals else None,
                        feature_importance=feature_importance,
                        prediction_time=prediction_time,
                        timestamp=time.time()
                    )
                else:
                    return float(predictions[0])
            else:
                results = []
                for i, (specimen_id, pred) in enumerate(zip(specimen_ids, predictions)):
                    result = PredictionResult(
                        specimen_id=specimen_id,
                        predicted_value=float(pred),
                        confidence_interval=confidence_intervals[i] if confidence_intervals else None,
                        feature_importance=feature_importance if i == 0 else None,  # Only include once
                        prediction_time=prediction_time / len(predictions),
                        timestamp=time.time()
                    )
                    results.append(result)
                return results
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _calculate_confidence_intervals(self, X: np.ndarray, confidence_level: float = 0.95) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions."""
        try:
            if self.model_type == 'random_forest':
                # Use prediction variance from individual trees
                regressor = self.pipeline.named_steps['regressor']
                tree_predictions = np.array([tree.predict(X) for tree in regressor.estimators_])
                
                # Calculate prediction variance
                pred_mean = np.mean(tree_predictions, axis=0)
                pred_std = np.std(tree_predictions, axis=0)
                
                # Calculate confidence intervals
                z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
                margin = z_score * pred_std
                
                confidence_intervals = [(float(mean - margin[i]), float(mean + margin[i])) 
                                      for i, mean in enumerate(pred_mean)]
                
                return confidence_intervals
            else:
                # Fallback: use simple standard deviation estimate
                predictions = self.pipeline.predict(X)
                std_estimate = np.std(predictions) if len(predictions) > 1 else 0.1
                margin = 1.96 * std_estimate
                
                return [(float(pred - margin), float(pred + margin)) for pred in predictions]
                
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return [(0.0, 0.0)] * X.shape[0]
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model."""
        try:
            regressor = self.pipeline.named_steps['regressor']
            
            if hasattr(regressor, 'feature_importances_'):
                # Tree-based models
                importances = regressor.feature_importances_
            elif hasattr(regressor, 'coef_'):
                # Linear models
                importances = np.abs(regressor.coef_)
            else:
                return None
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, importance in enumerate(importances):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] = float(importance)
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return None
    
    def save_model(self, filepath: Path) -> None:
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            model_data = {
                'pipeline': self.pipeline,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'scaler_type': self.scaler_type,
                'target_property': self.target_property,
                'random_state': self.random_state,
                'creation_time': self.creation_time,
                'last_training_time': self.last_training_time,
                'training_samples_count': self.training_samples_count,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics.to_dict() if self.performance_metrics else None
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: Path) -> None:
        """Load a trained model from file."""
        try:
            model_data = joblib.load(filepath)
            
            self.pipeline = model_data['pipeline']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.scaler_type = model_data['scaler_type']
            self.target_property = model_data['target_property']
            self.random_state = model_data['random_state']
            self.creation_time = model_data['creation_time']
            self.last_training_time = model_data['last_training_time']
            self.training_samples_count = model_data['training_samples_count']
            self.training_history = model_data['training_history']
            
            if model_data['performance_metrics']:
                self.performance_metrics = ModelPerformance(**model_data['performance_metrics'])
            
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': self.model_type,
            'scaler_type': self.scaler_type,
            'target_property': self.target_property,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'creation_time': self.creation_time.isoformat(),
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_samples_count': self.training_samples_count,
            'performance_metrics': self.performance_metrics.to_dict() if self.performance_metrics else None,
            'training_history_count': len(self.training_history)
        }