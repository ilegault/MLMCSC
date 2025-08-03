#!/usr/bin/env python3
"""
Shiny Region-Based Classifier

Complete classification system based on shiny region analysis.
This replaces your current system with one focused on the actual indicators.
"""

import cv2
import numpy as np
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from improved_fracture_detector import ImprovedFractureSurfaceDetector
from shiny_region_analyzer import ShinyRegionAnalyzer, ShinyRegionFeatures

logger = logging.getLogger(__name__)


class ShinyRegionBasedClassifier:
    """Complete classification system based on shiny region analysis."""

    def __init__(self):
        self.detector = ImprovedFractureSurfaceDetector()
        self.analyzer = ShinyRegionAnalyzer()
        self.model = None
        self.scaler = None
        self.is_trained = False

        logger.info("Shiny Region-Based Classifier initialized")

    def process_image(self, image: np.ndarray) -> Optional[ShinyRegionFeatures]:
        """Process a single image and extract shiny region features."""

        # Detect full fracture surface
        detection_result = self.detector.detect_full_fracture_surface(image)
        if detection_result is None:
            logger.warning("Could not detect fracture surface")
            return None

        surface_mask, surface_roi = detection_result

        # Extract shiny region features
        features = self.analyzer.extract_shiny_region_features(surface_mask, surface_roi)

        return features

    def create_training_dataset(self, training_data: Dict[float, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset from labeled images.

        Args:
            training_data: Dict mapping shear percentages to image file paths

        Returns:
            Tuple of (feature_matrix, label_array)
        """
        features_list = []
        labels_list = []

        logger.info("Creating training dataset with shiny region features...")

        for shear_percent, image_paths in training_data.items():
            logger.info(f"Processing {len(image_paths)} images for {shear_percent}% shear")

            for image_path in image_paths:
                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Could not load image: {image_path}")
                        continue

                    # Extract features
                    features = self.process_image(image)
                    if features is None:
                        logger.warning(f"Could not extract features from: {image_path}")
                        continue

                    features_list.append(features.to_array())
                    labels_list.append(shear_percent)

                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue

        if not features_list:
            raise ValueError("No valid features extracted from training data")

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Shear range: {np.min(y):.1f}% - {np.max(y):.1f}%")

        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the shiny region-based regression model."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_absolute_error, r2_score

        logger.info("Training shiny region-based model...")

        # Handle NaN values
        nan_mask = np.isnan(X).any(axis=1)
        if np.any(nan_mask):
            logger.warning(f"Removing {np.sum(nan_mask)} samples with NaN values")
            X = X[~nan_mask]
            y = y[~nan_mask]

        if X.shape[0] == 0:
            raise ValueError("No valid samples remaining after NaN removal")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create model with conservative parameters
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores

        # Train final model
        self.model.fit(X_scaled, y)

        # Training metrics
        y_pred_train = self.model.predict(X_scaled)
        train_mae = mean_absolute_error(y, y_pred_train)
        train_r2 = r2_score(y, y_pred_train)

        self.is_trained = True

        # Feature importance analysis
        feature_names = ShinyRegionFeatures.get_feature_names()
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        results = {
            'train_mae': float(train_mae),
            'train_r2': float(train_r2),
            'cv_mae_mean': float(np.mean(cv_mae)),
            'cv_mae_std': float(np.std(cv_mae)),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'feature_importance': sorted_importance
        }

        logger.info(f"Training completed!")
        logger.info(f"Training MAE: {train_mae:.2f}%")
        logger.info(f"Training R²: {train_r2:.3f}")
        logger.info(f"CV MAE: {np.mean(cv_mae):.2f} ± {np.std(cv_mae):.2f}%")

        return results

    def predict_shear_percentage(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict shear percentage from image using shiny region analysis."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        features = self.process_image(image)
        if features is None:
            return {
                'success': False,
                'error': 'Could not extract features from image',
                'prediction': None,
                'confidence': 0.0
            }

        try:
            # Scale features and predict
            feature_vector = features.to_array().reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)

            prediction = self.model.predict(feature_vector_scaled)[0]
            prediction = np.clip(prediction, 0.0, 100.0)  # Constrain to valid range

            # Calculate prediction confidence using tree variance
            if hasattr(self.model, 'estimators_'):
                tree_predictions = [tree.predict(feature_vector_scaled)[0] for tree in self.model.estimators_]
                prediction_std = np.std(tree_predictions)
                confidence = max(0.0, 1.0 - (prediction_std / 50.0))  # Normalize to 0-1
            else:
                confidence = 0.5  # Default confidence

            return {
                'success': True,
                'prediction': float(prediction),
                'confidence': float(confidence),
                'features': features.__dict__,
                'prediction_category': self._categorize_shear(prediction)
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {e}',
                'prediction': None,
                'confidence': 0.0
            }

    def _categorize_shear(self, prediction: float) -> str:
        """Categorize shear percentage into descriptive ranges."""
        if prediction <= 15:
            return "Low Shear (Brittle)"
        elif prediction <= 35:
            return "Low-Medium Shear"
        elif prediction <= 65:
            return "Medium Shear (Mixed)"
        elif prediction <= 85:
            return "High Shear"
        else:
            return "Very High Shear (Ductile)"

    def save_model(self, filepath: str) -> bool:
        """Save trained model to file."""
        if not self.is_trained:
            logger.error("No trained model to save")
            return False

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': ShinyRegionFeatures.get_feature_names(),
                'model_type': 'shiny_region_based'
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = True

            logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False