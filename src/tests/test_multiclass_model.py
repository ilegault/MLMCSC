#!/usr/bin/env python3
"""
Batch Model Tester

Tests both YOLO detection and shear regression models on sample images from a directory.
Perfect for testing when your camera is faulty.

RECENT UPDATES:
- Updated to use robust_shear_debug_model.pkl (new classification model)
- Added grayscale conversion for shear prediction (model was trained on grayscale images)
- YOLO detection still uses original color images
- Shear prediction now uses grayscale converted images
"""

import cv2
import numpy as np
import pickle
import json
import logging
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import classification modules
try:
    # Try direct imports from the classification directory
    classification_path = src_path / "mlmcsc" / "classification"
    sys.path.insert(0, str(classification_path))
    
    from improved_fracture_detector import ImprovedFractureSurfaceDetector
    from shiny_region_analyzer import ShinyRegionAnalyzer, ShinyRegionFeatures
    CLASSIFICATION_AVAILABLE = True
    print("✅ Classification modules available")
except ImportError as e:
    print(f"⚠️ Classification modules not available: {e}")
    print("Will use fallback prediction method")
    CLASSIFICATION_AVAILABLE = False

# Import improved shear regression model
try:
    # Add the regression model path
    regression_path = src_path.parent / "tools" / "regression_model"
    sys.path.insert(0, str(regression_path))
    
    from shear_surface_classifer import ImprovedShearRegressionModel, ImprovedShearFeatures
    IMPROVED_SHEAR_AVAILABLE = True
    print("✅ Improved shear regression model available")
except ImportError as e:
    print(f"⚠️ Improved shear regression model not available: {e}")
    IMPROVED_SHEAR_AVAILABLE = False

# YOLO imports
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
    print("✅ YOLO available")
except ImportError:
    print("⚠️ YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class ImprovedShearModelWrapper:
    """Wrapper for the new ImprovedShearRegressionModel saved as dictionary."""
    
    def __init__(self, model_dict: Dict[str, Any]):
        self.model_dict = model_dict
        # Handle both 'model' (singular) and 'models' (plural) formats
        self.model = model_dict.get('model') or model_dict.get('models')
        self.scaler = model_dict.get('scaler')
        self.feature_names = model_dict.get('feature_names', [])
        self.model_type = model_dict.get('model_type', 'ensemble')
        self.version = model_dict.get('version', '1.0')
        
        # Feature mapping for robust_shear_debug_model compatibility
        self.expected_features = [
            'surface_roughness', 'texture_energy', 'edge_density', 'edge_strength',
            'bright_area_ratio', 'dark_area_ratio', 'global_contrast', 'local_contrast_mean',
            'high_freq_ratio', 'intensity_skewness', 'intensity_kurtosis'
        ]
        
        # Initialize the improved shear regression model components if available
        if IMPROVED_SHEAR_AVAILABLE:
            # Create a new ImprovedShearRegressionModel instance
            self.shear_model = ImprovedShearRegressionModel(model_type=self.model_type)
            # Set the trained components
            self.shear_model.model = self.model
            self.shear_model.scaler = self.scaler
            self.shear_model.is_trained = True
            logger.info("✅ ImprovedShearRegressionModel wrapper initialized")
        else:
            self.shear_model = None
            logger.warning("⚠️ ImprovedShearRegressionModel not available - using fallback")
        
        logger.info(f"ImprovedShearModelWrapper initialized with version {self.version}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Feature count: {len(self.feature_names)}")
    

    
    def predict_shear_percentage(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict shear percentage from image using the improved shear model.
        """
        try:
            # Check if this is the robust_shear_debug_model (version 3.0_debug) that needs feature mapping
            if self.version == '3.0_debug' or len(self.feature_names) == 11:
                # Use direct prediction with feature mapping for robust_shear_debug_model
                logger.info("Using feature mapping for robust_shear_debug_model")
                return self._predict_with_feature_mapping(image)
            elif self.shear_model is not None and IMPROVED_SHEAR_AVAILABLE:
                # Use the proper ImprovedShearRegressionModel for other versions
                return self.shear_model.predict_shear_percentage(image)
            else:
                # Fallback to simple prediction
                return self._fallback_prediction(image)
                
        except Exception as e:
            logger.error(f"Improved shear model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Improved shear model prediction failed: {e}',
                'shear_percentage': None,
                'confidence': 0.0
            }
    
    def _predict_with_feature_mapping(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict using the robust feature extraction pipeline that matches the training.
        """
        try:
            # Use the same feature extraction pipeline as the training
            features_array = self._extract_robust_features(image)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_array.reshape(1, -1))
            else:
                features_scaled = features_array.reshape(1, -1)
                logger.warning("No scaler available - using raw features")
            
            # Make prediction using ensemble approach
            if self.model is None:
                raise ValueError("No model available for prediction")
            
            # Handle ensemble models (dictionary of models)
            if isinstance(self.model, dict):
                # Use ensemble prediction (average of all models)
                predictions = []
                for model_name, model in self.model.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(features_scaled)[0]
                        predictions.append(pred)
                        logger.debug(f"Model {model_name} prediction: {pred:.2f}")
                
                if predictions:
                    prediction = np.mean(predictions)  # Ensemble average
                    logger.info(f"Ensemble prediction from {len(predictions)} models: {prediction:.2f}")
                else:
                    raise ValueError("No valid models found in ensemble")
            else:
                # Single model prediction
                prediction = self.model.predict(features_scaled)[0]
            
            prediction = np.clip(prediction, 0.0, 100.0)  # Constrain to valid range
            
            # Calculate confidence if possible
            confidence = self._calculate_confidence(features_scaled)
            
            return {
                'success': True,
                'shear_percentage': float(prediction),
                'confidence': float(confidence),
                'features_extracted_correctly': True,
                'feature_count': features_array.shape[0],
                'prediction_category': self._categorize_shear(prediction),
                'method': f'robust_debug_model_{self.model_type}'
            }
            
        except Exception as e:
            logger.error(f"Feature mapping prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction(image)
    
    def _extract_robust_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract robust features using the same pipeline as the training.
        This matches the RobustFeatureExtractor from shear_surface_debug.py
        """
        from scipy import stats
        
        # Preprocess image (same as training)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to standard size (256x256 as in training)
        target_size = (256, 256)
        resized = cv2.resize(gray, target_size)
        
        # Apply CLAHE for contrast normalization (same as training)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(resized)
        
        # Extract the same 11 features as in training
        
        # 1. Surface roughness
        surface_roughness = np.std(processed)
        
        # 2. Texture energy (simplified GLCM)
        texture_energy = self._compute_texture_energy(processed)
        
        # 3. Edge density
        edges = cv2.Canny(processed, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 4. Edge strength
        grad_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_strength = np.mean(grad_mag)
        
        # 5-6. Bright and dark area ratios
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_pixels = np.sum(binary > 0)
        dark_pixels = np.sum(binary == 0)
        total_pixels = processed.size
        bright_area_ratio = bright_pixels / total_pixels
        dark_area_ratio = dark_pixels / total_pixels
        
        # 7. Global contrast
        p5, p95 = np.percentile(processed, [5, 95])
        global_contrast = p95 - p5
        
        # 8. Local contrast mean
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        local_mean = cv2.filter2D(processed.astype(float), -1, kernel)
        local_var = cv2.filter2D((processed.astype(float) - local_mean) ** 2, -1, kernel)
        local_contrast_mean = np.mean(np.sqrt(local_var))
        
        # 9. High frequency ratio
        f_transform = np.fft.fft2(processed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        center_y, center_x = processed.shape[0] // 2, processed.shape[1] // 2
        y, x = np.ogrid[:processed.shape[0], :processed.shape[1]]
        mask_low = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(processed.shape) // 4) ** 2
        
        low_freq_energy = np.sum(magnitude_spectrum[mask_low])
        high_freq_energy = np.sum(magnitude_spectrum[~mask_low])
        total_energy = low_freq_energy + high_freq_energy
        high_freq_ratio = high_freq_energy / (total_energy + 1e-6)
        
        # 10-11. Statistical moments
        intensity_skewness = stats.skew(processed.flatten())
        intensity_kurtosis = stats.kurtosis(processed.flatten())
        
        # Return as array matching the training feature order
        return np.array([
            surface_roughness, texture_energy,
            edge_density, edge_strength,
            bright_area_ratio, dark_area_ratio,
            global_contrast, local_contrast_mean,
            high_freq_ratio,
            intensity_skewness, intensity_kurtosis
        ])
    
    def _compute_texture_energy(self, image: np.ndarray) -> float:
        """Compute texture energy using simplified GLCM."""
        # Quantize to 16 levels for faster computation
        levels = 16
        quantized = (image // (256 // levels)).astype(np.uint8)
        
        # Compute co-occurrence matrix for horizontal direction
        glcm = np.zeros((levels, levels))
        for i in range(image.shape[0]):
            for j in range(image.shape[1] - 1):
                glcm[quantized[i, j], quantized[i, j + 1]] += 1
        
        # Normalize
        glcm = glcm / (np.sum(glcm) + 1e-6)
        
        # Return energy (sum of squared elements)
        return np.sum(glcm ** 2)
    
    def _calculate_confidence(self, feature_vector_scaled: np.ndarray) -> float:
        """Calculate prediction confidence."""
        try:
            # Handle ensemble models (dictionary of models)
            if isinstance(self.model, dict):
                # Calculate confidence based on agreement between ensemble models
                predictions = []
                for model_name, model in self.model.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(feature_vector_scaled)[0]
                        predictions.append(pred)
                
                if len(predictions) > 1:
                    prediction_std = np.std(predictions)
                    confidence = max(0.0, 1.0 - (prediction_std / 50.0))  # Normalize to 0-1
                    return confidence
                else:
                    return 0.7
            
            # For single ensemble models (like RandomForest), use tree variance
            elif hasattr(self.model, 'estimators_'):
                tree_predictions = [tree.predict(feature_vector_scaled)[0] for tree in self.model.estimators_]
                prediction_std = np.std(tree_predictions)
                confidence = max(0.0, 1.0 - (prediction_std / 50.0))  # Normalize to 0-1
                return confidence
            else:
                # For other models, return default confidence
                return 0.7
        except Exception:
            return 0.5
    
    def _categorize_shear(self, prediction: float) -> str:
        """Categorize shear percentage into descriptive ranges."""
        if prediction <= 15:
            return "Low Shear (Brittle)"
        elif prediction <= 35:
            return "Low-Medium Shear"
        elif prediction <= 65:
            return "Medium Shear"
        elif prediction <= 85:
            return "Medium-High Shear"
        else:
            return "High Shear (Ductile)"
    
    def _fallback_prediction(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback prediction when improved model is not available."""
        try:
            # Simple image-based heuristic as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate basic image statistics as a rough estimate
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Very rough heuristic: brighter, more uniform surfaces tend to be more ductile
            normalized_intensity = (mean_intensity - 50) / 100
            uniformity = max(0, 1 - (std_intensity / 100))
            
            # Combine factors for rough estimate
            rough_estimate = 50 + (normalized_intensity * 30) + (uniformity * 20)
            rough_estimate = np.clip(rough_estimate, 0, 100)
            
            logger.warning("Using fallback prediction for improved shear model - not scientifically accurate!")
            
            return {
                'success': True,
                'shear_percentage': float(rough_estimate),
                'confidence': 0.3,  # Low confidence for fallback
                'method': 'improved_fallback_heuristic',
                'note': 'Fallback prediction used - improved shear model components not available'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Improved model fallback prediction failed: {e}',
                'shear_percentage': None,
                'confidence': 0.0
            }


class DictModelWrapper:
    """Wrapper for dictionary-based models to provide predict_shear_percentage interface."""
    
    def __init__(self, model_dict: Dict[str, Any]):
        self.model_dict = model_dict
        # Handle both 'model' (singular) and 'models' (plural) formats
        self.model = model_dict.get('model') or model_dict.get('models')
        self.scaler = model_dict.get('scaler')
        self.feature_names = model_dict.get('feature_names', [])
        self.model_type = model_dict.get('model_type', 'unknown')
        
        # Initialize feature extraction components if available
        if CLASSIFICATION_AVAILABLE:
            self.detector = ImprovedFractureSurfaceDetector()
            self.analyzer = ShinyRegionAnalyzer()
            logger.info("✅ Feature extraction components initialized")
        else:
            self.detector = None
            self.analyzer = None
            logger.warning("⚠️ Classification modules not available - using fallback prediction")
        
        logger.info(f"DictModelWrapper initialized with keys: {list(model_dict.keys())}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Feature count: {len(self.feature_names)}")
    
    def predict_shear_percentage(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict shear percentage from image using dictionary-based model.
        """
        try:
            if not CLASSIFICATION_AVAILABLE or self.detector is None or self.analyzer is None:
                # Fallback to simple prediction if classification modules not available
                return self._fallback_prediction(image)
            
            # Extract features using the full pipeline
            features = self._extract_features(image)
            if features is None:
                return {
                    'success': False,
                    'error': 'Could not extract features from image',
                    'shear_percentage': None,
                    'confidence': 0.0
                }
            
            # Make prediction using the model
            return self._make_prediction(features)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Dictionary model prediction failed: {e}',
                'shear_percentage': None,
                'confidence': 0.0
            }
    
    def _extract_features(self, image: np.ndarray) -> Optional[ShinyRegionFeatures]:
        """Extract shiny region features from image."""
        try:
            # Detect fracture surface
            detection_result = self.detector.detect_full_fracture_surface(image)
            if detection_result is None:
                logger.warning("Could not detect fracture surface")
                return None
            
            surface_mask, surface_roi = detection_result
            
            # Extract shiny region features
            features = self.analyzer.extract_shiny_region_features(surface_mask, surface_roi)
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _make_prediction(self, features: ShinyRegionFeatures) -> Dict[str, Any]:
        """Make prediction using extracted features."""
        try:
            # Convert features to array
            feature_vector = features.to_array().reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                feature_vector_scaled = self.scaler.transform(feature_vector)
            else:
                feature_vector_scaled = feature_vector
                logger.warning("No scaler available - using raw features")
            
            # Make prediction
            if self.model is None:
                raise ValueError("No model available for prediction")
            
            prediction = self.model.predict(feature_vector_scaled)[0]
            prediction = np.clip(prediction, 0.0, 100.0)  # Constrain to valid range
            
            # Calculate confidence if possible
            confidence = self._calculate_confidence(feature_vector_scaled)
            
            return {
                'success': True,
                'shear_percentage': float(prediction),
                'confidence': float(confidence),
                'features': features.__dict__,
                'prediction_category': self._categorize_shear(prediction),
                'method': f'dictionary_model_{self.model_type}'
            }
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {
                'success': False,
                'error': f'Model prediction failed: {e}',
                'shear_percentage': None,
                'confidence': 0.0
            }
    
    def _calculate_confidence(self, feature_vector_scaled: np.ndarray) -> float:
        """Calculate prediction confidence."""
        try:
            # For ensemble models (like RandomForest), use tree variance
            if hasattr(self.model, 'estimators_'):
                tree_predictions = [tree.predict(feature_vector_scaled)[0] for tree in self.model.estimators_]
                prediction_std = np.std(tree_predictions)
                confidence = max(0.0, 1.0 - (prediction_std / 50.0))  # Normalize to 0-1
                return confidence
            else:
                # For other models, return default confidence
                return 0.7
        except Exception:
            return 0.5
    
    def _categorize_shear(self, prediction: float) -> str:
        """Categorize shear percentage into descriptive ranges."""
        if prediction <= 15:
            return "Low Shear (Brittle)"
        elif prediction <= 35:
            return "Low-Medium Shear"
        elif prediction <= 65:
            return "Medium Shear"
        elif prediction <= 85:
            return "Medium-High Shear"
        else:
            return "High Shear (Ductile)"
    
    def _fallback_prediction(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback prediction when classification modules are not available."""
        try:
            # Simple image-based heuristic as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate basic image statistics as a rough estimate
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Very rough heuristic: brighter, more uniform surfaces tend to be more ductile
            # This is a simplified approximation and not scientifically accurate
            normalized_intensity = (mean_intensity - 50) / 100  # Normalize around 50
            uniformity = max(0, 1 - (std_intensity / 100))  # Higher uniformity = lower std
            
            # Combine factors for rough estimate
            rough_estimate = 50 + (normalized_intensity * 30) + (uniformity * 20)
            rough_estimate = np.clip(rough_estimate, 0, 100)
            
            logger.warning("Using fallback prediction - not scientifically accurate!")
            
            return {
                'success': True,
                'shear_percentage': float(rough_estimate),
                'confidence': 0.3,  # Low confidence for fallback
                'method': 'fallback_heuristic',
                'note': 'Fallback prediction used - install classification modules for accurate results'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Fallback prediction failed: {e}',
                'shear_percentage': None,
                'confidence': 0.0
            }


class BatchModelTester:
    """Test both models on sample images from a directory."""

    def __init__(self, yolo_model_path: str, regression_model_path: str):
        """
        Initialize the batch model tester.

        Args:
            yolo_model_path: Path to YOLO model (.pt file) - can be relative to project root
            regression_model_path: Path to regression model (.pkl file) - can be relative to project root
        """
        # Convert relative paths to absolute paths from project root
        self.yolo_model_path = self._resolve_path(yolo_model_path)
        self.regression_model_path = self._resolve_path(regression_model_path)

        # Models
        self.yolo_model = None
        self.shear_classifier = None

        # Results storage
        self.test_results = []

        logger.info("Batch Model Tester initialized")

    def _resolve_path(self, path: str) -> str:
        """
        Resolve path relative to project root if it's not absolute.
        
        Args:
            path: Path string (can be relative or absolute)
            
        Returns:
            Absolute path string
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        else:
            return str(PROJECT_ROOT / path_obj)

    def load_models(self) -> bool:
        """Load both YOLO and regression models."""

        # Load YOLO model
        if not self._load_yolo_model():
            return False

        # Load regression model
        if not self._load_regression_model():
            return False

        logger.info("✅ All models loaded successfully")
        return True

    def _load_yolo_model(self) -> bool:
        """Load YOLO detection model."""
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available")
            return False

        try:
            if not Path(self.yolo_model_path).exists():
                logger.error(f"YOLO model not found: {self.yolo_model_path}")
                return False

            self.yolo_model = YOLO(self.yolo_model_path)
            logger.info("✅ YOLO model loaded")
            logger.info(f"YOLO classes: {list(self.yolo_model.names.values())}")
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def _load_regression_model(self) -> bool:
        """Load regression model from pickle file."""

        if not Path(self.regression_model_path).exists():
            logger.error(f"Regression model not found: {self.regression_model_path}")
            return False

        try:
            with open(self.regression_model_path, 'rb') as f:
                model_data = pickle.load(f)

            logger.info(f"Loaded model file. Type: {type(model_data)}")

            # Check if it's a ShinyRegionBasedClassifier object
            if hasattr(model_data, 'predict_shear_percentage'):
                self.shear_classifier = model_data
                logger.info("✅ Loaded ShinyRegionBasedClassifier object")
            elif isinstance(model_data, dict):
                # Handle dictionary format - might contain model components
                logger.info(f"Loaded dictionary with keys: {list(model_data.keys())}")
                
                # Check if it's the new improved shear model format
                if 'version' in model_data and model_data.get('version') == '2.0':
                    # This is the new ImprovedShearRegressionModel format
                    self.shear_classifier = ImprovedShearModelWrapper(model_data)
                    logger.info("✅ Created wrapper for ImprovedShearRegressionModel (v2.0)")
                elif 'model' in model_data and 'scaler' in model_data and 'feature_names' in model_data:
                    # This looks like the improved shear model format (even without version)
                    self.shear_classifier = ImprovedShearModelWrapper(model_data)
                    logger.info("✅ Created wrapper for ImprovedShearRegressionModel (detected format)")
                elif 'models' in model_data and 'scaler' in model_data and 'feature_names' in model_data:
                    # This is the new robust shear debug model format with 'models' (plural)
                    self.shear_classifier = ImprovedShearModelWrapper(model_data)
                    logger.info("✅ Created wrapper for robust shear debug model (models plural format)")
                elif 'model' in model_data or 'models' in model_data or 'classifier' in model_data:
                    # Create a simple wrapper for dictionary-based models
                    self.shear_classifier = DictModelWrapper(model_data)
                    logger.info("✅ Created wrapper for dictionary-based model")
                else:
                    logger.error(f"Dictionary model format not supported. Keys: {list(model_data.keys())}")
                    logger.info("Expected dictionary with 'model' key (ImprovedShearRegressionModel) or 'classifier' key (legacy)")
                    return False
            else:
                logger.error(f"Unknown model format: {type(model_data)}")
                logger.info("Expected model with 'predict_shear_percentage' method or dictionary format")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to load regression model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_image_directory(self, image_dir: str, max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Test both models on all images in a directory.

        Args:
            image_dir: Directory containing test images (can be relative to project root)
            max_images: Maximum number of images to process (None for all)

        Returns:
            Dictionary with test results and statistics
        """
        # Resolve image directory path relative to project root
        resolved_image_dir = self._resolve_path(image_dir)
        image_path = Path(resolved_image_dir)

        if not image_path.exists():
            return {'error': f'Directory not found: {image_dir}'}

        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []

        for ext in image_extensions:
            image_files.extend(image_path.glob(f'*{ext}'))
            image_files.extend(image_path.glob(f'*{ext.upper()}'))

        if not image_files:
            return {'error': f'No image files found in {image_dir}'}

        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]

        logger.info(f"Found {len(image_files)} images to test")

        # Process each image
        results = []
        start_time = time.time()

        for i, image_file in enumerate(image_files):
            print(f"Processing {i + 1}/{len(image_files)}: {image_file.name}")

            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(f"Could not load image: {image_file}")
                continue

            # Convert to grayscale for shear prediction (model was trained on grayscale)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Run YOLO detection (uses original color image)
            detections = self._detect_fractures(image)

            # Run shear prediction (uses grayscale image)
            shear_result = self._predict_shear(gray_image)

            # Store results
            result = {
                'filename': image_file.name,
                'filepath': str(image_file),
                'image_shape': image.shape,  # Original color image shape
                'gray_image_shape': gray_image.shape,  # Grayscale image shape used for shear prediction
                'detections': detections,
                'shear_prediction': shear_result,
                'processing_time': time.time() - start_time
            }

            results.append(result)
            self.test_results.append(result)

        # Calculate statistics
        processing_time = time.time() - start_time
        stats = self._calculate_statistics(results, processing_time)

        return {
            'results': results,
            'statistics': stats,
            'total_images': len(results),
            'processing_time': processing_time
        }

    def _detect_fractures(self, image: np.ndarray) -> List[Dict]:
        """Detect fractures using YOLO model."""
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model(image, conf=0.3, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        class_name = self.yolo_model.names.get(cls, f"class_{cls}")

                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': class_name,
                            'class_id': cls
                        })

            return detections

        except Exception as e:
            logger.debug(f"YOLO detection failed: {e}")
            return []

    def _predict_shear(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict shear percentage using regression model."""
        if self.shear_classifier is None:
            return {'success': False, 'error': 'No shear classifier loaded'}

        try:
            result = self.shear_classifier.predict_shear_percentage(image)
            return result

        except Exception as e:
            return {'success': False, 'error': f'Shear prediction failed: {e}'}

    def _calculate_statistics(self, results: List[Dict], processing_time: float) -> Dict[str, Any]:
        """Calculate test statistics."""

        total_images = len(results)
        images_with_detections = sum(1 for r in results if r['detections'])
        successful_predictions = sum(1 for r in results if r['shear_prediction'].get('success'))

        # Shear prediction statistics
        shear_values = []
        for result in results:
            shear_pred = result['shear_prediction']
            if shear_pred.get('success') and 'shear_percentage' in shear_pred:
                shear_values.append(shear_pred['shear_percentage'])

        stats = {
            'total_images': total_images,
            'images_with_detections': images_with_detections,
            'detection_rate': images_with_detections / total_images if total_images > 0 else 0,
            'successful_predictions': successful_predictions,
            'prediction_rate': successful_predictions / total_images if total_images > 0 else 0,
            'avg_processing_time': processing_time / total_images if total_images > 0 else 0,
            'fps': total_images / processing_time if processing_time > 0 else 0
        }

        if shear_values:
            stats['shear_statistics'] = {
                'count': len(shear_values),
                'mean': np.mean(shear_values),
                'std': np.std(shear_values),
                'min': np.min(shear_values),
                'max': np.max(shear_values),
                'median': np.median(shear_values)
            }

        return stats

    def save_results(self, filename: str = None) -> str:
        """Save test results to JSON file."""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_test_results_{timestamp}.json"
        
        # Ensure results directory exists and save there
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        
        # If filename is not absolute, save to results directory
        if not Path(filename).is_absolute():
            filename = str(results_dir / filename)

        # Prepare results for JSON serialization
        serializable_results = []
        for result in self.test_results:
            serializable_result = result.copy()
            # Convert numpy arrays to lists
            if 'image_shape' in serializable_result:
                serializable_result['image_shape'] = list(serializable_result['image_shape'])
            if 'gray_image_shape' in serializable_result:
                serializable_result['gray_image_shape'] = list(serializable_result['gray_image_shape'])
            serializable_results.append(serializable_result)

        output_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'yolo_model_path': self.yolo_model_path,
                'regression_model_path': self.regression_model_path,
                'total_results': len(serializable_results)
            },
            'results': serializable_results
        }

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Results saved to: {filename}")
        return filename

    def create_summary_visualization(self, save_path: str = None):
        """Create visualization of test results."""

        if not self.test_results:
            logger.warning("No test results to visualize")
            return

        # Extract data for visualization
        shear_values = []
        detection_counts = []
        filenames = []

        for result in self.test_results:
            filenames.append(result['filename'])
            detection_counts.append(len(result['detections']))

            shear_pred = result['shear_prediction']
            if shear_pred.get('success') and 'shear_percentage' in shear_pred:
                shear_values.append(shear_pred['shear_percentage'])
            else:
                shear_values.append(None)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Batch Model Test Results', fontsize=16)

        # Plot 1: Detection counts per image
        axes[0, 0].bar(range(len(detection_counts)), detection_counts)
        axes[0, 0].set_title('YOLO Detections per Image')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Number of Detections')

        # Plot 2: Shear predictions
        valid_shear = [s for s in shear_values if s is not None]
        if valid_shear:
            axes[0, 1].hist(valid_shear, bins=20, alpha=0.7, color='blue')
            axes[0, 1].set_title('Shear Percentage Distribution')
            axes[0, 1].set_xlabel('Shear Percentage (%)')
            axes[0, 1].set_ylabel('Frequency')

        # Plot 3: Shear predictions over time
        if valid_shear:
            valid_indices = [i for i, s in enumerate(shear_values) if s is not None]
            axes[1, 0].plot(valid_indices, valid_shear, 'o-', alpha=0.7)
            axes[1, 0].set_title('Shear Predictions by Image')
            axes[1, 0].set_xlabel('Image Index')
            axes[1, 0].set_ylabel('Shear Percentage (%)')

        # Plot 4: Success rates
        total_images = len(self.test_results)
        successful_detections = sum(1 for r in self.test_results if r['detections'])
        successful_predictions = sum(1 for r in self.test_results if r['shear_prediction'].get('success'))

        categories = ['YOLO Detections', 'Shear Predictions']
        success_rates = [
            successful_detections / total_images * 100,
            successful_predictions / total_images * 100
        ]

        axes[1, 1].bar(categories, success_rates, color=['green', 'blue'], alpha=0.7)
        axes[1, 1].set_title('Model Success Rates')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_ylim(0, 100)

        # Add value labels on bars
        for i, v in enumerate(success_rates):
            axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"batch_test_visualization_{timestamp}.png"
        
        # Ensure results directory exists and save there
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        
        # If save_path is not absolute, save to results directory
        if not Path(save_path).is_absolute():
            save_path = str(results_dir / save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Visualization saved to: {save_path}")

    def run_test(self, image_directory: str, max_images: Optional[int] = None, save_results: bool = True):
        """
        Main method to run the complete test.

        Args:
            image_directory: Directory containing test images
            max_images: Maximum number of images to process
            save_results: Whether to save results to file
        """

        print("🔬 BATCH MODEL TESTER")
        print("=" * 50)
        print(f"YOLO Model: {self.yolo_model_path}")
        print(f"Regression Model: {self.regression_model_path}")
        print(f"Test Directory: {self._resolve_path(image_directory)}")
        print(f"Project Root: {PROJECT_ROOT}")
        print()

        # Load models
        print("Loading models...")
        if not self.load_models():
            print("❌ Failed to load models")
            return None

        # Run tests
        print("Running tests...")
        results = self.test_image_directory(image_directory, max_images)

        if 'error' in results:
            print(f"❌ Testing failed: {results['error']}")
            return None

        # Print summary
        self._print_summary(results['statistics'])

        # Save results
        if save_results:
            results_file = self.save_results()
            print(f"\n💾 Results saved to: {results_file}")

        # Create visualization
        print("\n📊 Creating visualization...")
        self.create_summary_visualization()

        return results



    def _print_summary(self, stats: Dict[str, Any]):
        """Print test summary."""

        print("\n📊 TEST SUMMARY")
        print("=" * 30)
        print(f"Total Images Processed: {stats['total_images']}")
        print(f"Images with YOLO Detections: {stats['images_with_detections']}")
        print(f"YOLO Detection Rate: {stats['detection_rate']:.1%}")
        print(f"Successful Shear Predictions: {stats['successful_predictions']}")
        print(f"Shear Prediction Rate: {stats['prediction_rate']:.1%}")
        print(f"Average Processing Time: {stats['avg_processing_time']:.3f}s per image")
        print(f"Effective FPS: {stats['fps']:.1f}")

        if 'shear_statistics' in stats:
            shear_stats = stats['shear_statistics']
            print(f"\n🎯 SHEAR PREDICTION STATISTICS:")
            print(f"  Count: {shear_stats['count']}")
            print(f"  Mean: {shear_stats['mean']:.1f}%")
            print(f"  Std Dev: {shear_stats['std']:.1f}%")
            print(f"  Range: {shear_stats['min']:.1f}% - {shear_stats['max']:.1f}%")
            print(f"  Median: {shear_stats['median']:.1f}%")


def main():
    """Main function to run batch testing."""

    # Default paths relative to project root - adjust these to match your setup
    # These are example paths - update them to point to your actual model files
    yolo_model_path = "models/detection/charpy_3class/charpy_3class_20250729_110009/weights/best.pt"  # Update this path to your YOLO model
    regression_model_path = "models/classification/robust_shear_debug_model.pkl"  # Updated to use the new robust shear debug model
    test_images_dir = "data/raw/charpy_dataset_v2/images/test"  # Directory containing test images

    print("🔬 BATCH MODEL TESTER")
    print("=" * 50)
    print("Tests both YOLO detection and shear regression models")
    print("Perfect for when your camera is faulty!")
    print()

    # Check if files exist (resolve paths relative to project root)
    def resolve_path(path: str) -> str:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        else:
            return str(PROJECT_ROOT / path_obj)
    
    resolved_yolo_path = resolve_path(yolo_model_path)
    resolved_regression_path = resolve_path(regression_model_path)
    resolved_test_dir = resolve_path(test_images_dir)
    
    missing_files = []
    if not Path(resolved_yolo_path).exists():
        missing_files.append(f"YOLO model: {resolved_yolo_path}")
    if not Path(resolved_regression_path).exists():
        missing_files.append(f"Regression model: {resolved_regression_path}")
    if not Path(resolved_test_dir).exists():
        missing_files.append(f"Test images directory: {resolved_test_dir}")

    if missing_files:
        print("❌ Missing required files/directories:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 To fix this:")
        print("   1. Update the paths in main() to match your setup")
        print("   2. Train your models or copy existing model files to the specified paths")
        print("   3. Create a test images directory with sample images")
        print("   4. All paths are now resolved relative to the project root:")
        print(f"      Project root: {PROJECT_ROOT}")
        print("   5. You can use either relative paths (from project root) or absolute paths")
        return

    # Initialize and run tester
    tester = BatchModelTester(yolo_model_path, regression_model_path)

    try:
        # Run test on up to 20 images
        results = tester.run_test(
            image_directory=test_images_dir,
            max_images=20,
            save_results=True
        )

        if results:
            print(f"\n✅ TESTING COMPLETED!")
            print("Check the generated visualization and results file.")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()