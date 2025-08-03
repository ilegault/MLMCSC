#!/usr/bin/env python3
"""
Improved Shear Percentage Regression Model

This model focuses on texture-based features rather than brightness,
making it more robust to lighting changes and better at detecting
the actual physical characteristics of shear in concrete samples.
"""

import cv2
import numpy as np
import logging
import os
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
from pathlib import Path
from skimage import feature, morphology, filters
from skimage.measure import regionprops, label
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ImprovedShearFeatures:
    """Features specifically designed for shear percentage detection."""

    # Texture Pattern Features
    roughness_index: float  # Overall surface roughness
    texture_uniformity: float  # How uniform the texture is
    granularity: float  # Size of texture elements

    # Fracture Pattern Features
    fracture_complexity: float  # Complexity of fracture patterns
    edge_density: float  # Density of edges/cracks
    angular_variation: float  # Variation in edge angles

    # Surface Topology Features
    depth_variation: float  # Estimated depth variation from texture
    surface_irregularity: float  # Overall surface irregularity

    # Regional Features
    smooth_area_ratio: float  # Ratio of smooth to rough areas
    transition_zones: float  # Number of smooth-rough transitions

    # Statistical Features
    intensity_entropy: float  # Entropy of intensity distribution
    gradient_entropy: float  # Entropy of gradient distribution
    local_contrast: float  # Average local contrast

    # Frequency Domain Features
    high_freq_energy: float  # Energy in high frequencies (rough texture)
    low_freq_energy: float  # Energy in low frequencies (smooth areas)
    freq_ratio: float  # Ratio of high to low frequency energy

    def to_array(self) -> np.ndarray:
        """Convert to array for model input."""
        return np.array([
            self.roughness_index, self.texture_uniformity, self.granularity,
            self.fracture_complexity, self.edge_density, self.angular_variation,
            self.depth_variation, self.surface_irregularity,
            self.smooth_area_ratio, self.transition_zones,
            self.intensity_entropy, self.gradient_entropy, self.local_contrast,
            self.high_freq_energy, self.low_freq_energy, self.freq_ratio
        ])

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for analysis."""
        return [
            'roughness_index', 'texture_uniformity', 'granularity',
            'fracture_complexity', 'edge_density', 'angular_variation',
            'depth_variation', 'surface_irregularity',
            'smooth_area_ratio', 'transition_zones',
            'intensity_entropy', 'gradient_entropy', 'local_contrast',
            'high_freq_energy', 'low_freq_energy', 'freq_ratio'
        ]


class ImprovedShearDetector:
    """Improved detector for shear percentage in concrete samples."""

    def __init__(self):
        self.detector = None  # Will be initialized when needed
        logger.info("Improved Shear Detector initialized")

    def detect_fracture_surface(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract the fracture surface from the image.
        Uses adaptive thresholding to handle varying lighting conditions.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Use adaptive thresholding to handle varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )

        # Find the largest contour (assuming it's the sample)
        contours, _ = cv2.findContours(
            adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Create mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Extract the surface
        surface = cv2.bitwise_and(enhanced, enhanced, mask=mask)

        # Crop to bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        surface_cropped = surface[y:y + h, x:x + w]

        return surface_cropped


class ImprovedFeatureExtractor:
    """Extract texture-based features for shear detection."""

    def __init__(self):
        self.gabor_kernels = self._create_gabor_kernels()

    def _create_gabor_kernels(self) -> List[np.ndarray]:
        """Create Gabor kernels for texture analysis."""
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            for frequency in [0.1, 0.2, 0.3]:
                kernel = cv2.getGaborKernel(
                    (21, 21), 4.0, theta, 10.0 / frequency, 0.5, 0, ktype=cv2.CV_32F
                )
                kernels.append(kernel)
        return kernels

    def extract_features(self, surface: np.ndarray) -> ImprovedShearFeatures:
        """Extract all features from the fracture surface."""
        # Ensure surface is uint8
        if surface.dtype != np.uint8:
            surface = (surface * 255).astype(np.uint8)

        # Extract different feature groups
        texture_features = self._extract_texture_features(surface)
        fracture_features = self._extract_fracture_features(surface)
        topology_features = self._extract_topology_features(surface)
        regional_features = self._extract_regional_features(surface)
        statistical_features = self._extract_statistical_features(surface)
        frequency_features = self._extract_frequency_features(surface)

        return ImprovedShearFeatures(
            # Texture features
            roughness_index=texture_features['roughness'],
            texture_uniformity=texture_features['uniformity'],
            granularity=texture_features['granularity'],

            # Fracture features
            fracture_complexity=fracture_features['complexity'],
            edge_density=fracture_features['edge_density'],
            angular_variation=fracture_features['angular_variation'],

            # Topology features
            depth_variation=topology_features['depth_variation'],
            surface_irregularity=topology_features['irregularity'],

            # Regional features
            smooth_area_ratio=regional_features['smooth_ratio'],
            transition_zones=regional_features['transitions'],

            # Statistical features
            intensity_entropy=statistical_features['intensity_entropy'],
            gradient_entropy=statistical_features['gradient_entropy'],
            local_contrast=statistical_features['local_contrast'],

            # Frequency features
            high_freq_energy=frequency_features['high_freq'],
            low_freq_energy=frequency_features['low_freq'],
            freq_ratio=frequency_features['freq_ratio']
        )

    def _extract_texture_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features using Gabor filters and LBP."""
        features = {}

        # Apply Gabor filters
        gabor_responses = []
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(surface, cv2.CV_32F, kernel)
            gabor_responses.append(np.std(filtered))

        # Roughness as variation in Gabor responses
        features['roughness'] = np.mean(gabor_responses)
        features['uniformity'] = 1.0 / (1.0 + np.std(gabor_responses))

        # Granularity using morphological operations
        opened = cv2.morphologyEx(surface, cv2.MORPH_OPEN, np.ones((3, 3)))
        granularity = np.sum(np.abs(surface.astype(float) - opened.astype(float)))
        features['granularity'] = granularity / (surface.shape[0] * surface.shape[1])

        return features

    def _extract_fracture_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract features related to fracture patterns."""
        features = {}

        # Edge detection using Canny
        edges = cv2.Canny(surface, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

        # Hough transform for line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            # Calculate angular variation
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)

            features['angular_variation'] = np.std(angles) if angles else 0.0
            features['complexity'] = len(lines) / 100.0  # Normalized by area
        else:
            features['angular_variation'] = 0.0
            features['complexity'] = 0.0

        return features

    def _extract_topology_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract surface topology features."""
        features = {}

        # Estimate depth variation using gradient magnitude
        grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features['depth_variation'] = np.std(grad_mag)

        # Surface irregularity using local variance
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        local_mean = cv2.filter2D(surface.astype(float), -1, kernel)
        local_var = cv2.filter2D((surface.astype(float) - local_mean) ** 2, -1, kernel)

        features['irregularity'] = np.mean(np.sqrt(local_var))

        return features

    def _extract_regional_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract features based on regional analysis."""
        features = {}

        # Threshold to separate smooth and rough regions
        _, binary = cv2.threshold(surface, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate smooth area ratio
        smooth_pixels = np.sum(binary > 0)
        total_pixels = surface.size
        features['smooth_ratio'] = smooth_pixels / total_pixels

        # Count transitions between smooth and rough
        diff = np.diff(binary.astype(int), axis=1)
        transitions = np.sum(np.abs(diff) > 0)
        features['transitions'] = transitions / binary.shape[1]

        return features

    def _extract_statistical_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract statistical features."""
        features = {}

        # Intensity entropy
        hist, _ = np.histogram(surface, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        features['intensity_entropy'] = -np.sum(hist * np.log2(hist))

        # Gradient entropy
        grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        grad_hist, _ = np.histogram(grad_mag, bins=100)
        grad_hist = grad_hist / np.sum(grad_hist)
        grad_hist = grad_hist[grad_hist > 0]
        features['gradient_entropy'] = -np.sum(grad_hist * np.log2(grad_hist))

        # Local contrast
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        local_mean = cv2.filter2D(surface.astype(float), -1, kernel)
        local_contrast = np.abs(surface.astype(float) - local_mean)
        features['local_contrast'] = np.mean(local_contrast)

        return features

    def _extract_frequency_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT."""
        features = {}

        # Compute 2D FFT
        f = np.fft.fft2(surface)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        # Create frequency masks
        rows, cols = surface.shape
        crow, ccol = rows // 2, cols // 2

        # Low frequency mask (center region)
        low_freq_mask = np.zeros((rows, cols), dtype=bool)
        radius_low = min(rows, cols) // 4
        y, x = np.ogrid[:rows, :cols]
        low_freq_mask[(x - ccol) ** 2 + (y - crow) ** 2 <= radius_low ** 2] = True

        # High frequency mask (outer region)
        high_freq_mask = ~low_freq_mask

        # Calculate energy in different frequency bands
        features['low_freq'] = np.sum(magnitude_spectrum[low_freq_mask])
        features['high_freq'] = np.sum(magnitude_spectrum[high_freq_mask])
        features['freq_ratio'] = features['high_freq'] / (features['low_freq'] + 1e-6)

        return features


class ImprovedShearRegressionModel:
    """Improved regression model for shear percentage prediction."""

    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.detector = ImprovedShearDetector()
        self.extractor = ImprovedFeatureExtractor()
        self.model = None
        self.scaler = None
        self.is_trained = False

        logger.info(f"Improved Shear Regression Model initialized with {model_type}")

    def load_training_data_from_directory(self, data_dir: str) -> Dict[float, List[str]]:
        """
        Load training data from directory structure organized by shear percentage.
        
        Expected structure:
        data_dir/
        ├── 0_percent/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── 10_percent/
        │   ├── image1.jpg
        │   └── ...
        ├── 20_percent/
        │   └── ...
        └── ...
        
        Args:
            data_dir: Path to the directory containing shear percentage subdirectories
            
        Returns:
            Dictionary mapping shear percentages to lists of image paths
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {data_dir}")
        
        training_data = {}
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        logger.info(f"Loading training data from: {data_dir}")
        
        # Scan for subdirectories that match shear percentage patterns
        for subdir in data_dir.iterdir():
            if not subdir.is_dir():
                continue
                
            # Extract shear percentage from directory name
            # Supports formats like: "0_percent", "10_percent", "0%", "10%", "0", "10"
            dir_name = subdir.name.lower()
            
            # Try different patterns to extract percentage
            percentage = None
            
            # Pattern 1: "XX_percent" or "XX_per" or "XXpercent"
            match = re.search(r'(\d+)(?:_?percent|_?per|%)', dir_name)
            if match:
                percentage = float(match.group(1))
            
            # Pattern 2: Just numbers "XX" (assuming it's percentage)
            elif re.match(r'^\d+$', dir_name):
                percentage = float(dir_name)
            
            # Pattern 3: "shear_XX" or "XX_shear"
            elif 'shear' in dir_name:
                match = re.search(r'(\d+)', dir_name)
                if match:
                    percentage = float(match.group(1))
            
            if percentage is None:
                logger.warning(f"Could not extract shear percentage from directory: {subdir.name}")
                continue
                
            # Validate percentage range
            if not (0 <= percentage <= 100):
                logger.warning(f"Invalid shear percentage {percentage}% in directory: {subdir.name}")
                continue
            
            # Find all image files in this directory
            image_paths = []
            for file_path in subdir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    image_paths.append(str(file_path))
            
            if image_paths:
                training_data[percentage] = image_paths
                logger.info(f"Found {len(image_paths)} images for {percentage}% shear in {subdir.name}")
            else:
                logger.warning(f"No valid images found in directory: {subdir.name}")
        
        if not training_data:
            raise ValueError(f"No valid training data found in {data_dir}")
        
        # Sort by shear percentage for better logging
        total_images = sum(len(paths) for paths in training_data.values())
        logger.info(f"Successfully loaded training data:")
        logger.info(f"  - {len(training_data)} shear percentage classes")
        logger.info(f"  - {total_images} total images")
        logger.info(f"  - Shear percentages: {sorted(training_data.keys())}")
        
        return training_data

    def create_ensemble_model(self):
        """Create an ensemble model combining multiple regressors."""
        from sklearn.ensemble import VotingRegressor

        # Create individual models with different strengths
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

        # Ensemble them
        ensemble = VotingRegressor([
            ('rf', rf_model),
            ('gb', gb_model)
        ])

        return ensemble

    def process_image(self, image: np.ndarray) -> Optional[ImprovedShearFeatures]:
        """Process image and extract features."""
        # Detect fracture surface
        surface = self.detector.detect_fracture_surface(image)
        if surface is None:
            logger.warning("Could not detect fracture surface")
            return None

        # Extract features
        features = self.extractor.extract_features(surface)
        return features

    def train(self, training_data: Dict[float, List[str]]) -> Dict[str, Any]:
        """
        Train the model on labeled data.

        Args:
            training_data: Dict mapping shear percentages to image paths
        """
        # Extract features from all training images
        features_list = []
        labels_list = []

        logger.info("Extracting features from training data...")

        for shear_percent, image_paths in training_data.items():
            logger.info(f"Processing {len(image_paths)} images for {shear_percent}% shear")

            for img_path in image_paths:
                try:
                    # Load image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"Could not load {img_path}")
                        continue

                    # Extract features
                    features = self.process_image(image)
                    if features is not None:
                        features_list.append(features.to_array())
                        labels_list.append(shear_percent)

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue

        if not features_list:
            raise ValueError("No valid features extracted")

        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"Training on {len(X)} samples with {X.shape[1]} features")

        # Use RobustScaler for outlier resistance
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create and train model
        if self.model_type == 'ensemble':
            self.model = self.create_ensemble_model()
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )

        # Train with cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=5, scoring='neg_mean_absolute_error'
        )

        # Final training
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        results = {
            'mae': mae,
            'r2': r2,
            'cv_mae': -np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

        logger.info(f"Training complete - MAE: {mae:.2f}%, R²: {r2:.3f}")
        logger.info(f"Cross-validation MAE: {-np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

        return results

    def train_from_directory(self, data_dir: str) -> Dict[str, Any]:
        """
        Convenience method to train directly from a directory structure.
        
        Args:
            data_dir: Path to directory containing shear percentage subdirectories
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training model from directory: {data_dir}")
        
        # Load training data from directory
        training_data = self.load_training_data_from_directory(data_dir)
        
        # Train the model
        return self.train(training_data)

    def predict_shear_percentage(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict shear percentage for a single image."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Extract features
        features = self.process_image(image)
        if features is None:
            return {
                'success': False,
                'error': 'Could not extract features',
                'shear_percentage': None,
                'confidence': 0.0
            }

        try:
            # Scale and predict
            feature_vector = features.to_array().reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_vector)

            prediction = self.model.predict(feature_scaled)[0]

            # Clip to valid range
            prediction = np.clip(prediction, 0, 100)

            # Calculate confidence based on ensemble agreement
            if hasattr(self.model, 'estimators_'):
                # For ensemble models
                predictions = []
                for estimator in self.model.estimators_:
                    pred = estimator.predict(feature_scaled)[0]
                    predictions.append(pred)

                std_dev = np.std(predictions)
                confidence = max(0.0, 1.0 - (std_dev / 25.0))
            else:
                confidence = 0.8  # Default confidence

            return {
                'success': True,
                'shear_percentage': float(prediction),
                'confidence': float(confidence),
                'features': features.__dict__
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'shear_percentage': None,
                'confidence': 0.0
            }

    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': ImprovedShearFeatures.get_feature_names(),
            'version': '2.0'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data.get('model_type', 'ensemble')
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")


# Example usage and training script
if __name__ == "__main__":
    # Create model
    model = ImprovedShearRegressionModel(model_type='ensemble')
    
    # Method 1: Train directly from directory structure (RECOMMENDED)
    # This will automatically load all images from subdirectories named by shear percentage
    try:
        # Use absolute path from project root
        training_dir = "/data/samples/shiny_training_data"
        results = model.train_from_directory(training_dir)
        
        print(f"Training Results:")
        print(f"  - Mean Absolute Error: {results['mae']:.2f}%")
        print(f"  - R² Score: {results['r2']:.3f}")
        print(f"  - Cross-validation MAE: {results['cv_mae']:.2f} ± {results['cv_std']:.2f}")
        print(f"  - Training samples: {results['n_samples']}")
        
        # Save the trained model
        model.save_model("improved_shear_model.pkl")
        print("Model saved successfully!")
        
    except FileNotFoundError as e:
        print(f"Training directory not found: {e}")
        print("Please ensure your training data is organized in the following structure:")
        print("C:/Users/IGLeg/PycharmProjects/MLMCSC/data/raw/samples/shiny_training_data/")
        print("├── 0_percent/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── 10_percent/")
        print("│   └── ...")
        print("└── ...")
    
    # Test prediction on a sample image
    # Uncomment and modify path to test
    """
    test_image = cv2.imread("path/to/test_image.jpg")
    if test_image is not None:
        prediction = model.predict_shear_percentage(test_image)
        if prediction['success']:
            print(f"Predicted shear: {prediction['shear_percentage']:.1f}% "
                  f"(confidence: {prediction['confidence']:.2f})")
        else:
            print(f"Prediction failed: {prediction['error']}")
    """