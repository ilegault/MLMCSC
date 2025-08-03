#!/usr/bin/env python3
"""
Shear Model Debugging and Improved Training

This script helps debug why the model is predicting narrow ranges
and provides a more robust training approach.
"""

import cv2
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
from pathlib import Path
import seaborn as sns
from scipy import stats
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShearModelDebugger:
    """Debug and analyze shear model training issues."""

    def __init__(self):
        self.features_cache = {}
        self.labels_cache = {}

    def analyze_training_data(self, training_data: Dict[float, List[str]]) -> Dict[str, Any]:
        """Analyze the training data distribution and quality."""
        analysis = {
            'class_distribution': {},
            'total_samples': 0,
            'missing_classes': [],
            'image_issues': []
        }

        # Check expected vs actual classes
        expected_classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        actual_classes = sorted(training_data.keys())

        for expected in expected_classes:
            if expected not in actual_classes:
                analysis['missing_classes'].append(expected)

        # Count samples per class
        for shear_pct, image_paths in training_data.items():
            analysis['class_distribution'][shear_pct] = len(image_paths)
            analysis['total_samples'] += len(image_paths)

            # Check for image loading issues
            for img_path in image_paths[:3]:  # Check first 3 images per class
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        analysis['image_issues'].append(f"Cannot load: {img_path}")
                except Exception as e:
                    analysis['image_issues'].append(f"Error loading {img_path}: {e}")

        return analysis

    def visualize_sample_images(self, training_data: Dict[float, List[str]], max_per_class: int = 2):
        """Visualize sample images from each class."""
        classes = sorted(training_data.keys())
        n_classes = len(classes)
        
        if n_classes == 0:
            logger.warning("No classes to visualize")
            return

        # Create subplot grid
        cols = min(4, n_classes)
        rows = (n_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, shear_pct in enumerate(classes):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            # Load and display first image from this class
            image_paths = training_data[shear_pct]
            if image_paths:
                try:
                    img = cv2.imread(str(image_paths[0]))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img_rgb)
                        ax.set_title(f'{shear_pct}% Shear\n({len(image_paths)} images)')
                        ax.axis('off')
                    else:
                        ax.text(0.5, 0.5, f'{shear_pct}% Shear\nImage Load Failed', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f'{shear_pct}% Shear\nError: {str(e)[:20]}...', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'{shear_pct}% Shear\nNo Images', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        # Hide empty subplots
        for idx in range(n_classes, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig('training_samples_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved training samples visualization to 'training_samples_visualization.png'")


@dataclass
class RobustShearFeatures:
    """Simplified but robust features for shear detection."""

    # Basic texture metrics
    surface_roughness: float  # Standard deviation of intensities
    texture_energy: float  # Sum of squared elements in GLCM

    # Edge characteristics
    edge_density: float  # Proportion of edge pixels
    edge_strength: float  # Average gradient magnitude

    # Region analysis
    bright_area_ratio: float  # Ratio of bright to total pixels
    dark_area_ratio: float  # Ratio of dark to total pixels

    # Contrast metrics
    global_contrast: float  # Difference between percentiles
    local_contrast_mean: float  # Average local contrast

    # Frequency domain
    high_freq_ratio: float  # Ratio of high to total frequency content

    # Statistical moments
    intensity_skewness: float  # Skewness of intensity distribution
    intensity_kurtosis: float  # Kurtosis of intensity distribution

    def to_array(self) -> np.ndarray:
        return np.array([
            self.surface_roughness, self.texture_energy,
            self.edge_density, self.edge_strength,
            self.bright_area_ratio, self.dark_area_ratio,
            self.global_contrast, self.local_contrast_mean,
            self.high_freq_ratio,
            self.intensity_skewness, self.intensity_kurtosis
        ])

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return [
            'surface_roughness', 'texture_energy',
            'edge_density', 'edge_strength',
            'bright_area_ratio', 'dark_area_ratio',
            'global_contrast', 'local_contrast_mean',
            'high_freq_ratio',
            'intensity_skewness', 'intensity_kurtosis'
        ]


class RobustFeatureExtractor:
    """Robust feature extraction with better normalization."""

    def __init__(self):
        self.target_size = (256, 256)  # Standardize image size

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for consistent feature extraction."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize to standard size
        resized = cv2.resize(gray, self.target_size)

        # Apply CLAHE for contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        return enhanced

    def extract_features(self, image: np.ndarray) -> RobustShearFeatures:
        """Extract robust features from preprocessed image."""
        # Preprocess
        processed = self.preprocess_image(image)

        # Basic texture metrics
        surface_roughness = np.std(processed)

        # Compute GLCM for texture energy
        # Simplified GLCM calculation
        glcm = self._compute_glcm(processed)
        texture_energy = np.sum(glcm ** 2)

        # Edge detection
        edges = cv2.Canny(processed, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Gradient for edge strength
        grad_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_strength = np.mean(grad_mag)

        # Region analysis using Otsu thresholding
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_pixels = np.sum(binary > 0)
        dark_pixels = np.sum(binary == 0)
        total_pixels = processed.size

        bright_area_ratio = bright_pixels / total_pixels
        dark_area_ratio = dark_pixels / total_pixels

        # Contrast metrics
        p5, p95 = np.percentile(processed, [5, 95])
        global_contrast = p95 - p5

        # Local contrast using standard deviation in local windows
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        local_mean = cv2.filter2D(processed.astype(float), -1, kernel)
        local_var = cv2.filter2D((processed.astype(float) - local_mean) ** 2, -1, kernel)
        local_contrast_mean = np.mean(np.sqrt(local_var))

        # Frequency domain analysis
        f_transform = np.fft.fft2(processed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # High frequency energy (outer regions)
        center_y, center_x = processed.shape[0] // 2, processed.shape[1] // 2
        y, x = np.ogrid[:processed.shape[0], :processed.shape[1]]
        mask_low = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(processed.shape) // 4) ** 2
        
        low_freq_energy = np.sum(magnitude_spectrum[mask_low])
        high_freq_energy = np.sum(magnitude_spectrum[~mask_low])
        total_energy = low_freq_energy + high_freq_energy
        high_freq_ratio = high_freq_energy / (total_energy + 1e-6)

        # Statistical moments
        intensity_skewness = stats.skew(processed.flatten())
        intensity_kurtosis = stats.kurtosis(processed.flatten())

        return RobustShearFeatures(
            surface_roughness=surface_roughness,
            texture_energy=texture_energy,
            edge_density=edge_density,
            edge_strength=edge_strength,
            bright_area_ratio=bright_area_ratio,
            dark_area_ratio=dark_area_ratio,
            global_contrast=global_contrast,
            local_contrast_mean=local_contrast_mean,
            high_freq_ratio=high_freq_ratio,
            intensity_skewness=intensity_skewness,
            intensity_kurtosis=intensity_kurtosis
        )

    def _compute_glcm(self, image: np.ndarray) -> np.ndarray:
        """Simplified GLCM computation."""
        # Quantize to 16 levels for faster computation
        levels = 16
        quantized = (image // (256 // levels)).astype(np.uint8)

        # Compute co-occurrence matrix for horizontal direction
        glcm = np.zeros((levels, levels))
        for i in range(image.shape[0]):
            for j in range(image.shape[1] - 1):
                glcm[quantized[i, j], quantized[i, j + 1]] += 1

        # Normalize
        glcm = glcm / np.sum(glcm)
        return glcm


class ImprovedShearModel:
    """Improved model with better training strategy."""

    def __init__(self):
        self.extractor = RobustFeatureExtractor()
        self.scaler = MinMaxScaler()  # Use MinMaxScaler for bounded features
        self.models = {}  # Ensemble of models
        self.debugger = ShearModelDebugger()
        self.is_trained = False

        # Feature importance tracking
        self.feature_importance = None

    def create_ensemble(self):
        """Create ensemble of diverse models."""
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'et': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=1000,
                random_state=42
            )
        }

    def extract_all_features(self, training_data: Dict[float, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all training images with progress tracking."""
        features_list = []
        labels_list = []

        total_images = sum(len(paths) for paths in training_data.values())
        processed = 0

        logger.info(f"Extracting features from {total_images} images...")

        for shear_pct in sorted(training_data.keys()):
            image_paths = training_data[shear_pct]
            logger.info(f"\nProcessing {len(image_paths)} images for {shear_pct}% shear")

            class_features = []
            for img_path in image_paths:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Could not load: {img_path}")
                        continue

                    features = self.extractor.extract_features(img)
                    class_features.append(features.to_array())
                    features_list.append(features.to_array())
                    labels_list.append(shear_pct)

                    processed += 1
                    if processed % 10 == 0:
                        logger.info(f"Progress: {processed}/{total_images}")

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue

            if class_features:
                # Log feature statistics for this class
                class_features_array = np.array(class_features)
                logger.info(f"Class {shear_pct}% - Features mean: {np.mean(class_features_array, axis=0)[:3]}...")

        return np.array(features_list), np.array(labels_list)

    def train(self, training_data: Dict[float, List[str]], debug: bool = True) -> Dict[str, Any]:
        """Train the model with debugging capabilities."""

        if debug:
            # Analyze training data
            analysis = self.debugger.analyze_training_data(training_data)
            logger.info(f"\nTraining Data Analysis:")
            logger.info(f"Total samples: {analysis['total_samples']}")
            logger.info(f"Class distribution: {analysis['class_distribution']}")
            if analysis['missing_classes']:
                logger.warning(f"Missing classes: {analysis['missing_classes']}")

            # Visualize samples
            self.debugger.visualize_sample_images(training_data)

        # Extract features
        X, y = self.extract_all_features(training_data)

        if len(X) == 0:
            raise ValueError("No valid features extracted!")

        logger.info(f"\nExtracted features shape: {X.shape}")
        logger.info(f"Label distribution: {np.bincount(y.astype(int), minlength=101)}")

        # Check for feature issues
        if np.any(np.isnan(X)):
            logger.warning("Found NaN values in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        # Create and train ensemble
        self.create_ensemble()

        results = {}
        feature_importances = []

        for name, model in self.models.items():
            logger.info(f"\nTraining {name} model...")

            # Train model
            model.fit(X_train, y_train)

            # Validate
            y_pred_val = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)

            results[name] = {
                'val_mae': val_mae,
                'val_r2': val_r2,
                'predictions': y_pred_val
            }

            logger.info(f"{name} - Validation MAE: {val_mae:.2f}%, R²: {val_r2:.3f}")

            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)

        # Average feature importances
        if feature_importances:
            self.feature_importance = np.mean(feature_importances, axis=0)
            self._plot_feature_importance()

        # Plot validation results
        self._plot_validation_results(y_val, results)

        self.is_trained = True

        # Overall metrics
        ensemble_pred = np.mean([r['predictions'] for r in results.values()], axis=0)
        overall_mae = mean_absolute_error(y_val, ensemble_pred)
        overall_r2 = r2_score(y_val, ensemble_pred)

        return {
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'individual_results': results,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

    def _plot_feature_importance(self):
        """Plot feature importance."""
        if self.feature_importance is None:
            return

        feature_names = RobustShearFeatures.get_feature_names()

        plt.figure(figsize=(10, 6))
        indices = np.argsort(self.feature_importance)[::-1]

        plt.bar(range(len(indices)), self.feature_importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        logger.info("Saved feature importance plot to 'feature_importance.png'")

    def _plot_validation_results(self, y_true: np.ndarray, results: Dict[str, Any]):
        """Plot validation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Predictions vs True for each model
        ax = axes[0, 0]
        for name, result in results.items():
            ax.scatter(y_true, result['predictions'], alpha=0.5, label=name)
        ax.plot([0, 100], [0, 100], 'r--', label='Perfect')
        ax.set_xlabel('True Shear %')
        ax.set_ylabel('Predicted Shear %')
        ax.set_title('Predictions vs True Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error distribution
        ax = axes[0, 1]
        ensemble_pred = np.mean([r['predictions'] for r in results.values()], axis=0)
        errors = ensemble_pred - y_true
        ax.hist(errors, bins=20, edgecolor='black')
        ax.set_xlabel('Prediction Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution (MAE: {np.mean(np.abs(errors)):.2f}%)')
        ax.grid(True, alpha=0.3)

        # Plot 3: Predictions by class
        ax = axes[1, 0]
        unique_classes = np.unique(y_true)
        for cls in unique_classes:
            mask = y_true == cls
            class_preds = ensemble_pred[mask]
            ax.scatter([cls] * len(class_preds), class_preds, alpha=0.5)
        ax.plot([0, 100], [0, 100], 'r--')
        ax.set_xlabel('True Shear Class (%)')
        ax.set_ylabel('Predicted Shear %')
        ax.set_title('Predictions by Class')
        ax.grid(True, alpha=0.3)

        # Plot 4: Feature importance (if available)
        ax = axes[1, 1]
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            feature_names = RobustShearFeatures.get_feature_names()
            ax.bar(range(len(self.feature_importance)), self.feature_importance)
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig('validation_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved validation results plot to 'validation_results.png'")

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

    def train_from_directory(self, data_dir: str, debug: bool = True) -> Dict[str, Any]:
        """
        Convenience method to train directly from a directory structure.
        
        Args:
            data_dir: Path to directory containing shear percentage subdirectories
            debug: Whether to enable debugging features
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training model from directory: {data_dir}")
        
        # Load training data from directory
        training_data = self.load_training_data_from_directory(data_dir)
        
        # Train the model
        return self.train(training_data, debug=debug)

    def predict_shear_percentage(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict shear percentage for a single image."""
        if not self.is_trained:
            return {
                'success': False,
                'error': 'Model must be trained first',
                'shear_percentage': None,
                'confidence': 0.0
            }

        try:
            # Extract features
            features = self.extractor.extract_features(image)
            feature_vector = features.to_array().reshape(1, -1)
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_vector)

            # Get predictions from all models
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(feature_scaled)[0]
                predictions.append(pred)

            # Ensemble prediction (average)
            prediction = np.mean(predictions)
            
            # Clip to valid range
            prediction = np.clip(prediction, 0, 100)

            # Calculate confidence based on agreement between models
            std_dev = np.std(predictions)
            confidence = max(0.0, 1.0 - (std_dev / 25.0))  # Normalize by reasonable std range

            return {
                'success': True,
                'shear_percentage': float(prediction),
                'confidence': float(confidence),
                'individual_predictions': [float(p) for p in predictions],
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
            'models': self.models,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'feature_names': RobustShearFeatures.get_feature_names(),
            'version': '3.0_debug'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Debug model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = True

        logger.info(f"Debug model loaded from {filepath}")


# Example usage and training script
if __name__ == "__main__":
    # Create model
    model = ImprovedShearModel()
    
    # Method 1: Train directly from directory structure (RECOMMENDED)
    try:
        # Use absolute path from project root
        training_dir = "/src/database/samples/shiny_training_data"
        results = model.train_from_directory(training_dir, debug=True)
        
        print(f"\nTraining Results:")
        print(f"  - Overall MAE: {results['overall_mae']:.2f}%")
        print(f"  - Overall R² Score: {results['overall_r2']:.3f}")
        print(f"  - Training samples: {results['n_samples']}")
        
        # Save the trained model
        model.save_model("robust_shear_debug_model.pkl")
        print("Debug model saved successfully!")
        
    except FileNotFoundError as e:
        print(f"Training directory not found: {e}")
        print("Please ensure your training data is organized in the following structure:")
        print("C:/Users/IGLeg/PycharmProjects/MLMCSC/data/samples/shiny_training_data/")
        print("├── 0_percent/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── 10_percent/")
        print("│   └── ...")
        print("└── ...")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test prediction on a sample image
    # Uncomment and modify path to test
    """
    test_image = cv2.imread("path/to/test_image.jpg")
    if test_image is not None:
        prediction = model.predict_shear_percentage(test_image)
        if prediction['success']:
            print(f"Predicted shear: {prediction['shear_percentage']:.1f}% "
                  f"(confidence: {prediction['confidence']:.2f})")
            print(f"Individual model predictions: {prediction['individual_predictions']}")
        else:
            print(f"Prediction failed: {prediction['error']}")
    """