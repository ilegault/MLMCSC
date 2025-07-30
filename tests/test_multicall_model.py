#!/usr/bin/env python3
"""
Improved Model Tester

Tests both your YOLO detection model and new shiny region regression model.
Handles both old and new model formats properly.
"""

import cv2
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# YOLO imports
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

# Import new shiny region components
try:
    from improved_fracture_detector import ImprovedFractureSurfaceDetector
    from shiny_region_analyzer import ShinyRegionAnalyzer, ShinyRegionFeatures
    from shiny_region_classifier import ShinyRegionBasedClassifier

    NEW_MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ New shiny region models not available. Check file paths.")
    NEW_MODELS_AVAILABLE = False

# Try to import old models as fallback
try:
    import sys

    sys.path.append("../src/postprocessing")
    from shear_classification import FractureSurfaceDetector, ShearFeatureExtractor

    OLD_MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ Old models not available either.")
    OLD_MODELS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedModelTester:
    """Tests both YOLO detection and shear regression models with proper error handling."""

    def __init__(self, yolo_model_path: str, regression_model_path: str):
        """
        Initialize the improved model tester.

        Args:
            yolo_model_path: Path to YOLO model (.pt file)
            regression_model_path: Path to regression model (.pkl file)
        """
        self.yolo_model_path = yolo_model_path
        self.regression_model_path = regression_model_path

        # Models
        self.yolo_model = None
        self.regression_classifier = None
        self.model_type = None  # 'new' or 'old'

        # Results storage
        self.test_results = []

        logger.info("Improved Model Tester initialized")
        logger.info(f"YOLO model: {yolo_model_path}")
        logger.info(f"Regression model: {regression_model_path}")

    def load_models(self) -> bool:
        """Load both YOLO and regression models with smart fallback."""

        # Load YOLO model
        if not self._load_yolo_model():
            return False

        # Load regression model (try new format first, then old)
        if not self._load_regression_model():
            return False

        logger.info("✅ All models loaded successfully")
        return True

    def _load_yolo_model(self) -> bool:
        """Load YOLO model."""
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available")
            return False

        try:
            if not Path(self.yolo_model_path).exists():
                logger.error(f"YOLO model not found: {self.yolo_model_path}")
                return False

            self.yolo_model = YOLO(self.yolo_model_path)
            logger.info("✅ YOLO model loaded")

            # Print model info
            if hasattr(self.yolo_model, 'names'):
                logger.info(f"YOLO classes: {list(self.yolo_model.names.values())}")

            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def _load_regression_model(self) -> bool:
        """Load regression model (try new format first, then old)."""

        if not Path(self.regression_model_path).exists():
            logger.error(f"Regression model not found: {self.regression_model_path}")
            return False

        # Try to load as new shiny region model
        if NEW_MODELS_AVAILABLE:
            try:
                self.regression_classifier = ShinyRegionBasedClassifier()
                if self.regression_classifier.load_model(self.regression_model_path):
                    self.model_type = 'new'
                    logger.info("✅ New shiny region model loaded")
                    return True
            except Exception as e:
                logger.debug(f"Failed to load as new model: {e}")

        # Try to load as old model format
        if OLD_MODELS_AVAILABLE:
            try:
                with open(self.regression_model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Create old-style components
                self.old_detector = FractureSurfaceDetector()
                self.old_feature_extractor = ShearFeatureExtractor()

                # Handle different pickle formats
                if isinstance(model_data, dict):
                    self.old_model = model_data.get('model')
                    self.old_scaler = model_data.get('scaler')
                else:
                    self.old_model = model_data
                    self.old_scaler = None

                self.model_type = 'old'
                logger.info("✅ Old model format loaded")
                return True

            except Exception as e:
                logger.debug(f"Failed to load as old model: {e}")

        logger.error("Could not load regression model in any supported format")
        return False

    def predict_shear_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict shear percentage using the loaded model."""

        if self.model_type == 'new':
            return self._predict_with_new_model(image)
        elif self.model_type == 'old':
            return self._predict_with_old_model(image)
        else:
            return {
                'success': False,
                'error': 'No regression model loaded',
                'prediction': None,
                'confidence': 0.0
            }

    def _predict_with_new_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict using new shiny region model."""
        try:
            return self.regression_classifier.predict_shear_percentage(image)
        except Exception as e:
            return {
                'success': False,
                'error': f'New model prediction failed: {e}',
                'prediction': None,
                'confidence': 0.0
            }

    def _predict_with_old_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict using old model format."""
        try:
            # Detect fracture surface
            surface = self.old_detector.detect_fracture_surface(image)
            if surface is None:
                return {
                    'success': False,
                    'error': 'Could not detect fracture surface',
                    'prediction': None,
                    'confidence': 0.0
                }

            # Extract features
            features = self.old_feature_extractor.extract_features(surface)
            feature_vector = features.to_array().reshape(1, -1)

            # Scale features if scaler available
            if self.old_scaler is not None:
                feature_vector = self.old_scaler.transform(feature_vector)

            # Make prediction
            prediction = self.old_model.predict(feature_vector)[0]
            prediction = np.clip(prediction, 0.0, 100.0)

            # Calculate confidence (rough estimate for old models)
            confidence = 0.7  # Default confidence for old models

            return {
                'success': True,
                'prediction': float(prediction),
                'confidence': float(confidence),
                'model_type': 'old'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Old model prediction failed: {e}',
                'prediction': None,
                'confidence': 0.0
            }

    def detect_fractures_with_yolo(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """Detect fractures using YOLO model."""

        if self.yolo_model is None:
            return []

        try:
            # Run YOLO detection
            results = self.yolo_model(image, conf=conf_threshold, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
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
            logger.error(f"YOLO detection failed: {e}")
            return []

    def test_single_image(self, image_path: str) -> Dict[str, Any]:
        """Test both models on a single image."""

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'Could not load image: {image_path}',
                'file': image_path
            }

        logger.info(f"Testing image: {Path(image_path).name}")

        # YOLO detection
        detections = self.detect_fractures_with_yolo(image)

        # Shear prediction
        shear_result = self.predict_shear_from_image(image)

        return {
            'success': True,
            'file': image_path,
            'image_shape': image.shape,
            'yolo_detections': detections,
            'shear_prediction': shear_result,
            'model_type': self.model_type
        }

    def test_image_directory(self, test_dir: str, max_images: int = 20) -> Dict[str, Any]:
        """Test models on a directory of images."""

        test_path = Path(test_dir)
        if not test_path.exists():
            logger.error(f"Test directory not found: {test_dir}")
            return {'error': f'Directory not found: {test_dir}'}

        # Find image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(test_path.glob(ext)))

        image_files = image_files[:max_images]  # Limit number of test images

        if not image_files:
            logger.error(f"No images found in {test_dir}")
            return {'error': 'No images found'}

        logger.info(f"Testing {len(image_files)} images...")

        # Test each image
        results = []
        detection_count = 0
        prediction_count = 0
        successful_predictions = []

        for i, img_file in enumerate(image_files):
            logger.info(f"Testing {i + 1}/{len(image_files)}: {img_file.name}")

            result = self.test_single_image(str(img_file))
            results.append(result)

            if result['success']:
                # Count detections
                if result['yolo_detections']:
                    detection_count += 1

                # Count successful predictions
                if result['shear_prediction']['success']:
                    prediction_count += 1
                    successful_predictions.append(result['shear_prediction']['prediction'])

        # Calculate statistics
        prediction_stats = {}
        if successful_predictions:
            predictions = np.array(successful_predictions)
            prediction_stats = {
                'count': len(predictions),
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions)),
                'distribution': self._calculate_distribution(predictions)
            }

        summary = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_directory': test_dir,
                'total_images': len(image_files),
                'model_type': self.model_type
            },
            'detection_stats': {
                'total': len(image_files),
                'with_detections': detection_count,
                'detection_rate': detection_count / len(image_files) if image_files else 0
            },
            'prediction_stats': prediction_stats,
            'detailed_results': results
        }

        return summary

    def _calculate_distribution(self, predictions: np.ndarray) -> Dict[str, int]:
        """Calculate distribution of predictions."""
        return {
            '0-20%': int(np.sum((predictions >= 0) & (predictions < 20))),
            '20-40%': int(np.sum((predictions >= 20) & (predictions < 40))),
            '40-60%': int(np.sum((predictions >= 40) & (predictions < 60))),
            '60-80%': int(np.sum((predictions >= 60) & (predictions < 80))),
            '80-100%': int(np.sum((predictions >= 80) & (predictions <= 100)))
        }

    def save_results(self, results: Dict[str, Any], output_path: str = "test_results.json"):
        """Save test results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def create_visualizations(self, results: Dict[str, Any], output_dir: str = "test_output"):
        """Create visualization plots."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        prediction_stats = results.get('prediction_stats', {})

        if prediction_stats.get('count', 0) > 0:
            # Get all predictions
            predictions = []
            for result in results.get('detailed_results', []):
                if result['success'] and result['shear_prediction']['success']:
                    predictions.append(result['shear_prediction']['prediction'])

            if predictions:
                # Create histogram
                plt.figure(figsize=(10, 6))
                plt.hist(predictions, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Predicted Shear Percentage (%)')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of Shear Predictions (n={len(predictions)})')
                plt.grid(True, alpha=0.3)

                # Add statistics
                mean_pred = np.mean(predictions)
                std_pred = np.std(predictions)
                plt.axvline(mean_pred, color='red', linestyle='--',
                            label=f'Mean: {mean_pred:.1f}%')
                plt.axvline(mean_pred - std_pred, color='orange', linestyle=':', alpha=0.7)
                plt.axvline(mean_pred + std_pred, color='orange', linestyle=':', alpha=0.7,
                            label=f'±1 SD: {std_pred:.1f}%')
                plt.legend()

                plt.tight_layout()
                plt.savefig(output_path / 'prediction_distribution.png', dpi=300)
                plt.show()

                logger.info("Visualization saved to test_output/prediction_distribution.png")


def main():
    """Main function to run the improved model tester."""

    print("🔬 IMPROVED MODEL TESTER")
    print("=" * 40)
    print("Tests both YOLO detection and shear regression models")
    print("Supports both old and new model formats")
    print()

    # Default paths - update these for your setup
    yolo_model_path = "models/charpy_3class/charpy_3class_20250729_110009/weights/best.pt"
    regression_model_path = "../src/postprocessing/charpy_shear_regressor.pkl"  # Try new model first
    test_images_dir = "data/charpy_dataset_v2/images/test"  # Test on your manual samples

    # Check if new model exists, otherwise try old model
    if not Path(regression_model_path).exists():
        old_model_path = "../src/postprocessing/charpy_shear_regressor.pkl"
        if Path(old_model_path).exists():
            regression_model_path = old_model_path
            print(f"Using old model: {regression_model_path}")
        else:
            print("❌ No regression model found!")
            print(f"Looking for: {regression_model_path}")
            print(f"Or: {old_model_path}")
            return

    print(f"🎯 YOLO Model: {yolo_model_path}")
    print(f"🧮 Regression Model: {regression_model_path}")
    print(f"📁 Test Images: {test_images_dir}")
    print()

    # Check if files exist
    missing_files = []
    if not Path(yolo_model_path).exists():
        missing_files.append(f"YOLO model: {yolo_model_path}")
    if not Path(regression_model_path).exists():
        missing_files.append(f"Regression model: {regression_model_path}")
    if not Path(test_images_dir).exists():
        missing_files.append(f"Test images: {test_images_dir}")

    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   • {file}")
        return

    # Initialize tester
    tester = ImprovedModelTester(yolo_model_path, regression_model_path)

    # Load models
    if not tester.load_models():
        print("❌ Failed to load models")
        return

    # Run testing
    print("\n🔄 Running tests...")
    results = tester.test_image_directory(test_images_dir, max_images=20)

    if 'error' in results:
        print(f"❌ Testing failed: {results['error']}")
        return

    # Print summary
    print(f"\n📊 TEST RESULTS:")
    print("=" * 30)

    test_info = results['test_info']
    detection_stats = results['detection_stats']
    prediction_stats = results['prediction_stats']

    print(f"Model Type: {test_info['model_type'].upper()}")
    print(f"Total Images: {detection_stats['total']}")
    print(f"Images with YOLO Detections: {detection_stats['with_detections']}")
    print(f"YOLO Detection Rate: {detection_stats['detection_rate']:.1%}")

    if prediction_stats:
        print(f"\nShear Prediction Results:")
        print(f"  Successful Predictions: {prediction_stats['count']}")
        print(f"  Mean Shear: {prediction_stats['mean']:.1f}%")
        print(f"  Standard Deviation: {prediction_stats['std']:.1f}%")
        print(f"  Range: {prediction_stats['min']:.1f}% - {prediction_stats['max']:.1f}%")

        print(f"\n  Distribution:")
        for range_name, count in prediction_stats['distribution'].items():
            print(f"    {range_name}: {count} images")

    # Save results
    tester.save_results(results, "improved_test_results.json")

    # Create visualizations
    tester.create_visualizations(results)

    print(f"\n✅ TESTING COMPLETED!")
    print("Results saved to: improved_test_results.json")


if __name__ == "__main__":
    main()