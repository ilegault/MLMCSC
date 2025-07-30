#!/usr/bin/env python3
"""
Multi-Call Model Testing Script
Tests the combination of YOLO object detection and shear regression models

This script:
1. Uses YOLO model to detect fracture surfaces in Charpy specimens
2. Extracts detected fracture surface regions
3. Uses the regression model to predict shear percentages
4. Provides comprehensive analysis and visualization
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import json
import os
import glob
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import the shear classification components
import sys
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.postprocessing.shear_classification import (
        FractureSurfaceDetector, 
        ShearFeatureExtractor, 
        ShearFeatures
    )
except ImportError:
    print("⚠️ Could not import shear classification modules. Some features may be limited.")
    FractureSurfaceDetector = None
    ShearFeatureExtractor = None
    ShearFeatures = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiCallModelTester:
    """Combined YOLO detection + Shear regression testing system."""
    
    def __init__(self, yolo_model_path: str, regression_model_path: str):
        """
        Initialize the multi-call model tester.
        
        Args:
            yolo_model_path: Path to trained YOLO model (.pt file)
            regression_model_path: Path to trained regression model (.pkl file)
        """
        self.yolo_model_path = yolo_model_path
        self.regression_model_path = regression_model_path
        
        # Models
        self.yolo_model = None
        self.regression_model = None
        self.scaler = None
        
        # Components
        self.surface_detector = None
        self.feature_extractor = None
        
        # Results storage
        self.test_results = []
        self.detection_stats = {}
        self.regression_stats = {}
        
        print("🔬 Multi-Call Model Tester Initialized")
        print(f"📊 YOLO Model: {yolo_model_path}")
        print(f"🧮 Regression Model: {regression_model_path}")
    
    def load_models(self) -> bool:
        """Load both YOLO and regression models."""
        print("\n🔄 Loading Models...")
        print("=" * 40)
        
        # Load YOLO model
        try:
            if not os.path.exists(self.yolo_model_path):
                print(f"❌ YOLO model not found: {self.yolo_model_path}")
                return False
            
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"✅ YOLO model loaded successfully")
            print(f"   Classes: {list(self.yolo_model.names.values())}")
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return False
        
        # Load regression model
        try:
            if not os.path.exists(self.regression_model_path):
                print(f"❌ Regression model not found: {self.regression_model_path}")
                return False
            
            with open(self.regression_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different pickle formats
            if isinstance(model_data, dict):
                self.regression_model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                print(f"✅ Regression model loaded from dict format")
            else:
                self.regression_model = model_data
                print(f"✅ Regression model loaded (no scaler found)")
            
            print(f"   Model type: {type(self.regression_model).__name__}")
            
        except Exception as e:
            print(f"❌ Failed to load regression model: {e}")
            return False
        
        # Initialize feature extraction components
        if FractureSurfaceDetector and ShearFeatureExtractor:
            self.surface_detector = FractureSurfaceDetector()
            self.feature_extractor = ShearFeatureExtractor()
            print("✅ Feature extraction components initialized")
        else:
            print("⚠️ Feature extraction components not available")
        
        return True
    
    def detect_fracture_surfaces(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """
        Detect fracture surfaces using YOLO model.
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection dictionaries with bbox and confidence
        """
        if self.yolo_model is None:
            return []
        
        # Run YOLO inference
        results = self.yolo_model(image, conf=conf_threshold, iou=0.45)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class name
                    class_name = self.yolo_model.names[cls] if cls < len(self.yolo_model.names) else f"class_{cls}"
                    
                    # Only process fracture surface detections
                    if 'fracture' in class_name.lower() or 'surface' in class_name.lower():
                        detections.append({
                            'class_id': cls,
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'area': (x2 - x1) * (y2 - y1)
                        })
        
        return detections
    
    def extract_fracture_region(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract fracture surface region from image using bounding box.
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Extracted fracture surface region
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding around detection
        padding = 20
        h, w = image.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract region
        if len(image.shape) == 3:
            region = image[y1:y2, x1:x2]
        else:
            region = image[y1:y2, x1:x2]
        
        return region if region.size > 0 else None
    
    def extract_shear_features(self, fracture_region: np.ndarray) -> Optional[ShearFeatures]:
        """
        Extract shear features from fracture surface region.
        
        Args:
            fracture_region: Fracture surface image region
            
        Returns:
            Extracted features or None if extraction fails
        """
        if self.feature_extractor is None:
            print("⚠️ Feature extractor not available")
            return None
        
        try:
            # Convert to grayscale if needed
            if len(fracture_region.shape) == 3:
                gray_region = cv2.cvtColor(fracture_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = fracture_region
            
            # Extract features
            features = self.feature_extractor.extract_features(gray_region)
            return features
            
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            return None
    
    def predict_shear_percentage(self, features: ShearFeatures) -> Dict[str, float]:
        """
        Predict shear percentage using regression model.
        
        Args:
            features: Extracted shear features
            
        Returns:
            Dictionary with prediction and confidence information
        """
        if self.regression_model is None:
            return {'prediction': 0.0, 'confidence': 0.0}
        
        try:
            # Convert features to array
            feature_array = features.to_array().reshape(1, -1)
            
            # Apply scaling if available
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.regression_model.predict(feature_array)[0]
            
            # Calculate confidence (if model supports it)
            confidence = 0.8  # Default confidence
            if hasattr(self.regression_model, 'predict_proba'):
                try:
                    proba = self.regression_model.predict_proba(feature_array)
                    confidence = np.max(proba)
                except:
                    pass
            elif hasattr(self.regression_model, 'score'):
                # Use a simple confidence measure for regression
                confidence = min(1.0, max(0.1, 1.0 - abs(prediction - 50) / 100))
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'features_used': len(feature_array[0])
            }
            
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Complete processing results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Could not load image: {image_path}'}
        
        result = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'image_shape': image.shape,
            'timestamp': datetime.now().isoformat(),
            'detections': [],
            'predictions': [],
            'best_prediction': None
        }
        
        # Step 1: Detect fracture surfaces
        detections = self.detect_fracture_surfaces(image)
        result['detections'] = detections
        
        if not detections:
            result['status'] = 'no_fracture_detected'
            return result
        
        # Step 2: Process each detection
        predictions = []
        for i, detection in enumerate(detections):
            # Extract fracture region
            fracture_region = self.extract_fracture_region(image, detection['bbox'])
            
            if fracture_region is None:
                continue
            
            # Extract features
            features = self.extract_shear_features(fracture_region)
            
            if features is None:
                continue
            
            # Predict shear percentage
            prediction_result = self.predict_shear_percentage(features)
            
            # Combine detection and prediction info
            combined_result = {
                'detection_id': i,
                'detection_info': detection,
                'fracture_region_shape': fracture_region.shape,
                'prediction': prediction_result
            }
            
            predictions.append(combined_result)
        
        result['predictions'] = predictions
        result['status'] = 'success' if predictions else 'feature_extraction_failed'
        
        # Find best prediction (highest confidence)
        if predictions:
            best_pred = max(predictions, key=lambda x: x['prediction'].get('confidence', 0))
            result['best_prediction'] = {
                'shear_percentage': best_pred['prediction']['prediction'],
                'confidence': best_pred['prediction']['confidence'],
                'detection_confidence': best_pred['detection_info']['confidence']
            }
        
        return result
    
    def test_on_dataset(self, test_images_dir: str, output_dir: str = "multicall_results") -> Dict[str, Any]:
        """
        Test the multi-call model on a dataset of images.
        
        Args:
            test_images_dir: Directory containing test images
            output_dir: Directory to save results
            
        Returns:
            Complete testing results
        """
        print(f"\n🧪 MULTI-CALL MODEL TESTING")
        print("=" * 50)
        print(f"📁 Test images: {test_images_dir}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find test images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_images = []
        
        for ext in image_extensions:
            test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
        
        if not test_images:
            print(f"❌ No test images found in: {test_images_dir}")
            return {}
        
        print(f"📊 Found {len(test_images)} test images")
        
        # Process all images
        all_results = []
        successful_predictions = []
        detection_counts = {'total': 0, 'with_fracture': 0, 'with_prediction': 0}
        
        for i, img_path in enumerate(test_images):
            print(f"\n📸 Processing {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
            
            result = self.process_single_image(img_path)
            all_results.append(result)
            
            # Update statistics
            detection_counts['total'] += 1
            if result.get('detections'):
                detection_counts['with_fracture'] += 1
            if result.get('best_prediction'):
                detection_counts['with_prediction'] += 1
                successful_predictions.append(result['best_prediction']['shear_percentage'])
                
                # Print prediction
                pred = result['best_prediction']
                print(f"   🎯 Predicted shear: {pred['shear_percentage']:.1f}% "
                      f"(confidence: {pred['confidence']:.2f})")
        
        # Generate comprehensive results
        test_summary = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_images_dir': test_images_dir,
                'total_images': len(test_images),
                'yolo_model': self.yolo_model_path,
                'regression_model': self.regression_model_path
            },
            'detection_stats': detection_counts,
            'prediction_stats': self._calculate_prediction_stats(successful_predictions),
            'detailed_results': all_results
        }
        
        # Save results
        results_file = os.path.join(output_dir, "multicall_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(test_summary, f, indent=2, default=str)
        
        # Create visualizations
        self._create_visualizations(test_summary, output_dir)
        
        # Save annotated images
        self._save_annotated_images(all_results, output_dir)
        
        print(f"\n✅ TESTING COMPLETED!")
        print(f"📊 Results saved to: {results_file}")
        
        return test_summary
    
    def _calculate_prediction_stats(self, predictions: List[float]) -> Dict[str, Any]:
        """Calculate statistics for predictions."""
        if not predictions:
            return {'count': 0}
        
        predictions = np.array(predictions)
        
        return {
            'count': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75)),
            'distribution': {
                '0-20%': int(np.sum((predictions >= 0) & (predictions < 20))),
                '20-40%': int(np.sum((predictions >= 20) & (predictions < 40))),
                '40-60%': int(np.sum((predictions >= 40) & (predictions < 60))),
                '60-80%': int(np.sum((predictions >= 60) & (predictions < 80))),
                '80-100%': int(np.sum((predictions >= 80) & (predictions <= 100)))
            }
        }
    
    def _create_visualizations(self, test_summary: Dict, output_dir: str):
        """Create visualization plots for test results."""
        try:
            predictions = []
            confidences = []
            
            for result in test_summary['detailed_results']:
                if result.get('best_prediction'):
                    predictions.append(result['best_prediction']['shear_percentage'])
                    confidences.append(result['best_prediction']['confidence'])
            
            if not predictions:
                print("⚠️ No predictions to visualize")
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Multi-Call Model Test Results', fontsize=16)
            
            # Prediction distribution histogram
            axes[0, 0].hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Shear Percentage Predictions Distribution')
            axes[0, 0].set_xlabel('Predicted Shear Percentage (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Confidence distribution
            axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Prediction Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Scatter plot: Prediction vs Confidence
            axes[1, 0].scatter(predictions, confidences, alpha=0.6, color='coral')
            axes[1, 0].set_title('Prediction vs Confidence')
            axes[1, 0].set_xlabel('Predicted Shear Percentage (%)')
            axes[1, 0].set_ylabel('Confidence Score')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Detection success rates
            stats = test_summary['detection_stats']
            categories = ['Total Images', 'Fracture Detected', 'Prediction Made']
            values = [stats['total'], stats['with_fracture'], stats['with_prediction']]
            
            bars = axes[1, 1].bar(categories, values, color=['lightblue', 'orange', 'lightgreen'])
            axes[1, 1].set_title('Detection and Prediction Success Rates')
            axes[1, 1].set_ylabel('Count')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(value), ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(output_dir, "test_results_visualization.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualization saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to create visualizations: {e}")
    
    def _save_annotated_images(self, results: List[Dict], output_dir: str, max_images: int = 20):
        """Save annotated images showing detections and predictions."""
        try:
            annotated_dir = os.path.join(output_dir, "annotated_images")
            os.makedirs(annotated_dir, exist_ok=True)
            
            saved_count = 0
            for result in results:
                if saved_count >= max_images:
                    break
                
                if not result.get('best_prediction'):
                    continue
                
                # Load original image
                image = cv2.imread(result['image_path'])
                if image is None:
                    continue
                
                # Draw detections and predictions
                for detection in result.get('detections', []):
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw class label
                    label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw prediction
                pred = result['best_prediction']
                pred_text = f"Shear: {pred['shear_percentage']:.1f}% (conf: {pred['confidence']:.2f})"
                
                # Add background rectangle for text
                text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
                cv2.putText(image, pred_text, (15, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Save annotated image
                output_filename = f"annotated_{os.path.basename(result['image_path'])}"
                output_path = os.path.join(annotated_dir, output_filename)
                cv2.imwrite(output_path, image)
                
                saved_count += 1
            
            print(f"📸 Saved {saved_count} annotated images to: {annotated_dir}")
            
        except Exception as e:
            print(f"⚠️ Failed to save annotated images: {e}")


def main():
    """Main function to run multi-call model testing."""
    print("🔬 MULTI-CALL MODEL TESTER")
    print("=" * 50)
    print("This script tests the combination of:")
    print("1. YOLO model for fracture surface detection")
    print("2. Regression model for shear percentage prediction")
    print()
    
    # Default paths - update these based on your setup
    yolo_model_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/models/charpy_3class/charpy_3class_20250729_110009/weights/best.pt"
    regression_model_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/src/postprocessing/shiny_region_charpy_model.pkl"
    test_images_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/data/charpy_dataset_v2/images/test"
    
    print(f"🎯 YOLO Model: {yolo_model_path}")
    print(f"🧮 Regression Model: {regression_model_path}")
    print(f"📁 Test Images: {test_images_dir}")
    print()
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(yolo_model_path):
        missing_files.append(f"YOLO model: {yolo_model_path}")
    if not os.path.exists(regression_model_path):
        missing_files.append(f"Regression model: {regression_model_path}")
    if not os.path.exists(test_images_dir):
        missing_files.append(f"Test images directory: {test_images_dir}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nPlease check the paths and ensure all models are trained.")
        return
    
    # Initialize tester
    tester = MultiCallModelTester(yolo_model_path, regression_model_path)
    
    # Load models
    if not tester.load_models():
        print("❌ Failed to load models. Exiting.")
        return
    
    # Run testing
    try:
        results = tester.test_on_dataset(test_images_dir)
        
        # Print summary
        print(f"\n📊 TESTING SUMMARY:")
        print("=" * 30)
        
        detection_stats = results.get('detection_stats', {})
        prediction_stats = results.get('prediction_stats', {})
        
        print(f"Total Images Processed: {detection_stats.get('total', 0)}")
        print(f"Images with Fracture Detection: {detection_stats.get('with_fracture', 0)}")
        print(f"Images with Shear Prediction: {detection_stats.get('with_prediction', 0)}")
        
        if prediction_stats.get('count', 0) > 0:
            print(f"\nPrediction Statistics:")
            print(f"  Mean Shear: {prediction_stats['mean']:.1f}%")
            print(f"  Std Dev: {prediction_stats['std']:.1f}%")
            print(f"  Range: {prediction_stats['min']:.1f}% - {prediction_stats['max']:.1f}%")
            
            print(f"\nShear Distribution:")
            for range_name, count in prediction_stats['distribution'].items():
                print(f"  {range_name}: {count} images")
        
        print(f"\n✅ TESTING COMPLETED SUCCESSFULLY!")
        print(f"📁 Check 'multicall_results' folder for detailed results and visualizations")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()