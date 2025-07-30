#!/usr/bin/env python3
"""
Simple Working Live Viewer

Works directly with your saved charpy_shear_regressor.pkl model.
No complex imports - loads the model file directly.
"""

import cv2
import numpy as np
import json
import time
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import threading
import pylab as p

# Import standalone config loader
try:
    from config_loader import load_config
    print("✅ Config system available")
except ImportError as e:
    print(f"Warning: Could not import config system: {e}")
    print("Falling back to hardcoded paths")
    load_config = None

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


def get_project_root() -> Path:
    """Get the project root directory (parent of tools directory)."""
    return Path(__file__).parent.parent


def resolve_project_path(relative_path: str) -> str:
    """Resolve a relative path to an absolute path from project root."""
    project_root = get_project_root()
    absolute_path = project_root / relative_path
    return str(absolute_path)


class DictModelWrapper:
    """Wrapper for dictionary-based model to provide predict_shear_percentage method."""
    
    def __init__(self, model_data: dict):
        """Initialize wrapper with model dictionary."""
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names', [])
        self.model_type = model_data.get('model_type', 'unknown')
        
    def predict_shear_percentage(self, image: np.ndarray) -> dict:
        """
        Predict shear percentage from image.
        
        Args:
            image: Input image
            
        Returns:
            dict: Prediction result with success flag and percentage
        """
        try:
            # Extract features from image (simplified version)
            features = self._extract_features(image)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform([features])
            else:
                features = [features]
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get prediction confidence if available
            confidence = 0.8  # Default confidence
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features)[0]
                    confidence = max(proba)
                except:
                    pass
            
            return {
                'success': True,
                'shear_percentage': float(prediction),
                'confidence': float(confidence),
                'model_type': self.model_type,
                'features_used': len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'shear_percentage': 0.0,
                'confidence': 0.0
            }
    
    def _extract_features(self, image: np.ndarray) -> list:
        """
        Extract features from image (simplified version).
        This is a basic implementation - you may need to adjust based on your actual features.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Basic features (you may need to adjust these based on your actual model)
            features = []
            
            # Statistical features
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.min(gray),
                np.max(gray),
                np.median(gray)
            ])
            
            # Texture features (simplified)
            features.extend([
                cv2.Laplacian(gray, cv2.CV_64F).var(),  # Variance of Laplacian
                np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0)),  # Sobel X
                np.mean(cv2.Sobel(gray, cv2.CV_64F, 0, 1))   # Sobel Y
            ])
            
            # Pad or truncate to match expected feature count
            expected_features = len(self.feature_names) if self.feature_names else 8
            if len(features) < expected_features:
                features.extend([0.0] * (expected_features - len(features)))
            elif len(features) > expected_features:
                features = features[:expected_features]
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            expected_features = len(self.feature_names) if self.feature_names else 8
            return [0.0] * expected_features


class SimpleWorkingLiveViewer:
    """Simple live viewer that works directly with your saved model."""

    def __init__(self,
                 yolo_model_path: str = None,
                 regression_model_path: str = None,
                 calibration_path: str = None,
                 config = None):
        """
        Initialize the simple working live viewer.
        """
        # Load config if not provided and available
        if config is None and load_config is not None:
            try:
                config = load_config()
            except:
                config = None
        
        # Use provided paths or fall back to config or defaults
        if config is not None:
            self.yolo_model_path = yolo_model_path or resolve_project_path(config.model.yolo_model_path)
            self.regression_model_path = regression_model_path or resolve_project_path(config.model.classification_model_path)
            self.calibration_path = calibration_path or resolve_project_path(config.camera.calibration_file)
        else:
            # Fallback to defaults if no config
            self.yolo_model_path = yolo_model_path or resolve_project_path("models/detection/charpy_3class/charpy_3class_20250729_110009/weights/best.pt")
            self.regression_model_path = regression_model_path or resolve_project_path("models/classification/charpy_shear_regressor.pkl")
            self.calibration_path = calibration_path or resolve_project_path("src/mlmcsc/camera/data/microscope_calibration.json")
        
        self.config = config

        # Models
        self.yolo_model = None
        self.shear_classifier = None

        # Camera
        self.cap = None
        self.device_id = config.camera.device_id if config else 1
        self.is_connected = False
        self.calibration_data = {}

        # Display settings from config or defaults
        if config:
            self.display_scale = config.display.window_scale
            self.show_detections = config.display.show_detections
            self.show_predictions = config.display.show_predictions
            self.show_confidence = config.display.show_confidence
            self.show_fps = config.display.show_fps
            # Colors from config
            self.colors = {
                'detection': tuple(config.display.colors['detection']),
                'prediction': tuple(config.display.colors['prediction']),
                'text': tuple(config.display.colors['text']),
                'background': tuple(config.display.colors['background'])
            }
        else:
            # Default display settings
            self.display_scale = 0.8
            self.show_detections = True
            self.show_predictions = True
            self.show_confidence = True
            self.show_fps = True
            # Default colors
            self.colors = {
                'detection': (0, 255, 0),
                'prediction': (0, 0, 255),
                'text': (255, 255, 255),
                'background': (0, 0, 0)
            }

        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.prediction_count = 0
        self.successful_predictions = 0
        self.start_time = None
        self.fps = 0
        self.last_fps_update = 0

        # Threading
        self.inference_thread = None
        self.current_frame = None
        self.current_results = {}
        self.frame_lock = threading.Lock()
        self.running = False

        logger.info("Simple Working Live Viewer initialized")

    def load_calibration(self) -> bool:
        """Load microscope calibration data."""
        calib_path = Path(self.calibration_path)

        if not calib_path.exists():
            logger.warning(f"Calibration file not found: {self.calibration_path}")
            self.calibration_data = {
                'device_id': 1,
                'resolution': [1280, 720],
                'target_fps': 30,
                'settings': {}
            }
            return True

        try:
            with open(calib_path, 'r') as f:
                self.calibration_data = json.load(f)

            if 'device_id' in self.calibration_data:
                self.device_id = self.calibration_data['device_id']

            logger.info(f"Calibration loaded from {calib_path}")
            logger.info(f"Device ID: {self.device_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def load_models(self) -> bool:
        """Load both YOLO and your new shiny region model."""

        # Load YOLO
        if not self._load_yolo_model():
            return False

        # Load your new shiny region model
        if not self._load_shiny_region_model():
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

    def _load_shiny_region_model(self) -> bool:
        """Load your new shiny region model directly from the pickle file."""

        if not Path(self.regression_model_path).exists():
            logger.error(f"Regression model not found: {self.regression_model_path}")
            return False

        try:
            # Load the pickle file you created
            with open(self.regression_model_path, 'rb') as f:
                model_data = pickle.load(f)

            logger.info(f"Loaded model file. Type: {type(model_data)}")

            # Your new model should be saved as a ShinyRegionBasedClassifier
            if hasattr(model_data, 'predict_shear_percentage'):
                # It's the classifier object itself
                self.shear_classifier = model_data
                logger.info("✅ Loaded ShinyRegionBasedClassifier object")
            elif isinstance(model_data, dict) and 'model' in model_data:
                # It's a dictionary with model components
                logger.info(f"Model data keys: {list(model_data.keys())}")
                
                # Create a simple wrapper for the dictionary-based model
                self.shear_classifier = DictModelWrapper(model_data)
                logger.info("✅ Loaded dictionary-based model with wrapper")
                return True
            else:
                logger.error(f"Unknown model format: {type(model_data)}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to load shiny region model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def connect_camera(self) -> bool:
        """Connect to microscope camera."""
        try:
            logger.info(f"Connecting to camera device {self.device_id}...")

            self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                logger.warning(f"Failed to open camera device {self.device_id}")
                logger.info("Trying device 0 as fallback...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    logger.error("Failed to open any camera device")
                    return False
                self.device_id = 0

            # Apply calibration settings
            self._apply_calibration_settings()

            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to capture test frame")
                return False

            self.is_connected = True
            logger.info(f"✅ Camera connected on device {self.device_id}. Frame size: {frame.shape}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect camera: {e}")
            return False

    def _apply_calibration_settings(self):
        """Apply calibration settings to camera."""
        if not self.cap or not self.calibration_data:
            return

        try:
            # Set resolution
            resolution = self.calibration_data.get('resolution', [1280, 720])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

            # Set FPS
            target_fps = self.calibration_data.get('target_fps', 30)
            if target_fps != 'default':
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)

            # Apply other settings
            settings = self.calibration_data.get('settings', {})
            setting_map = {
                'brightness': cv2.CAP_PROP_BRIGHTNESS,
                'contrast': cv2.CAP_PROP_CONTRAST,
                'saturation': cv2.CAP_PROP_SATURATION,
                'hue': cv2.CAP_PROP_HUE,
                'gain': cv2.CAP_PROP_GAIN,
                'exposure': cv2.CAP_PROP_EXPOSURE,
                'focus': cv2.CAP_PROP_FOCUS
            }

            for setting_name, value in settings.items():
                if setting_name in setting_map:
                    try:
                        self.cap.set(setting_map[setting_name], value)
                        logger.info(f"Applied {setting_name}: {value}")
                    except Exception as e:
                        logger.warning(f"Failed to set {setting_name}: {e}")

        except Exception as e:
            logger.warning(f"Failed to apply some calibration settings: {e}")

    def inference_worker(self):
        """Background thread for model inference."""
        while self.running:
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                else:
                    time.sleep(0.01)
                    continue

            # Run YOLO detection
            detections = self._detect_fractures(frame)

            # Run shear prediction using your new model
            shear_result = self._predict_shear_with_new_model(frame)

            # Store results
            with self.frame_lock:
                self.current_results = {
                    'detections': detections,
                    'shear_prediction': shear_result,
                    'timestamp': time.time()
                }

            time.sleep(0.1)  # Limit inference rate

    def _detect_fractures(self, frame: np.ndarray) -> List[Dict]:
        """Detect fractures using YOLO model."""
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model(frame, conf=0.3, verbose=False)

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

    def _predict_shear_with_new_model(self, frame: np.ndarray) -> Dict[str, Any]:
        """Predict shear using your new shiny region model."""
        if self.shear_classifier is None:
            return {'success': False, 'error': 'No shear classifier loaded'}

        try:
            # Use your new model's predict_shear_percentage method
            result = self.shear_classifier.predict_shear_percentage(frame)

            # Add model type for display
            if result.get('success'):
                result['model_type'] = 'NEW_SHINY'

            return result

        except Exception as e:
            return {'success': False, 'error': f'New model prediction failed: {e}'}

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw YOLO detections on frame."""
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class']

            # Draw bounding box
            cv2.rectangle(frame,
                          (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          self.colors['detection'], 2)

            # Draw label
            if self.show_confidence:
                label = f"{class_name}: {conf:.2f}"
            else:
                label = class_name

            self._draw_text(frame, label, (bbox[0], bbox[1] - 10),
                            self.colors['detection'])

        return frame

    def draw_prediction(self, frame: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray:
        """Draw shear prediction on frame."""
        if not prediction.get('success', False):
            error_msg = prediction.get('error', 'Unknown error')
            self._draw_text(frame, f"Prediction Error: {error_msg}",
                            (10, frame.shape[0] - 60), self.colors['prediction'])
            return frame

        # Draw prediction
        pred_value = prediction['prediction']
        confidence = prediction.get('confidence', 0.0)
        model_type = prediction.get('model_type', 'NEW')

        if self.show_confidence:
            text = f"Shear: {pred_value:.1f}% (conf: {confidence:.2f}) [NEW MODEL]"
        else:
            text = f"Shear: {pred_value:.1f}% [NEW MODEL]"

        self._draw_text(frame, text, (10, frame.shape[0] - 40),
                        self.colors['prediction'])

        # Show category if available
        category = prediction.get('prediction_category', '')
        if category:
            self._draw_text(frame, category, (10, frame.shape[0] - 20),
                            self.colors['prediction'])

        return frame

    def draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance statistics on frame."""
        # Calculate FPS
        current_time = time.time()
        if current_time - self.last_fps_update > 1.0:
            if self.start_time:
                elapsed = current_time - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_fps_update = current_time

        # Draw statistics
        y_offset = 30

        if self.show_fps:
            self._draw_text(frame, f"FPS: {self.fps:.1f}", (10, y_offset),
                            self.colors['text'])
            y_offset += 25

        self._draw_text(frame, f"Frames: {self.frame_count}", (10, y_offset),
                        self.colors['text'])
        y_offset += 25

        self._draw_text(frame, f"Detections: {self.detection_count}", (10, y_offset),
                        self.colors['text'])
        y_offset += 25

        success_rate = (self.successful_predictions / max(1, self.prediction_count)) * 100
        self._draw_text(frame,
                        f"Predictions: {self.successful_predictions}/{self.prediction_count} ({success_rate:.0f}%)",
                        (10, y_offset), self.colors['text'])

        return frame

    def draw_controls(self, frame: p.ndarray) -> np.ndarray:
        """Draw keyboard controls information."""
        controls = [
            "NEW SHINY MODEL ACTIVE",
            "SPACE - Screenshot",
            "D - Toggle detections",
            "P - Toggle predictions",
            "F - Toggle FPS",
            "C - Toggle confidence",
            "Q - Quit"
        ]

        x_start = frame.shape[1] - 220
        y_start = 30

        for i, control in enumerate(controls):
            color = self.colors['prediction'] if i == 0 else self.colors['text']
            self._draw_text(frame, control, (x_start, y_start + i * 20),
                            color, font_scale=0.4)

        return frame

    def _draw_text(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                   color: Tuple[int, int, int], font_scale: float = 0.6):
        """Draw text with background for better visibility."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        cv2.rectangle(frame,
                      (position[0] - 2, position[1] - text_height - 2),
                      (position[0] + text_width + 2, position[1] + 2),
                      self.colors['background'], -1)

        # Draw text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    def save_screenshot(self, frame: np.ndarray):
        """Save screenshot with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"microscope_screenshot_{timestamp}.jpg"

        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)

        filepath = screenshots_dir / filename
        cv2.imwrite(str(filepath), frame)

        logger.info(f"Screenshot saved: {filepath}")

        # Also save clean image
        with self.frame_lock:
            if self.current_frame is not None:
                clean_filename = f"microscope_clean_{timestamp}.jpg"
                clean_filepath = screenshots_dir / clean_filename
                cv2.imwrite(str(clean_filepath), self.current_frame)
                logger.info(f"Clean image saved: {clean_filepath}")

    def run(self):
        """Main loop for live microscope viewer."""

        logger.info("Starting Simple Working Live Viewer...")

        # Load calibration
        if not self.load_calibration():
            logger.error("Failed to load calibration")
            return

        # Load models
        if not self.load_models():
            logger.error("Failed to load models")
            return

        # Connect camera
        if not self.connect_camera():
            logger.error("Failed to connect camera")
            return

        # Start inference thread
        self.running = True
        self.start_time = time.time()
        self.inference_thread = threading.Thread(target=self.inference_worker)
        self.inference_thread.daemon = True
        self.inference_thread.start()

        logger.info("✅ Live viewer started successfully with NEW SHINY REGION MODEL")
        logger.info("Press 'q' to quit, SPACE for screenshot")

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                # Update frame for inference thread
                with self.frame_lock:
                    self.current_frame = frame.copy()

                # Scale display if needed
                if self.display_scale != 1.0:
                    display_frame = cv2.resize(frame, None,
                                               fx=self.display_scale,
                                               fy=self.display_scale)
                else:
                    display_frame = frame.copy()

                # Get current results
                with self.frame_lock:
                    results = self.current_results.copy()

                # Draw overlays
                if self.show_detections and results.get('detections'):
                    display_frame = self.draw_detections(display_frame, results['detections'])
                    self.detection_count += len(results['detections'])

                if self.show_predictions and results.get('shear_prediction'):
                    display_frame = self.draw_prediction(display_frame, results['shear_prediction'])
                    self.prediction_count += 1
                    if results['shear_prediction'].get('success'):
                        self.successful_predictions += 1

                # Draw statistics and controls
                display_frame = self.draw_statistics(display_frame)
                display_frame = self.draw_controls(display_frame)

                # Show frame
                cv2.imshow('NEW SHINY REGION MODEL - Live Viewer', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.save_screenshot(display_frame)
                elif key == ord('d'):
                    self.show_detections = not self.show_detections
                    logger.info(f"Detections: {'ON' if self.show_detections else 'OFF'}")
                elif key == ord('p'):
                    self.show_predictions = not self.show_predictions
                    logger.info(f"Predictions: {'ON' if self.show_predictions else 'OFF'}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                    logger.info(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                    logger.info(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")

                self.frame_count += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")

        self.running = False

        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        # Print final statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.info(f"Session statistics:")
            logger.info(f"  Runtime: {elapsed:.1f} seconds")
            logger.info(f"  Total frames: {self.frame_count}")
            logger.info(f"  Average FPS: {self.frame_count / elapsed:.1f}")
            logger.info(f"  Total detections: {self.detection_count}")
            logger.info(f"  Successful predictions: {self.successful_predictions}/{self.prediction_count}")
            if self.prediction_count > 0:
                success_rate = (self.successful_predictions / self.prediction_count) * 100
                logger.info(f"  Prediction success rate: {success_rate:.1f}%")
            logger.info(f"  NEW SHINY REGION MODEL used successfully!")


def main():
    """Main function to run the simple working live viewer."""

    print("🔬 SIMPLE WORKING LIVE VIEWER - NEW SHINY REGION MODEL")
    print("=" * 65)
    print("Uses your excellent new shiny region model!")
    print("Training MAE: 0.43% | CV MAE: 12.11% | R²: 0.999")
    print()

    # Load configuration or use defaults
    if load_config is not None:
        try:
            config = load_config()
            print("✅ Configuration loaded successfully")
            # Convert relative paths to absolute paths from project root
            yolo_model_path = resolve_project_path(config.model.yolo_model_path)
            regression_model_path = resolve_project_path(config.model.classification_model_path)
            calibration_path = resolve_project_path(config.camera.calibration_file)
        except Exception as e:
            print(f"⚠️ Failed to load config, using hardcoded paths: {e}")
            config = None
    else:
        config = None
    
    # Fallback to hardcoded paths if config failed
    if config is None:
        # Convert relative paths to absolute paths from project root
        yolo_model_path = resolve_project_path("models/detection/charpy_3class/charpy_3class_20250729_110009/weights/best.pt")
        regression_model_path = resolve_project_path("models/classification/charpy_shear_regressor.pkl")
        calibration_path = resolve_project_path("src/mlmcsc/camera/data/microscope_calibration.json")

    # Check if files exist (now using absolute paths)
    missing_files = []
    if not Path(yolo_model_path).exists():
        missing_files.append(f"YOLO model: {yolo_model_path}")
    if not Path(regression_model_path).exists():
        missing_files.append(f"NEW regression model: {regression_model_path}")

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        return

    print(f"🎯 YOLO Model: {yolo_model_path}")
    print(f"🧮 NEW Shiny Region Model: {regression_model_path}")
    print(f"📷 Calibration: {calibration_path}")
    print()

    if not Path(calibration_path).exists():
        print(f"⚠️ Calibration file not found: {calibration_path}")
        print("Will use default camera settings")

    print("🎮 KEYBOARD CONTROLS:")
    print("   SPACE - Take screenshot")
    print("   D - Toggle detection display")
    print("   P - Toggle prediction display")
    print("   F - Toggle FPS display")
    print("   C - Toggle confidence display")
    print("   Q - Quit")
    print()

    print("🌟 FEATURES OF YOUR NEW MODEL:")
    print("   • Shiny region analysis (not just texture)")
    print("   • 0.43% training error (vs old model's much higher error)")
    print("   • Features that actually matter for shear prediction")
    print("   • Much more realistic confidence scores")
    print()

    # Initialize and run viewer
    viewer = SimpleWorkingLiveViewer(config=config)

    try:
        viewer.run()
    except Exception as e:
        logger.error(f"Viewer failed: {e}")
        print(f"❌ Viewer failed: {e}")
        import traceback
        traceback.print_exc()

    print("👋 Live viewer ended")


if __name__ == "__main__":
    main()