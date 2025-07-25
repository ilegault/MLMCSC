#!/usr/bin/env python3
"""
Basic Object Detection Test with Microscope Camera

This script provides a simple test of the object detection system using your microscope camera.
It captures a few frames, runs detection, and saves the results for analysis.

This is useful for:
- Testing if the camera connection works
- Verifying object detection functionality
- Debugging without live video complexity
- Generating test data for analysis
"""

import cv2
import numpy as np
import json
import time
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.models.object_detector import SpecimenDetector, DetectionResult
    from src.camera.microscope_interface import MicroscopeCapture
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have installed all required dependencies:")
    print("pip install ultralytics opencv-python torch numpy scipy")
    IMPORTS_AVAILABLE = False
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicObjectDetectionTest:
    """Basic object detection test with microscope camera."""
    
    def __init__(self, device_id: int = 1):
        """
        Initialize the basic detection test.
        
        Args:
            device_id: Camera device ID (1 for your microscope)
        """
        self.device_id = device_id
        self.microscope: Optional[MicroscopeCapture] = None
        self.detector: Optional[SpecimenDetector] = None
        
        # Test settings
        self.num_test_frames = 5
        self.frame_delay = 1.0  # seconds between captures
        
        logger.info(f"BasicObjectDetectionTest initialized for device {device_id}")
    
    def load_camera_calibration(self) -> Dict[str, Any]:
        """Load camera calibration from your calibration file."""
        calibration_file = Path("data/camera_calibration.json")
        
        if calibration_file.exists():
            try:
                with open(calibration_file, 'r') as f:
                    calibration = json.load(f)
                logger.info(f"Loaded camera calibration from {calibration_file}")
                logger.info(f"Device ID: {calibration.get('device_id')}")
                logger.info(f"Resolution: {calibration.get('resolution')}")
                logger.info(f"Target FPS: {calibration.get('target_fps')}")
                return calibration
            except Exception as e:
                logger.error(f"Error loading calibration: {e}")
        else:
            logger.warning(f"Calibration file not found: {calibration_file}")
        
        return {}
    
    def test_camera_connection(self) -> bool:
        """Test camera connection and basic functionality."""
        logger.info("=== Testing Camera Connection ===")
        
        try:
            # Load calibration settings
            calibration = self.load_camera_calibration()
            
            # Get settings from calibration or use defaults
            resolution = tuple(calibration.get('resolution', [1280, 720]))
            target_fps = calibration.get('target_fps', 30)
            settings = calibration.get('settings', {})
            
            logger.info(f"Attempting to connect to device {self.device_id}")
            logger.info(f"Target resolution: {resolution}")
            logger.info(f"Target FPS: {target_fps}")
            
            # Initialize microscope capture
            self.microscope = MicroscopeCapture(
                device_id=self.device_id,
                target_fps=target_fps,
                resolution=resolution
            )
            
            # Connect to camera
            if not self.microscope.connect():
                logger.error("❌ Failed to connect to microscope camera")
                return False
            
            logger.info("✅ Successfully connected to microscope camera")
            
            # Test frame capture
            logger.info("Testing frame capture...")
            frame = self.microscope.get_frame()
            
            if frame is None:
                logger.error("❌ Failed to capture test frame")
                return False
            
            logger.info(f"✅ Successfully captured frame: {frame.shape}")
            
            # Apply calibration settings if available
            if settings:
                logger.info("Applying calibration settings...")
                results = self.microscope.adjust_settings(**settings)
                success_count = sum(results.values())
                logger.info(f"Applied {success_count}/{len(results)} settings successfully")
                
                for setting, success in results.items():
                    status = "✅" if success else "❌"
                    logger.info(f"  {status} {setting}: {settings[setting]}")
            
            # Save test frame
            test_frame_path = "data/test_camera_frame.jpg"
            cv2.imwrite(test_frame_path, frame)
            logger.info(f"✅ Saved test frame: {test_frame_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera connection test failed: {e}")
            return False
    
    def test_object_detector(self) -> bool:
        """Test object detector initialization."""
        logger.info("=== Testing Object Detector ===")
        
        try:
            logger.info("Initializing object detector...")
            
            # Initialize detector with settings optimized for microscopy
            self.detector = SpecimenDetector(
                model_path=None,  # Use pre-trained YOLOv8
                confidence_threshold=0.25,  # Lower threshold for microscopy
                nms_threshold=0.4,
                device='auto',  # Use GPU if available
                max_detections=20  # Allow more detections for specimens
            )
            
            logger.info("✅ Object detector initialized successfully")
            
            # Get detector stats
            stats = self.detector.get_detection_stats()
            logger.info(f"Detector device: {stats['device']}")
            logger.info(f"Confidence threshold: {stats['confidence_threshold']}")
            logger.info(f"NMS threshold: {stats['nms_threshold']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Object detector test failed: {e}")
            return False
    
    def run_detection_test(self) -> bool:
        """Run detection on captured frames."""
        logger.info("=== Running Detection Test ===")
        
        if not self.microscope or not self.detector:
            logger.error("❌ Camera or detector not initialized")
            return False
        
        try:
            all_results = []
            
            for i in range(self.num_test_frames):
                logger.info(f"Capturing and analyzing frame {i+1}/{self.num_test_frames}...")
                
                # Capture frame
                frame = self.microscope.get_frame()
                if frame is None:
                    logger.warning(f"❌ Failed to capture frame {i+1}")
                    continue
                
                # Run detection
                start_time = time.time()
                detections = self.detector.detect_specimen(frame)
                detection_time = time.time() - start_time
                
                # Log results
                logger.info(f"Frame {i+1}: {len(detections)} detections in {detection_time:.3f}s")
                
                # Process each detection
                for j, detection in enumerate(detections):
                    logger.info(f"  Detection {j+1}:")
                    logger.info(f"    Specimen ID: {detection.specimen_id}")
                    logger.info(f"    Confidence: {detection.confidence:.3f}")
                    logger.info(f"    Bbox: {[f'{x:.1f}' for x in detection.bbox]}")
                    logger.info(f"    Stable: {detection.is_stable}")
                    logger.info(f"    Rotation: {detection.rotation_angle:.1f}°")
                    logger.info(f"    Center offset: {[f'{x:.1f}' for x in detection.center_offset]}")
                
                # Save frame with detections
                annotated_frame = self.draw_detections(frame, detections)
                frame_filename = f"detection_test_frame_{i+1}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                
                # Store results
                frame_results = {
                    'frame_number': i + 1,
                    'detection_time': detection_time,
                    'num_detections': len(detections),
                    'detections': [detection.to_dict() for detection in detections]
                }
                all_results.append(frame_results)
                
                logger.info(f"✅ Saved annotated frame: {frame_filename}")
                
                # Wait before next frame
                if i < self.num_test_frames - 1:
                    time.sleep(self.frame_delay)
            
            # Save all results
            results_filename = "data/detection_test_results.json"
            with open(results_filename, 'w') as f:
                json.dump({
                    'test_timestamp': time.time(),
                    'device_id': self.device_id,
                    'num_frames': len(all_results),
                    'detector_stats': self.detector.get_detection_stats(),
                    'results': all_results
                }, f, indent=2)
            
            logger.info(f"✅ Saved test results: {results_filename}")
            
            # Summary statistics
            total_detections = sum(r['num_detections'] for r in all_results)
            avg_detections = total_detections / len(all_results) if all_results else 0
            avg_time = sum(r['detection_time'] for r in all_results) / len(all_results) if all_results else 0
            
            logger.info("=== Test Summary ===")
            logger.info(f"Frames processed: {len(all_results)}")
            logger.info(f"Total detections: {total_detections}")
            logger.info(f"Average detections per frame: {avg_detections:.2f}")
            logger.info(f"Average detection time: {avg_time:.3f}s")
            logger.info(f"Average FPS: {1/avg_time:.1f}" if avg_time > 0 else "N/A")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Detection test failed: {e}")
            return False
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection results on the frame."""
        annotated_frame = frame.copy()
        
        # Colors for different specimens
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for detection in detections:
            # Get bounding box coordinates
            x, y, w, h = [int(coord) for coord in detection.bbox]
            
            # Choose color based on specimen ID
            color = colors[detection.specimen_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label = f"ID:{detection.specimen_id} {detection.confidence:.2f}"
            if detection.is_stable:
                label += " STABLE"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)
        
        # Add frame info
        info_text = f"Detections: {len(detections)}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        return annotated_frame
    
    def run(self) -> bool:
        """Run the complete basic detection test."""
        logger.info("Starting basic object detection test...")
        
        success = True
        
        try:
            # Test camera connection
            if not self.test_camera_connection():
                success = False
                return success
            
            # Test object detector
            if not self.test_object_detector():
                success = False
                return success
            
            # Run detection test
            if not self.run_detection_test():
                success = False
                return success
            
            logger.info("✅ All tests completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            success = False
        finally:
            self.cleanup()
        
        return success
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self.microscope:
            self.microscope.disconnect()
        
        logger.info("Cleanup completed")


def main():
    """Main function to run the basic object detection test."""
    print("=== Basic Object Detection Test ===")
    print("Testing object detection with microscope camera (device_id: 1)")
    print("This will capture a few frames and test detection functionality.")
    print()
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available. Please install dependencies.")
        return
    
    # Create and run the test
    test = BasicObjectDetectionTest(device_id=1)  # Your microscope camera
    
    try:
        success = test.run()
        if success:
            print("\n✅ Test completed successfully!")
            print("Check the generated files:")
            print("  - test_camera_frame.jpg (camera test)")
            print("  - detection_test_frame_*.jpg (detection results)")
            print("  - detection_test_results.json (detailed results)")
        else:
            print("\n❌ Test failed. Check the logs above for details.")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()