#!/usr/bin/env python3
"""
Live Object Detection Test with Microscope Camera

This script tests the object detection system using your microscope camera (device_id: 1).
It captures live video from the microscope and runs real-time object detection,
displaying results with bounding boxes, tracking IDs, and detection statistics.

Features:
- Live video capture from microscope camera
- Real-time object detection using YOLOv8
- Visual feedback with bounding boxes and labels
- Detection statistics and performance monitoring
- Keyboard controls for interaction
- Automatic camera calibration loading
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


class LiveObjectDetectionTest:
    """Live object detection test with microscope camera."""
    
    def __init__(self, device_id: int = 1):
        """
        Initialize the live detection test.
        
        Args:
            device_id: Camera device ID (1 for your microscope)
        """
        self.device_id = device_id
        self.microscope: Optional[MicroscopeCapture] = None
        self.detector: Optional[SpecimenDetector] = None
        self.is_running = False
        
        # Display settings
        self.display_scale = 0.8  # Scale down display for better fit
        self.show_fps = True
        self.show_stats = True
        self.show_tracking_ids = True
        
        # Colors for visualization
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        logger.info(f"LiveObjectDetectionTest initialized for device {device_id}")
    
    def load_camera_calibration(self) -> Dict[str, Any]:
        """Load camera calibration from your calibration file."""
        calibration_file = Path("data/camera_calibration.json")
        
        if calibration_file.exists():
            try:
                with open(calibration_file, 'r') as f:
                    calibration = json.load(f)
                logger.info(f"Loaded camera calibration: {calibration}")
                return calibration
            except Exception as e:
                logger.error(f"Error loading calibration: {e}")
        else:
            logger.warning(f"Calibration file not found: {calibration_file}")
        
        return {}
    
    def initialize_camera(self) -> bool:
        """Initialize the microscope camera."""
        try:
            logger.info("Initializing microscope camera...")
            
            # Load calibration settings
            calibration = self.load_camera_calibration()
            
            # Get settings from calibration or use defaults
            resolution = tuple(calibration.get('resolution', [1280, 720]))
            target_fps = calibration.get('target_fps', 30)
            settings = calibration.get('settings', {})
            
            # Initialize microscope capture
            self.microscope = MicroscopeCapture(
                device_id=self.device_id,
                target_fps=target_fps,
                resolution=resolution
            )
            
            # Connect to camera
            if not self.microscope.connect():
                logger.error("Failed to connect to microscope camera")
                return False
            
            # Apply calibration settings if available
            if settings:
                logger.info("Applying calibration settings...")
                results = self.microscope.adjust_settings(**settings)
                for setting, success in results.items():
                    if success:
                        logger.info(f"Applied {setting}: {settings[setting]}")
                    else:
                        logger.warning(f"Failed to apply {setting}")
            
            logger.info("Microscope camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def initialize_detector(self) -> bool:
        """Initialize the object detector."""
        try:
            logger.info("Initializing object detector...")
            
            # Initialize detector with optimized settings for microscopy
            self.detector = SpecimenDetector(
                model_path=None,  # Use pre-trained YOLOv8
                confidence_threshold=0.3,  # Lower threshold for microscopy
                nms_threshold=0.4,
                device='auto',  # Use GPU if available
                max_detections=20  # Allow more detections for specimens
            )
            
            logger.info("Object detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            return False
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection results on the frame."""
        display_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            # Get bounding box coordinates
            x, y, w, h = [int(coord) for coord in detection.bbox]
            
            # Choose color based on specimen ID
            color = self.colors[detection.specimen_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label_parts = []
            if self.show_tracking_ids:
                label_parts.append(f"ID:{detection.specimen_id}")
            label_parts.append(f"{detection.confidence:.2f}")
            if detection.is_stable:
                label_parts.append("STABLE")
            
            label = " ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                display_frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                display_frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(display_frame, (center_x, center_y), 3, color, -1)
            
            # Draw rotation indicator if significant
            if abs(detection.rotation_angle) > 5:
                angle_rad = np.radians(detection.rotation_angle)
                line_length = min(w, h) // 3
                end_x = int(center_x + line_length * np.cos(angle_rad))
                end_y = int(center_y + line_length * np.sin(angle_rad))
                cv2.line(display_frame, (center_x, center_y), (end_x, end_y), color, 2)
        
        return display_frame
    
    def draw_statistics(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw statistics on the frame."""
        if not (self.show_fps or self.show_stats):
            return frame
        
        display_frame = frame.copy()
        y_offset = 30
        
        # FPS and performance stats
        if self.show_fps and self.detector:
            fps = self.detector.get_fps()
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30
        
        # Detection stats
        if self.show_stats:
            stats_text = [
                f"Detections: {len(detections)}",
                f"Frame: {self.frame_count}",
                f"Total Detections: {self.detection_count}",
            ]
            
            if self.detector:
                detector_stats = self.detector.get_detection_stats()
                stats_text.append(f"Active Tracks: {detector_stats['active_tracks']}")
                stats_text.append(f"Device: {detector_stats['device']}")
            
            for stat in stats_text:
                cv2.putText(
                    display_frame,
                    stat,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_offset += 25
        
        # Instructions
        instructions = [
            "Controls:",
            "ESC/Q - Quit",
            "S - Toggle Stats",
            "T - Toggle Tracking IDs",
            "F - Toggle FPS",
            "SPACE - Capture Frame"
        ]
        
        y_start = frame.shape[0] - len(instructions) * 20 - 10
        for i, instruction in enumerate(instructions):
            cv2.putText(
                display_frame,
                instruction,
                (10, y_start + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
        
        return display_frame
    
    def save_detection_frame(self, frame: np.ndarray, detections: List[DetectionResult]) -> None:
        """Save a frame with detections for analysis."""
        timestamp = int(time.time())
        filename = f"detection_capture_{timestamp}.jpg"
        
        # Save original frame
        cv2.imwrite(filename, frame)
        
        # Save detection data
        detection_data = {
            'timestamp': timestamp,
            'frame_count': self.frame_count,
            'detections': [detection.to_dict() for detection in detections]
        }
        
        json_filename = f"detection_data_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logger.info(f"Saved detection frame: {filename} and data: {json_filename}")
    
    def run(self) -> None:
        """Run the live object detection test."""
        logger.info("Starting live object detection test...")
        
        # Initialize components
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return
        
        if not self.initialize_detector():
            logger.error("Failed to initialize detector")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        logger.info("Live detection started. Press ESC or Q to quit.")
        
        try:
            while self.is_running:
                # Capture frame
                frame = self.microscope.get_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Run object detection
                detections = self.detector.detect_specimen(frame)
                self.detection_count += len(detections)
                
                # Create display frame
                display_frame = self.draw_detections(frame, detections)
                display_frame = self.draw_statistics(display_frame, detections)
                
                # Scale for display
                if self.display_scale != 1.0:
                    height, width = display_frame.shape[:2]
                    new_width = int(width * self.display_scale)
                    new_height = int(height * self.display_scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                # Show frame
                cv2.imshow('Live Object Detection - Microscope', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    break
                elif key == ord('s'):  # Toggle stats
                    self.show_stats = not self.show_stats
                    logger.info(f"Stats display: {'ON' if self.show_stats else 'OFF'}")
                elif key == ord('t'):  # Toggle tracking IDs
                    self.show_tracking_ids = not self.show_tracking_ids
                    logger.info(f"Tracking IDs: {'ON' if self.show_tracking_ids else 'OFF'}")
                elif key == ord('f'):  # Toggle FPS
                    self.show_fps = not self.show_fps
                    logger.info(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
                elif key == ord(' '):  # Space - capture frame
                    self.save_detection_frame(frame, detections)
                
                # Log detection results periodically
                if self.frame_count % 100 == 0:
                    elapsed_time = time.time() - self.start_time
                    avg_detections = self.detection_count / self.frame_count if self.frame_count > 0 else 0
                    logger.info(f"Frame {self.frame_count}: {len(detections)} detections, "
                              f"Avg: {avg_detections:.2f}, Runtime: {elapsed_time:.1f}s")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        self.is_running = False
        
        if self.microscope:
            self.microscope.disconnect()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_detections = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        logger.info("=== Final Statistics ===")
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total detections: {self.detection_count}")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Average detections per frame: {avg_detections:.2f}")
        logger.info(f"Total runtime: {elapsed_time:.2f} seconds")


def main():
    """Main function to run the live object detection test."""
    print("=== Live Object Detection Test ===")
    print("Testing object detection with microscope camera (device_id: 1)")
    print()
    
    if not IMPORTS_AVAILABLE:
        print("Required modules not available. Please install dependencies.")
        return
    
    # Create and run the test
    test = LiveObjectDetectionTest(device_id=1)  # Your microscope camera
    
    try:
        test.run()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Error: {e}")
    finally:
        print("Test completed.")


if __name__ == "__main__":
    main()