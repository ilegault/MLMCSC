#!/usr/bin/env python3
"""
Example usage of the SpecimenDetector for real-time microscope specimen detection.

This script demonstrates how to integrate the object detection module with
the microscope capture system for real-time specimen detection and tracking.
"""

import sys
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.object_detector import SpecimenDetector, DetectionResult
from camera.microscope_capture import MicroscopeCapture  # Assuming this exists

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeDetectionDemo:
    """Demo class for real-time specimen detection."""
    
    def __init__(self, camera_id: int = 0, model_path: str = None):
        """
        Initialize the demo.
        
        Args:
            camera_id: Camera device ID
            model_path: Path to custom trained model (optional)
        """
        self.camera_id = camera_id
        
        # Initialize detector
        logger.info("Initializing specimen detector...")
        self.detector = SpecimenDetector(
            model_path=model_path,
            confidence_threshold=0.7,
            device='auto'
        )
        
        # Initialize camera (using OpenCV for this example)
        logger.info(f"Initializing camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Detection parameters
        self.auto_capture_enabled = True
        self.stability_frames_required = 10
        self.stable_frame_count = {}
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        logger.info("Demo initialized successfully!")
    
    def run_detection_demo(self, duration: int = 60):
        """
        Run real-time detection demo.
        
        Args:
            duration: Duration to run demo in seconds
        """
        logger.info(f"Starting detection demo for {duration} seconds...")
        logger.info("Press 'q' to quit, 's' to save current frame, 'c' to toggle auto-capture")
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            self.frame_count += 1
            
            # Run detection
            start_detect = time.time()
            results = self.detector.detect_specimen(frame)
            detect_time = time.time() - start_detect
            
            if results:
                self.detection_count += 1
                
                # Process each detection
                for result in results:
                    self._process_detection(result, frame)
            
            # Draw results on frame
            annotated_frame = self._draw_detections(frame, results)
            
            # Add performance info
            self._add_performance_info(annotated_frame, detect_time)
            
            # Display frame
            cv2.imshow('Specimen Detection Demo', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_frame(annotated_frame, results)
            elif key == ord('c'):
                self.auto_capture_enabled = not self.auto_capture_enabled
                logger.info(f"Auto-capture {'enabled' if self.auto_capture_enabled else 'disabled'}")
        
        self._print_statistics()
        self._cleanup()
    
    def _process_detection(self, result: DetectionResult, frame: np.ndarray):
        """Process a single detection result."""
        specimen_id = result.specimen_id
        
        # Track stability for auto-capture
        if result.is_stable:
            if specimen_id not in self.stable_frame_count:
                self.stable_frame_count[specimen_id] = 0
            self.stable_frame_count[specimen_id] += 1
            
            # Trigger auto-capture if specimen is stable long enough
            if (self.auto_capture_enabled and 
                self.stable_frame_count[specimen_id] >= self.stability_frames_required):
                
                self._auto_capture_specimen(result, frame)
                self.stable_frame_count[specimen_id] = 0  # Reset counter
        else:
            self.stable_frame_count[specimen_id] = 0
        
        # Log detection info
        logger.debug(f"Specimen {specimen_id}: conf={result.confidence:.2f}, "
                    f"stable={result.is_stable}, rotation={result.rotation_angle:.1f}°, "
                    f"offset=({result.center_offset[0]:.1f}, {result.center_offset[1]:.1f})")
    
    def _auto_capture_specimen(self, result: DetectionResult, frame: np.ndarray):
        """Auto-capture a stable specimen."""
        logger.info(f"Auto-capturing stable specimen {result.specimen_id}")
        
        # Extract ROI
        roi = self.detector.extract_roi(frame, result, padding=50)
        if roi is not None:
            # Save ROI
            timestamp = int(time.time())
            filename = f"auto_capture_specimen_{result.specimen_id}_{timestamp}.jpg"
            cv2.imwrite(filename, roi)
            logger.info(f"Saved auto-capture: {filename}")
            
            # Save detection metadata
            metadata = {
                'timestamp': timestamp,
                'specimen_id': result.specimen_id,
                'confidence': result.confidence,
                'bbox': result.bbox,
                'rotation_angle': result.rotation_angle,
                'center_offset': result.center_offset,
                'is_stable': result.is_stable
            }
            
            import json
            metadata_file = f"auto_capture_specimen_{result.specimen_id}_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _draw_detections(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Draw detection results on frame."""
        annotated_frame = frame.copy()
        
        for result in results:
            x, y, w, h = result.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Choose color based on stability
            color = (0, 255, 0) if result.is_stable else (0, 255, 255)  # Green if stable, yellow if not
            thickness = 3 if result.is_stable else 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw specimen ID and confidence
            label = f"ID:{result.specimen_id} ({result.confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(annotated_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw stability indicator
            if result.is_stable:
                cv2.circle(annotated_frame, (x + w - 15, y + 15), 8, (0, 255, 0), -1)
                cv2.putText(annotated_frame, "S", (x + w - 20, y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw rotation angle
            if abs(result.rotation_angle) > 5:  # Only show if significantly rotated
                angle_text = f"{result.rotation_angle:.1f}°"
                cv2.putText(annotated_frame, angle_text, (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center offset arrow (for auto-centering)
            center_x = x + w // 2
            center_y = y + h // 2
            frame_center_x = annotated_frame.shape[1] // 2
            frame_center_y = annotated_frame.shape[0] // 2
            
            # Draw line from specimen center to frame center
            cv2.arrowedLine(annotated_frame, (center_x, center_y), 
                           (frame_center_x, frame_center_y), (255, 0, 0), 2)
        
        # Draw frame center crosshair
        h, w = annotated_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
        cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
        
        return annotated_frame
    
    def _add_performance_info(self, frame: np.ndarray, detect_time: float):
        """Add performance information to frame."""
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Get detector FPS
        detector_fps = self.detector.get_fps()
        
        # Add text overlay
        info_lines = [
            f"Frame: {self.frame_count}",
            f"FPS: {fps:.1f}",
            f"Detector FPS: {detector_fps:.1f}",
            f"Detect Time: {detect_time*1000:.1f}ms",
            f"Detections: {self.detection_count}",
            f"Auto-capture: {'ON' if self.auto_capture_enabled else 'OFF'}"
        ]
        
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _save_frame(self, frame: np.ndarray, results: list):
        """Save current frame with detections."""
        timestamp = int(time.time())
        filename = f"detection_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save detection data
        detection_data = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'detections': [result.to_dict() for result in results]
        }
        
        import json
        json_filename = f"detection_data_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logger.info(f"Saved frame: {filename} and data: {json_filename}")
    
    def _print_statistics(self):
        """Print detection statistics."""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        detection_rate = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        print("\n" + "="*50)
        print("DETECTION STATISTICS")
        print("="*50)
        print(f"Total runtime: {elapsed:.1f} seconds")
        print(f"Total frames: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Frames with detections: {self.detection_count}")
        print(f"Detection rate: {detection_rate:.1%}")
        print(f"Detector stats: {self.detector.get_detection_stats()}")
        print("="*50)
    
    def _cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Demo cleanup completed")


def test_with_sample_images():
    """Test detector with sample images."""
    logger.info("Testing detector with sample images...")
    
    detector = SpecimenDetector(confidence_threshold=0.5)
    
    # Test with sample images (you would replace these with actual microscope images)
    sample_images = [
        "sample_microscope_1.jpg",
        "sample_microscope_2.jpg",
        "sample_microscope_3.jpg"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            logger.info(f"Processing {image_path}...")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load {image_path}")
                continue
            
            # Run detection
            start_time = time.time()
            results = detector.detect_specimen(image)
            detect_time = time.time() - start_time
            
            logger.info(f"Detection completed in {detect_time*1000:.1f}ms")
            logger.info(f"Found {len(results)} specimens")
            
            # Print results
            for result in results:
                print(f"  Specimen {result.specimen_id}: "
                      f"confidence={result.confidence:.3f}, "
                      f"stable={result.is_stable}, "
                      f"rotation={result.rotation_angle:.1f}°")
            
            # Save annotated image
            annotated = image.copy()
            for result in results:
                x, y, w, h = [int(v) for v in result.bbox]
                color = (0, 255, 0) if result.is_stable else (0, 255, 255)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, f"ID:{result.specimen_id}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            output_path = f"annotated_{Path(image_path).name}"
            cv2.imwrite(output_path, annotated)
            logger.info(f"Saved annotated image: {output_path}")
        else:
            logger.warning(f"Sample image not found: {image_path}")


def benchmark_detector():
    """Benchmark detector performance."""
    logger.info("Benchmarking detector performance...")
    
    detector = SpecimenDetector(confidence_threshold=0.7)
    
    # Create test image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(10):
        detector.detect_specimen(test_image)
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    
    for i in range(num_iterations):
        results = detector.detect_specimen(test_image)
        if i % 20 == 0:
            logger.info(f"Iteration {i}/{num_iterations}")
    
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    print(f"\nBenchmark Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per frame: {avg_time*1000:.1f} ms")
    print(f"Theoretical FPS: {fps:.1f}")
    print(f"Detector reported FPS: {detector.get_fps():.1f}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Specimen Detection Demo")
    parser.add_argument('--mode', choices=['demo', 'test', 'benchmark'], 
                       default='demo', help='Demo mode')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device ID')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to custom trained model')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Demo duration in seconds')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'demo':
            demo = RealTimeDetectionDemo(args.camera, args.model)
            demo.run_detection_demo(args.duration)
        elif args.mode == 'test':
            test_with_sample_images()
        elif args.mode == 'benchmark':
            benchmark_detector()
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        raise


if __name__ == "__main__":
    main()