#!/usr/bin/env python3
"""
Charpy Live Detection Test Script
Real-time detection of Charpy specimens using your trained model and microscope

Features:
- Live video feed from microscope
- Real-time detection display
- Fracture surface measurement
- Screenshot capture
- Detection statistics
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

# Import standalone config loader
try:
    from config_loader import load_config
    print("✅ Config system available")
except ImportError as e:
    print(f"Warning: Could not import config system: {e}")
    print("Falling back to hardcoded paths")
    load_config = None


class CharpyLiveDetector:
    """Live detection system for Charpy specimens."""

    def __init__(self, model_path, device_id=1):
        """
        Initialize the live detector.

        Args:
            model_path: Path to trained YOLO model
            device_id: Camera device ID (1 for your microscope)
        """
        self.model_path = Path(model_path)
        self.device_id = device_id

        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        print(f"📦 Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✅ Model loaded successfully")

        # Class information
        self.class_names = {
            0: "specimen",
            1: "corner",
            2: "fracture"
        }

        self.class_colors = {
            0: (0, 255, 0),  # Green - Specimen
            1: (255, 255, 0),  # Cyan - Corners
            2: (0, 0, 255)  # Red - Fracture surface
        }

        # Camera setup
        self.cap = None
        self.is_connected = False

        # Detection settings
        self.confidence_threshold = 0.25
        self.show_labels = True
        self.show_confidence = True
        self.show_measurement = True

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None

        # Calibration (pixels per mm - adjust based on your microscope)
        self.pixels_per_mm = 50  # Default value, calibrate for accuracy

    def connect_camera(self):
        """Connect to the microscope camera."""
        print(f"🔌 Connecting to camera (device {self.device_id})...")

        # Try different backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for backend in backends:
            self.cap = cv2.VideoCapture(self.device_id, backend)
            if self.cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None and test_frame.shape[0] > 0 and test_frame.shape[1] > 0:
                    print(f"✅ Connected using backend {backend}")
                    self.is_connected = True

                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Get actual resolution from a real frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"📐 Resolution: {width}x{height}")
                    else:
                        # Fallback to property values
                        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"📐 Resolution: {width}x{height}")

                    return True
                else:
                    # Camera opened but can't read frames
                    self.cap.release()

        print("❌ Failed to connect to camera")
        return False

    def disconnect_camera(self):
        """Disconnect from camera."""
        if self.cap:
            self.cap.release()
        self.is_connected = False
        cv2.destroyAllWindows()
        print("📷 Camera disconnected")

    def detect_frame(self, frame):
        """Run detection on a single frame."""
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        # Process detections
        detections = {
            'specimen': [],
            'corners': [],
            'fracture': None
        }

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get detection info
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())

                # Store detection
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': cls
                }

                if cls == 0:  # Specimen
                    detections['specimen'].append(detection)
                elif cls == 1:  # Corner
                    detections['corners'].append(detection)
                elif cls == 2:  # Fracture surface
                    detections['fracture'] = detection

        return detections

    def draw_detections(self, frame, detections):
        """Draw detection results on frame."""
        overlay = frame.copy()

        # Draw specimen boxes
        for det in detections['specimen']:
            self._draw_box(overlay, det, self.class_colors[0], "Specimen")

        # Draw corner boxes
        for i, det in enumerate(detections['corners']):
            self._draw_box(overlay, det, self.class_colors[1], f"C{i + 1}")

        # Draw fracture surface with measurement
        if detections['fracture']:
            det = detections['fracture']
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]

            # Draw box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.class_colors[2], 2)

            # Draw measurement line at top edge
            cv2.line(overlay, (x1, y1), (x2, y1), (255, 255, 255), 3)

            # Calculate and display measurement
            if self.show_measurement:
                width_pixels = x2 - x1
                width_mm = width_pixels / self.pixels_per_mm

                # Draw measurement text
                text = f"Fracture: {width_mm:.2f}mm"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = x1 + (width_pixels - text_size[0]) // 2
                text_y = y1 - 10

                # Background for text
                cv2.rectangle(overlay,
                              (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5),
                              (0, 0, 0), -1)

                cv2.putText(overlay, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return overlay

    def _draw_box(self, image, detection, color, label):
        """Draw a single detection box."""
        x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
        conf = detection['confidence']

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        if self.show_labels:
            text = label
            if self.show_confidence:
                text += f" {conf:.2f}"

            # Draw label
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image,
                          (x1, y1 - text_size[1] - 4),
                          (x1 + text_size[0], y1),
                          color, -1)
            cv2.putText(image, text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_info_overlay(self, frame, detections, fps):
        """Draw information overlay on frame."""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Prepare info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Specimens: {len(detections['specimen'])}",
            f"Corners: {len(detections['corners'])}",
            f"Fracture: {'Yes' if detections['fracture'] else 'No'}",
            f"Threshold: {self.confidence_threshold:.2f}"
        ]

        # Draw text
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw controls
        controls = [
            "CONTROLS:",
            "Space - Screenshot",
            "+/- - Adjust threshold",
            "L - Toggle labels",
            "C - Toggle confidence",
            "M - Toggle measurement",
            "R - Reset calibration",
            "Q/Esc - Quit"
        ]

        y_start = h - len(controls) * 20 - 20
        for i, line in enumerate(controls):
            cv2.putText(frame, line, (10, y_start + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def calibrate_measurement(self, frame):
        """Interactive calibration for pixel-to-mm ratio."""
        print("\n📏 CALIBRATION MODE")
        print("Click two points on a known distance")
        print("Press ESC to cancel")

        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)

        while len(points) < 2:
            display = frame.copy()

            # Draw existing points
            for i, pt in enumerate(points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, f"P{i + 1}", (pt[0] + 10, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw line if we have 2 points
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 255, 0), 2)

                # Calculate distance
                dist_pixels = np.sqrt((points[1][0] - points[0][0]) ** 2 +
                                      (points[1][1] - points[0][1]) ** 2)

                cv2.putText(display, f"Distance: {dist_pixels:.1f} pixels",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display, "Click 2 points, then enter distance in mm",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Calibration', display)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cv2.destroyWindow('Calibration')
                return False

        # Get distance in mm
        dist_pixels = np.sqrt((points[1][0] - points[0][0]) ** 2 +
                              (points[1][1] - points[0][1]) ** 2)

        cv2.destroyWindow('Calibration')

        try:
            dist_mm = float(input(f"Enter actual distance in mm: "))
            self.pixels_per_mm = dist_pixels / dist_mm
            print(f"✅ Calibration set: {self.pixels_per_mm:.2f} pixels/mm")
            return True
        except:
            print("❌ Invalid input")
            return False

    def save_screenshot(self, frame, detections):
        """Save annotated screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save image
        filename = f"charpy_detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        # Save detection data
        data = {
            'timestamp': timestamp,
            'detections': detections,
            'pixels_per_mm': self.pixels_per_mm,
            'confidence_threshold': self.confidence_threshold
        }

        json_filename = f"charpy_detection_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📸 Saved: {filename} and {json_filename}")

    def run(self):
        """Run live detection."""
        if not self.connect_camera():
            return

        print("\n🎬 Starting live detection...")
        print("Press 'Q' or ESC to quit\n")

        self.start_time = time.time()
        fps_time = time.time()
        fps = 0

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Failed to capture frame")
                    continue

                self.frame_count += 1

                # Run detection
                detections = self.detect_frame(frame)

                # Update detection count
                if any([detections['specimen'], detections['corners'], detections['fracture']]):
                    self.detection_count += 1

                # Draw results
                display_frame = self.draw_detections(frame, detections)

                # Calculate FPS
                if self.frame_count % 10 == 0:
                    current_time = time.time()
                    fps = 10 / (current_time - fps_time)
                    fps_time = current_time

                # Add info overlay
                self.draw_info_overlay(display_frame, detections, fps)

                # Show frame
                cv2.imshow('Charpy Live Detection', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # Space - screenshot
                    self.save_screenshot(display_frame, detections)
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('l'):
                    self.show_labels = not self.show_labels
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                elif key == ord('m'):
                    self.show_measurement = not self.show_measurement
                elif key == ord('r'):
                    self.calibrate_measurement(frame)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        finally:
            self.print_statistics()
            self.disconnect_camera()

    def print_statistics(self):
        """Print session statistics."""
        if self.start_time:
            duration = time.time() - self.start_time
            avg_fps = self.frame_count / duration if duration > 0 else 0
            detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0

            print("\n📊 SESSION STATISTICS")
            print("=" * 40)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Frames processed: {self.frame_count}")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Frames with detections: {self.detection_count}")
            print(f"Detection rate: {detection_rate:.1f}%")
            print("=" * 40)


def main():
    """Main function to run live detection test."""
    print("🔬 CHARPY LIVE DETECTION TEST")
    print("=" * 50)

    # Load configuration or use defaults
    model_path = None
    if load_config is not None:
        try:
            config = load_config()
            print("✅ Configuration loaded successfully")
            model_path = Path(config.model.yolo_model_path)
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}")
    
    # Fallback to default path if config failed
    if model_path is None:
        print("Using default model path")
        model_path = Path("models/detection/best.pt")

    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please check the path in config/default.yaml or enter a valid path")
        return

    print(f"📦 Using model: {model_path}")

    # Create detector
    try:
        detector = CharpyLiveDetector(model_path, device_id=1)

        # Run live detection
        detector.run()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()