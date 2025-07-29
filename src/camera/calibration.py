#!/usr/bin/env python3
"""
Fixed Calibrated Measurement System for Charpy Specimens

This script:
1. Uses YOUR trained Charpy model (no downloads!)
2. Loads YOUR actual calibration data
3. Lets you measure specimens you place in front of the camera
4. Works with your existing project structure

Usage:
1. First run calibration: python microscope_calibration.py  
2. Then use this for measurements: python calibrated_measurement.py
"""

import cv2
import numpy as np
import json
import time
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add your project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

try:
    from src.camera.microscope_interface import MicroscopeCapture
    # Only import if you want to use your trained model
    USE_DETECTION = False  # Set to True if you want automated detection
    if USE_DETECTION:
        from ultralytics import YOLO
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibratedMeasurementSystem:
    """Measurement system using YOUR calibrated microscope."""

    def __init__(self, device_id=1):
        """Initialize measurement system."""
        self.device_id = device_id
        self.microscope = None
        self.detector = None
        
        # Calibration data (loaded from your calibration file)
        self.calibration = None
        self.pixel_scale = None  # pixels per mm
        self.mm_per_pixel = None
        self.calibration_error = None
        self.calibration_date = None
        
        # Measurement history
        self.measurements = []
        self.session_start = time.time()
        
        # Your trained model path (update this!)
        self.model_path = project_root / "models" / "charpy_3class" / "best.pt"
        
        print("📏 Calibrated Measurement System Initialized")
        print(f"📷 Camera device: {self.device_id}")
        print(f"🤖 Model path: {self.model_path}")

    def load_calibration(self, calibration_file=None):
        """Load YOUR actual calibration data."""
        if calibration_file is None:
            # Try multiple possible locations
            possible_files = [
                "data/microscope_calibration.json",
                "data/camera_calibration.json", 
                "calibration_data.json",
                project_root / "data" / "microscope_calibration.json"
            ]
        else:
            possible_files = [calibration_file]
        
        for cal_file in possible_files:
            try:
                cal_path = Path(cal_file)
                if cal_path.exists():
                    with open(cal_path, 'r') as f:
                        self.calibration = json.load(f)
                    
                    # Extract calibration values
                    self.pixel_scale = self.calibration['pixel_scale']
                    self.mm_per_pixel = self.calibration['mm_per_pixel'] 
                    self.calibration_error = self.calibration['calibration_error_mm']
                    self.calibration_date = self.calibration['timestamp']
                    
                    print("✅ Calibration loaded successfully!")
                    print(f"   File: {cal_path}")
                    print(f"   Scale: {self.pixel_scale:.2f} pixels/mm")
                    print(f"   Resolution: {self.mm_per_pixel:.4f} mm/pixel")
                    print(f"   Accuracy: ±{self.calibration_error:.4f} mm")
                    print(f"   Date: {self.calibration_date}")
                    
                    return True
                    
            except Exception as e:
                print(f"Could not load {cal_file}: {e}")
                continue
        
        print("❌ No calibration file found!")
        print("Please run the calibration script first:")
        print("   python microscope_calibration.py")
        return False

    def connect_microscope(self):
        """Connect to microscope with calibrated settings."""
        print("🔌 Connecting to microscope...")
        
        self.microscope = MicroscopeCapture(
            device_id=self.device_id,
            target_fps=30,
            resolution=(1280, 720)
        )
        
        if self.microscope.connect():
            # Apply calibration camera settings if available
            if self.calibration and 'camera_settings' in self.calibration:
                settings = self.calibration['camera_settings']
                if settings:
                    print("🔧 Applying calibrated camera settings...")
                    self.microscope.adjust_settings(**settings)
            
            print("✅ Microscope connected")
            return True
        else:
            print("❌ Failed to connect to microscope")
            return False

    def load_your_model(self):
        """Load YOUR trained Charpy model (optional)."""
        if not USE_DETECTION:
            print("🤖 Detection disabled - manual measurement only")
            return False
            
        try:
            if self.model_path.exists():
                print(f"🤖 Loading YOUR trained model: {self.model_path}")
                self.detector = YOLO(str(self.model_path))
                print("✅ Your model loaded successfully!")
                return True
            else:
                print(f"❌ Model not found: {self.model_path}")
                print("Available options:")
                print("1. Train your model first")
                print("2. Update model_path in the script")
                print("3. Use manual measurement mode")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def pixels_to_mm(self, pixels):
        """Convert pixels to millimeters using YOUR calibration."""
        if self.pixel_scale is None:
            raise ValueError("Calibration not loaded!")
        return pixels / self.pixel_scale

    def mm_to_pixels(self, mm):
        """Convert millimeters to pixels using YOUR calibration."""
        if self.pixel_scale is None:
            raise ValueError("Calibration not loaded!")
        return mm * self.pixel_scale

    def measure_distance(self, point1, point2):
        """Measure distance between two points in mm."""
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        mm_distance = self.pixels_to_mm(pixel_distance)
        
        return {
            'distance_mm': mm_distance,
            'distance_pixels': pixel_distance,
            'accuracy_mm': self.calibration_error,
            'points': [point1, point2]
        }

    def interactive_measurement(self, image, instruction="Click two points to measure distance"):
        """Interactive measurement - click two points on your specimen."""
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
        
        # Create window and set callback
        cv2.namedWindow('Measurement', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Measurement', mouse_callback)
        
        print(f"\n📏 {instruction}")
        print("Left-click two points on your specimen, then press ENTER")
        
        while True:
            display_image = image.copy()
            
            # Draw existing points
            for i, point in enumerate(points):
                cv2.circle(display_image, point, 8, (0, 255, 0), -1)
                cv2.putText(display_image, f"P{i+1}", (point[0]+15, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw line if we have 2 points
            if len(points) == 2:
                cv2.line(display_image, points[0], points[1], (0, 255, 0), 3)
                
                # Calculate and display distance
                measurement = self.measure_distance(points[0], points[1])
                distance_text = f"Distance: {measurement['distance_mm']:.3f} mm"
                accuracy_text = f"Accuracy: +/-{self.calibration_error:.3f} mm"
                
                cv2.putText(display_image, distance_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(display_image, accuracy_text, (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Instructions
            cv2.putText(display_image, instruction, (10, image.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, "Click 2 points, ENTER=Accept, C=Clear, ESC=Cancel", 
                       (10, image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Measurement', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) == 2:  # Enter
                measurement = self.measure_distance(points[0], points[1])
                cv2.destroyAllWindows()
                return measurement
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == ord('c'):  # Clear points
                points.clear()

    def automated_detection(self, image):
        """Use YOUR trained model to detect fracture surfaces."""
        if not self.detector:
            return []
        
        try:
            # Run YOUR model on the image
            results = self.detector(image, conf=0.25, verbose=False)
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Your class names (update based on your model)
                    class_names = {0: "specimen", 1: "corner", 2: "fracture_surface"}
                    
                    detection = {
                        'class': cls,
                        'class_name': class_names.get(cls, f"class_{cls}"),
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'width_pixels': float(x2 - x1),
                        'height_pixels': float(y2 - y1)
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def live_measurement_mode(self):
        """Live measurement mode - place specimens and measure them."""
        if not self.microscope:
            print("❌ Microscope not connected")
            return
        
        print("\n🎬 LIVE MEASUREMENT MODE")
        print("=" * 50)
        print("📋 Instructions:")
        print("1. Place your Charpy specimen in front of the camera")
        print("2. Press SPACE to capture and measure")
        print("3. Click two points on the specimen to measure distance")
        print("4. Results will be saved automatically")
        print()
        print("🎮 Controls:")
        print("  SPACE - Capture specimen and measure")
        print("  A - Auto detection (if model loaded)")
        print("  S - Save current frame without measuring")
        print("  Q - Quit")
        print()
        
        specimen_count = 0
        
        while True:
            frame = self.microscope.get_frame()
            if frame is None:
                continue
            
            display_frame = frame.copy()
            
            # Add live info overlay
            self.add_live_overlay(display_frame, specimen_count)
            
            # Auto detection if available
            detections = []
            if self.detector:
                detections = self.automated_detection(frame)
                self.draw_detections(display_frame, detections)
            
            cv2.imshow('Live Measurement - Place Your Specimen', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - Capture and measure
                specimen_count += 1
                self.capture_and_measure_specimen(frame, specimen_count)
            
            elif key == ord('a') and self.detector:  # Auto measurement
                auto_measurements = self.auto_measure_detections(detections, specimen_count)
                if auto_measurements:
                    specimen_count += len(auto_measurements)
                    print(f"🤖 Auto-measured {len(auto_measurements)} features")
                else:
                    print("🔍 No detectable features for auto-measurement")
            
            elif key == ord('s'):  # Save frame only
                timestamp = int(time.time())
                filename = f"specimen_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved: {filename}")
            
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        print(f"\n✅ Measurement session completed!")
        print(f"📊 Total specimens measured: {specimen_count}")

    def capture_and_measure_specimen(self, frame, specimen_number):
        """Capture a specimen and measure it interactively."""
        print(f"\n🔬 SPECIMEN {specimen_number}")
        print("=" * 30)
        
        # Save the specimen image
        timestamp = int(time.time())
        image_filename = f"specimen_{specimen_number:03d}_{timestamp}.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"📸 Specimen image saved: {image_filename}")
        
        # Interactive measurement
        measurement = self.interactive_measurement(
            frame, 
            f"Measure Specimen {specimen_number} - Click two points"
        )
        
        if measurement:
            # Create measurement record
            measurement_record = {
                'specimen_number': specimen_number,
                'distance_mm': measurement['distance_mm'],
                'distance_pixels': measurement['distance_pixels'],
                'accuracy_mm': measurement['accuracy_mm'],
                'points': measurement['points'],
                'image_file': image_filename,
                'timestamp': datetime.now().isoformat(),
                'calibration_scale': self.pixel_scale
            }
            
            # Add to measurements list
            self.measurements.append(measurement_record)
            
            # Save individual measurement
            measurement_filename = f"measurement_{specimen_number:03d}_{timestamp}.json"
            with open(measurement_filename, 'w') as f:
                json.dump(measurement_record, f, indent=2)
            
            # Display results
            print(f"✅ MEASUREMENT RESULTS:")
            print(f"   Distance: {measurement['distance_mm']:.3f} ±{self.calibration_error:.3f} mm")
            print(f"   Image: {image_filename}")
            print(f"   Data: {measurement_filename}")
            
        else:
            print("❌ Measurement cancelled")

    def auto_measure_detections(self, detections, base_specimen_number):
        """Automatically measure detected features."""
        measurements = []
        
        for i, detection in enumerate(detections):
            if detection['class_name'] == 'fracture_surface':
                # Measure fracture surface
                width_mm = self.pixels_to_mm(detection['width_pixels'])
                height_mm = self.pixels_to_mm(detection['height_pixels'])
                
                auto_measurement = {
                    'specimen_number': f"{base_specimen_number}_auto_{i+1}",
                    'type': 'auto_fracture_surface',
                    'width_mm': width_mm,
                    'height_mm': height_mm,
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox'],
                    'timestamp': datetime.now().isoformat(),
                    'accuracy_mm': self.calibration_error
                }
                
                measurements.append(auto_measurement)
                self.measurements.append(auto_measurement)
                
                print(f"🤖 Auto-measurement {i+1}:")
                print(f"   Fracture width: {width_mm:.3f} ±{self.calibration_error:.3f} mm")
                print(f"   Fracture height: {height_mm:.3f} ±{self.calibration_error:.3f} mm")
                print(f"   Confidence: {detection['confidence']:.2f}")
        
        return measurements

    def draw_detections(self, image, detections):
        """Draw detection results on the image."""
        for detection in detections:
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            
            # Color based on class
            if detection['class_name'] == 'fracture_surface':
                color = (0, 0, 255)  # Red for fracture
            elif detection['class_name'] == 'specimen':
                color = (0, 255, 0)  # Green for specimen
            else:
                color = (255, 255, 0)  # Yellow for others
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def add_live_overlay(self, frame, specimen_count):
        """Add information overlay to live frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Info text
        info_lines = [
            f"Calibration: {self.pixel_scale:.1f} px/mm",
            f"Accuracy: ±{self.calibration_error:.3f} mm",
            f"Specimens measured: {specimen_count}",
            f"Total measurements: {len(self.measurements)}",
            "SPACE=Measure | A=Auto | S=Save | Q=Quit"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def export_measurements(self, filename=None):
        """Export all measurements to Excel."""
        if not self.measurements:
            print("📊 No measurements to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"charpy_measurements_{timestamp}.xlsx"
        
        # Convert to DataFrame
        df = pd.DataFrame(self.measurements)
        
        # Create summary
        summary_data = {
            'session_start': datetime.fromtimestamp(self.session_start).isoformat(),
            'session_duration_minutes': (time.time() - self.session_start) / 60,
            'total_measurements': len(self.measurements),
            'calibration_date': self.calibration_date,
            'calibration_accuracy_mm': self.calibration_error,
            'pixel_scale': self.pixel_scale,
            'mm_per_pixel': self.mm_per_pixel
        }
        
        # Statistical analysis
        distances = [m.get('distance_mm', 0) for m in self.measurements if 'distance_mm' in m]
        if distances:
            summary_data.update({
                'mean_distance_mm': np.mean(distances),
                'std_distance_mm': np.std(distances),
                'min_distance_mm': np.min(distances),
                'max_distance_mm': np.max(distances)
            })
        
        # Save to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Measurements', index=False)
            
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"📊 Measurements exported to: {filename}")
        print(f"   Total measurements: {len(self.measurements)}")
        
        return filename

    def run_measurement_session(self):
        """Run complete measurement session."""
        print("🚀 STARTING CALIBRATED MEASUREMENT SESSION")
        print("=" * 60)
        
        # Step 1: Load calibration
        if not self.load_calibration():
            print("❌ Cannot proceed without calibration!")
            print("Run: python microscope_calibration.py")
            return False
        
        # Step 2: Connect microscope
        if not self.connect_microscope():
            return False
        
        # Step 3: Load model (optional)
        self.load_your_model()
        
        try:
            # Step 4: Start measurement session
            self.live_measurement_mode()
            
            # Step 5: Export results
            if self.measurements:
                export = input("\nExport measurements to Excel? (Y/n): ")
                if export.lower() != 'n':
                    filename = self.export_measurements()
                    if filename:
                        print(f"✅ Session data exported to: {filename}")
            
            print("\n✅ Measurement session completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error during measurement session: {e}")
            return False
        finally:
            if self.microscope:
                self.microscope.disconnect()


def main():
    """Main function."""
    print("📏 CHARPY SPECIMEN MEASUREMENT SYSTEM")
    print("=" * 60)
    print("This system uses YOUR calibrated microscope to measure Charpy specimens.")
    print()
    print("📋 Prerequisites:")
    print("1. Microscope must be calibrated (run microscope_calibration.py first)")
    print("2. Camera connected to device 1")
    print("3. Specimens ready for measurement")
    print()
    
    # Check for calibration
    calibration_files = [
        "data/microscope_calibration.json",
        "data/camera_calibration.json"
    ]
    
    calibration_found = any(Path(f).exists() for f in calibration_files)
    
    if not calibration_found:
        print("⚠️ WARNING: No calibration file found!")
        print("You must calibrate your microscope first:")
        print("   python microscope_calibration.py")
        print()
        proceed = input("Continue anyway with default values? (y/N): ")
        if proceed.lower() != 'y':
            return
    
    # Initialize and run system
    system = CalibratedMeasurementSystem(device_id=1)
    system.run_measurement_session()


if __name__ == "__main__":
    main()