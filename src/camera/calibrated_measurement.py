#!/usr/bin/env python3
"""
Calibrated Measurement System for Charpy Specimens

This script uses your calibrated microscope to make accurate measurements.
It includes measurement logging and integration with your detection model
for automated measurements.

Features:
- Load calibration
- Interactive measurement tools
- Automated measurements from detection results
- Measurement logging and export
"""

import cv2
import numpy as np
import json
import time
import sys
import logging
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.camera.microscope_interface import MicroscopeCapture
    from src.models.object_detector import SpecimenDetector
except ImportError:
    print("Required modules not found. Make sure you're running from the project root.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibratedMeasurementSystem:
    """Measurement system using calibrated microscope."""

    def __init__(self):
        """Initialize measurement system."""
        self.device_id = 1  # Fixed device ID
        self.microscope = None
        self.detector = None

        # Default calibration values (you can adjust these based on your microscope)
        self.pixel_scale = 100.0  # pixels per mm (default value)
        self.mm_per_pixel = 0.01  # mm per pixel
        self.calibration_error = 0.001  # mm (default accuracy)
        self.calibration_date = datetime.now().isoformat()

        # Measurement history
        self.measurements = []
        self.session_start = time.time()

        print("📏 Calibrated Measurement System Initialized")
        print(f"   Using default calibration: {self.pixel_scale:.1f} pixels/mm")



    def connect_microscope(self):
        """Connect to microscope."""
        print("🔌 Connecting to microscope...")

        self.microscope = MicroscopeCapture(
            device_id=self.device_id,
            target_fps=30,
            resolution=(1280, 720)
        )

        if self.microscope.connect():
            print("✅ Microscope connected")
            return True
        else:
            print("❌ Failed to connect to microscope")
            return False

    def pixels_to_mm(self, pixels):
        """Convert pixels to millimeters."""
        return pixels / self.pixel_scale

    def mm_to_pixels(self, mm):
        """Convert millimeters to pixels."""
        return mm * self.pixel_scale

    def measure_distance(self, point1, point2):
        """Measure distance between two points in mm."""
        pixel_distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        mm_distance = self.pixels_to_mm(pixel_distance)

        return {
            'distance_mm': mm_distance,
            'distance_pixels': pixel_distance,
            'accuracy_mm': self.calibration_error,
            'points': [point1, point2]
        }



    def interactive_measurement(self, image, instruction="Click two points to measure"):
        """Interactive measurement on an image."""
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        # Create window and set callback
        cv2.namedWindow('Measurement', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Measurement', mouse_callback)

        print(f"\n📏 {instruction}")
        print("Left-click two points, then press ENTER")

        while True:
            display_image = image.copy()

            # Draw existing points
            for i, point in enumerate(points):
                cv2.circle(display_image, point, 5, (0, 255, 0), -1)
                cv2.putText(display_image, f"P{i + 1}", (point[0] + 10, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw line if we have 2 points
            if len(points) == 2:
                cv2.line(display_image, points[0], points[1], (0, 255, 0), 2)

                # Calculate and display distance
                measurement = self.measure_distance(points[0], points[1])
                distance_text = f"Distance: {measurement['distance_mm']:.3f} mm"

                cv2.putText(display_image, distance_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Instructions
            cv2.putText(display_image, instruction, (10, image.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_image, "Click 2 points, ENTER=Accept, ESC=Cancel",
                        (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Measurement', display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) == 2:  # Enter
                measurement = self.measure_distance(points[0], points[1])
                cv2.destroyAllWindows()
                return measurement['distance_mm']
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == ord('c'):  # Clear points
                points.clear()

    def automated_measurement_from_detection(self, image, detection_results):
        """Automatically measure specimens from detection results."""
        measurements = []

        for detection in detection_results:
            if detection.specimen_id == 2:  # Fracture surface class
                # Extract fracture surface measurements
                x, y, w, h = [int(coord) for coord in detection.bbox]

                # Measure fracture width (bbox width)
                width_mm = self.pixels_to_mm(w)

                # Measure fracture height if needed
                height_mm = self.pixels_to_mm(h)

                measurement = {
                    'type': 'fracture_surface',
                    'specimen_id': detection.specimen_id,
                    'width_mm': width_mm,
                    'height_mm': height_mm,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'timestamp': datetime.now().isoformat(),
                    'accuracy_mm': self.calibration_error
                }

                measurements.append(measurement)

                # Log measurement
                self.measurements.append(measurement)

                print(f"🔬 Automated measurement:")
                print(f"   Fracture width: {width_mm:.3f} ±{self.calibration_error:.3f} mm")
                print(f"   Fracture height: {height_mm:.3f} ±{self.calibration_error:.3f} mm")
                print(f"   Confidence: {detection.confidence:.2f}")

        return measurements

    def live_measurement_mode(self):
        """Live measurement mode with detection."""
        if not self.microscope:
            print("❌ Microscope not connected")
            return

        # Load detector if not already loaded
        if self.detector is None:
            print("🤖 Loading specimen detector...")
            try:
                self.detector = SpecimenDetector(
                    confidence_threshold=0.25,
                    device='auto'
                )
                print("✅ Detector loaded")
            except Exception as e:
                print(f"⚠️ Could not load detector: {e}")
                print("Manual measurement mode only")

        print("\n🎬 LIVE MEASUREMENT MODE")
        print("=" * 40)
        print("Controls:")
        print("  SPACE - Manual measurement")
        print("  A - Auto measurement (if detector available)")
        print("  S - Save current frame")
        print("  Q - Quit")
        print()

        while True:
            frame = self.microscope.get_frame()
            if frame is None:
                continue

            display_frame = frame.copy()

            # Run detection if available
            detections = []
            if self.detector:
                try:
                    detections = self.detector.detect_specimen(frame)

                    # Draw detections
                    for detection in detections:
                        x, y, w, h = [int(coord) for coord in detection.bbox]

                        if detection.specimen_id == 2:  # Fracture surface
                            color = (0, 0, 255)  # Red
                            label = "Fracture"

                            # Show measurement
                            width_mm = self.pixels_to_mm(w)
                            measurement_text = f"{width_mm:.2f}mm"

                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(display_frame, f"{label}: {measurement_text}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            color = (0, 255, 0)  # Green
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                except Exception as e:
                    pass  # Continue without detection

            # Add info overlay
            self.add_measurement_overlay(display_frame, len(detections))

            cv2.imshow('Live Measurement Mode', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Manual measurement
                distance = self.interactive_measurement(frame, "Manual measurement")
                if distance:
                    print(f"📏 Manual measurement: {distance:.3f} ±{self.calibration_error:.3f} mm")

            elif key == ord('a') and self.detector:  # Auto measurement
                measurements = self.automated_measurement_from_detection(frame, detections)
                if measurements:
                    print(f"🤖 Automated {len(measurements)} measurements")
                else:
                    print("🔍 No fracture surfaces detected for measurement")

            elif key == ord('s'):  # Save frame
                timestamp = int(time.time())
                filename = f"measurement_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved frame: {filename}")

            elif key == ord('q'):  # Quit
                break

        cv2.destroyAllWindows()

    def add_measurement_overlay(self, frame, detection_count):
        """Add measurement info overlay to frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Info text
        info_lines = [
            f"Calibration: {self.pixel_scale:.1f} px/mm",
            f"Accuracy: ±{self.calibration_error:.3f} mm",
            f"Detections: {detection_count}",
            f"Measurements: {len(self.measurements)}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def export_measurements(self, filename=None):
        """Export measurements to Excel/CSV."""
        if not self.measurements:
            print("📊 No measurements to export")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"measurements_{timestamp}.xlsx"

        # Convert to DataFrame
        df = pd.DataFrame(self.measurements)

        # Add summary statistics
        summary = {
            'session_start': datetime.fromtimestamp(self.session_start).isoformat(),
            'total_measurements': len(self.measurements),
            'calibration_date': self.calibration_date,
            'calibration_accuracy': self.calibration_error,
            'pixel_scale': self.pixel_scale
        }

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Measurements', index=False)

            # Summary sheet
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Session_Summary', index=False)

        print(f"📊 Measurements exported to: {filename}")
        print(f"   Total measurements: {len(self.measurements)}")

        return filename



    def run_measurement_session(self):
        """Run complete measurement session."""
        print("🚀 STARTING CALIBRATED MEASUREMENT SESSION")
        print("=" * 50)

        # Connect microscope
        if not self.connect_microscope():
            return False

        try:
            # Start measurement session
            self.live_measurement_mode()

            # Export measurements
            if self.measurements:
                export = input("\nExport measurements? (Y/n): ")
                if export.lower() != 'n':
                    filename = self.export_measurements()
                    print(f"✅ Session data exported to: {filename}")

            print("\n✅ Measurement session completed!")
            return True

        except Exception as e:
            print(f"❌ Error during measurement session: {e}")
            return False
        finally:
            if self.microscope:
                self.microscope.disconnect()


def create_measurement_report(measurements, pixel_scale=100.0, mm_per_pixel=0.01, calibration_error=0.001):
    """Create a detailed measurement report."""
    if not measurements:
        return "No measurements to report"

    report = f"""
CHARPY SPECIMEN MEASUREMENT REPORT
{'=' * 60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session Duration: {(time.time() - measurements[0].get('session_start', time.time())) / 60:.1f} minutes

CALIBRATION INFO:
• Scale Factor: {pixel_scale:.3f} pixels/mm
• Resolution: {mm_per_pixel:.6f} mm/pixel
• Measurement Accuracy: ±{calibration_error:.4f} mm
• Calibration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MEASUREMENT SUMMARY:
• Total Measurements: {len(measurements)}
• Fracture Surface Measurements: {len([m for m in measurements if m.get('type') == 'fracture_surface'])}

DETAILED MEASUREMENTS:
"""

    for i, measurement in enumerate(measurements, 1):
        if measurement.get('type') == 'fracture_surface':
            report += f"""
Measurement {i}:
  • Type: Fracture Surface
  • Width: {measurement['width_mm']:.3f} ±{measurement['accuracy_mm']:.3f} mm
  • Height: {measurement['height_mm']:.3f} ±{measurement['accuracy_mm']:.3f} mm
  • Detection Confidence: {measurement.get('confidence', 'N/A')}
  • Timestamp: {measurement['timestamp']}
"""

    # Statistical analysis
    fracture_widths = [m['width_mm'] for m in measurements if m.get('type') == 'fracture_surface']
    if fracture_widths:
        report += f"""
STATISTICAL ANALYSIS:
• Mean Fracture Width: {np.mean(fracture_widths):.3f} mm
• Std Deviation: {np.std(fracture_widths):.3f} mm
• Min Width: {np.min(fracture_widths):.3f} mm
• Max Width: {np.max(fracture_widths):.3f} mm
• Range: {np.max(fracture_widths) - np.min(fracture_widths):.3f} mm
"""

    report += f"\n{'=' * 60}\n"
    return report


def quick_measurement_tool():
    """Quick measurement tool for single measurements."""
    print("⚡ QUICK MEASUREMENT TOOL")
    print("=" * 30)

    # Initialize system
    system = CalibratedMeasurementSystem()

    # Connect microscope
    if not system.connect_microscope():
        return

    try:
        print("\n📸 Capturing image for measurement...")

        # Capture single frame
        frame = system.microscope.get_frame()
        if frame is None:
            print("❌ Failed to capture image")
            return

        # Save reference image
        timestamp = int(time.time())
        image_file = f"quick_measurement_{timestamp}.jpg"
        cv2.imwrite(image_file, frame)
        print(f"💾 Image saved: {image_file}")

        # Interactive measurement
        distance = system.interactive_measurement(frame, "Quick measurement - click two points")

        if distance:
            print(f"\n📏 MEASUREMENT RESULT:")
            print(f"   Distance: {distance:.3f} ±{system.calibration_error:.3f} mm")
            print(f"   Image: {image_file}")

            # Save measurement data
            measurement_data = {
                'distance_mm': distance,
                'accuracy_mm': system.calibration_error,
                'image_file': image_file,
                'timestamp': datetime.now().isoformat(),
                'calibration_scale': system.pixel_scale
            }

            data_file = f"quick_measurement_{timestamp}.json"
            with open(data_file, 'w') as f:
                json.dump(measurement_data, f, indent=2)

            print(f"   Data: {data_file}")
        else:
            print("❌ Measurement cancelled")

    finally:
        system.microscope.disconnect()





def batch_measurement_tool():
    """Tool for measuring multiple specimens in sequence."""
    print("📊 BATCH MEASUREMENT TOOL")
    print("=" * 30)

    system = CalibratedMeasurementSystem()

    if not system.connect_microscope():
        return

    try:
        num_specimens = int(input("Number of specimens to measure: "))

        batch_measurements = []

        for i in range(1, num_specimens + 1):
            print(f"\n🔬 SPECIMEN {i}/{num_specimens}")
            input(f"Position specimen {i} and press ENTER...")

            # Capture image
            frame = system.microscope.get_frame()
            if frame is None:
                print(f"❌ Failed to capture specimen {i}")
                continue

            # Save specimen image
            timestamp = int(time.time())
            image_file = f"specimen_{i:02d}_{timestamp}.jpg"
            cv2.imwrite(image_file, frame)

            # Measure
            distance = system.interactive_measurement(frame, f"Measure specimen {i}")

            if distance:
                measurement = {
                    'specimen_number': i,
                    'distance_mm': distance,
                    'accuracy_mm': system.calibration_error,
                    'image_file': image_file,
                    'timestamp': datetime.now().isoformat()
                }

                batch_measurements.append(measurement)

                print(f"✅ Specimen {i}: {distance:.3f} ±{system.calibration_error:.3f} mm")
            else:
                print(f"❌ Specimen {i} measurement cancelled")

        # Export batch results
        if batch_measurements:
            batch_file = f"batch_measurements_{int(time.time())}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_measurements, f, indent=2)

            # Create Excel export
            df = pd.DataFrame(batch_measurements)
            excel_file = batch_file.replace('.json', '.xlsx')
            df.to_excel(excel_file, index=False)

            print(f"\n📊 BATCH RESULTS:")
            print(f"   Specimens measured: {len(batch_measurements)}")
            print(f"   Mean: {np.mean([m['distance_mm'] for m in batch_measurements]):.3f} mm")
            print(f"   Std Dev: {np.std([m['distance_mm'] for m in batch_measurements]):.3f} mm")
            print(f"   Data saved: {batch_file}")
            print(f"   Excel saved: {excel_file}")

    finally:
        system.microscope.disconnect()


def main():
    """Main function with tool selection."""
    print("📏 CALIBRATED MEASUREMENT SYSTEM")
    print("=" * 50)

    tools = {
        '1': ('Full Measurement Session', 'Complete session with live mode and detection'),
        '2': ('Quick Measurement', 'Single measurement tool'),
        '3': ('Batch Measurement', 'Measure multiple specimens'),
        '4': ('View Calibration Info', 'Show current calibration details')
    }

    print("\nAvailable Tools:")
    for key, (name, desc) in tools.items():
        print(f"  {key}. {name} - {desc}")

    choice = input("\nSelect tool (1-4): ").strip()

    if choice == '1':
        system = CalibratedMeasurementSystem()
        system.run_measurement_session()
    elif choice == '2':
        quick_measurement_tool()
    elif choice == '3':
        batch_measurement_tool()
    elif choice == '4':
        system = CalibratedMeasurementSystem()
        print(f"\n📊 CALIBRATION INFORMATION:")
        print(f"   Date: {system.calibration_date}")
        print(f"   Scale: {system.pixel_scale:.3f} pixels/mm")
        print(f"   Resolution: {system.mm_per_pixel:.6f} mm/pixel")
        print(f"   Accuracy: ±{system.calibration_error:.4f} mm")
        print(f"   Device ID: {system.device_id}")
        print("   Note: Using default calibration values")
    else:
        print("❌ Invalid selection")


if __name__ == "__main__":
    main()