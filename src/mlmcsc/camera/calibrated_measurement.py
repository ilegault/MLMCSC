#!/usr/bin/env python3
"""
Standalone Microscope Calibrator System

Simple calibration tool for microscope on device ID 1.
Creates calibration data that can be used by other measurement tools.

Features:
- Connect to microscope on device ID 1
- Interactive calibration using known reference objects
- Save/load calibration data
- Quick measurement verification

Usage:
    python microscope_calibrator.py
"""

import cv2
import numpy as np
import json
import time
import math
from pathlib import Path
from datetime import datetime


class MicroscopeCalibrator:
    """Standalone microscope calibrator for device ID 1."""

    def __init__(self):
        """Initialize calibrator."""
        self.device_id = 1
        self.cap = None

        # Calibration data
        self.pixel_scale = None  # pixels per mm
        self.mm_per_pixel = None
        self.calibration_error = None
        self.reference_distance_mm = None
        self.reference_distance_pixels = None

        # Calibration file
        self.calibration_file = Path("data/microscope_calibration.json")
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)

        print("🔬 Microscope Calibrator - Device ID 1")
        print("=" * 40)

    def connect_microscope(self):
        """Connect to microscope on device ID 1."""
        print(f"📡 Connecting to microscope (device {self.device_id})...")

        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            print("❌ Failed to connect to microscope")
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Test capture
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture from microscope")
            return False

        print("✅ Microscope connected successfully")
        print(
            f"   Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True

    def get_frame(self):
        """Get current frame from microscope."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def interactive_calibration(self):
        """Interactive calibration using reference object."""
        print("\n🎯 INTERACTIVE CALIBRATION")
        print("=" * 30)
        print("Steps:")
        print("1. Place a reference object with known dimension")
        print("2. Click two points to measure the reference")
        print("3. Enter the actual dimension in mm")
        print()

        # Get reference distance
        while True:
            try:
                ref_mm = float(input("Enter reference object size in mm (e.g., 10.0): "))
                if ref_mm > 0:
                    self.reference_distance_mm = ref_mm
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

        print(f"\n📏 Reference object: {self.reference_distance_mm} mm")
        print("\nCapturing live view...")
        print("Position your reference object and press SPACE to measure")

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            # Add instructions to frame
            cv2.putText(frame, f"Reference: {self.reference_distance_mm} mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE = Measure, ESC = Exit",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Calibration - Position Reference Object', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space - start measurement
                cv2.destroyAllWindows()
                pixel_distance = self.measure_reference(frame)
                if pixel_distance:
                    self.calculate_calibration(pixel_distance)
                    return True
                else:
                    print("❌ Measurement failed, trying again...")
                    continue
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return False

    def measure_reference(self, image):
        """Measure reference object interactively."""
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.namedWindow('Measure Reference Object', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Measure Reference Object', mouse_callback)

        print(f"\n📐 Click two points on the {self.reference_distance_mm}mm reference")
        print("Left-click two points, then press ENTER")

        while True:
            display_image = image.copy()

            # Draw points
            for i, point in enumerate(points):
                cv2.circle(display_image, point, 5, (0, 255, 0), -1)
                cv2.putText(display_image, f"P{i + 1}", (point[0] + 10, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw line if we have 2 points
            if len(points) == 2:
                cv2.line(display_image, points[0], points[1], (0, 255, 0), 2)

                # Calculate pixel distance
                pixel_dist = math.sqrt((points[1][0] - points[0][0]) ** 2 +
                                       (points[1][1] - points[0][1]) ** 2)

                cv2.putText(display_image, f"Distance: {pixel_dist:.1f} pixels",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Reference: {self.reference_distance_mm} mm",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Instructions
            cv2.putText(display_image, f"Click 2 points on {self.reference_distance_mm}mm object",
                        (10, display_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_image, "ENTER = Accept, C = Clear, ESC = Cancel",
                        (10, display_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Measure Reference Object', display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) == 2:  # Enter
                pixel_distance = math.sqrt((points[1][0] - points[0][0]) ** 2 +
                                           (points[1][1] - points[0][1]) ** 2)
                cv2.destroyAllWindows()
                return pixel_distance
            elif key == ord('c'):  # Clear
                points.clear()
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None

    def calculate_calibration(self, pixel_distance):
        """Calculate calibration from reference measurement."""
        self.reference_distance_pixels = pixel_distance
        self.pixel_scale = pixel_distance / self.reference_distance_mm
        self.mm_per_pixel = self.reference_distance_mm / pixel_distance

        # Estimate calibration error (typically 1-2% for good calibration)
        self.calibration_error = self.reference_distance_mm * 0.02  # 2% error estimate

        print(f"\n✅ CALIBRATION CALCULATED")
        print(f"   Reference: {self.reference_distance_mm} mm = {pixel_distance:.1f} pixels")
        print(f"   Scale: {self.pixel_scale:.3f} pixels/mm")
        print(f"   Resolution: {self.mm_per_pixel:.6f} mm/pixel")
        print(f"   Estimated accuracy: ±{self.calibration_error:.3f} mm")

    def save_calibration(self):
        """Save calibration to file."""
        if self.pixel_scale is None:
            print("❌ No calibration data to save")
            return False

        calibration_data = {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'reference_distance_mm': self.reference_distance_mm,
            'reference_distance_pixels': self.reference_distance_pixels,
            'pixel_scale': self.pixel_scale,
            'mm_per_pixel': self.mm_per_pixel,
            'calibration_error_mm': self.calibration_error,
            'resolution': [
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ],
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS))
        }

        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)

            print(f"💾 Calibration saved to: {self.calibration_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to save calibration: {e}")
            return False

    def load_calibration(self):
        """Load existing calibration."""
        if not self.calibration_file.exists():
            print("❌ No calibration file found")
            return False

        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)

            self.pixel_scale = data['pixel_scale']
            self.mm_per_pixel = data['mm_per_pixel']
            self.calibration_error = data['calibration_error_mm']
            self.reference_distance_mm = data['reference_distance_mm']
            self.reference_distance_pixels = data['reference_distance_pixels']

            print("✅ Calibration loaded successfully")
            print(f"   Date: {data['timestamp']}")
            print(f"   Scale: {self.pixel_scale:.3f} pixels/mm")
            print(f"   Resolution: {self.mm_per_pixel:.6f} mm/pixel")
            print(f"   Accuracy: ±{self.calibration_error:.3f} mm")
            return True
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False

    def test_measurement(self):
        """Test measurement with current calibration."""
        if self.pixel_scale is None:
            print("❌ No calibration loaded")
            return

        print("\n🧪 TEST MEASUREMENT")
        print("Position an object and measure it")

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            cv2.putText(frame, f"Scale: {self.pixel_scale:.2f} px/mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE = Measure, ESC = Exit",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Test Measurement', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space - measure
                cv2.destroyAllWindows()
                distance_mm = self.quick_measure(frame)
                if distance_mm:
                    print(f"📏 Measurement: {distance_mm:.3f} ±{self.calibration_error:.3f} mm")
                print("Press SPACE for another measurement, ESC to exit")
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                break

    def quick_measure(self, image):
        """Quick measurement tool."""
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.namedWindow('Quick Measure', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Quick Measure', mouse_callback)

        while True:
            display_image = image.copy()

            # Draw points
            for i, point in enumerate(points):
                cv2.circle(display_image, point, 5, (255, 0, 0), -1)
                cv2.putText(display_image, f"P{i + 1}", (point[0] + 10, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw line and show measurement
            if len(points) == 2:
                cv2.line(display_image, points[0], points[1], (255, 0, 0), 2)

                pixel_dist = math.sqrt((points[1][0] - points[0][0]) ** 2 +
                                       (points[1][1] - points[0][1]) ** 2)
                mm_dist = pixel_dist / self.pixel_scale

                cv2.putText(display_image, f"Distance: {mm_dist:.3f} mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(display_image, "Click 2 points, ENTER = Accept, ESC = Cancel",
                        (10, display_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Quick Measure', display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) == 2:  # Enter
                pixel_distance = math.sqrt((points[1][0] - points[0][0]) ** 2 +
                                           (points[1][1] - points[0][1]) ** 2)
                mm_distance = pixel_distance / self.pixel_scale
                cv2.destroyAllWindows()
                return mm_distance
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None

    def disconnect(self):
        """Disconnect from microscope."""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            print("📴 Microscope disconnected")

    def run(self):
        """Run calibrator main menu."""
        if not self.connect_microscope():
            return

        try:
            while True:
                print("\n🔬 MICROSCOPE CALIBRATOR MENU")
                print("=" * 35)
                print("1. Create new calibration")
                print("2. Load existing calibration")
                print("3. Test measurement")
                print("4. View calibration info")
                print("5. Exit")

                choice = input("\nSelect option (1-5): ").strip()

                if choice == '1':
                    if self.interactive_calibration():
                        self.save_calibration()

                elif choice == '2':
                    self.load_calibration()

                elif choice == '3':
                    self.test_measurement()

                elif choice == '4':
                    if self.pixel_scale:
                        print(f"\n📊 CURRENT CALIBRATION:")
                        print(f"   Scale: {self.pixel_scale:.3f} pixels/mm")
                        print(f"   Resolution: {self.mm_per_pixel:.6f} mm/pixel")
                        print(f"   Accuracy: ±{self.calibration_error:.3f} mm")
                        print(f"   Reference: {self.reference_distance_mm} mm")
                    else:
                        print("❌ No calibration loaded")

                elif choice == '5':
                    break

                else:
                    print("❌ Invalid choice")

        finally:
            self.disconnect()


def main():
    """Main function."""
    calibrator = MicroscopeCalibrator()
    calibrator.run()


if __name__ == "__main__":
    main()