#!/usr/bin/env python3
"""
Charpy Specimen Systematic Photo Capture System

Designed specifically for capturing training data from your 8 Charpy specimens
with systematic positioning, orientations, and quality control.

Features:
- Guided photo capture workflow
- Systematic specimen positioning
- Quality assessment
- Automatic file naming and organization
- Progress tracking
- Preview mode for setup verification
"""

import cv2
import numpy as np
import os
import time
import json
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import threading


class CharpyPhotoCaptureSystem:
    """Systematic photo capture system for Charpy specimens."""

    def __init__(self, device_id=1, output_dir="data/charpy_training_v2"):
        """
        Initialize the capture system.

        Args:
            device_id: Camera device ID (1 for your microscope)
            output_dir: Directory to save captured photos
        """
        self.device_id = device_reid
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Camera setup
        self.cap = None
        self.is_connected = False

        # Capture configuration
        self.total_specimens = 8
        self.current_specimen = 1
        self.photos_per_specimen = 10  # Start with essential 10, can increase
        self.current_photo = 1

        # Photo capture plan
        self.photo_plan = self._create_photo_plan()
        self.captured_photos = []

        # Quality settings
        self.min_specimen_area = 0.1  # Minimum 10% of frame
        self.max_specimen_area = 0.8  # Maximum 80% of frame

        # Statistics
        self.session_start_time = time.time()
        self.total_captured = 0
        self.quality_rejected = 0

        print(f"🔬 Charpy Photo Capture System Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(
            f"🎯 Plan: {self.total_specimens} specimens × {self.photos_per_specimen} photos = {self.total_specimens * self.photos_per_specimen} total photos")

    def _create_photo_plan(self):
        """Create systematic photo capture plan."""
        positions = [
            ("center", "Center of frame"),
            ("top", "Upper portion of frame"),
            ("bottom", "Lower portion of frame"),
            ("left", "Left side of frame"),
            ("right", "Right side of frame")
        ]

        orientations = [
            ("horizontal_up", "Horizontal, notch up", 0),
            ("horizontal_down", "Horizontal, notch down", 180),
            ("angle_left", "Angled 15° left", -15),
            ("angle_right", "Angled 15° right", 15),
            ("vertical", "Vertical orientation", 90)
        ]

        distances = [
            ("medium", "Standard distance", "40-60% of frame"),
            ("close", "Close-up", "60-80% of frame"),
            ("far", "Distant", "20-40% of frame")
        ]

        # Create photo plan (10 essential photos per specimen)
        plan = [
            (1, "center", "horizontal_up", "medium", "Standard horizontal orientation"),
            (2, "center", "horizontal_down", "medium", "Flipped horizontal orientation"),
            (3, "center", "angle_left", "medium", "15° counter-clockwise rotation"),
            (4, "center", "angle_right", "medium", "15° clockwise rotation"),
            (5, "center", "vertical", "medium", "90° vertical orientation"),
            (6, "top", "horizontal_up", "medium", "Upper position, horizontal"),
            (7, "bottom", "horizontal_up", "medium", "Lower position, horizontal"),
            (8, "center", "horizontal_up", "close", "Close-up, standard orientation"),
            (9, "center", "horizontal_up", "far", "Distant, standard orientation"),
            (10, "center", "horizontal_up", "medium", "Final standard shot")
        ]

        return plan

    def connect_camera(self):
        """Connect to the microscope camera."""
        try:
            print(f"🔌 Connecting to camera device {self.device_id}...")

            # Try different backends for better USB camera support
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.device_id, backend)
                    if self.cap.isOpened():
                        print(f"✅ Connected using backend {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    continue

            if self.cap is None or not self.cap.isOpened():
                raise Exception(f"Could not connect to camera {self.device_id}")

            # Configure camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Fix brightness/exposure issues for microscope
            print("🔧 Optimizing camera settings for microscope...")
            self.optimize_microscope_settings()

            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Camera connected but cannot capture frames")

            self.is_connected = True
            print(f"✅ Camera ready - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            return True

        except Exception as e:
            print(f"❌ Camera connection failed: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def disconnect_camera(self):
        """Disconnect from camera."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print("📷 Camera disconnected")

    def optimize_microscope_settings(self):
        """Optimize camera settings specifically for microscope lighting."""
        if not self.cap:
            return

        print("🔧 Adjusting microscope camera settings...")

        # Brightness and contrast adjustments
        brightness_settings = [
            (cv2.CAP_PROP_BRIGHTNESS, 0.7, "Brightness"),
            (cv2.CAP_PROP_CONTRAST, 0.8, "Contrast"),
            (cv2.CAP_PROP_SATURATION, 0.6, "Saturation"),
            (cv2.CAP_PROP_EXPOSURE, -1, "Auto Exposure"),  # -1 for auto
            (cv2.CAP_PROP_GAIN, 0.3, "Gain"),
            (cv2.CAP_PROP_GAMMA, 100, "Gamma"),
            (cv2.CAP_PROP_AUTO_WB, 1, "Auto White Balance")
        ]

        for prop, value, name in brightness_settings:
            try:
                current = self.cap.get(prop)
                success = self.cap.set(prop, value)
                new_value = self.cap.get(prop)

                if success:
                    print(f"   ✅ {name}: {current:.3f} → {new_value:.3f}")
                else:
                    print(f"   ⚠️ {name}: Could not set (current: {current:.3f})")

            except Exception as e:
                print(f"   ❌ {name}: Error setting - {e}")

        # Test the settings with a sample frame
        time.sleep(0.5)  # Let camera adjust
        ret, test_frame = self.cap.read()

        if ret:
            brightness = np.mean(test_frame)
            print(f"📊 Current image brightness: {brightness:.1f}")

            if brightness < 80:
                print("⚠️ Image still too dark, trying manual brightness boost...")
                self._boost_brightness_manual()
            elif brightness > 200:
                print("⚠️ Image too bright, reducing exposure...")
                self._reduce_brightness_manual()
            else:
                print("✅ Brightness looks good!")

    def _boost_brightness_manual(self):
        """Manually boost brightness if auto settings don't work."""
        # Try different brightness values
        brightness_values = [0.8, 0.9, 1.0]

        for brightness in brightness_values:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            time.sleep(0.3)

            ret, test_frame = self.cap.read()
            if ret:
                frame_brightness = np.mean(test_frame)
                print(f"   🔧 Trying brightness {brightness}: frame brightness = {frame_brightness:.1f}")

                if 80 <= frame_brightness <= 200:
                    print(f"   ✅ Good brightness found: {brightness}")
                    break

        # If still too dark, try adjusting exposure
        exposure_values = [-1, -2, -3, -4]  # Try different exposure levels
        for exposure in exposure_values:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            time.sleep(0.3)

            ret, test_frame = self.cap.read()
            if ret:
                frame_brightness = np.mean(test_frame)
                print(f"   🔧 Trying exposure {exposure}: frame brightness = {frame_brightness:.1f}")

                if frame_brightness >= 100:
                    print(f"   ✅ Good exposure found: {exposure}")
                    break

    def _reduce_brightness_manual(self):
        """Manually reduce brightness if image is too bright."""
        brightness_values = [0.5, 0.4, 0.3]

        for brightness in brightness_values:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            time.sleep(0.3)

            ret, test_frame = self.cap.read()
            if ret:
                frame_brightness = np.mean(test_frame)
                print(f"   🔧 Trying brightness {brightness}: frame brightness = {frame_brightness:.1f}")

                if 80 <= frame_brightness <= 200:
                    print(f"   ✅ Good brightness found: {brightness}")
                    break

    def manual_brightness_adjustment(self):
        """Interactive brightness adjustment mode."""
        if not self.is_connected:
            print("❌ Camera not connected")
            return

        print("\n🔧 MANUAL BRIGHTNESS ADJUSTMENT MODE")
        print("=" * 50)
        print("Controls:")
        print("   + : Increase brightness")
        print("   - : Decrease brightness")
        print("   e : Adjust exposure")
        print("   c : Adjust contrast")
        print("   g : Adjust gain")
        print("   r : Reset to defaults")
        print("   q : Quit adjustment mode")
        print("   s : Save current settings")

        current_brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
        current_contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)
        current_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        current_gain = self.cap.get(cv2.CAP_PROP_GAIN)

        while True:
            # Get current frame
            frame = self.get_frame()
            if frame is None:
                continue

            # Add brightness info overlay
            display_frame = frame.copy()
            frame_brightness = np.mean(frame)

            # Add brightness info
            info_text = [
                f"Frame Brightness: {frame_brightness:.1f}",
                f"Camera Brightness: {current_brightness:.3f}",
                f"Contrast: {current_contrast:.3f}",
                f"Exposure: {current_exposure:.3f}",
                f"Gain: {current_gain:.3f}",
                "Use +/- to adjust brightness"
            ]

            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Brightness Adjustment - Press Q when satisfied', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('+') or key == ord('='):
                current_brightness = min(1.0, current_brightness + 0.05)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, current_brightness)
                print(f"🔆 Brightness: {current_brightness:.3f}")

            elif key == ord('-'):
                current_brightness = max(0.0, current_brightness - 0.05)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, current_brightness)
                print(f"🔅 Brightness: {current_brightness:.3f}")

            elif key == ord('e'):
                current_exposure = current_exposure - 1 if current_exposure > -10 else -1
                self.cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
                print(f"📷 Exposure: {current_exposure:.3f}")

            elif key == ord('c'):
                current_contrast = min(1.0, current_contrast + 0.05) if current_contrast < 0.8 else max(0.0,
                                                                                                        current_contrast - 0.05)
                self.cap.set(cv2.CAP_PROP_CONTRAST, current_contrast)
                print(f"🎨 Contrast: {current_contrast:.3f}")

            elif key == ord('g'):
                current_gain = min(1.0, current_gain + 0.05) if current_gain < 0.5 else max(0.0, current_gain - 0.05)
                self.cap.set(cv2.CAP_PROP_GAIN, current_gain)
                print(f"📶 Gain: {current_gain:.3f}")

            elif key == ord('r'):
                # Reset to defaults
                current_brightness = 0.5
                current_contrast = 0.5
                current_exposure = -1
                current_gain = 0.3
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, current_brightness)
                self.cap.set(cv2.CAP_PROP_CONTRAST, current_contrast)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
                self.cap.set(cv2.CAP_PROP_GAIN, current_gain)
                print("🔄 Reset to defaults")

            elif key == ord('s'):
                # Save current settings
                settings = {
                    'brightness': current_brightness,
                    'contrast': current_contrast,
                    'exposure': current_exposure,
                    'gain': current_gain,
                    'timestamp': time.time()
                }

                settings_file = self.output_dir / "camera_settings.json"
                with open(settings_file, 'w') as f:
                    json.dump(settings, f, indent=2)
                print(f"💾 Settings saved to {settings_file}")

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        print("✅ Brightness adjustment completed")

    def get_frame(self):
        """Capture a frame from the camera."""
        if not self.is_connected or not self.cap:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def assess_image_quality(self, frame):
        """Assess if the captured image meets quality standards."""
        if frame is None:
            return False, "No image captured"

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check image sharpness (variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Adjust threshold as needed
            return False, f"Image too blurry (sharpness: {laplacian_var:.1f})"

        # Check brightness (avoid too dark/bright images)
        brightness = np.mean(gray)
        if brightness < 40:  # Lowered threshold for microscope images
            return False, f"Image too dark (brightness: {brightness:.1f})"
        if brightness > 220:  # Raised threshold for bright microscope images
            return False, f"Image too bright (brightness: {brightness:.1f})"

        # Check contrast
        contrast = gray.std()
        if contrast < 30:
            return False, f"Low contrast (contrast: {contrast:.1f})"

        # Estimate specimen area (simple thresholding)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        specimen_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        specimen_ratio = specimen_pixels / total_pixels

        if specimen_ratio < self.min_specimen_area:
            return False, f"Specimen too small ({specimen_ratio:.1%} of frame)"

        if specimen_ratio > self.max_specimen_area:
            return False, f"Specimen too large ({specimen_ratio:.1%} of frame)"

        return True, f"Good quality (sharpness: {laplacian_var:.1f}, brightness: {brightness:.1f}, contrast: {contrast:.1f}, size: {specimen_ratio:.1%})"

    def save_photo(self, frame, specimen_num, photo_num, description):
        """Save a captured photo with systematic naming."""
        timestamp = int(time.time())
        filename = f"charpy_s{specimen_num:02d}_p{photo_num:02d}_{timestamp}.jpg"
        filepath = self.output_dir / filename

        # Save image
        success = cv2.imwrite(str(filepath), frame)

        if success:
            # Save metadata
            metadata = {
                'filename': filename,
                'specimen_number': specimen_num,
                'photo_number': photo_num,
                'description': description,
                'timestamp': timestamp,
                'capture_time': datetime.now().isoformat(),
                'image_shape': frame.shape,
                'quality_check': self.assess_image_quality(frame)
            }

            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.captured_photos.append(metadata)
            self.total_captured += 1

            print(f"✅ Saved: {filename}")
            return filename
        else:
            print(f"❌ Failed to save photo")
            return None

    def run_interactive_capture(self):
        """Run interactive photo capture session."""
        if not self.connect_camera():
            return False

        print("\n🎬 STARTING INTERACTIVE CAPTURE SESSION")
        print("=" * 60)
        print("📋 Instructions:")
        print("   - Position your specimen according to the prompts")
        print("   - Press SPACE to capture photo")
        print("   - Press 'r' to retake current photo")
        print("   - Press 'q' to quit session")
        print("   - Press 's' to skip current photo")
        print("=" * 60)

        try:
            for specimen_num in range(1, self.total_specimens + 1):
                print(f"\n🔬 SPECIMEN {specimen_num}/{self.total_specimens}")
                print("=" * 40)

                input(f"📋 Place Specimen {specimen_num} and press ENTER to continue...")

                for photo_num, (plan_num, position, orientation, distance, description) in enumerate(self.photo_plan,
                                                                                                     1):
                    print(f"\n📸 Photo {photo_num}/{len(self.photo_plan)}: {description}")
                    print(f"   Position: {position}")
                    print(f"   Orientation: {orientation}")
                    print(f"   Distance: {distance}")

                    captured = False
                    while not captured:
                        # Show live preview
                        frame = self.get_frame()
                        if frame is None:
                            print("❌ Failed to get camera frame")
                            continue

                        # Add overlay information
                        display_frame = self.add_capture_overlay(frame, specimen_num, photo_num, description)

                        # Show preview
                        cv2.imshow('Charpy Capture System - Press SPACE to capture', display_frame)

                        key = cv2.waitKey(1) & 0xFF

                        if key == ord(' '):  # Space to capture
                            # Quality check
                            quality_ok, quality_msg = self.assess_image_quality(frame)

                            if quality_ok:
                                filename = self.save_photo(frame, specimen_num, photo_num, description)
                                if filename:
                                    print(f"✅ Captured: {filename}")
                                    print(f"   Quality: {quality_msg}")
                                    captured = True
                                else:
                                    print("❌ Failed to save photo, try again")
                            else:
                                print(f"⚠️ Quality issue: {quality_msg}")
                                print("   Adjust setup and try again, or press 's' to skip")
                                self.quality_rejected += 1

                        elif key == ord('r'):  # Retake
                            print("🔄 Retaking photo...")
                            continue

                        elif key == ord('s'):  # Skip
                            print("⏭️ Skipping this photo")
                            captured = True

                        elif key == ord('q'):  # Quit
                            print("🛑 Session cancelled by user")
                            return self.cleanup_session()

                print(f"✅ Completed Specimen {specimen_num}")

            print(f"\n🎉 CAPTURE SESSION COMPLETED!")
            return self.cleanup_session()

        except KeyboardInterrupt:
            print("\n⚠️ Session interrupted by user")
            return self.cleanup_session()
        except Exception as e:
            print(f"\n❌ Error during capture: {e}")
            return self.cleanup_session()

    def add_capture_overlay(self, frame, specimen_num, photo_num, description):
        """Add information overlay to the preview frame."""
        overlay_frame = frame.copy()
        h, w = overlay_frame.shape[:2]

        # Add semi-transparent background for text
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, overlay_frame, 0.3, 0, overlay_frame)

        # Add text information
        texts = [
            f"Specimen {specimen_num}/{self.total_specimens} - Photo {photo_num}/{len(self.photo_plan)}",
            f"Setup: {description}",
            f"Progress: {self.total_captured} photos captured",
            "SPACE=Capture | R=Retake | S=Skip | Q=Quit"
        ]

        for i, text in enumerate(texts):
            y = 30 + i * 20
            cv2.putText(overlay_frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add frame guides (3x3 grid)
        self.draw_position_guides(overlay_frame)

        # Add quality indicators
        quality_ok, quality_msg = self.assess_image_quality(frame)
        quality_color = (0, 255, 0) if quality_ok else (0, 0, 255)
        cv2.putText(overlay_frame, f"Quality: {'OK' if quality_ok else 'CHECK'}",
                    (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

        return overlay_frame

    def draw_position_guides(self, frame):
        """Draw positioning guides on the frame."""
        h, w = frame.shape[:2]

        # Draw 3x3 grid lines
        grid_color = (0, 255, 255)  # Yellow

        # Vertical lines
        cv2.line(frame, (w // 3, 0), (w // 3, h), grid_color, 1)
        cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), grid_color, 1)

        # Horizontal lines
        cv2.line(frame, (0, h // 3), (w, h // 3), grid_color, 1)
        cv2.line(frame, (0, 2 * h // 3), (w, 2 * h // 3), grid_color, 1)

        # Center crosshair
        cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), grid_color, 2)
        cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), grid_color, 2)

    def cleanup_session(self):
        """Clean up and show session summary."""
        cv2.destroyAllWindows()
        self.disconnect_camera()

        session_duration = time.time() - self.session_start_time

        print("\n" + "=" * 60)
        print("📊 CAPTURE SESSION SUMMARY")
        print("=" * 60)
        print(f"⏱️ Session duration: {session_duration / 60:.1f} minutes")
        print(f"📸 Total photos captured: {self.total_captured}")
        print(f"❌ Quality rejected: {self.quality_rejected}")
        print(f"📁 Photos saved to: {self.output_dir}")

        if self.captured_photos:
            print(f"🎯 Average photos per specimen: {self.total_captured / self.total_specimens:.1f}")
            print(f"📈 Capture rate: {self.total_captured / (session_duration / 60):.1f} photos/minute")

        # Save session summary
        summary = {
            'session_start': datetime.fromtimestamp(self.session_start_time).isoformat(),
            'session_duration_minutes': session_duration / 60,
            'total_specimens': self.total_specimens,
            'photos_per_specimen_target': self.photos_per_specimen,
            'total_captured': self.total_captured,
            'quality_rejected': self.quality_rejected,
            'captured_photos': self.captured_photos
        }

        summary_file = self.output_dir / f"capture_session_{int(self.session_start_time)}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📋 Session summary saved: {summary_file}")
        print("=" * 60)

        if self.total_captured >= self.total_specimens * 5:  # At least 5 per specimen
            print("✅ SUCCESS: You have enough photos to start training!")
            print("📝 Next steps:")
            print("   1. Review captured photos for quality")
            print("   2. Start annotation process")
            print("   3. Train your model")
        else:
            print("⚠️ Consider capturing more photos for better training data")

        return True


def main():
    """Main function to run the capture system."""
    print("📷 CHARPY SPECIMEN PHOTO CAPTURE SYSTEM")
    print("=" * 60)
    print("This system will guide you through capturing systematic")
    print("training photos of your 8 Charpy specimens.")
    print()

    # Initialize capture system
    capture_system = CharpyPhotoCaptureSystem(
        device_id=1,  # Your microscope camera
        output_dir="data/charpy_training_v2"
    )

    print("🔧 Setup Tips:")
    print("   1. Use consistent lighting and background")
    print("   2. Keep camera position fixed")
    print("   3. Have all 8 specimens ready")
    print("   4. Follow the positioning guides")
    print()

    # Check if camera connects and adjust brightness if needed
    if capture_system.connect_camera():
        print("✅ Camera connected successfully!")

        # Show brightness adjustment option
        print("\n🔧 Camera Settings:")
        print("   The system has optimized settings for your microscope.")

        adjust_brightness = input("Would you like to manually adjust brightness? (y/N): ")
        if adjust_brightness.lower() == 'y':
            capture_system.manual_brightness_adjustment()

        capture_system.disconnect_camera()  # Disconnect for now
    else:
        print("❌ Could not connect to camera")
        return

    response = input("\nReady to start capture session? (y/N): ")
    if response.lower() != 'y':
        print("❌ Capture cancelled")
        return

    # Run the interactive capture session
    success = capture_system.run_interactive_capture()

    if success:
        print("\n🎉 Capture session completed successfully!")
        print("Your photos are ready for annotation and training.")
    else:
        print("\n❌ Capture session ended early.")


if __name__ == "__main__":
    main()