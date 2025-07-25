#!/usr/bin/env python3
"""
Standalone test script for MicroscopeCapture interface.
This version includes all dependencies inline for maximum portability.
"""

import sys
import time
import json
import threading
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

# Check for required dependencies
try:
    import cv2
    import numpy as np

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicroscopeCapture:
    """
    Standalone microscope interface for testing USB cameras.
    This version is self-contained for easy distribution.
    """

    def __init__(self, device_id: int = 1, target_fps: int = 30, resolution: Tuple[int, int] = (1920, 1080)):
        """Initialize microscope capture interface."""
        self.device_id = device_id
        self.target_fps = target_fps
        self.resolution = resolution
        self.cap: Optional = None
        self.is_connected = False
        self.is_streaming = False

        # Camera settings
        self.camera_settings = {
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'hue': 0.5,
            'gain': 0.5,
            'exposure': -1,
            'focus': -1,
            'white_balance': -1
        }

        # Streaming control
        self.stream_thread: Optional[threading.Thread] = None
        self.stream_callback: Optional[Callable] = None
        self.frame_buffer: Optional = None
        self.buffer_lock = threading.Lock()

        logger.info(f"MicroscopeCapture initialized for device {device_id}")

    def connect(self) -> bool:
        """Establish connection to the microscope camera."""
        if not DEPENDENCIES_AVAILABLE:
            logger.error(f"Required dependencies not available: {MISSING_DEPS}")
            return False

        try:
            logger.info(f"Attempting to connect to camera device {self.device_id}")

            # Try different backends for better USB camera support
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.device_id, backend)
                    if self.cap.isOpened():
                        logger.info(f"Connected using backend {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    continue

            if self.cap is None or not self.cap.isOpened():
                logger.error(f"Failed to connect to camera {self.device_id}")
                return False

            # Configure camera settings
            self._configure_camera()
            self.is_connected = True

            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Camera connected but cannot capture frames")
                self.disconnect()
                return False

            logger.info(f"Successfully connected to microscope camera {self.device_id}")
            logger.info(f"Camera resolution: {self.get_frame_dimensions()}")

            return True

        except Exception as e:
            logger.error(f"Error connecting to camera: {e}")
            self.disconnect()
            return False

    def disconnect(self) -> None:
        """Disconnect from the microscope camera and cleanup resources."""
        try:
            self.stop_stream()

            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.is_connected = False
            logger.info("Disconnected from microscope camera")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def get_frame(self):
        """Capture a single frame from the microscope."""
        if not self.is_connected or self.cap is None:
            logger.error("Camera not connected")
            return None

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                logger.debug("Frame captured successfully")
                return frame
            else:
                logger.error("Failed to capture frame")
                return None

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None

    def start_stream(self, callback: Optional[Callable] = None) -> bool:
        """Begin continuous video stream from the microscope."""
        if not self.is_connected:
            logger.error("Cannot start stream: camera not connected")
            return False

        if self.is_streaming:
            logger.warning("Stream already running")
            return True

        try:
            self.stream_callback = callback
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.stream_thread.start()

            logger.info(f"Video stream started at {self.target_fps} FPS")
            return True

        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            self.is_streaming = False
            return False

    def stop_stream(self) -> None:
        """Stop the continuous video stream."""
        if not self.is_streaming:
            return

        try:
            self.is_streaming = False

            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)

            self.stream_thread = None
            self.stream_callback = None

            logger.info("Video stream stopped")

        except Exception as e:
            logger.error(f"Error stopping stream: {e}")

    def adjust_settings(self, **settings) -> Dict[str, bool]:
        """Control camera settings like brightness, contrast, focus, etc."""
        if not self.is_connected or self.cap is None:
            logger.error("Cannot adjust settings: camera not connected")
            return {key: False for key in settings.keys()}

        results = {}

        # Mapping of setting names to OpenCV properties
        property_map = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'hue': cv2.CAP_PROP_HUE,
            'gain': cv2.CAP_PROP_GAIN,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'focus': cv2.CAP_PROP_FOCUS,
            'white_balance': cv2.CAP_PROP_WHITE_BALANCE_BLUE_U
        }

        for setting_name, value in settings.items():
            if setting_name not in property_map:
                logger.warning(f"Unknown setting: {setting_name}")
                results[setting_name] = False
                continue

            try:
                prop_id = property_map[setting_name]
                success = self.cap.set(prop_id, value)

                if success:
                    self.camera_settings[setting_name] = value
                    logger.info(f"Set {setting_name} to {value}")
                else:
                    logger.warning(f"Failed to set {setting_name} to {value}")

                results[setting_name] = success

            except Exception as e:
                logger.error(f"Error setting {setting_name}: {e}")
                results[setting_name] = False

        return results

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get current camera frame dimensions."""
        if not self.is_connected or self.cap is None:
            return (0, 0)

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        except:
            return (0, 0)

    def get_current_fps(self) -> float:
        """Get current camera frame rate."""
        if not self.is_connected or self.cap is None:
            return 0.0

        try:
            return self.cap.get(cv2.CAP_PROP_FPS)
        except:
            return 0.0

    def save_calibration(self, filename: Optional[str] = None) -> bool:
        """Save current camera calibration settings to file."""
        try:
            if filename:
                calib_file = Path(filename)
            else:
                calib_file = Path("camera_calibration.json")

            calibration_data = {
                'device_id': self.device_id,
                'resolution': self.resolution,
                'target_fps': self.target_fps,
                'settings': self.camera_settings.copy(),
                'timestamp': time.time()
            }

            with open(calib_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)

            logger.info(f"Calibration saved to {calib_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return False

    def _configure_camera(self) -> None:
        """Configure camera with initial settings."""
        if not self.cap:
            return

        try:
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.info("Camera configured with initial settings")

        except Exception as e:
            logger.warning(f"Error configuring camera: {e}")

    def _stream_worker(self) -> None:
        """Worker thread for continuous video streaming."""
        frame_time = 1.0 / self.target_fps

        while self.is_streaming:
            start_time = time.time()

            try:
                frame = self.get_frame()
                if frame is not None:
                    # Update frame buffer
                    with self.buffer_lock:
                        self.frame_buffer = frame.copy() if hasattr(frame, 'copy') else frame

                    # Call callback if provided
                    if self.stream_callback:
                        try:
                            self.stream_callback(frame)
                        except Exception as e:
                            logger.error(f"Error in stream callback: {e}")

                # Control frame rate
                elapsed = time.time() - start_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in stream worker: {e}")
                time.sleep(0.1)


def check_system_info():
    """Check system information and available cameras."""
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")

    if DEPENDENCIES_AVAILABLE:
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")

        # Check available cameras
        print("\nScanning for available cameras...")
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        if available_cameras:
            print(f"Available camera devices: {available_cameras}")
        else:
            print("No cameras detected")
    else:
        print(f"❌ Missing dependencies: {MISSING_DEPS}")
        print("\nTo install dependencies, run:")
        print("pip install opencv-python numpy")


def test_microscope_interface():
    """Test the microscope interface functionality."""
    print("\n" + "=" * 50)
    print("Testing Microscope Interface")
    print("=" * 50)

    if not DEPENDENCIES_AVAILABLE:
        print(f"❌ Cannot run tests - missing dependencies: {MISSING_DEPS}")
        print("\nPlease install required packages:")
        print("pip install opencv-python numpy")
        return False

    # Test with different device IDs to find available cameras
    for device_id in [0, 1, 2]:
        print(f"\nTesting device ID: {device_id}")

        microscope = MicroscopeCapture(device_id=device_id, target_fps=30, resolution=(1280, 720))

        try:
            if microscope.connect():
                print(f"✅ Successfully connected to camera {device_id}")

                # Get camera info
                dimensions = microscope.get_frame_dimensions()
                fps = microscope.get_current_fps()
                print(f"  📐 Resolution: {dimensions[0]}x{dimensions[1]}")
                print(f"  🎬 FPS: {fps}")

                # Test single frame capture
                frame = microscope.get_frame()
                if frame is not None:
                    print(f"✅ Frame captured successfully: {frame.shape}")

                    # Save test image
                    test_image_path = f"test_frame_device_{device_id}.jpg"
                    cv2.imwrite(test_image_path, frame)
                    print(f"💾 Test image saved: {test_image_path}")
                else:
                    print("❌ Failed to capture frame")

                # Test camera settings adjustment
                print("\n🔧 Testing camera settings...")
                settings_result = microscope.adjust_settings(
                    brightness=0.6,
                    contrast=0.7,
                    saturation=0.5
                )
                print(f"Settings adjustment results: {settings_result}")

                # Test calibration save
                if microscope.save_calibration(f"test_calibration_device_{device_id}.json"):
                    print("✅ Calibration saved")
                else:
                    print("❌ Failed to save calibration")

                # Test streaming for a short duration
                print("\n🎥 Testing video streaming...")
                frame_count = 0

                def stream_callback(frame):
                    nonlocal frame_count
                    frame_count += 1
                    if frame_count <= 3:  # Print first 3 frames
                        print(f"  📹 Stream frame {frame_count}: {frame.shape}")

                if microscope.start_stream(callback=stream_callback):
                    print("✅ Streaming started")
                    time.sleep(2)  # Stream for 2 seconds
                    microscope.stop_stream()
                    print(f"✅ Streaming stopped. Total frames: {frame_count}")
                else:
                    print("❌ Failed to start streaming")

                microscope.disconnect()
                print(f"✅ Disconnected from camera {device_id}")

                # If we successfully tested one camera, we can stop
                return True

            else:
                print(f"❌ Failed to connect to camera {device_id}")

        except Exception as e:
            print(f"❌ Error testing camera {device_id}: {e}")

        finally:
            microscope.disconnect()

    print("\n❌ No working cameras found")
    return False


def main():
    """Main function to run all tests."""
    print("🔬 Standalone Microscope Test Script")
    print("=" * 60)

    # Check system info
    check_system_info()

    # Run interface tests
    success = test_microscope_interface()

    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
        print("\n📝 Generated files:")
        print("  - test_frame_device_X.jpg (camera test images)")
        print("  - test_calibration_device_X.json (camera settings)")
    else:
        print("❌ Test failed or no cameras available")

    print("\n🚀 To use this in your application:")
    print("microscope = MicroscopeCapture(device_id=1)")
    print("microscope.connect()")
    print("frame = microscope.get_frame()")

    # Keep window open on Windows
    if sys.platform == "win32":
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()