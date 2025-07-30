"""
Microscope interface for capturing images from Koolertron microscope cameras via USB.
Provides comprehensive control over camera settings, streaming, and calibration.
"""

import cv2
import numpy as np
import json
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import camera detector
try:
    from .camera_detector import CameraDetector
except ImportError:
    # Fallback if camera_detector is not available
    class CameraDetector:
        @staticmethod
        def get_best_microscope_device_id():
            return None
        @staticmethod
        def scan_cameras():
            return []


class MicroscopeCapture:
    """
    Advanced interface for connecting to and controlling Koolertron microscope cameras.
    
    Features:
    - USB connection via OpenCV
    - Continuous video streaming
    - Camera settings control (brightness, contrast, focus)
    - Calibration save/load
    - Multiple camera support
    - Error handling and reconnection
    - Frame rate control
    - Auto-exposure and auto-focus
    """
    
    def __init__(self, device_id: int = 1, target_fps: int = 30, resolution: Tuple[int, int] = (1920, 1080)):
        """
        Initialize microscope capture interface.
        
        Args:
            device_id: Camera device ID (default: 1 for USB microscope, 0 for built-in)
            target_fps: Target frame rate (default: 30)
            resolution: Camera resolution as (width, height) (default: 1920x1080)
        """
        self.device_id = device_id
        self.target_fps = target_fps
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.is_streaming = False
        
        # Camera settings
        self.camera_settings = {
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'hue': 0.5,
            'gain': 0.5,
            'exposure': -1,  # Auto exposure
            'focus': -1,     # Auto focus
            'white_balance': -1  # Auto white balance
        }
        
        # Streaming control
        self.stream_thread: Optional[threading.Thread] = None
        self.stream_callback: Optional[Callable[[np.ndarray], None]] = None
        self.frame_buffer: Optional[np.ndarray] = None
        self.buffer_lock = threading.Lock()
        
        # Calibration file path
        self.calibration_file = Path("data/camera_calibration.json")
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MicroscopeCapture initialized for device {device_id}")
    
    @classmethod
    def auto_detect_microscope(cls, target_fps: int = 30, resolution: Tuple[int, int] = (1920, 1080)) -> Optional['MicroscopeCapture']:
        """
        Automatically detect and connect to the best available microscope camera.
        
        Args:
            target_fps: Target frame rate
            resolution: Desired resolution
            
        Returns:
            MicroscopeCapture instance connected to microscope, or None if not found
        """
        logger.info("Auto-detecting microscope cameras...")
        
        # Try to find the best microscope camera
        best_device_id = CameraDetector.get_best_microscope_device_id()
        
        if best_device_id is not None:
            logger.info(f"Found potential microscope on device {best_device_id}")
            microscope = cls(device_id=best_device_id, target_fps=target_fps, resolution=resolution)
            
            if microscope.connect():
                logger.info("Successfully connected to auto-detected microscope")
                return microscope
            else:
                logger.warning("Failed to connect to auto-detected microscope")
        
        # Fallback: try common microscope device IDs
        logger.info("Trying common microscope device IDs...")
        for device_id in [1, 2, 3, 4, 5]:  # Skip 0 (usually built-in camera)
            try:
                microscope = cls(device_id=device_id, target_fps=target_fps, resolution=resolution)
                if microscope.connect():
                    logger.info(f"Successfully connected to microscope on device {device_id}")
                    return microscope
                else:
                    microscope.disconnect()
            except Exception as e:
                logger.debug(f"Device {device_id} failed: {e}")
                continue
        
        logger.error("No microscope cameras found")
        return None
    
    def connect(self) -> bool:
        """
        Establish connection to the microscope camera.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Attempting to connect to camera device {self.device_id}")
            
            # Try to connect with different backends for better USB camera support
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
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the microscope.
        
        Returns:
            np.ndarray: Captured frame or None if capture failed
        """
        if not self.is_connected or self.cap is None:
            logger.error("Camera not connected")
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                logger.debug("Frame captured successfully")
                return frame
            else:
                logger.error("Failed to capture frame")
                # Try to reconnect if frame capture fails
                self._handle_connection_error()
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            self._handle_connection_error()
            return None
    
    def start_stream(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """
        Begin continuous video stream from the microscope.
        
        Args:
            callback: Optional callback function to process each frame
            
        Returns:
            bool: True if stream started successfully
        """
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
        """
        Control camera settings like brightness, contrast, focus, etc.
        
        Args:
            **settings: Camera settings to adjust
                - brightness: 0.0 to 1.0
                - contrast: 0.0 to 1.0
                - saturation: 0.0 to 1.0
                - hue: 0.0 to 1.0
                - gain: 0.0 to 1.0
                - exposure: -1 for auto, or specific value
                - focus: -1 for auto, or specific value
                - white_balance: -1 for auto, or specific value
        
        Returns:
            Dict[str, bool]: Success status for each setting
        """
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
    
    def save_calibration(self, filename: Optional[str] = None) -> bool:
        """
        Save current camera calibration settings to file.
        
        Args:
            filename: Optional custom filename for calibration file
            
        Returns:
            bool: True if calibration saved successfully
        """
        try:
            if filename:
                calib_file = Path(filename)
            else:
                calib_file = self.calibration_file
            
            # Get current camera properties
            calibration_data = {
                'device_id': self.device_id,
                'resolution': self.resolution,
                'target_fps': self.target_fps,
                'settings': self.camera_settings.copy(),
                'timestamp': time.time()
            }
            
            # Add current OpenCV properties if camera is connected
            if self.is_connected and self.cap is not None:
                opencv_props = {}
                prop_ids = [
                    cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_CONTRAST,
                    cv2.CAP_PROP_SATURATION, cv2.CAP_PROP_HUE,
                    cv2.CAP_PROP_GAIN, cv2.CAP_PROP_EXPOSURE,
                    cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_WHITE_BALANCE_BLUE_U
                ]
                
                for prop_id in prop_ids:
                    try:
                        opencv_props[prop_id] = self.cap.get(prop_id)
                    except:
                        pass
                
                calibration_data['opencv_properties'] = opencv_props
            
            # Save to file
            calib_file.parent.mkdir(parents=True, exist_ok=True)
            with open(calib_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {calib_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filename: Optional[str] = None) -> bool:
        """
        Load camera calibration settings from file.
        
        Args:
            filename: Optional custom filename for calibration file
            
        Returns:
            bool: True if calibration loaded successfully
        """
        try:
            if filename:
                calib_file = Path(filename)
            else:
                calib_file = self.calibration_file
            
            if not calib_file.exists():
                logger.warning(f"Calibration file not found: {calib_file}")
                return False
            
            with open(calib_file, 'r') as f:
                calibration_data = json.load(f)
            
            # Apply loaded settings
            if 'settings' in calibration_data:
                self.camera_settings.update(calibration_data['settings'])
                
                # Apply settings to camera if connected
                if self.is_connected:
                    self.adjust_settings(**self.camera_settings)
            
            # Update other parameters
            if 'resolution' in calibration_data:
                self.resolution = tuple(calibration_data['resolution'])
                if self.is_connected:
                    self.set_resolution(*self.resolution)
            
            if 'target_fps' in calibration_data:
                self.target_fps = calibration_data['target_fps']
                if self.is_connected:
                    self.set_fps(self.target_fps)
            
            logger.info(f"Calibration loaded from {calib_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return False
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            bool: True if resolution set successfully
        """
        if not self.is_connected or self.cap is None:
            logger.error("Cannot set resolution: camera not connected")
            return False
        
        try:
            success_w = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            success_h = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if success_w and success_h:
                self.resolution = (width, height)
                logger.info(f"Resolution set to {width}x{height}")
                return True
            else:
                logger.warning(f"Failed to set resolution to {width}x{height}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting resolution: {e}")
            return False
    
    def set_fps(self, fps: int) -> bool:
        """
        Set camera frame rate.
        
        Args:
            fps: Target frames per second
            
        Returns:
            bool: True if FPS set successfully
        """
        if not self.is_connected or self.cap is None:
            logger.error("Cannot set FPS: camera not connected")
            return False
        
        try:
            success = self.cap.set(cv2.CAP_PROP_FPS, fps)
            if success:
                self.target_fps = fps
                logger.info(f"FPS set to {fps}")
                return True
            else:
                logger.warning(f"Failed to set FPS to {fps}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting FPS: {e}")
            return False
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get current camera frame dimensions.
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        if not self.is_connected or self.cap is None:
            return (0, 0)
        
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        except:
            return (0, 0)
    
    def get_current_fps(self) -> float:
        """
        Get current camera frame rate.
        
        Returns:
            float: Current FPS
        """
        if not self.is_connected or self.cap is None:
            return 0.0
        
        try:
            return self.cap.get(cv2.CAP_PROP_FPS)
        except:
            return 0.0
    
    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto-exposure.
        
        Args:
            enable: True to enable auto-exposure, False to disable
            
        Returns:
            bool: True if setting applied successfully
        """
        exposure_value = -1 if enable else 0.5  # -1 for auto, manual value otherwise
        result = self.adjust_settings(exposure=exposure_value)
        return result.get('exposure', False)
    
    def enable_auto_focus(self, enable: bool = True) -> bool:
        """
        Enable or disable auto-focus.
        
        Args:
            enable: True to enable auto-focus, False to disable
            
        Returns:
            bool: True if setting applied successfully
        """
        focus_value = -1 if enable else 0.5  # -1 for auto, manual value otherwise
        result = self.adjust_settings(focus=focus_value)
        return result.get('focus', False)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the buffer (useful during streaming).
        
        Returns:
            np.ndarray: Latest frame or None if no frame available
        """
        with self.buffer_lock:
            return self.frame_buffer.copy() if self.frame_buffer is not None else None
    
    def is_camera_connected(self) -> bool:
        """
        Check if camera is currently connected and responsive.
        
        Returns:
            bool: True if camera is connected and responsive
        """
        if not self.is_connected or self.cap is None:
            return False
        
        try:
            # Try to get a camera property to test connection
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            return True
        except:
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
            
            # Apply default settings
            self.adjust_settings(**self.camera_settings)
            
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
                        self.frame_buffer = frame.copy()
                    
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
                time.sleep(0.1)  # Brief pause before retry
    
    def _handle_connection_error(self) -> None:
        """Handle camera connection errors and attempt reconnection."""
        logger.warning("Camera connection error detected, attempting reconnection...")
        
        try:
            self.disconnect()
            time.sleep(1)  # Brief pause before reconnection
            
            if self.connect():
                logger.info("Camera reconnected successfully")
            else:
                logger.error("Failed to reconnect camera")
                
        except Exception as e:
            logger.error(f"Error during reconnection attempt: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.disconnect()
        except:
            pass


# Legacy compatibility class
class MicroscopeInterface(MicroscopeCapture):
    """Legacy compatibility class - use MicroscopeCapture instead."""
    
    def __init__(self, camera_id: int = 0):
        super().__init__(device_id=camera_id)
        logger.warning("MicroscopeInterface is deprecated, use MicroscopeCapture instead")
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Legacy method name - use get_frame() instead."""
        return self.get_frame()


# Example usage and testing
if __name__ == "__main__":
    # Example usage as requested
    print("Microscope Interface Example Usage:")
    print("=" * 50)
    
    # Initialize microscope capture
    microscope = MicroscopeCapture(device_id=1)
    
    try:
        # Connect to microscope
        if microscope.connect():
            print("✓ Connected to microscope")
            
            # Get single frame
            frame = microscope.get_frame()
            if frame is not None:
                print(f"✓ Captured frame: {frame.shape}")
            
            # Adjust camera settings
            settings_result = microscope.adjust_settings(
                brightness=0.6,
                contrast=0.7,
                exposure=-1  # Auto exposure
            )
            print(f"✓ Settings adjusted: {settings_result}")
            
            # Enable auto-focus
            if microscope.enable_auto_focus(True):
                print("✓ Auto-focus enabled")
            
            # Save calibration
            if microscope.save_calibration():
                print("✓ Calibration saved")
            
            # Start streaming (example with callback)
            def frame_callback(frame):
                print(f"Stream frame: {frame.shape}")
            
            if microscope.start_stream(callback=frame_callback):
                print("✓ Streaming started")
                time.sleep(2)  # Stream for 2 seconds
                microscope.stop_stream()
                print("✓ Streaming stopped")
            
        else:
            print("✗ Failed to connect to microscope")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        
    finally:
        # Cleanup
        microscope.disconnect()
        print("✓ Disconnected from microscope")