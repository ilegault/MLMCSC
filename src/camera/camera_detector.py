"""
Camera detection utility for finding connected cameras including Koolertron microscopes.
"""

import cv2
import logging
import time
from typing import List, Dict, Tuple, Optional
import platform

logger = logging.getLogger(__name__)


class CameraDetector:
    """Utility class for detecting and identifying connected cameras."""
    
    @staticmethod
    def scan_cameras(max_cameras: int = 10) -> List[Dict]:
        """
        Scan for available cameras and return detailed information.
        
        Args:
            max_cameras: Maximum number of camera indices to check
            
        Returns:
            List of dictionaries containing camera information
        """
        available_cameras = []
        
        logger.info(f"Scanning for cameras (checking indices 0-{max_cameras-1})...")
        
        for device_id in range(max_cameras):
            camera_info = CameraDetector._test_camera(device_id)
            if camera_info:
                available_cameras.append(camera_info)
                logger.info(f"Found camera {device_id}: {camera_info['name']}")
        
        return available_cameras
    
    @staticmethod
    def _test_camera(device_id: int) -> Optional[Dict]:
        """
        Test a specific camera device ID and return information if available.
        
        Args:
            device_id: Camera device ID to test
            
        Returns:
            Dictionary with camera info or None if not available
        """
        # Try different backends for better compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
        
        if platform.system() == "Windows":
            # On Windows, prioritize DirectShow and Media Foundation
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif platform.system() == "Linux":
            # On Linux, prioritize V4L2
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(device_id, backend)
                
                if cap.isOpened():
                    # Try to read a frame to verify the camera works
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Try to get camera name (not always available)
                        camera_name = CameraDetector._identify_camera(device_id, width, height)
                        
                        camera_info = {
                            'device_id': device_id,
                            'name': camera_name,
                            'backend': backend,
                            'resolution': (width, height),
                            'fps': fps,
                            'frame_shape': frame.shape,
                            'is_microscope': CameraDetector._is_likely_microscope(camera_name, width, height)
                        }
                        
                        cap.release()
                        return camera_info
                
                cap.release()
                
            except Exception as e:
                logger.debug(f"Backend {backend} failed for device {device_id}: {e}")
                continue
        
        return None
    
    @staticmethod
    def _identify_camera(device_id: int, width: int, height: int) -> str:
        """
        Try to identify the camera type based on available information.
        
        Args:
            device_id: Camera device ID
            width: Camera width
            height: Camera height
            
        Returns:
            String describing the camera
        """
        # Common microscope resolutions and characteristics
        microscope_indicators = [
            (1920, 1080, "USB Microscope (Full HD)"),
            (1280, 720, "USB Microscope (HD)"),
            (1600, 1200, "USB Microscope (UXGA)"),
            (2592, 1944, "USB Microscope (5MP)"),
            (3264, 2448, "USB Microscope (8MP)"),
        ]
        
        # Check for common microscope resolutions
        for mic_width, mic_height, name in microscope_indicators:
            if width == mic_width and height == mic_height and device_id > 0:
                return f"Koolertron {name}"
        
        # Default naming based on device ID and resolution
        if device_id == 0:
            return f"Built-in Camera ({width}x{height})"
        else:
            return f"USB Camera {device_id} ({width}x{height})"
    
    @staticmethod
    def _is_likely_microscope(camera_name: str, width: int, height: int) -> bool:
        """
        Determine if a camera is likely a microscope based on characteristics.
        
        Args:
            camera_name: Camera name/description
            width: Camera width
            height: Camera height
            
        Returns:
            True if likely a microscope
        """
        # Check name for microscope indicators
        microscope_keywords = ['microscope', 'koolertron', 'usb camera']
        name_lower = camera_name.lower()
        
        for keyword in microscope_keywords:
            if keyword in name_lower:
                return True
        
        # High resolution USB cameras are often microscopes
        if width >= 1920 and height >= 1080:
            return True
        
        return False
    
    @staticmethod
    def find_microscope_cameras() -> List[Dict]:
        """
        Find cameras that are likely microscopes.
        
        Returns:
            List of camera info dictionaries for likely microscopes
        """
        all_cameras = CameraDetector.scan_cameras()
        microscope_cameras = [cam for cam in all_cameras if cam['is_microscope']]
        
        logger.info(f"Found {len(microscope_cameras)} potential microscope cameras")
        return microscope_cameras
    
    @staticmethod
    def get_best_microscope_device_id() -> Optional[int]:
        """
        Get the device ID of the best microscope camera candidate.
        
        Returns:
            Device ID of the best microscope camera or None if not found
        """
        microscope_cameras = CameraDetector.find_microscope_cameras()
        
        if not microscope_cameras:
            logger.warning("No microscope cameras detected")
            return None
        
        # Prefer higher resolution cameras
        best_camera = max(microscope_cameras, 
                         key=lambda cam: cam['resolution'][0] * cam['resolution'][1])
        
        logger.info(f"Best microscope camera: {best_camera['name']} (device {best_camera['device_id']})")
        return best_camera['device_id']
    
    @staticmethod
    def print_camera_report():
        """Print a detailed report of all detected cameras."""
        print("\n" + "="*60)
        print("CAMERA DETECTION REPORT")
        print("="*60)
        
        cameras = CameraDetector.scan_cameras()
        
        if not cameras:
            print("❌ No cameras detected")
            return
        
        print(f"📷 Found {len(cameras)} camera(s):")
        print()
        
        for i, cam in enumerate(cameras, 1):
            microscope_indicator = "🔬" if cam['is_microscope'] else "📹"
            print(f"{microscope_indicator} Camera {i}:")
            print(f"   Device ID: {cam['device_id']}")
            print(f"   Name: {cam['name']}")
            print(f"   Resolution: {cam['resolution'][0]}x{cam['resolution'][1]}")
            print(f"   FPS: {cam['fps']:.1f}")
            print(f"   Backend: {cam['backend']}")
            print(f"   Microscope: {'Yes' if cam['is_microscope'] else 'No'}")
            print()
        
        # Recommend best microscope
        microscope_cameras = [cam for cam in cameras if cam['is_microscope']]
        if microscope_cameras:
            best = max(microscope_cameras, 
                      key=lambda cam: cam['resolution'][0] * cam['resolution'][1])
            print(f"🎯 Recommended microscope: Device {best['device_id']} ({best['name']})")
        else:
            print("⚠️  No microscope cameras detected. USB cameras may be on higher device IDs.")
        
        print("="*60)


if __name__ == "__main__":
    # Run camera detection when script is executed directly
    CameraDetector.print_camera_report()