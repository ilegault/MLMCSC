"""
Camera interface and calibration modules for microscope integration.
"""

from .microscope_interface import MicroscopeCapture, MicroscopeInterface
from .camera_detector import CameraDetector

__all__ = ['MicroscopeCapture', 'MicroscopeInterface']