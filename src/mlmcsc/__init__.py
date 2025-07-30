"""
MLMCSC - Machine Learning Microscope Control System for Charpy Fracture Classification

A comprehensive system for:
- Live microscope control and image capture
- YOLO-based fracture surface detection
- Shear percentage classification using machine learning
- Real-time analysis and visualization

Main Components:
- Core: Base functionality and interfaces
- Detection: Object detection models and utilities
- Classification: Shear classification models
- Camera: Microscope interface and calibration
- Preprocessing: Image processing and data preparation
- Postprocessing: Results analysis and visualization
"""

__version__ = "2.0.0"
__author__ = "MLMCSC Team"
__description__ = "Machine Learning Microscope Control System for Charpy Fracture Classification"

# Core imports for easy access
try:
    from .core import *
except ImportError as e:
    print(f"Warning: Could not import core module: {e}")

try:
    from .detection import *
except ImportError as e:
    print(f"Warning: Could not import detection module: {e}")

try:
    from .classification import *
except ImportError as e:
    print(f"Warning: Could not import classification module: {e}")

# Version info
VERSION_INFO = {
    'version': __version__,
    'description': __description__,
    'components': [
        'Core functionality',
        'YOLO detection',
        'Shear classification',
        'Camera interface',
        'Live viewer application'
    ]
}

def get_version():
    """Get version information."""
    return VERSION_INFO

def print_info():
    """Print system information."""
    print(f"MLMCSC v{__version__}")
    print(f"Description: {__description__}")
    print("Components:")
    for component in VERSION_INFO['components']:
        print(f"  - {component}")