"""
Classification module for MLMCSC system.

This module provides shear percentage classification capabilities
using both traditional ML and deep learning approaches.
"""

# Import components with error handling
_available_components = []

try:
    from .improved_fracture_detector import ImprovedFractureSurfaceDetector
    _available_components.append('ImprovedFractureSurfaceDetector')
except ImportError as e:
    print(f"Warning: Could not import ImprovedFractureSurfaceDetector: {e}")

try:
    from .shiny_region_analyzer import ShinyRegionAnalyzer, ShinyRegionFeatures
    _available_components.extend(['ShinyRegionAnalyzer', 'ShinyRegionFeatures'])
except ImportError as e:
    print(f"Warning: Could not import ShinyRegionAnalyzer: {e}")

try:
    from .shiny_region_classifier import ShinyRegionBasedClassifier
    _available_components.append('ShinyRegionBasedClassifier')
except ImportError as e:
    print(f"Warning: Could not import ShinyRegionBasedClassifier: {e}")

__all__ = _available_components