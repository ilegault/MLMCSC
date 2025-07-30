"""
Classification module for MLMCSC system.

This module provides shear percentage classification capabilities
using both traditional ML and deep learning approaches.
"""

try:
    from .improved_fracture_detector import ImprovedFractureSurfaceDetector
    from .shiny_region_analyzer import ShinyRegionAnalyzer, ShinyRegionFeatures
    from .shiny_region_classifier import ShinyRegionBasedClassifier
except ImportError:
    pass

__all__ = [
    'ImprovedFractureSurfaceDetector',
    'ShinyRegionAnalyzer', 
    'ShinyRegionFeatures',
    'ShinyRegionBasedClassifier',
]