"""
Object detection module for MLMCSC system.

This module provides YOLO-based fracture surface detection capabilities.
"""

try:
    from .object_detector import *
    from .annotation_utils import *
except ImportError:
    pass

__all__ = [
    'SpecimenDetector',
    'AnnotationConverter',
    'SimpleAnnotationTool',
    'Annotation',
]