"""
Models package for MLMCSC - Machine Learning Microscope Control System.

This package contains machine learning models for microscope automation,
including object detection, specimen tracking, and image analysis.
"""

from .object_detector import (
    SpecimenDetector,
    DetectionResult,
    SpecimenTracker,
    MotionDetector,
    RotationDetector
)

from .train_specimen_detector import (
    SpecimenTrainer,
    DatasetPreparer
)

from .annotation_utils import (
    Annotation,
    AnnotationConverter,
    SimpleAnnotationTool,
    validate_annotations,
    create_class_names_file
)

__all__ = [
    # Main detector classes
    'SpecimenDetector',
    'DetectionResult',
    'SpecimenTracker',
    'MotionDetector',
    'RotationDetector',
    
    # Training utilities
    'SpecimenTrainer',
    'DatasetPreparer',
    
    # Annotation utilities
    'Annotation',
    'AnnotationConverter',
    'SimpleAnnotationTool',
    'validate_annotations',
    'create_class_names_file'
]