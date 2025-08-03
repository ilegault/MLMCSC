"""
Regression Module for Fracture Surface Analysis

This module provides regression models and online learning capabilities
for predicting material properties from extracted features.
"""

from .regression_model import FractureRegressionModel
from .online_learning import OnlineLearningSystem
from .model_versioning import ModelVersionManager

__all__ = [
    'FractureRegressionModel',
    'OnlineLearningSystem', 
    'ModelVersionManager'
]