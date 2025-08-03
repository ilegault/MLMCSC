#!/usr/bin/env python3
"""
MLMCSC Database Management System

This package provides comprehensive database management for the MLMCSC system,
including data storage, versioning, backup, privacy compliance, and efficient
querying for machine learning training.

Key Components:
- DatabaseManager: Core database operations and management
- DataPipeline: Automated data processing and maintenance
- Models: SQLAlchemy database models
- Configuration: System configuration management
- CLI: Command-line interface for database operations

Example Usage:
    from src.database import DatabaseManager, DataPipeline
    from src.database.config import load_config
    
    # Initialize database
    config = load_config()
    db_manager = DatabaseManager(config.database.get_connection_url())
    
    # Start data pipeline
    pipeline = DataPipeline(db_manager)
    pipeline.start()
"""

from .manager import DatabaseManager
from .pipeline import DataPipeline, PipelineConfig
from .config import DataManagementConfig, load_config
from .models import (
    Image, ModelVersion, Prediction, Technician, Label, 
    Feature, DataVersion, AuditLog, SystemMetric
)

__version__ = "1.0.0"
__author__ = "MLMCSC Team"

__all__ = [
    # Core classes
    "DatabaseManager",
    "DataPipeline", 
    "PipelineConfig",
    "DataManagementConfig",
    
    # Database models
    "Image",
    "ModelVersion", 
    "Prediction",
    "Technician",
    "Label",
    "Feature",
    "DataVersion",
    "AuditLog",
    "SystemMetric",
    
    # Utilities
    "load_config",
]