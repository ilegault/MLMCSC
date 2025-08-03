#!/usr/bin/env python3
"""
Enhanced Database Models for MLMCSC Data Management

This module provides comprehensive database models for handling growing datasets
with proper schema design, data versioning, and efficient querying capabilities.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, 
    Boolean, ForeignKey, Index, LargeBinary, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import numpy as np
from pathlib import Path

Base = declarative_base()


class Image(Base):
    """Images table - stores image metadata and paths."""
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String(500), nullable=False, unique=True)
    filename = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Image metadata
    width = Column(Integer)
    height = Column(Integer)
    channels = Column(Integer)
    file_size = Column(Integer)  # in bytes
    format = Column(String(10))  # jpg, png, etc.
    
    # Microscope metadata
    magnification = Column(Float)
    specimen_id = Column(String(100))
    acquisition_settings = Column(JSON)  # Store camera settings as JSON
    
    # Data quality flags
    is_valid = Column(Boolean, default=True)
    quality_score = Column(Float)  # 0-1 quality assessment
    
    # Relationships
    predictions = relationship("Prediction", back_populates="image", cascade="all, delete-orphan")
    labels = relationship("Label", back_populates="image", cascade="all, delete-orphan")
    features = relationship("Feature", back_populates="image", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_images_timestamp', 'timestamp'),
        Index('idx_images_specimen', 'specimen_id'),
        Index('idx_images_path', 'path'),
        Index('idx_images_valid', 'is_valid'),
    )


class ModelVersion(Base):
    """Model versions table - tracks different model iterations."""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500))
    
    # Version metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    created_by = Column(String(100))
    is_active = Column(Boolean, default=False)
    
    # Performance metrics
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float)
    test_accuracy = Column(Float)
    training_loss = Column(Float)
    
    # Training metadata
    training_data_version = Column(String(50))
    training_duration = Column(Float)  # in hours
    hyperparameters = Column(JSON)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model_version")
    
    # Indexes
    __table_args__ = (
        Index('idx_model_versions_version', 'version'),
        Index('idx_model_versions_active', 'is_active'),
        Index('idx_model_versions_created', 'created_at'),
    )


class Prediction(Base):
    """Predictions table - stores model predictions with confidence scores."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    model_version_id = Column(Integer, ForeignKey('model_versions.id'), nullable=False)
    
    # Prediction results
    prediction = Column(Float, nullable=False)  # Main prediction value
    confidence = Column(Float, nullable=False)  # Confidence score 0-1
    
    # Detection results (for object detection models)
    detection_bbox = Column(JSON)  # Bounding box coordinates
    detection_confidence = Column(Float)
    detection_class = Column(String(50))
    
    # Processing metadata
    processing_time = Column(Float)  # in seconds
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Additional prediction data
    raw_output = Column(JSON)  # Store full model output
    preprocessing_params = Column(JSON)  # Parameters used for preprocessing
    
    # Relationships
    image = relationship("Image", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
    
    # Indexes
    __table_args__ = (
        Index('idx_predictions_image', 'image_id'),
        Index('idx_predictions_model', 'model_version_id'),
        Index('idx_predictions_timestamp', 'timestamp'),
        Index('idx_predictions_confidence', 'confidence'),
    )


class Technician(Base):
    """Technicians table - stores technician information."""
    __tablename__ = 'technicians'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    technician_id = Column(String(50), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100))
    department = Column(String(100))
    experience_level = Column(String(20))  # beginner, intermediate, expert
    
    # Activity tracking
    created_at = Column(DateTime, default=func.now(), nullable=False)
    last_active = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Performance metrics
    total_labels = Column(Integer, default=0)
    accuracy_score = Column(Float)  # Agreement with consensus
    
    # Relationships
    labels = relationship("Label", back_populates="technician")
    
    # Indexes
    __table_args__ = (
        Index('idx_technicians_id', 'technician_id'),
        Index('idx_technicians_active', 'is_active'),
    )


class Label(Base):
    """Labels table - stores human annotations and corrections."""
    __tablename__ = 'labels'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    technician_id = Column(Integer, ForeignKey('technicians.id'), nullable=False)
    
    # Label data
    label = Column(Float, nullable=False)  # Ground truth value
    label_type = Column(String(50), default='manual')  # manual, corrected, consensus
    
    # Context from prediction (if correcting a prediction)
    original_prediction = Column(Float)
    original_confidence = Column(Float)
    model_version_used = Column(String(50))
    
    # Labeling metadata
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    time_spent = Column(Float)  # seconds spent labeling
    difficulty_rating = Column(Integer)  # 1-5 scale
    notes = Column(Text)
    
    # Quality assurance
    is_verified = Column(Boolean, default=False)
    verified_by = Column(String(50))
    verification_timestamp = Column(DateTime)
    
    # Additional annotation data
    annotation_data = Column(JSON)  # Store additional annotations
    
    # Relationships
    image = relationship("Image", back_populates="labels")
    technician = relationship("Technician", back_populates="labels")
    
    # Indexes
    __table_args__ = (
        Index('idx_labels_image', 'image_id'),
        Index('idx_labels_technician', 'technician_id'),
        Index('idx_labels_timestamp', 'timestamp'),
        Index('idx_labels_type', 'label_type'),
        Index('idx_labels_verified', 'is_verified'),
    )


class Feature(Base):
    """Features table - stores extracted feature vectors for efficient training."""
    __tablename__ = 'features'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    
    # Feature metadata
    feature_type = Column(String(50), nullable=False)  # cnn, histogram, texture, etc.
    extractor_version = Column(String(50), nullable=False)
    extraction_timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Feature data
    feature_vector = Column(LargeBinary)  # Serialized numpy array
    feature_dimension = Column(Integer)
    
    # Processing metadata
    extraction_time = Column(Float)  # seconds
    preprocessing_params = Column(JSON)
    
    # Relationships
    image = relationship("Image", back_populates="features")
    
    # Indexes
    __table_args__ = (
        Index('idx_features_image', 'image_id'),
        Index('idx_features_type', 'feature_type'),
        Index('idx_features_extractor', 'extractor_version'),
        Index('idx_features_timestamp', 'extraction_timestamp'),
    )


class DataVersion(Base):
    """Data versions table - tracks dataset versions for reproducibility."""
    __tablename__ = 'data_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Version metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    created_by = Column(String(100))
    
    # Dataset statistics
    total_images = Column(Integer)
    total_labels = Column(Integer)
    train_split = Column(Float)
    val_split = Column(Float)
    test_split = Column(Float)
    
    # Data quality metrics
    avg_quality_score = Column(Float)
    label_distribution = Column(JSON)
    
    # Backup information
    backup_path = Column(String(500))
    backup_size = Column(Integer)  # in bytes
    backup_checksum = Column(String(64))  # SHA-256
    
    # Indexes
    __table_args__ = (
        Index('idx_data_versions_version', 'version'),
        Index('idx_data_versions_created', 'created_at'),
    )


class AuditLog(Base):
    """Audit log table - tracks all data operations for compliance."""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Operation details
    operation = Column(String(50), nullable=False)  # CREATE, UPDATE, DELETE, EXPORT
    table_name = Column(String(50), nullable=False)
    record_id = Column(Integer)
    
    # User and timestamp
    user_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Change details
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    # Context
    reason = Column(Text)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_logs_timestamp', 'timestamp'),
        Index('idx_audit_logs_user', 'user_id'),
        Index('idx_audit_logs_operation', 'operation'),
        Index('idx_audit_logs_table', 'table_name'),
    )


class SystemMetric(Base):
    """System metrics table - tracks system performance and health."""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metric details
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    
    # Context
    component = Column(String(50))  # database, model, api, etc.
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Additional data
    metric_metadata = Column(JSON)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_name', 'metric_name'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
        Index('idx_system_metrics_component', 'component'),
    )