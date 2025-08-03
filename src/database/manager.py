#!/usr/bin/env python3
"""
Enhanced Database Manager for MLMCSC Data Management

This module provides comprehensive database management capabilities including
data versioning, backup management, privacy compliance, and efficient querying.
"""

import os
import json
import hashlib
import shutil
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager
import sqlite3
import pickle

from sqlalchemy import create_engine, text, and_, or_, desc, asc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from .models import (
    Base, Image, ModelVersion, Prediction, Technician, Label, 
    Feature, DataVersion, AuditLog, SystemMetric
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Enhanced database manager with comprehensive data management capabilities."""
    
    def __init__(self, 
                 db_url: Optional[str] = None,
                 backup_dir: Optional[Path] = None,
                 enable_audit: bool = True):
        """
        Initialize the database manager.
        
        Args:
            db_url: Database URL (defaults to SQLite)
            backup_dir: Directory for backups
            enable_audit: Enable audit logging
        """
        if db_url is None:
            # Default to SQLite database
            db_path = Path(__file__).parent / "data" / "mlmcsc.db"
            db_path.parent.mkdir(exist_ok=True)
            db_url = f"sqlite:///{db_path}"
        
        self.db_url = db_url
        self.backup_dir = backup_dir or Path(__file__).parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.enable_audit = enable_audit
        
        # Create engine with connection pooling for SQLite
        if "sqlite" in db_url:
            self.engine = create_engine(
                db_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                echo=False
            )
        else:
            self.engine = create_engine(db_url, echo=False)
        
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize database
        self.init_database()
        
        logger.info(f"Database manager initialized with URL: {db_url}")
    
    def init_database(self):
        """Initialize database tables and indexes."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def log_audit(self, session: Session, operation: str, table_name: str, 
                  record_id: Optional[int] = None, old_values: Optional[Dict] = None,
                  new_values: Optional[Dict] = None, user_id: str = "system",
                  reason: Optional[str] = None):
        """Log audit trail for data operations."""
        if not self.enable_audit:
            return
        
        audit_log = AuditLog(
            operation=operation,
            table_name=table_name,
            record_id=record_id,
            user_id=user_id,
            old_values=old_values,
            new_values=new_values,
            reason=reason,
            timestamp=datetime.now()
        )
        session.add(audit_log)
    
    # ========================================================================
    # IMAGE MANAGEMENT
    # ========================================================================
    
    def store_image(self, image_path: Union[str, Path], 
                   specimen_id: Optional[str] = None,
                   magnification: Optional[float] = None,
                   acquisition_settings: Optional[Dict] = None,
                   user_id: str = "system") -> int:
        """Store image metadata in database."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with self.get_session() as session:
            # Check if image already exists
            existing = session.query(Image).filter(Image.path == str(image_path)).first()
            if existing:
                logger.warning(f"Image already exists: {image_path}")
                return existing.id
            
            # Get image metadata
            file_stats = image_path.stat()
            
            # Try to get image dimensions (requires PIL/OpenCV)
            width, height, channels = None, None, None
            try:
                from PIL import Image as PILImage
                with PILImage.open(image_path) as img:
                    width, height = img.size
                    channels = len(img.getbands())
            except ImportError:
                logger.warning("PIL not available for image dimension detection")
            except Exception as e:
                logger.warning(f"Could not read image dimensions: {e}")
            
            # Create image record
            image = Image(
                path=str(image_path),
                filename=image_path.name,
                timestamp=datetime.now(),
                width=width,
                height=height,
                channels=channels,
                file_size=file_stats.st_size,
                format=image_path.suffix.lower().lstrip('.'),
                magnification=magnification,
                specimen_id=specimen_id,
                acquisition_settings=acquisition_settings,
                is_valid=True
            )
            
            session.add(image)
            session.flush()
            
            # Log audit trail
            self.log_audit(session, "CREATE", "images", image.id, 
                          new_values={"path": str(image_path)}, user_id=user_id)
            
            logger.info(f"Stored image {image.id}: {image_path}")
            return image.id
    
    def get_images(self, limit: int = 100, offset: int = 0,
                   specimen_id: Optional[str] = None,
                   valid_only: bool = True) -> List[Image]:
        """Get images with filtering and pagination."""
        with self.get_session() as session:
            query = session.query(Image)
            
            if valid_only:
                query = query.filter(Image.is_valid == True)
            
            if specimen_id:
                query = query.filter(Image.specimen_id == specimen_id)
            
            query = query.order_by(desc(Image.timestamp))
            query = query.offset(offset).limit(limit)
            
            images = query.all()
            # Expunge objects from session to avoid lazy loading issues
            for img in images:
                session.expunge(img)
            return images
    
    def update_image_quality(self, image_id: int, quality_score: float,
                           is_valid: bool = True, user_id: str = "system"):
        """Update image quality assessment."""
        with self.get_session() as session:
            image = session.query(Image).filter(Image.id == image_id).first()
            if not image:
                raise ValueError(f"Image {image_id} not found")
            
            old_values = {"quality_score": image.quality_score, "is_valid": image.is_valid}
            
            image.quality_score = quality_score
            image.is_valid = is_valid
            image.updated_at = datetime.now()
            
            self.log_audit(session, "UPDATE", "images", image_id,
                          old_values=old_values,
                          new_values={"quality_score": quality_score, "is_valid": is_valid},
                          user_id=user_id)
    
    # ========================================================================
    # MODEL VERSION MANAGEMENT
    # ========================================================================
    
    def register_model_version(self, version: str, name: str, model_path: str,
                             description: Optional[str] = None,
                             config_path: Optional[str] = None,
                             hyperparameters: Optional[Dict] = None,
                             performance_metrics: Optional[Dict] = None,
                             user_id: str = "system") -> int:
        """Register a new model version."""
        with self.get_session() as session:
            # Check if version already exists
            existing = session.query(ModelVersion).filter(ModelVersion.version == version).first()
            if existing:
                raise ValueError(f"Model version {version} already exists")
            
            model_version = ModelVersion(
                version=version,
                name=name,
                description=description,
                model_path=model_path,
                config_path=config_path,
                created_by=user_id,
                hyperparameters=hyperparameters,
                **performance_metrics if performance_metrics else {}
            )
            
            session.add(model_version)
            session.flush()
            
            self.log_audit(session, "CREATE", "model_versions", model_version.id,
                          new_values={"version": version, "name": name}, user_id=user_id)
            
            logger.info(f"Registered model version {version}")
            return model_version.id
    
    def set_active_model(self, version: str, user_id: str = "system"):
        """Set a model version as active."""
        with self.get_session() as session:
            # Deactivate all models
            session.query(ModelVersion).update({"is_active": False})
            
            # Activate specified model
            model = session.query(ModelVersion).filter(ModelVersion.version == version).first()
            if not model:
                raise ValueError(f"Model version {version} not found")
            
            model.is_active = True
            
            self.log_audit(session, "UPDATE", "model_versions", model.id,
                          new_values={"is_active": True}, user_id=user_id,
                          reason=f"Set as active model")
    
    def get_active_model(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        with self.get_session() as session:
            return session.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    
    # ========================================================================
    # PREDICTION MANAGEMENT
    # ========================================================================
    
    def store_prediction(self, image_id: int, model_version_id: int,
                        prediction: float, confidence: float,
                        detection_bbox: Optional[List[float]] = None,
                        detection_confidence: Optional[float] = None,
                        detection_class: Optional[str] = None,
                        processing_time: Optional[float] = None,
                        raw_output: Optional[Dict] = None,
                        preprocessing_params: Optional[Dict] = None,
                        user_id: str = "system") -> int:
        """Store model prediction."""
        with self.get_session() as session:
            prediction_record = Prediction(
                image_id=image_id,
                model_version_id=model_version_id,
                prediction=prediction,
                confidence=confidence,
                detection_bbox=detection_bbox,
                detection_confidence=detection_confidence,
                detection_class=detection_class,
                processing_time=processing_time,
                raw_output=raw_output,
                preprocessing_params=preprocessing_params,
                timestamp=datetime.now()
            )
            
            session.add(prediction_record)
            session.flush()
            
            self.log_audit(session, "CREATE", "predictions", prediction_record.id,
                          new_values={"image_id": image_id, "prediction": prediction},
                          user_id=user_id)
            
            return prediction_record.id
    
    def get_predictions(self, image_id: Optional[int] = None,
                       model_version_id: Optional[int] = None,
                       confidence_threshold: Optional[float] = None,
                       limit: int = 100, offset: int = 0) -> List[Prediction]:
        """Get predictions with filtering."""
        with self.get_session() as session:
            query = session.query(Prediction)
            
            if image_id:
                query = query.filter(Prediction.image_id == image_id)
            
            if model_version_id:
                query = query.filter(Prediction.model_version_id == model_version_id)
            
            if confidence_threshold:
                query = query.filter(Prediction.confidence >= confidence_threshold)
            
            query = query.order_by(desc(Prediction.timestamp))
            query = query.offset(offset).limit(limit)
            
            return query.all()
    
    # ========================================================================
    # TECHNICIAN AND LABEL MANAGEMENT
    # ========================================================================
    
    def register_technician(self, technician_id: str, name: str,
                          email: Optional[str] = None,
                          department: Optional[str] = None,
                          experience_level: str = "intermediate",
                          user_id: str = "system") -> int:
        """Register a new technician."""
        with self.get_session() as session:
            # Check if technician already exists
            existing = session.query(Technician).filter(
                Technician.technician_id == technician_id
            ).first()
            if existing:
                raise ValueError(f"Technician {technician_id} already exists")
            
            technician = Technician(
                technician_id=technician_id,
                name=name,
                email=email,
                department=department,
                experience_level=experience_level,
                last_active=datetime.now()
            )
            
            session.add(technician)
            session.flush()
            
            self.log_audit(session, "CREATE", "technicians", technician.id,
                          new_values={"technician_id": technician_id, "name": name},
                          user_id=user_id)
            
            return technician.id
    
    def store_label(self, image_id: int, technician_id: int, label: float,
                   label_type: str = "manual",
                   original_prediction: Optional[float] = None,
                   original_confidence: Optional[float] = None,
                   model_version_used: Optional[str] = None,
                   time_spent: Optional[float] = None,
                   difficulty_rating: Optional[int] = None,
                   notes: Optional[str] = None,
                   annotation_data: Optional[Dict] = None,
                   user_id: str = "system") -> int:
        """Store human label/annotation."""
        with self.get_session() as session:
            label_record = Label(
                image_id=image_id,
                technician_id=technician_id,
                label=label,
                label_type=label_type,
                original_prediction=original_prediction,
                original_confidence=original_confidence,
                model_version_used=model_version_used,
                time_spent=time_spent,
                difficulty_rating=difficulty_rating,
                notes=notes,
                annotation_data=annotation_data,
                timestamp=datetime.now()
            )
            
            session.add(label_record)
            session.flush()
            
            # Update technician statistics
            technician = session.query(Technician).filter(Technician.id == technician_id).first()
            if technician:
                technician.total_labels = (technician.total_labels or 0) + 1
                technician.last_active = datetime.now()
            
            self.log_audit(session, "CREATE", "labels", label_record.id,
                          new_values={"image_id": image_id, "label": label},
                          user_id=user_id)
            
            return label_record.id
    
    def verify_label(self, label_id: int, verified_by: str, user_id: str = "system"):
        """Mark a label as verified."""
        with self.get_session() as session:
            label = session.query(Label).filter(Label.id == label_id).first()
            if not label:
                raise ValueError(f"Label {label_id} not found")
            
            label.is_verified = True
            label.verified_by = verified_by
            label.verification_timestamp = datetime.now()
            
            self.log_audit(session, "UPDATE", "labels", label_id,
                          new_values={"is_verified": True, "verified_by": verified_by},
                          user_id=user_id)
    
    # ========================================================================
    # FEATURE MANAGEMENT
    # ========================================================================
    
    def store_features(self, image_id: int, feature_type: str,
                      extractor_version: str, feature_vector: np.ndarray,
                      preprocessing_params: Optional[Dict] = None,
                      extraction_time: Optional[float] = None,
                      user_id: str = "system") -> int:
        """Store extracted feature vectors."""
        with self.get_session() as session:
            # Serialize feature vector
            feature_bytes = pickle.dumps(feature_vector)
            
            feature_record = Feature(
                image_id=image_id,
                feature_type=feature_type,
                extractor_version=extractor_version,
                feature_vector=feature_bytes,
                feature_dimension=len(feature_vector),
                preprocessing_params=preprocessing_params,
                extraction_time=extraction_time,
                extraction_timestamp=datetime.now()
            )
            
            session.add(feature_record)
            session.flush()
            
            self.log_audit(session, "CREATE", "features", feature_record.id,
                          new_values={"image_id": image_id, "feature_type": feature_type},
                          user_id=user_id)
            
            return feature_record.id
    
    def get_features(self, image_id: Optional[int] = None,
                    feature_type: Optional[str] = None,
                    extractor_version: Optional[str] = None) -> List[Tuple[int, np.ndarray]]:
        """Get feature vectors with optional filtering."""
        with self.get_session() as session:
            query = session.query(Feature)
            
            if image_id:
                query = query.filter(Feature.image_id == image_id)
            
            if feature_type:
                query = query.filter(Feature.feature_type == feature_type)
            
            if extractor_version:
                query = query.filter(Feature.extractor_version == extractor_version)
            
            features = query.all()
            
            # Deserialize feature vectors
            result = []
            for feature in features:
                feature_vector = pickle.loads(feature.feature_vector)
                result.append((feature.image_id, feature_vector))
            
            return result
    
    # ========================================================================
    # DATA VERSIONING AND BACKUP
    # ========================================================================
    
    def create_data_version(self, version: str, name: str,
                          description: Optional[str] = None,
                          train_split: float = 0.7,
                          val_split: float = 0.15,
                          test_split: float = 0.15,
                          user_id: str = "system") -> int:
        """Create a new data version snapshot."""
        with self.get_session() as session:
            # Check if version already exists
            existing = session.query(DataVersion).filter(DataVersion.version == version).first()
            if existing:
                raise ValueError(f"Data version {version} already exists")
            
            # Calculate dataset statistics
            total_images = session.query(func.count(Image.id)).filter(Image.is_valid == True).scalar()
            total_labels = session.query(func.count(Label.id)).scalar()
            
            # Calculate average quality score
            avg_quality = session.query(func.avg(Image.quality_score)).filter(
                Image.is_valid == True, Image.quality_score.isnot(None)
            ).scalar()
            
            # Get label distribution
            label_stats = session.query(
                func.min(Label.label).label('min_label'),
                func.max(Label.label).label('max_label'),
                func.avg(Label.label).label('avg_label'),
                func.count(Label.label).label('count_label')
            ).first()
            
            label_distribution = {
                "min": float(label_stats.min_label) if label_stats.min_label else None,
                "max": float(label_stats.max_label) if label_stats.max_label else None,
                "mean": float(label_stats.avg_label) if label_stats.avg_label else None,
                "count": int(label_stats.count_label) if label_stats.count_label else 0
            }
            
            data_version = DataVersion(
                version=version,
                name=name,
                description=description,
                created_by=user_id,
                total_images=total_images,
                total_labels=total_labels,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
                avg_quality_score=avg_quality,
                label_distribution=label_distribution
            )
            
            session.add(data_version)
            session.flush()
            
            self.log_audit(session, "CREATE", "data_versions", data_version.id,
                          new_values={"version": version, "total_images": total_images},
                          user_id=user_id)
            
            logger.info(f"Created data version {version} with {total_images} images, {total_labels} labels")
            return data_version.id
    
    def backup_database(self, backup_name: Optional[str] = None) -> Path:
        """Create a backup of the database."""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / f"{backup_name}.db"
        
        # For SQLite, copy the database file
        if "sqlite" in self.db_url:
            db_path = self.db_url.replace("sqlite:///", "")
            shutil.copy2(db_path, backup_path)
        else:
            # For other databases, use pg_dump or similar
            raise NotImplementedError("Backup for non-SQLite databases not implemented")
        
        # Calculate checksum
        checksum = self._calculate_file_checksum(backup_path)
        file_size = backup_path.stat().st_size
        
        # Update data version with backup info if exists
        with self.get_session() as session:
            latest_version = session.query(DataVersion).order_by(desc(DataVersion.created_at)).first()
            if latest_version:
                latest_version.backup_path = str(backup_path)
                latest_version.backup_size = file_size
                latest_version.backup_checksum = checksum
        
        logger.info(f"Database backup created: {backup_path} (size: {file_size} bytes)")
        return backup_path
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    # ========================================================================
    # PRIVACY AND COMPLIANCE
    # ========================================================================
    
    def anonymize_data(self, retention_days: int = 365, user_id: str = "system"):
        """Anonymize old data for privacy compliance."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with self.get_session() as session:
            # Anonymize old audit logs
            old_logs = session.query(AuditLog).filter(AuditLog.timestamp < cutoff_date)
            anonymized_count = 0
            
            for log in old_logs:
                if log.user_id != "system":
                    log.user_id = f"anonymized_{log.id}"
                    log.ip_address = None
                    log.user_agent = None
                    anonymized_count += 1
            
            self.log_audit(session, "UPDATE", "audit_logs", None,
                          new_values={"anonymized_count": anonymized_count},
                          user_id=user_id, reason="Privacy compliance anonymization")
            
            logger.info(f"Anonymized {anonymized_count} audit log entries older than {retention_days} days")
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a specific user (GDPR compliance)."""
        with self.get_session() as session:
            # Get technician data
            technician = session.query(Technician).filter(
                Technician.technician_id == user_id
            ).first()
            
            user_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "technician_info": None,
                "labels": [],
                "audit_logs": []
            }
            
            if technician:
                user_data["technician_info"] = {
                    "name": technician.name,
                    "email": technician.email,
                    "department": technician.department,
                    "experience_level": technician.experience_level,
                    "total_labels": technician.total_labels,
                    "created_at": technician.created_at.isoformat(),
                    "last_active": technician.last_active.isoformat() if technician.last_active else None
                }
                
                # Get user's labels
                labels = session.query(Label).filter(Label.technician_id == technician.id).all()
                for label in labels:
                    user_data["labels"].append({
                        "id": label.id,
                        "image_id": label.image_id,
                        "label": label.label,
                        "timestamp": label.timestamp.isoformat(),
                        "notes": label.notes
                    })
            
            # Get audit logs
            audit_logs = session.query(AuditLog).filter(AuditLog.user_id == user_id).all()
            for log in audit_logs:
                user_data["audit_logs"].append({
                    "operation": log.operation,
                    "table_name": log.table_name,
                    "timestamp": log.timestamp.isoformat(),
                    "reason": log.reason
                })
            
            return user_data
    
    def delete_user_data(self, user_id: str, admin_user_id: str = "admin"):
        """Delete all data for a specific user (GDPR right to be forgotten)."""
        with self.get_session() as session:
            # Get technician
            technician = session.query(Technician).filter(
                Technician.technician_id == user_id
            ).first()
            
            if technician:
                # Delete labels
                labels_deleted = session.query(Label).filter(
                    Label.technician_id == technician.id
                ).delete()
                
                # Delete technician
                session.delete(technician)
                
                # Anonymize audit logs
                session.query(AuditLog).filter(AuditLog.user_id == user_id).update({
                    "user_id": f"deleted_{user_id}",
                    "ip_address": None,
                    "user_agent": None
                })
                
                self.log_audit(session, "DELETE", "technicians", technician.id,
                              reason=f"GDPR deletion request for user {user_id}",
                              user_id=admin_user_id)
                
                logger.info(f"Deleted user data for {user_id}: {labels_deleted} labels")
    
    # ========================================================================
    # EFFICIENT QUERYING FOR TRAINING
    # ========================================================================
    
    def get_training_data(self, 
                         data_version: Optional[str] = None,
                         train_split: float = 0.7,
                         val_split: float = 0.15,
                         test_split: float = 0.15,
                         min_confidence: Optional[float] = None,
                         verified_only: bool = False,
                         feature_type: Optional[str] = None) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get training data with efficient querying and splitting.
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing (image_id, label) tuples
        """
        with self.get_session() as session:
            # Base query for labeled images
            query = session.query(Image.id, Label.label).join(Label)
            
            # Apply filters
            query = query.filter(Image.is_valid == True)
            
            if verified_only:
                query = query.filter(Label.is_verified == True)
            
            if min_confidence:
                # Join with predictions to filter by confidence
                query = query.join(Prediction).filter(Prediction.confidence >= min_confidence)
            
            # Get all data
            all_data = query.all()
            
            if not all_data:
                return {"train": [], "val": [], "test": []}
            
            # Shuffle and split data
            import random
            random.shuffle(all_data)
            
            total_samples = len(all_data)
            train_end = int(total_samples * train_split)
            val_end = train_end + int(total_samples * val_split)
            
            splits = {
                "train": all_data[:train_end],
                "val": all_data[train_end:val_end],
                "test": all_data[val_end:]
            }
            
            logger.info(f"Training data split: train={len(splits['train'])}, "
                       f"val={len(splits['val'])}, test={len(splits['test'])}")
            
            return splits
    
    def get_features_for_training(self, image_ids: List[int], 
                                 feature_type: str = "cnn",
                                 extractor_version: Optional[str] = None) -> np.ndarray:
        """Get feature vectors for a list of image IDs."""
        with self.get_session() as session:
            query = session.query(Feature).filter(
                Feature.image_id.in_(image_ids),
                Feature.feature_type == feature_type
            )
            
            if extractor_version:
                query = query.filter(Feature.extractor_version == extractor_version)
            
            features = query.all()
            
            # Create feature matrix
            if not features:
                return np.array([])
            
            # Sort by image_id to maintain order
            features_dict = {f.image_id: pickle.loads(f.feature_vector) for f in features}
            feature_matrix = np.array([features_dict[img_id] for img_id in image_ids if img_id in features_dict])
            
            return feature_matrix
    
    # ========================================================================
    # SYSTEM METRICS AND MONITORING
    # ========================================================================
    
    def record_metric(self, metric_name: str, metric_value: float,
                     component: str = "system", metric_unit: Optional[str] = None,
                     metadata: Optional[Dict] = None):
        """Record system performance metric."""
        with self.get_session() as session:
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                component=component,
                metric_metadata=metadata,
                timestamp=datetime.now()
            )
            
            session.add(metric)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        with self.get_session() as session:
            # Database size and counts
            total_images = session.query(func.count(Image.id)).scalar()
            total_labels = session.query(func.count(Label.id)).scalar()
            total_predictions = session.query(func.count(Prediction.id)).scalar()
            
            # Recent activity (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            recent_labels = session.query(func.count(Label.id)).filter(
                Label.timestamp > yesterday
            ).scalar()
            recent_predictions = session.query(func.count(Prediction.id)).filter(
                Prediction.timestamp > yesterday
            ).scalar()
            
            # Data quality metrics
            avg_quality = session.query(func.avg(Image.quality_score)).filter(
                Image.quality_score.isnot(None)
            ).scalar()
            
            # Active technicians
            active_technicians = session.query(func.count(Technician.id)).filter(
                Technician.is_active == True
            ).scalar()
            
            return {
                "database_stats": {
                    "total_images": total_images,
                    "total_labels": total_labels,
                    "total_predictions": total_predictions,
                    "active_technicians": active_technicians
                },
                "recent_activity": {
                    "labels_24h": recent_labels,
                    "predictions_24h": recent_predictions
                },
                "data_quality": {
                    "avg_quality_score": float(avg_quality) if avg_quality else None
                },
                "timestamp": datetime.now().isoformat()
            }