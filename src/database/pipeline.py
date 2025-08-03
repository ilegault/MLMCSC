#!/usr/bin/env python3
"""
Data Pipeline for MLMCSC - Automated Data Management

This module provides automated data pipeline functionality including:
- Automated backups
- Data versioning
- Privacy compliance
- Efficient data processing for training
- Data quality monitoring
"""

import os
import json
import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass, asdict

from .manager import DatabaseManager
from .models import Image, Label, Prediction, DataVersion, AuditLog, Technician

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline operations."""
    # Backup settings
    backup_enabled: bool = True
    backup_schedule: str = "daily"  # daily, weekly, monthly
    backup_retention_days: int = 30
    backup_compression: bool = True
    
    # Data versioning
    auto_versioning: bool = True
    version_trigger_threshold: int = 100  # New labels to trigger versioning
    
    # Privacy compliance
    anonymization_enabled: bool = True
    data_retention_days: int = 365
    
    # Data quality
    quality_monitoring: bool = True
    quality_threshold: float = 0.7
    
    # Performance
    max_workers: int = 4
    batch_size: int = 1000
    
    # Monitoring
    metrics_collection: bool = True
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "low_quality_images_percent": 20.0,
                "prediction_accuracy_drop": 10.0,
                "database_size_gb": 10.0
            }


class DataPipeline:
    """Automated data pipeline for MLMCSC system."""
    
    def __init__(self, db_manager: DatabaseManager, config: Optional[PipelineConfig] = None):
        """
        Initialize data pipeline.
        
        Args:
            db_manager: Database manager instance
            config: Pipeline configuration
        """
        self.db_manager = db_manager
        self.config = config or PipelineConfig()
        self.is_running = False
        self.scheduler_thread = None
        
        # Initialize pipeline
        self._setup_scheduler()
        
        logger.info("Data pipeline initialized")
    
    def start(self):
        """Start the data pipeline."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        
        # Start scheduler in separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Data pipeline started")
    
    def stop(self):
        """Stop the data pipeline."""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Data pipeline stopped")
    
    def _setup_scheduler(self):
        """Setup scheduled tasks."""
        schedule.clear()
        
        if self.config.backup_enabled:
            if self.config.backup_schedule == "daily":
                schedule.every().day.at("02:00").do(self._scheduled_backup)
            elif self.config.backup_schedule == "weekly":
                schedule.every().week.do(self._scheduled_backup)
            elif self.config.backup_schedule == "monthly":
                schedule.every(30).days.do(self._scheduled_backup)
        
        if self.config.anonymization_enabled:
            # Run anonymization weekly
            schedule.every().week.do(self._scheduled_anonymization)
        
        if self.config.quality_monitoring:
            # Run quality monitoring daily
            schedule.every().day.at("06:00").do(self._scheduled_quality_check)
        
        if self.config.metrics_collection:
            # Collect metrics every hour
            schedule.every().hour.do(self._scheduled_metrics_collection)
        
        if self.config.auto_versioning:
            # Check for versioning daily
            schedule.every().day.at("01:00").do(self._scheduled_version_check)
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    # ========================================================================
    # SCHEDULED TASKS
    # ========================================================================
    
    def _scheduled_backup(self):
        """Scheduled backup task."""
        try:
            backup_path = self.create_backup()
            self._cleanup_old_backups()
            logger.info(f"Scheduled backup completed: {backup_path}")
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")
    
    def _scheduled_anonymization(self):
        """Scheduled anonymization task."""
        try:
            self.db_manager.anonymize_data(
                retention_days=self.config.data_retention_days,
                user_id="pipeline"
            )
            logger.info("Scheduled anonymization completed")
        except Exception as e:
            logger.error(f"Scheduled anonymization failed: {e}")
    
    def _scheduled_quality_check(self):
        """Scheduled data quality check."""
        try:
            quality_report = self.check_data_quality()
            self._process_quality_alerts(quality_report)
            logger.info("Scheduled quality check completed")
        except Exception as e:
            logger.error(f"Scheduled quality check failed: {e}")
    
    def _scheduled_metrics_collection(self):
        """Scheduled metrics collection."""
        try:
            self.collect_system_metrics()
            logger.debug("Scheduled metrics collection completed")
        except Exception as e:
            logger.error(f"Scheduled metrics collection failed: {e}")
    
    def _scheduled_version_check(self):
        """Scheduled version check."""
        try:
            if self._should_create_version():
                version_name = f"auto_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.create_data_version(version_name, "Automated version")
                logger.info(f"Automated data version created: {version_name}")
        except Exception as e:
            logger.error(f"Scheduled version check failed: {e}")
    
    # ========================================================================
    # BACKUP MANAGEMENT
    # ========================================================================
    
    def create_backup(self, backup_name: Optional[str] = None) -> Path:
        """Create a database backup."""
        backup_path = self.db_manager.backup_database(backup_name)
        
        # Record backup metrics
        backup_size = backup_path.stat().st_size
        self.db_manager.record_metric(
            "backup_size_bytes", backup_size, "pipeline", "bytes"
        )
        self.db_manager.record_metric(
            "backup_created", 1, "pipeline", "count"
        )
        
        return backup_path
    
    def _cleanup_old_backups(self):
        """Clean up old backup files."""
        backup_dir = self.db_manager.backup_dir
        cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
        
        deleted_count = 0
        for backup_file in backup_dir.glob("*.db"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backup files")
    
    # ========================================================================
    # DATA VERSIONING
    # ========================================================================
    
    def create_data_version(self, version_name: str, description: str = "") -> int:
        """Create a new data version."""
        version_id = self.db_manager.create_data_version(
            version=version_name,
            name=version_name,
            description=description,
            user_id="pipeline"
        )
        
        # Create backup for this version
        backup_path = self.create_backup(f"version_{version_name}")
        
        # Record versioning metrics
        self.db_manager.record_metric(
            "data_version_created", 1, "pipeline", "count"
        )
        
        return version_id
    
    def _should_create_version(self) -> bool:
        """Check if a new data version should be created."""
        with self.db_manager.get_session() as session:
            # Get latest version
            latest_version = session.query(DataVersion).order_by(
                DataVersion.created_at.desc()
            ).first()
            
            if not latest_version:
                return True  # No versions exist
            
            # Count new labels since last version
            new_labels = session.query(Label).filter(
                Label.timestamp > latest_version.created_at
            ).count()
            
            return new_labels >= self.config.version_trigger_threshold
    
    # ========================================================================
    # DATA QUALITY MONITORING
    # ========================================================================
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Perform comprehensive data quality check."""
        with self.db_manager.get_session() as session:
            # Image quality metrics
            total_images = session.query(Image).filter(Image.is_valid == True).count()
            low_quality_images = session.query(Image).filter(
                Image.is_valid == True,
                Image.quality_score < self.config.quality_threshold
            ).count()
            
            low_quality_percent = (low_quality_images / total_images * 100) if total_images > 0 else 0
            
            # Label consistency metrics
            total_labels = session.query(Label).count()
            verified_labels = session.query(Label).filter(Label.is_verified == True).count()
            verification_rate = (verified_labels / total_labels * 100) if total_labels > 0 else 0
            
            # Prediction accuracy (recent)
            recent_date = datetime.now() - timedelta(days=7)
            recent_labels = session.query(Label).filter(
                Label.timestamp > recent_date,
                Label.original_prediction.isnot(None)
            ).all()
            
            accuracy_metrics = self._calculate_prediction_accuracy(recent_labels)
            
            # Data completeness
            images_without_labels = session.query(Image).outerjoin(Label).filter(
                Image.is_valid == True,
                Label.id.is_(None)
            ).count()
            
            completeness_rate = ((total_images - images_without_labels) / total_images * 100) if total_images > 0 else 0
            
            quality_report = {
                "timestamp": datetime.now().isoformat(),
                "image_quality": {
                    "total_images": total_images,
                    "low_quality_count": low_quality_images,
                    "low_quality_percent": low_quality_percent
                },
                "label_quality": {
                    "total_labels": total_labels,
                    "verified_labels": verified_labels,
                    "verification_rate": verification_rate
                },
                "prediction_accuracy": accuracy_metrics,
                "data_completeness": {
                    "images_without_labels": images_without_labels,
                    "completeness_rate": completeness_rate
                }
            }
            
            # Record quality metrics
            self.db_manager.record_metric(
                "data_quality_low_quality_percent", low_quality_percent, "pipeline", "percent"
            )
            self.db_manager.record_metric(
                "data_quality_verification_rate", verification_rate, "pipeline", "percent"
            )
            self.db_manager.record_metric(
                "data_quality_completeness_rate", completeness_rate, "pipeline", "percent"
            )
            
            return quality_report
    
    def _calculate_prediction_accuracy(self, labels: List[Label]) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        if not labels:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "sample_count": 0}
        
        technician_labels = [label.label for label in labels]
        model_predictions = [label.original_prediction for label in labels]
        
        # Calculate metrics
        differences = [abs(t - m) for t, m in zip(technician_labels, model_predictions)]
        mae = sum(differences) / len(differences)
        rmse = (sum(d**2 for d in differences) / len(differences)) ** 0.5
        
        # Calculate R² score
        mean_technician = sum(technician_labels) / len(technician_labels)
        ss_res = sum((t - m)**2 for t, m in zip(technician_labels, model_predictions))
        ss_tot = sum((t - mean_technician)**2 for t in technician_labels)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "sample_count": len(labels)
        }
    
    def _process_quality_alerts(self, quality_report: Dict[str, Any]):
        """Process quality report and generate alerts."""
        alerts = []
        
        # Check low quality images threshold
        low_quality_percent = quality_report["image_quality"]["low_quality_percent"]
        if low_quality_percent > self.config.alert_thresholds["low_quality_images_percent"]:
            alerts.append({
                "type": "data_quality",
                "severity": "warning",
                "message": f"High percentage of low quality images: {low_quality_percent:.1f}%"
            })
        
        # Check prediction accuracy
        if quality_report["prediction_accuracy"]["sample_count"] > 0:
            mae = quality_report["prediction_accuracy"]["mae"]
            # This would need baseline comparison for meaningful alerts
            # For now, just log the metric
            self.db_manager.record_metric(
                "prediction_accuracy_mae", mae, "pipeline", "error"
            )
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Quality Alert: {alert['message']}")
            self.db_manager.record_metric(
                f"alert_{alert['type']}", 1, "pipeline", "count",
                metadata=alert
            )
    
    # ========================================================================
    # SYSTEM METRICS COLLECTION
    # ========================================================================
    
    def collect_system_metrics(self):
        """Collect comprehensive system metrics."""
        # Database metrics
        health_data = self.db_manager.get_system_health()
        
        for category, metrics in health_data.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.db_manager.record_metric(
                            f"{category}_{metric_name}", value, "system"
                        )
        
        # Storage metrics
        self._collect_storage_metrics()
        
        # Performance metrics
        self._collect_performance_metrics()
    
    def _collect_storage_metrics(self):
        """Collect storage-related metrics."""
        try:
            # Database size
            if "sqlite" in self.db_manager.db_url:
                db_path = Path(self.db_manager.db_url.replace("sqlite:///", ""))
                if db_path.exists():
                    db_size_bytes = db_path.stat().st_size
                    db_size_gb = db_size_bytes / (1024**3)
                    
                    self.db_manager.record_metric(
                        "database_size_bytes", db_size_bytes, "storage", "bytes"
                    )
                    self.db_manager.record_metric(
                        "database_size_gb", db_size_gb, "storage", "gb"
                    )
            
            # Backup directory size
            backup_size = sum(f.stat().st_size for f in self.db_manager.backup_dir.glob("*") if f.is_file())
            self.db_manager.record_metric(
                "backup_total_size_bytes", backup_size, "storage", "bytes"
            )
            
        except Exception as e:
            logger.error(f"Error collecting storage metrics: {e}")
    
    def _collect_performance_metrics(self):
        """Collect performance-related metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.db_manager.record_metric("system_cpu_percent", cpu_percent, "system", "percent")
            self.db_manager.record_metric("system_memory_percent", memory.percent, "system", "percent")
            self.db_manager.record_metric("system_disk_percent", disk.percent, "system", "percent")
            
        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    # ========================================================================
    # EFFICIENT DATA PROCESSING
    # ========================================================================
    
    def process_training_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Process a batch of data for training with efficient querying."""
        batch_size = batch_size or self.config.batch_size
        
        # Get training data splits
        training_splits = self.db_manager.get_training_data(
            verified_only=True,
            min_confidence=0.5
        )
        
        # Process in batches
        processed_data = {
            "train": self._process_data_batch(training_splits["train"], batch_size),
            "val": self._process_data_batch(training_splits["val"], batch_size),
            "test": self._process_data_batch(training_splits["test"], batch_size)
        }
        
        return processed_data
    
    def _process_data_batch(self, data_list: List[tuple], batch_size: int) -> List[Dict]:
        """Process a batch of data with parallel processing."""
        batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
        processed_batches = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_single_batch, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    result = future.result()
                    processed_batches.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        
        return processed_batches
    
    def _process_single_batch(self, batch: List[tuple]) -> Dict[str, Any]:
        """Process a single batch of data."""
        image_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Get features for this batch
        features = self.db_manager.get_features_for_training(
            image_ids, feature_type="cnn"
        )
        
        return {
            "image_ids": image_ids,
            "labels": labels,
            "features": features.tolist() if len(features) > 0 else [],
            "batch_size": len(batch)
        }
    
    # ========================================================================
    # PRIVACY COMPLIANCE UTILITIES
    # ========================================================================
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report."""
        with self.db_manager.get_session() as session:
            # Data retention analysis
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
            
            old_images = session.query(Image).filter(Image.timestamp < cutoff_date).count()
            old_labels = session.query(Label).filter(Label.timestamp < cutoff_date).count()
            old_predictions = session.query(Prediction).filter(Prediction.timestamp < cutoff_date).count()
            
            # Anonymization status
            anonymized_logs = session.query(AuditLog).filter(
                AuditLog.user_id.like("anonymized_%")
            ).count()
            
            # Active technicians
            active_technicians = session.query(Technician).filter(
                Technician.is_active == True
            ).count()
            
            privacy_report = {
                "timestamp": datetime.now().isoformat(),
                "data_retention": {
                    "retention_days": self.config.data_retention_days,
                    "old_images": old_images,
                    "old_labels": old_labels,
                    "old_predictions": old_predictions
                },
                "anonymization": {
                    "anonymized_audit_logs": anonymized_logs,
                    "anonymization_enabled": self.config.anonymization_enabled
                },
                "active_users": {
                    "active_technicians": active_technicians
                },
                "compliance_status": {
                    "gdpr_ready": True,
                    "data_retention_policy": "active",
                    "anonymization_policy": "active" if self.config.anonymization_enabled else "disabled"
                }
            }
            
            return privacy_report
    
    # ========================================================================
    # PIPELINE STATUS AND CONTROL
    # ========================================================================
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "is_running": self.is_running,
            "config": asdict(self.config),
            "last_backup": self._get_last_backup_info(),
            "next_scheduled_tasks": self._get_next_scheduled_tasks(),
            "system_health": self.db_manager.get_system_health()
        }
    
    def _get_last_backup_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last backup."""
        backup_files = list(self.db_manager.backup_dir.glob("*.db"))
        if not backup_files:
            return None
        
        latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
        return {
            "path": str(latest_backup),
            "size_bytes": latest_backup.stat().st_size,
            "created_at": datetime.fromtimestamp(latest_backup.stat().st_mtime).isoformat()
        }
    
    def _get_next_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Get information about next scheduled tasks."""
        tasks = []
        for job in schedule.jobs:
            tasks.append({
                "task": str(job.job_func),
                "next_run": job.next_run.isoformat() if job.next_run else None,
                "interval": str(job.interval)
            })
        return tasks
    
    def update_config(self, new_config: PipelineConfig):
        """Update pipeline configuration."""
        self.config = new_config
        self._setup_scheduler()
        logger.info("Pipeline configuration updated")


# ========================================================================
# PIPELINE FACTORY AND UTILITIES
# ========================================================================

def create_pipeline(db_manager: DatabaseManager, 
                   config_dict: Optional[Dict[str, Any]] = None) -> DataPipeline:
    """Factory function to create a data pipeline."""
    config = PipelineConfig(**config_dict) if config_dict else PipelineConfig()
    return DataPipeline(db_manager, config)


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from file."""
    if not config_path.exists():
        return PipelineConfig()
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return PipelineConfig(**config_data)


def save_pipeline_config(config: PipelineConfig, config_path: Path):
    """Save pipeline configuration to file."""
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)