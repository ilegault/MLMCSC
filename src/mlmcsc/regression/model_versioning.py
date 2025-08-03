#!/usr/bin/env python3
"""
Model Versioning System for Online Learning

This module manages different versions of the regression model,
tracks performance improvements, and handles model rollbacks.
"""

import json
import shutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Container for model version information."""
    version_id: str
    version_number: int
    creation_time: datetime
    model_type: str
    performance_metrics: Dict[str, float]
    training_samples_count: int
    model_file_path: str
    metadata: Dict[str, Any]
    parent_version: Optional[str] = None
    is_active: bool = False
    is_baseline: bool = False
    creation_reason: str = "manual"
    notes: str = ""
    feature_names: List[str] = None
    scaler_type: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['creation_time'] = self.creation_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary format."""
        data = data.copy()
        data['creation_time'] = datetime.fromisoformat(data['creation_time'])
        return cls(**data)


class ModelVersionManager:
    """
    Manages different versions of the online learning model.
    
    Provides functionality for:
    - Creating new model versions
    - Tracking performance improvements
    - Rolling back to previous versions
    - Comparing model versions
    - Automatic model backup and cleanup
    """
    
    def __init__(self, models_directory: Path):
        """
        Initialize the model version manager.
        
        Args:
            models_directory: Directory to store model versions
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Version tracking
        self.versions_file = self.models_directory / 'versions.json'
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version_id: Optional[str] = None
        
        # Load existing versions
        self._load_versions()
        
        logger.info(f"ModelVersionManager initialized with {len(self.versions)} versions")
    
    def _load_versions(self):
        """Load existing model versions from disk."""
        try:
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                
                self.versions = {
                    version_id: ModelVersion.from_dict(version_data)
                    for version_id, version_data in data['versions'].items()
                }
                
                self.active_version_id = data.get('active_version_id')
                
                logger.info(f"Loaded {len(self.versions)} model versions")
            else:
                logger.info("No existing versions file found")
                
        except Exception as e:
            logger.error(f"Failed to load versions: {e}")
            self.versions = {}
            self.active_version_id = None
    
    def _save_versions(self):
        """Save model versions to disk."""
        try:
            data = {
                'versions': {
                    version_id: version.to_dict()
                    for version_id, version in self.versions.items()
                },
                'active_version_id': self.active_version_id,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
            raise
    
    def _generate_version_id(self, model_data: bytes) -> str:
        """Generate a unique version ID based on model content."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        content_hash = hashlib.md5(model_data).hexdigest()[:8]
        return f"v_{timestamp}_{content_hash}"
    
    def create_version(self, 
                      model_path: Path,
                      model_type: str,
                      performance_metrics: Dict[str, float],
                      training_samples_count: int,
                      metadata: Optional[Dict[str, Any]] = None,
                      parent_version_id: Optional[str] = None,
                      set_as_active: bool = True) -> str:
        """
        Create a new model version.
        
        Args:
            model_path: Path to the model file to version
            model_type: Type of the model (e.g., 'sgd', 'passive_aggressive')
            performance_metrics: Performance metrics for this version
            training_samples_count: Number of training samples used
            metadata: Additional metadata
            parent_version_id: ID of the parent version (for tracking lineage)
            set_as_active: Whether to set this as the active version
            
        Returns:
            Version ID of the created version
        """
        try:
            # Read model file to generate version ID
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            version_id = self._generate_version_id(model_data)
            version_number = len(self.versions) + 1
            
            # Create version directory
            version_dir = self.models_directory / version_id
            version_dir.mkdir(exist_ok=True)
            
            # Copy model file to version directory
            version_model_path = version_dir / 'model.joblib'
            shutil.copy2(model_path, version_model_path)
            
            # Create version object
            version = ModelVersion(
                version_id=version_id,
                version_number=version_number,
                creation_time=datetime.now(),
                model_type=model_type,
                performance_metrics=performance_metrics.copy(),
                training_samples_count=training_samples_count,
                model_file_path=str(version_model_path),
                metadata=metadata or {},
                parent_version=parent_version_id,
                is_active=set_as_active
            )
            
            # Add to versions
            self.versions[version_id] = version
            
            # Update active version
            if set_as_active:
                # Deactivate previous active version
                if self.active_version_id and self.active_version_id in self.versions:
                    self.versions[self.active_version_id].is_active = False
                
                self.active_version_id = version_id
                version.is_active = True
            
            # Save versions
            self._save_versions()
            
            # Save version metadata
            version_metadata_file = version_dir / 'metadata.json'
            with open(version_metadata_file, 'w') as f:
                json.dump(version.to_dict(), f, indent=2)
            
            logger.info(f"Created model version {version_id} (v{version_number})")
            logger.info(f"Performance: {performance_metrics}")
            
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            raise
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self.versions.get(version_id)
    
    def get_active_version(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        if self.active_version_id:
            return self.versions.get(self.active_version_id)
        return None
    
    def list_versions(self, sort_by: str = 'creation_time') -> List[ModelVersion]:
        """
        List all model versions.
        
        Args:
            sort_by: Field to sort by ('creation_time', 'version_number', 'performance')
            
        Returns:
            List of model versions sorted by specified field
        """
        versions = list(self.versions.values())
        
        if sort_by == 'creation_time':
            versions.sort(key=lambda v: v.creation_time, reverse=True)
        elif sort_by == 'version_number':
            versions.sort(key=lambda v: v.version_number, reverse=True)
        elif sort_by == 'performance':
            # Sort by R² score if available, otherwise by first metric
            def get_performance_key(v):
                if 'r2' in v.performance_metrics:
                    return v.performance_metrics['r2']
                elif v.performance_metrics:
                    return list(v.performance_metrics.values())[0]
                return 0
            versions.sort(key=get_performance_key, reverse=True)
        
        return versions
    
    def set_active_version(self, version_id: str) -> bool:
        """
        Set a specific version as active.
        
        Args:
            version_id: ID of the version to activate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if version_id not in self.versions:
                logger.error(f"Version {version_id} not found")
                return False
            
            # Deactivate current active version
            if self.active_version_id and self.active_version_id in self.versions:
                self.versions[self.active_version_id].is_active = False
            
            # Activate new version
            self.versions[version_id].is_active = True
            self.active_version_id = version_id
            
            # Save changes
            self._save_versions()
            
            logger.info(f"Set version {version_id} as active")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set active version: {e}")
            return False
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Comparison results
        """
        try:
            version1 = self.versions.get(version_id1)
            version2 = self.versions.get(version_id2)
            
            if not version1 or not version2:
                raise ValueError("One or both versions not found")
            
            comparison = {
                'version1': {
                    'id': version_id1,
                    'number': version1.version_number,
                    'creation_time': version1.creation_time.isoformat(),
                    'performance': version1.performance_metrics,
                    'samples_count': version1.training_samples_count
                },
                'version2': {
                    'id': version_id2,
                    'number': version2.version_number,
                    'creation_time': version2.creation_time.isoformat(),
                    'performance': version2.performance_metrics,
                    'samples_count': version2.training_samples_count
                },
                'differences': {}
            }
            
            # Compare performance metrics
            all_metrics = set(version1.performance_metrics.keys()) | set(version2.performance_metrics.keys())
            for metric in all_metrics:
                val1 = version1.performance_metrics.get(metric, 0)
                val2 = version2.performance_metrics.get(metric, 0)
                comparison['differences'][metric] = val2 - val1
            
            # Compare sample counts
            comparison['differences']['samples_count'] = version2.training_samples_count - version1.training_samples_count
            
            # Time difference
            time_diff = version2.creation_time - version1.creation_time
            comparison['differences']['time_diff_hours'] = time_diff.total_seconds() / 3600
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history across all versions."""
        versions = self.list_versions(sort_by='creation_time')
        
        history = []
        for version in versions:
            history.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'creation_time': version.creation_time.isoformat(),
                'performance_metrics': version.performance_metrics,
                'training_samples_count': version.training_samples_count,
                'is_active': version.is_active
            })
        
        return history
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            version_id: ID of the version to rollback to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if version_id not in self.versions:
                logger.error(f"Version {version_id} not found")
                return False
            
            # Set as active version
            success = self.set_active_version(version_id)
            
            if success:
                logger.info(f"Rolled back to version {version_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def delete_version(self, version_id: str, force: bool = False) -> bool:
        """
        Delete a model version.
        
        Args:
            version_id: ID of the version to delete
            force: Whether to force deletion of active version
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if version_id not in self.versions:
                logger.error(f"Version {version_id} not found")
                return False
            
            version = self.versions[version_id]
            
            # Prevent deletion of active version unless forced
            if version.is_active and not force:
                logger.error(f"Cannot delete active version {version_id} without force=True")
                return False
            
            # Delete version directory
            version_dir = Path(version.model_file_path).parent
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # Remove from versions
            del self.versions[version_id]
            
            # Update active version if deleted
            if self.active_version_id == version_id:
                self.active_version_id = None
                # Optionally set most recent version as active
                if self.versions:
                    latest_version = max(self.versions.values(), key=lambda v: v.creation_time)
                    self.set_active_version(latest_version.version_id)
            
            # Save changes
            self._save_versions()
            
            logger.info(f"Deleted version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False
    
    def cleanup_old_versions(self, keep_count: int = 10) -> int:
        """
        Clean up old model versions, keeping only the most recent ones.
        
        Args:
            keep_count: Number of versions to keep
            
        Returns:
            Number of versions deleted
        """
        try:
            if len(self.versions) <= keep_count:
                logger.info("No cleanup needed")
                return 0
            
            # Get versions sorted by creation time (newest first)
            versions = self.list_versions(sort_by='creation_time')
            
            # Keep the most recent versions and the active version
            versions_to_keep = set()
            
            # Always keep active version
            if self.active_version_id:
                versions_to_keep.add(self.active_version_id)
            
            # Keep most recent versions
            for version in versions[:keep_count]:
                versions_to_keep.add(version.version_id)
            
            # Delete old versions
            deleted_count = 0
            for version in versions[keep_count:]:
                if version.version_id not in versions_to_keep:
                    if self.delete_version(version.version_id, force=False):
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old versions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    def export_version_history(self, output_file: Path) -> None:
        """Export version history to JSON file."""
        try:
            history_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_versions': len(self.versions),
                'active_version_id': self.active_version_id,
                'performance_history': self.get_performance_history(),
                'versions': {
                    version_id: version.to_dict()
                    for version_id, version in self.versions.items()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f"Exported version history to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export version history: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about model versions."""
        if not self.versions:
            return {'total_versions': 0}
        
        versions = list(self.versions.values())
        
        # Performance statistics
        r2_scores = [v.performance_metrics.get('r2', 0) for v in versions if 'r2' in v.performance_metrics]
        
        stats = {
            'total_versions': len(versions),
            'active_version_id': self.active_version_id,
            'oldest_version': min(versions, key=lambda v: v.creation_time).creation_time.isoformat(),
            'newest_version': max(versions, key=lambda v: v.creation_time).creation_time.isoformat(),
            'model_types': list(set(v.model_type for v in versions)),
            'total_training_samples': sum(v.training_samples_count for v in versions),
        }
        
        if r2_scores:
            stats['performance_stats'] = {
                'best_r2': max(r2_scores),
                'worst_r2': min(r2_scores),
                'avg_r2': sum(r2_scores) / len(r2_scores),
                'r2_improvement': max(r2_scores) - min(r2_scores) if len(r2_scores) > 1 else 0
            }
        
        return stats


@dataclass
class ABTestConfig:
    """Configuration for A/B testing between model versions."""
    test_id: str
    version_a: str
    version_b: str
    traffic_split: float  # 0.0-1.0, percentage for version A
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics_tracked: List[str] = None
    is_active: bool = True
    results: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTestConfig':
        """Create from dictionary."""
        data = data.copy()
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)


class EnhancedModelVersionManager(ModelVersionManager):
    """
    Enhanced model version manager with advanced features:
    - Automatic checkpointing after N updates
    - Performance degradation detection and rollback
    - A/B testing capability
    - Comprehensive metadata tracking
    """
    
    def __init__(self, 
                 models_directory: Path,
                 checkpoint_frequency: int = 10,
                 performance_threshold: float = 0.05,
                 max_versions: int = 50):
        """
        Initialize enhanced model version manager.
        
        Args:
            models_directory: Directory to store model versions
            checkpoint_frequency: Create checkpoint after N updates
            performance_threshold: Minimum performance drop to trigger rollback warning
            max_versions: Maximum number of versions to keep
        """
        super().__init__(models_directory)
        
        self.checkpoint_frequency = checkpoint_frequency
        self.performance_threshold = performance_threshold
        self.max_versions = max_versions
        self.update_count = 0
        self.baseline_version_id: Optional[str] = None
        
        # A/B testing
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.ab_tests_file = self.models_directory / 'ab_tests.json'
        self._load_ab_tests()
        
        logger.info(f"EnhancedModelVersionManager initialized with checkpoint frequency: {checkpoint_frequency}")
    
    def _load_ab_tests(self):
        """Load existing A/B tests from disk."""
        try:
            if self.ab_tests_file.exists():
                with open(self.ab_tests_file, 'r') as f:
                    data = json.load(f)
                
                self.ab_tests = {
                    test_id: ABTestConfig.from_dict(test_data)
                    for test_id, test_data in data.items()
                }
                
                logger.info(f"Loaded {len(self.ab_tests)} A/B tests")
            else:
                logger.info("No existing A/B tests file found")
                
        except Exception as e:
            logger.error(f"Failed to load A/B tests: {e}")
            self.ab_tests = {}
    
    def _save_ab_tests(self):
        """Save A/B tests to disk."""
        try:
            data = {
                test_id: test.to_dict()
                for test_id, test in self.ab_tests.items()
            }
            
            with open(self.ab_tests_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save A/B tests: {e}")
            raise
    
    def should_create_checkpoint(self) -> bool:
        """Check if a checkpoint should be created based on update frequency."""
        self.update_count += 1
        return self.update_count % self.checkpoint_frequency == 0
    
    def create_checkpoint_version(self,
                                 model_path: Path,
                                 scaler_path: Optional[Path],
                                 model_type: str,
                                 performance_metrics: Dict[str, float],
                                 training_samples_count: int,
                                 feature_names: List[str],
                                 scaler_type: str = "standard",
                                 reason: str = "checkpoint",
                                 notes: str = "") -> str:
        """
        Create a checkpoint version with enhanced metadata.
        
        Args:
            model_path: Path to the model file
            scaler_path: Path to the scaler file (optional)
            model_type: Type of model
            performance_metrics: Performance metrics
            training_samples_count: Number of training samples
            feature_names: List of feature names
            scaler_type: Type of scaler used
            reason: Reason for creating checkpoint
            notes: Additional notes
            
        Returns:
            Version ID of the created checkpoint
        """
        try:
            # Enhanced metadata
            metadata = {
                'feature_names': feature_names,
                'scaler_type': scaler_type,
                'creation_reason': reason,
                'notes': notes,
                'update_count': self.update_count,
                'checkpoint_frequency': self.checkpoint_frequency
            }
            
            # Create version using parent method
            version_id = self.create_version(
                model_path=model_path,
                model_type=model_type,
                performance_metrics=performance_metrics,
                training_samples_count=training_samples_count,
                metadata=metadata,
                parent_version_id=self.active_version_id,
                set_as_active=True
            )
            
            # Copy scaler file if provided
            if scaler_path and scaler_path.exists():
                version_dir = Path(self.versions[version_id].model_file_path).parent
                scaler_dest = version_dir / 'scaler.joblib'
                shutil.copy2(scaler_path, scaler_dest)
                
                # Update metadata with scaler path
                self.versions[version_id].metadata['scaler_file_path'] = str(scaler_dest)
                self._save_versions()
            
            # Set as baseline if it's the first version or performance significantly improved
            if self._should_set_as_baseline(version_id):
                self.set_baseline_version(version_id)
            
            # Cleanup old versions if needed
            if len(self.versions) > self.max_versions:
                self.cleanup_old_versions(keep_count=self.max_versions)
            
            logger.info(f"Created checkpoint {version_id} - R²: {performance_metrics.get('r2', 0):.3f}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint version: {e}")
            raise
    
    def set_baseline_version(self, version_id: str) -> bool:
        """Set a version as the baseline for performance comparison."""
        try:
            if version_id not in self.versions:
                logger.error(f"Version {version_id} not found")
                return False
            
            # Remove baseline flag from current baseline
            if self.baseline_version_id and self.baseline_version_id in self.versions:
                self.versions[self.baseline_version_id].is_baseline = False
            
            # Set new baseline
            self.versions[version_id].is_baseline = True
            self.baseline_version_id = version_id
            
            # Save changes
            self._save_versions()
            
            logger.info(f"Set version {version_id} as baseline")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set baseline version: {e}")
            return False
    
    def check_performance_degradation(self, current_metrics: Dict[str, float]) -> Optional[str]:
        """
        Check if current performance has degraded compared to baseline.
        
        Args:
            current_metrics: Current model performance metrics
            
        Returns:
            Warning message if degradation detected, None otherwise
        """
        if not self.baseline_version_id or self.baseline_version_id not in self.versions:
            return None
        
        baseline_version = self.versions[self.baseline_version_id]
        baseline_metrics = baseline_version.performance_metrics
        
        # Check R² score degradation
        current_r2 = current_metrics.get('r2', 0)
        baseline_r2 = baseline_metrics.get('r2', 0)
        
        if baseline_r2 - current_r2 > self.performance_threshold:
            warning = (f"Performance degradation detected! "
                      f"R² dropped from {baseline_r2:.3f} to {current_r2:.3f} "
                      f"(>{self.performance_threshold:.3f} threshold). "
                      f"Consider rolling back to baseline version {self.baseline_version_id}")
            logger.warning(warning)
            return warning
        
        return None
    
    def auto_rollback_if_degraded(self, current_metrics: Dict[str, float]) -> bool:
        """
        Automatically rollback if performance has degraded significantly.
        
        Args:
            current_metrics: Current model performance metrics
            
        Returns:
            True if rollback was performed, False otherwise
        """
        warning = self.check_performance_degradation(current_metrics)
        
        if warning and self.baseline_version_id:
            logger.warning("Performing automatic rollback due to performance degradation")
            success = self.rollback_to_version(self.baseline_version_id)
            
            if success:
                # Add note about automatic rollback
                baseline_version = self.versions[self.baseline_version_id]
                baseline_version.notes += f" [Auto-rollback on {datetime.now().isoformat()}]"
                self._save_versions()
            
            return success
        
        return False
    
    def start_ab_test(self,
                     version_a: str,
                     version_b: str,
                     traffic_split: float = 0.5,
                     duration_hours: int = 24,
                     metrics_tracked: Optional[List[str]] = None) -> str:
        """
        Start an A/B test between two model versions.
        
        Args:
            version_a: First version ID
            version_b: Second version ID
            traffic_split: Percentage of traffic for version A (0.0-1.0)
            duration_hours: Duration of test in hours
            metrics_tracked: List of metrics to track
            
        Returns:
            A/B test ID
        """
        if version_a not in self.versions or version_b not in self.versions:
            raise ValueError("Both versions must exist")
        
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("Traffic split must be between 0.0 and 1.0")
        
        # Generate test ID
        test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate end time
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Create A/B test configuration
        ab_test = ABTestConfig(
            test_id=test_id,
            version_a=version_a,
            version_b=version_b,
            traffic_split=traffic_split,
            start_time=start_time,
            end_time=end_time,
            metrics_tracked=metrics_tracked or ['r2', 'rmse', 'mae'],
            is_active=True
        )
        
        self.ab_tests[test_id] = ab_test
        self._save_ab_tests()
        
        logger.info(f"Started A/B test {test_id}: {version_a} vs {version_b} "
                   f"({traffic_split:.1%} split for {duration_hours}h)")
        return test_id
    
    def get_ab_test_version(self, test_id: str, user_hash: Optional[str] = None) -> str:
        """
        Get the version to use for a specific user in an A/B test.
        
        Args:
            test_id: A/B test ID
            user_hash: Hash of user identifier for consistent assignment
            
        Returns:
            Version ID to use for this user
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        ab_test = self.ab_tests[test_id]
        
        if not ab_test.is_active:
            raise ValueError(f"A/B test {test_id} is not active")
        
        # Check if test has expired
        if ab_test.end_time and datetime.now() > ab_test.end_time:
            ab_test.is_active = False
            self._save_ab_tests()
            raise ValueError(f"A/B test {test_id} has expired")
        
        # Determine version based on traffic split
        if user_hash:
            # Consistent assignment based on user hash
            hash_value = int(hashlib.md5(user_hash.encode()).hexdigest()[:8], 16)
            use_version_a = (hash_value % 100) < (ab_test.traffic_split * 100)
        else:
            # Random assignment
            use_version_a = np.random.random() < ab_test.traffic_split
        
        return ab_test.version_a if use_version_a else ab_test.version_b
    
    def stop_ab_test(self, test_id: str) -> Dict[str, Any]:
        """
        Stop an A/B test and return results summary.
        
        Args:
            test_id: A/B test ID
            
        Returns:
            Test results summary
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        ab_test = self.ab_tests[test_id]
        ab_test.is_active = False
        ab_test.end_time = datetime.now()
        
        # Get version performance for comparison
        version_a_perf = self.versions[ab_test.version_a].performance_metrics
        version_b_perf = self.versions[ab_test.version_b].performance_metrics
        
        # Determine winner based on R² score
        winner = "version_a" if version_a_perf.get('r2', 0) > version_b_perf.get('r2', 0) else "version_b"
        
        results = {
            'test_id': test_id,
            'version_a': ab_test.version_a,
            'version_b': ab_test.version_b,
            'traffic_split': ab_test.traffic_split,
            'duration_hours': (ab_test.end_time - ab_test.start_time).total_seconds() / 3600,
            'version_a_performance': version_a_perf,
            'version_b_performance': version_b_perf,
            'winner': winner,
            'performance_difference': {
                metric: version_b_perf.get(metric, 0) - version_a_perf.get(metric, 0)
                for metric in ab_test.metrics_tracked
            }
        }
        
        ab_test.results = results
        self._save_ab_tests()
        
        logger.info(f"Stopped A/B test {test_id}, winner: {winner}")
        return results
    
    def get_active_ab_tests(self) -> List[Dict[str, Any]]:
        """Get list of currently active A/B tests."""
        active_tests = []
        current_time = datetime.now()
        
        for test_id, ab_test in self.ab_tests.items():
            if ab_test.is_active:
                # Check if test has expired
                if ab_test.end_time and current_time > ab_test.end_time:
                    ab_test.is_active = False
                    self._save_ab_tests()
                else:
                    active_tests.append({
                        'test_id': test_id,
                        'version_a': ab_test.version_a,
                        'version_b': ab_test.version_b,
                        'traffic_split': ab_test.traffic_split,
                        'start_time': ab_test.start_time.isoformat(),
                        'end_time': ab_test.end_time.isoformat() if ab_test.end_time else None,
                        'time_remaining_hours': (ab_test.end_time - current_time).total_seconds() / 3600 if ab_test.end_time else None
                    })
        
        return active_tests
    
    def load_version_with_scaler(self, version_id: str) -> Tuple[Any, Any, ModelVersion]:
        """
        Load a model version along with its scaler.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            Tuple of (model, scaler, version_metadata)
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        
        # Load model
        model = joblib.load(version.model_file_path)
        
        # Load scaler if available
        scaler = None
        scaler_path = version.metadata.get('scaler_file_path')
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded version {version_id} with scaler")
        return model, scaler, version
    
    def _should_set_as_baseline(self, version_id: str) -> bool:
        """Determine if a version should be set as baseline."""
        if not self.baseline_version_id:
            return True
        
        if self.baseline_version_id not in self.versions:
            return True
        
        current_version = self.versions[version_id]
        baseline_version = self.versions[self.baseline_version_id]
        
        current_r2 = current_version.performance_metrics.get('r2', 0)
        baseline_r2 = baseline_version.performance_metrics.get('r2', 0)
        
        # Set as baseline if R² improved by more than 5%
        return current_r2 > baseline_r2 + 0.05
    
    def get_performance_comparison(self, version_ids: List[str]) -> Dict[str, Any]:
        """
        Compare performance metrics across multiple versions.
        
        Args:
            version_ids: List of version IDs to compare
            
        Returns:
            Performance comparison results
        """
        try:
            comparison_results = {
                'versions': {},
                'metrics': ['r2', 'rmse', 'mae', 'mse'],
                'best_performers': {},
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            # Collect metrics for each version
            for version_id in version_ids:
                if version_id in self.versions:
                    version = self.versions[version_id]
                    comparison_results['versions'][version_id] = {
                        'performance_metrics': version.performance_metrics,
                        'creation_time': version.creation_time,
                        'training_samples_count': version.training_samples_count,
                        'model_type': version.model_type
                    }
            
            # Find best performers for each metric
            for metric in comparison_results['metrics']:
                best_version = None
                best_value = None
                
                for version_id, data in comparison_results['versions'].items():
                    if metric in data['performance_metrics']:
                        value = data['performance_metrics'][metric]
                        
                        # For R², higher is better; for errors, lower is better
                        is_better = False
                        if metric == 'r2':
                            is_better = best_value is None or value > best_value
                        else:  # rmse, mae, mse
                            is_better = best_value is None or value < best_value
                        
                        if is_better:
                            best_value = value
                            best_version = version_id
                
                if best_version:
                    comparison_results['best_performers'][metric] = {
                        'version_id': best_version,
                        'value': best_value
                    }
            
            logger.info(f"Compared performance across {len(version_ids)} versions")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            raise