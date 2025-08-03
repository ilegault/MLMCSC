#!/usr/bin/env python3
"""
Database Configuration for MLMCSC Data Management

This module provides configuration management for the database system,
including connection settings, backup policies, and data management rules.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class BackupSchedule(Enum):
    """Backup schedule options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class DatabaseConfig:
    """Database connection and settings configuration."""
    # Connection settings
    db_type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 5432
    database: str = "mlmcsc"
    username: str = ""
    password: str = ""
    
    # SQLite specific
    sqlite_path: str = "src/database/data/mlmcsc.db"
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Performance settings
    echo_sql: bool = False
    query_timeout: int = 30
    
    def get_connection_url(self, project_root: Optional[Path] = None) -> str:
        """Generate database connection URL."""
        if self.db_type == DatabaseType.SQLITE:
            if project_root:
                db_path = project_root / self.sqlite_path
            else:
                db_path = Path(self.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path}"
        
        elif self.db_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        elif self.db_type == DatabaseType.MYSQL:
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@dataclass
class BackupConfig:
    """Backup configuration settings."""
    enabled: bool = True
    schedule: BackupSchedule = BackupSchedule.DAILY
    backup_dir: str = "src/database/backups"
    retention_days: int = 30
    compression: bool = True
    encryption: bool = False
    encryption_key_path: str = ""
    
    # Backup validation
    verify_backups: bool = True
    test_restore: bool = False
    
    # Remote backup
    remote_backup: bool = False
    remote_location: str = ""
    remote_credentials: Dict[str, str] = None
    
    def __post_init__(self):
        if self.remote_credentials is None:
            self.remote_credentials = {}


@dataclass
class DataRetentionConfig:
    """Data retention and privacy configuration."""
    # General retention
    default_retention_days: int = 365
    
    # Specific retention policies
    image_retention_days: int = 730  # 2 years
    label_retention_days: int = 1095  # 3 years
    prediction_retention_days: int = 365  # 1 year
    audit_log_retention_days: int = 2555  # 7 years
    
    # Anonymization settings
    anonymization_enabled: bool = True
    anonymization_delay_days: int = 90
    
    # GDPR compliance
    gdpr_compliance: bool = True
    data_export_enabled: bool = True
    right_to_be_forgotten: bool = True
    
    # Archive settings
    archive_old_data: bool = True
    archive_threshold_days: int = 180
    archive_location: str = "src/database/archive"


@dataclass
class DataQualityConfig:
    """Data quality monitoring configuration."""
    enabled: bool = True
    
    # Quality thresholds
    min_image_quality_score: float = 0.7
    min_label_confidence: float = 0.8
    max_prediction_error: float = 0.1
    
    # Monitoring intervals
    quality_check_interval_hours: int = 24
    alert_threshold_percent: float = 20.0
    
    # Validation rules
    require_image_validation: bool = True
    require_label_verification: bool = False
    auto_flag_outliers: bool = True
    
    # Quality metrics
    track_inter_annotator_agreement: bool = True
    track_model_drift: bool = True
    track_data_distribution: bool = True


@dataclass
class SecurityConfig:
    """Security and access control configuration."""
    # Authentication
    require_authentication: bool = True
    session_timeout_minutes: int = 480  # 8 hours
    
    # Authorization
    role_based_access: bool = True
    default_role: str = "technician"
    
    # Audit logging
    audit_all_operations: bool = True
    audit_sensitive_operations: bool = True
    audit_retention_days: int = 2555  # 7 years
    
    # Data encryption
    encrypt_sensitive_data: bool = False
    encryption_algorithm: str = "AES-256"
    
    # Access control
    ip_whitelist: list = None
    max_failed_logins: int = 5
    lockout_duration_minutes: int = 30
    
    def __post_init__(self):
        if self.ip_whitelist is None:
            self.ip_whitelist = []


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    # Query optimization
    enable_query_cache: bool = True
    cache_size_mb: int = 256
    query_timeout_seconds: int = 30
    
    # Batch processing
    default_batch_size: int = 1000
    max_batch_size: int = 10000
    parallel_workers: int = 4
    
    # Indexing
    auto_create_indexes: bool = True
    rebuild_indexes_schedule: str = "weekly"
    
    # Memory management
    max_memory_usage_mb: int = 2048
    gc_threshold: int = 1000
    
    # Connection management
    connection_pool_size: int = 10
    max_connections: int = 50


@dataclass
class MonitoringConfig:
    """System monitoring configuration."""
    enabled: bool = True
    
    # Metrics collection
    collect_system_metrics: bool = True
    collect_database_metrics: bool = True
    collect_application_metrics: bool = True
    
    # Collection intervals
    metrics_interval_seconds: int = 60
    health_check_interval_seconds: int = 300
    
    # Alerting
    enable_alerts: bool = True
    alert_email: str = ""
    alert_webhook: str = ""
    
    # Thresholds
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 90.0
    database_size_threshold_gb: float = 10.0
    
    # Logging
    log_level: str = "INFO"
    log_file_path: str = "logs/database.log"
    log_rotation_size_mb: int = 100
    log_retention_days: int = 30


@dataclass
class DataManagementConfig:
    """Complete data management configuration."""
    database: DatabaseConfig = None
    backup: BackupConfig = None
    retention: DataRetentionConfig = None
    quality: DataQualityConfig = None
    security: SecurityConfig = None
    performance: PerformanceConfig = None
    monitoring: MonitoringConfig = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.backup is None:
            self.backup = BackupConfig()
        if self.retention is None:
            self.retention = DataRetentionConfig()
        if self.quality is None:
            self.quality = DataQualityConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'DataManagementConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            return cls()  # Return default configuration
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataManagementConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'database' in data:
            db_data = data['database'].copy()
            if 'db_type' in db_data:
                db_data['db_type'] = DatabaseType(db_data['db_type'])
            config.database = DatabaseConfig(**db_data)
        
        if 'backup' in data:
            backup_data = data['backup'].copy()
            if 'schedule' in backup_data:
                backup_data['schedule'] = BackupSchedule(backup_data['schedule'])
            config.backup = BackupConfig(**backup_data)
        
        if 'retention' in data:
            config.retention = DataRetentionConfig(**data['retention'])
        
        if 'quality' in data:
            config.quality = DataQualityConfig(**data['quality'])
        
        if 'security' in data:
            config.security = SecurityConfig(**data['security'])
        
        if 'performance' in data:
            config.performance = PerformanceConfig(**data['performance'])
        
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        
        if self.database:
            db_dict = asdict(self.database)
            db_dict['db_type'] = self.database.db_type.value
            result['database'] = db_dict
        
        if self.backup:
            backup_dict = asdict(self.backup)
            backup_dict['schedule'] = self.backup.schedule.value
            result['backup'] = backup_dict
        
        if self.retention:
            result['retention'] = asdict(self.retention)
        
        if self.quality:
            result['quality'] = asdict(self.quality)
        
        if self.security:
            result['security'] = asdict(self.security)
        
        if self.performance:
            result['performance'] = asdict(self.performance)
        
        if self.monitoring:
            result['monitoring'] = asdict(self.monitoring)
        
        return result
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Database validation
        if self.database.db_type == DatabaseType.SQLITE:
            sqlite_path = Path(self.database.sqlite_path)
            if not sqlite_path.parent.exists():
                try:
                    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create SQLite directory: {e}")
        
        # Backup validation
        if self.backup.enabled:
            backup_dir = Path(self.backup.backup_dir)
            if not backup_dir.exists():
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create backup directory: {e}")
        
        # Retention validation
        if self.retention.image_retention_days < 1:
            issues.append("Image retention days must be at least 1")
        
        if self.retention.audit_log_retention_days < self.retention.default_retention_days:
            issues.append("Audit log retention should be longer than default retention")
        
        # Quality validation
        if not (0 <= self.quality.min_image_quality_score <= 1):
            issues.append("Image quality score must be between 0 and 1")
        
        # Performance validation
        if self.performance.parallel_workers < 1:
            issues.append("Parallel workers must be at least 1")
        
        if self.performance.default_batch_size > self.performance.max_batch_size:
            issues.append("Default batch size cannot exceed max batch size")
        
        return issues


def load_config(config_path: Optional[Union[str, Path]] = None) -> DataManagementConfig:
    """Load data management configuration."""
    if config_path is None:
        # Try to find config file in standard locations
        project_root = Path(__file__).parent.parent.parent
        database_dir = Path(__file__).parent
        possible_paths = [
            database_dir / "database_config.json",  # In database directory (preferred)
            project_root / "config" / "database.json",
            project_root / "src" / "database" / "config.json",
            Path.cwd() / "database_config.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and Path(config_path).exists():
        return DataManagementConfig.from_file(config_path)
    else:
        # Return default configuration
        return DataManagementConfig()


def create_default_config(config_path: Union[str, Path]):
    """Create a default configuration file."""
    config = DataManagementConfig()
    config.save_to_file(config_path)


def get_environment_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    env_config = {}
    
    # Database settings from environment
    if os.getenv('DATABASE_URL'):
        env_config['database'] = {'connection_url': os.getenv('DATABASE_URL')}
    
    if os.getenv('DB_HOST'):
        env_config.setdefault('database', {})['host'] = os.getenv('DB_HOST')
    
    if os.getenv('DB_PORT'):
        env_config.setdefault('database', {})['port'] = int(os.getenv('DB_PORT'))
    
    if os.getenv('DB_NAME'):
        env_config.setdefault('database', {})['database'] = os.getenv('DB_NAME')
    
    if os.getenv('DB_USER'):
        env_config.setdefault('database', {})['username'] = os.getenv('DB_USER')
    
    if os.getenv('DB_PASSWORD'):
        env_config.setdefault('database', {})['password'] = os.getenv('DB_PASSWORD')
    
    # Backup settings
    if os.getenv('BACKUP_ENABLED'):
        env_config.setdefault('backup', {})['enabled'] = os.getenv('BACKUP_ENABLED').lower() == 'true'
    
    if os.getenv('BACKUP_SCHEDULE'):
        env_config.setdefault('backup', {})['schedule'] = os.getenv('BACKUP_SCHEDULE')
    
    # Security settings
    if os.getenv('REQUIRE_AUTH'):
        env_config.setdefault('security', {})['require_authentication'] = os.getenv('REQUIRE_AUTH').lower() == 'true'
    
    return env_config


# Default configuration instance
DEFAULT_CONFIG = DataManagementConfig()