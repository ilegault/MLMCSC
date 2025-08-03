#!/usr/bin/env python3
"""
Configuration for Human-in-the-Loop Web Interface

This module provides configuration settings for the web interface,
including model paths, database settings, and API parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEB_ROOT = Path(__file__).parent
DATA_DIR = WEB_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
STATIC_DIR = WEB_ROOT / "static"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


@dataclass
class WebConfig:
    """Configuration for the web interface."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True
    
    # Database settings
    database_path: Path = DATA_DIR / "labeling_history.db"
    
    # Model paths
    detection_model_path: Optional[Path] = None
    classification_model_path: Optional[Path] = None
    regression_model_path: Optional[Path] = None
    online_model_path: Path = DATA_DIR / "online_models"
    
    # Online learning settings
    online_learning_enabled: bool = True
    update_threshold: int = 10  # Minimum new labels before updating
    update_interval: int = 300  # Seconds between update checks
    
    # API settings
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: list = None
    prediction_timeout: int = 30  # Seconds
    
    # UI settings
    default_confidence_threshold: float = 0.8
    history_page_size: int = 20
    metrics_refresh_interval: int = 30  # Seconds
    
    # Security settings
    cors_origins: list = None
    api_key_required: bool = False
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.allowed_image_types is None:
            self.allowed_image_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
        
        if self.cors_origins is None:
            self.cors_origins = ["*"]  # In production, specify exact origins
        
        # Set model paths if they exist
        if self.detection_model_path is None:
            # Try multiple possible detection model paths
            detection_paths = [
                MODELS_DIR / "detection" / "charpy_3class" / "charpy_3class_20250729_110009" / "weights" / "best.pt",
                MODELS_DIR / "detection" / "charpy_v9" / "best.pt",
                MODELS_DIR / "detection" / "best.pt"
            ]
            for detection_path in detection_paths:
                if detection_path.exists():
                    self.detection_model_path = detection_path
                    break
        
        if self.classification_model_path is None:
            classification_path = MODELS_DIR / "classification" / "shiny_classifier.pkl"
            if classification_path.exists():
                self.classification_model_path = classification_path
        
        if self.regression_model_path is None:
            regression_path = MODELS_DIR / "shear_prediction" / "regression_model.pkl"
            if regression_path.exists():
                self.regression_model_path = regression_path


def load_config(config_file: Optional[Path] = None) -> WebConfig:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_file: Optional path to JSON config file
        
    Returns:
        WebConfig instance
    """
    config = WebConfig()
    
    # Load from JSON file if provided
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update config with file data
            for key, value in config_data.items():
                if hasattr(config, key):
                    # Convert path strings to Path objects
                    if key.endswith('_path') and isinstance(value, str):
                        value = Path(value)
                    setattr(config, key, value)
                    
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    # Override with environment variables
    env_mappings = {
        'MLMCSC_WEB_HOST': 'host',
        'MLMCSC_WEB_PORT': 'port',
        'MLMCSC_WEB_DEBUG': 'debug',
        'MLMCSC_DATABASE_PATH': 'database_path',
        'MLMCSC_DETECTION_MODEL': 'detection_model_path',
        'MLMCSC_CLASSIFICATION_MODEL': 'classification_model_path',
        'MLMCSC_REGRESSION_MODEL': 'regression_model_path',
        'MLMCSC_ONLINE_LEARNING': 'online_learning_enabled',
        'MLMCSC_UPDATE_THRESHOLD': 'update_threshold',
        'MLMCSC_UPDATE_INTERVAL': 'update_interval',
        'MLMCSC_API_KEY': 'api_key',
    }
    
    for env_var, config_attr in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Type conversion
            if config_attr in ['port', 'update_threshold', 'update_interval']:
                env_value = int(env_value)
            elif config_attr in ['debug', 'online_learning_enabled', 'api_key_required']:
                env_value = env_value.lower() in ('true', '1', 'yes', 'on')
            elif config_attr.endswith('_path'):
                env_value = Path(env_value)
            
            setattr(config, config_attr, env_value)
    
    return config


def save_config(config: WebConfig, config_file: Path) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: WebConfig instance to save
        config_file: Path to save config file
    """
    try:
        config_data = {}
        
        for key, value in config.__dict__.items():
            if isinstance(value, Path):
                config_data[key] = str(value)
            else:
                config_data[key] = value
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
            
    except Exception as e:
        print(f"Error saving config file {config_file}: {e}")


# Default configuration instance
DEFAULT_CONFIG = WebConfig()

# Configuration file path
CONFIG_FILE = WEB_ROOT / "config.json"

# Load configuration
def get_config() -> WebConfig:
    """Get the current configuration."""
    return load_config(CONFIG_FILE)


# Model configuration
MODEL_CONFIG = {
    'detection': {
        'confidence_threshold': 0.8,
        'nms_threshold': 0.4,
        'max_detections': 10,
        'device': 'auto'
    },
    'classification': {
        'model_type': 'shiny_region',
        'confidence_threshold': 0.7
    },
    'regression': {
        'model_type': 'random_forest',
        'scaler_type': 'standard'
    }
}

# Database configuration
DATABASE_CONFIG = {
    'cleanup_interval_days': 90,  # Clean up old predictions after 90 days
    'backup_interval_hours': 24,  # Backup database every 24 hours
    'max_image_storage_mb': 1000  # Maximum storage for images in MB
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(DATA_DIR / 'web_interface.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}