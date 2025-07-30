"""
Configuration settings for MLMCSC system.

This module provides centralized configuration management for all system components.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
from dataclasses import dataclass, asdict

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"

@dataclass
class CameraConfig:
    """Camera configuration settings."""
    device_id: int = 1
    resolution: tuple = (1280, 720)
    target_fps: int = 30
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    calibration_file: str = "src/mlmcsc/camera/data/microscope_calibration.json"

@dataclass
class ModelConfig:
    """Model configuration settings."""
    yolo_model_path: str = "models/detection/best.pt"
    classification_model_path: str = "models/classification/charpy_shear_regressor.pkl"
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.5

@dataclass
class DisplayConfig:
    """Display configuration settings."""
    window_scale: float = 0.8
    show_detections: bool = True
    show_predictions: bool = True
    show_confidence: bool = True
    show_fps: bool = True
    colors: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'detection': (0, 255, 0),  # Green
                'prediction': (0, 0, 255),  # Red
                'text': (255, 255, 255),  # White
                'background': (0, 0, 0)  # Black
            }

@dataclass
class SystemConfig:
    """Main system configuration."""
    camera: CameraConfig
    model: ModelConfig
    display: DisplayConfig
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Return default configuration
            return cls.default()
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create configuration from dictionary."""
        camera_data = data.get('camera', {})
        model_data = data.get('model', {})
        display_data = data.get('display', {})
        
        return cls(
            camera=CameraConfig(**camera_data),
            model=ModelConfig(**model_data),
            display=DisplayConfig(**display_data),
            debug=data.get('debug', False),
            log_level=data.get('log_level', 'INFO')
        )
    
    @classmethod
    def default(cls) -> 'SystemConfig':
        """Create default configuration."""
        return cls(
            camera=CameraConfig(),
            model=ModelConfig(),
            display=DisplayConfig()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

# Default paths
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
USER_CONFIG_PATH = PROJECT_ROOT / "config" / "user.yaml"

def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load system configuration."""
    if config_path is None:
        # Try user config first, then default, then create default
        if USER_CONFIG_PATH.exists():
            config_path = USER_CONFIG_PATH
        elif DEFAULT_CONFIG_PATH.exists():
            config_path = DEFAULT_CONFIG_PATH
        else:
            return SystemConfig.default()
    
    return SystemConfig.from_file(config_path)

def save_default_config():
    """Save default configuration file."""
    config = SystemConfig.default()
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.save(DEFAULT_CONFIG_PATH)
    return DEFAULT_CONFIG_PATH