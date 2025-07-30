"""
Path management for MLMCSC system.

This module provides centralized path management and ensures consistent
file organization across the system.
"""

from pathlib import Path
from typing import Optional, Union
import os

class MLMCSCPaths:
    """Centralized path management for MLMCSC system."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """Initialize path manager."""
        if project_root is None:
            # Auto-detect project root
            self.project_root = Path(__file__).parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        # Ensure project root exists
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")
    
    # Core directories
    @property
    def src(self) -> Path:
        """Source code directory."""
        return self.project_root / "src"
    
    @property
    def mlmcsc(self) -> Path:
        """Main package directory."""
        return self.src / "mlmcsc"
    
    @property
    def apps(self) -> Path:
        """Applications directory."""
        return self.src / "apps"
    
    # Data directories
    @property
    def data(self) -> Path:
        """Data root directory."""
        return self.project_root / "data"
    
    @property
    def data_raw(self) -> Path:
        """Raw data directory."""
        return self.data / "raw"
    
    @property
    def data_processed(self) -> Path:
        """Processed data directory."""
        return self.data / "processed"
    
    @property
    def data_annotations(self) -> Path:
        """Annotations directory."""
        return self.data / "annotations"
    
    @property
    def data_samples(self) -> Path:
        """Sample data directory."""
        return self.data / "samples"
    
    # Model directories
    @property
    def models(self) -> Path:
        """Models root directory."""
        return self.project_root / "models"
    
    @property
    def models_detection(self) -> Path:
        """Detection models directory."""
        return self.models / "detection"
    
    @property
    def models_classification(self) -> Path:
        """Classification models directory."""
        return self.models / "classification"
    
    @property
    def models_archived(self) -> Path:
        """Archived models directory."""
        return self.models / "archived"
    
    # Experiment directories
    @property
    def experiments(self) -> Path:
        """Experiments root directory."""
        return self.project_root / "experiments"
    
    @property
    def experiments_detection(self) -> Path:
        """Detection experiments directory."""
        return self.experiments / "detection"
    
    @property
    def experiments_classification(self) -> Path:
        """Classification experiments directory."""
        return self.experiments / "classification"
    
    @property
    def experiments_notebooks(self) -> Path:
        """Notebooks directory."""
        return self.experiments / "examples"
    
    @property
    def experiments_configs(self) -> Path:
        """Experiment configurations directory."""
        return self.experiments / "configs"
    
    # Results and output directories
    @property
    def results(self) -> Path:
        """Results directory."""
        return self.project_root / "results"
    
    @property
    def tools(self) -> Path:
        """Tools directory."""
        return self.project_root / "tools"
    
    @property
    def tests(self) -> Path:
        """Tests directory."""
        return self.project_root / "tests"
    
    @property
    def docs(self) -> Path:
        """Documentation directory."""
        return self.project_root / "docs"
    
    # Configuration directories
    @property
    def config(self) -> Path:
        """Configuration directory."""
        return self.project_root / "config"
    
    # Camera-specific paths
    @property
    def camera_data(self) -> Path:
        """Camera data directory."""
        return self.mlmcsc / "camera" / "data"
    
    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get path for a specific model."""
        if model_type == "detection":
            return self.models_detection / model_name
        elif model_type == "classification":
            return self.models_classification / model_name
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_data_path(self, data_type: str, dataset_name: str) -> Path:
        """Get path for a specific dataset."""
        if data_type == "raw":
            return self.data_raw / dataset_name
        elif data_type == "processed":
            return self.data_processed / dataset_name
        elif data_type == "annotations":
            return self.data_annotations / dataset_name
        elif data_type == "samples":
            return self.data_samples / dataset_name
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def ensure_directories(self):
        """Create all necessary directories."""
        directories = [
            self.src, self.mlmcsc, self.apps,
            self.data, self.data_raw, self.data_processed, 
            self.data_annotations, self.data_samples,
            self.models, self.models_detection, self.models_classification, self.models_archived,
            self.experiments, self.experiments_detection, self.experiments_classification,
            self.experiments_notebooks, self.experiments_configs,
            self.results, self.tools, self.tests, self.docs, self.config
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation."""
        return f"MLMCSCPaths(project_root={self.project_root})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()

# Global path manager instance
paths = MLMCSCPaths()

# Convenience functions
def get_project_root() -> Path:
    """Get project root directory."""
    return paths.project_root

def get_model_path(model_type: str, model_name: str) -> Path:
    """Get path for a specific model."""
    return paths.get_model_path(model_type, model_name)

def get_data_path(data_type: str, dataset_name: str) -> Path:
    """Get path for a specific dataset."""
    return paths.get_data_path(data_type, dataset_name)

def ensure_project_structure():
    """Ensure all project directories exist."""
    paths.ensure_directories()