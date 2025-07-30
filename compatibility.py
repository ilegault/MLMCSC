"""
Backward compatibility layer for MLMCSC restructuring.

This module provides import compatibility for existing scripts and maintains
the same functionality while using the new organized structure.
"""

import sys
import warnings
from pathlib import Path

# Add new src structure to Python path
current_dir = Path(__file__).parent
new_src_dir = current_dir / "src"
sys.path.insert(0, str(new_src_dir))

# Compatibility imports - maintain old import paths
try:
    # Old postprocessing imports
    from src.mlmcsc.classification.improved_fracture_detector import ImprovedFractureSurfaceDetector
    from src.mlmcsc.classification.shiny_region_analyzer import ShinyRegionAnalyzer, ShinyRegionFeatures
    from src.mlmcsc.classification.shiny_region_classifier import ShinyRegionBasedClassifier
    
    # Old models imports
    from src.mlmcsc.detection.object_detector import *
    from src.mlmcsc.detection.annotation_utils import *
    
    # Camera imports
    from src.mlmcsc.camera.microscope_interface import *
    from src.mlmcsc.camera.calibrated_measurement import *
    
    # Preprocessing imports
    from src.mlmcsc.preprocessing.image_processor import *
    
    # Utils imports
    from src.mlmcsc.utils import *
    
except ImportError as e:
    warnings.warn(f"Some compatibility imports failed: {e}")

# Compatibility functions for old script paths
def get_old_postprocessing_path():
    """Get old postprocessing path for compatibility."""
    warnings.warn("Using old postprocessing path. Please update to use new structure.", DeprecationWarning)
    return current_dir / "src" / "mlmcsc" / "classification"

def get_old_models_path():
    """Get old models path for compatibility."""
    warnings.warn("Using old models path. Please update to use new structure.", DeprecationWarning)
    return current_dir / "src" / "mlmcsc" / "detection"

# Legacy path mappings
LEGACY_PATHS = {
    'src/postprocessing': 'src/mlmcsc/classification',
    'src/models': 'src/mlmcsc/detection',
    'src/camera': 'src/mlmcsc/camera',
    'src/preprocessing': 'src/mlmcsc/preprocessing',
    'src/utils': 'src/mlmcsc/utils',
    'scripts': 'tools',
    'examples': 'experiments/examples',
    'models/charpy_*': 'models/detection',
}

def show_migration_guide():
    """Show migration guide for updating existing scripts."""
    print("=" * 60)
    print("MLMCSC RESTRUCTURING - MIGRATION GUIDE")
    print("=" * 60)
    print()
    print("Your MLMCSC directory has been restructured for better organization.")
    print("Here's how to update your existing scripts:")
    print()
    print("OLD IMPORTS → NEW IMPORTS")
    print("-" * 40)
    print("from postprocessing.* → from mlmcsc.classification.*")
    print("from models.* → from mlmcsc.detection.*")
    print("from camera.* → from mlmcsc.camera.*")
    print("from preprocessing.* → from mlmcsc.preprocessing.*")
    print()
    print("PATH CHANGES")
    print("-" * 40)
    for old_path, new_path in LEGACY_PATHS.items():
        print(f"{old_path} → {new_path}")
    print()
    print("APPLICATIONS")
    print("-" * 40)
    print("Training: src/apps/trainer/")
    print("Analysis: src/apps/analyzer/")
    print()
    print("For full compatibility, use: import compatibility")
    print("=" * 60)

if __name__ == "__main__":
    show_migration_guide()