#!/usr/bin/env python3
"""
Quick script to check if model files exist at expected paths.
"""

import os
from pathlib import Path

def get_project_root():
    """Get the project root directory (parent of tools directory)."""
    return Path(__file__).parent.parent

def check_path(relative_path: str, description: str):
    """Check if a path exists and print result."""
    project_root = get_project_root()
    full_path = project_root / relative_path
    
    print(f"\n{description}:")
    print(f"  Looking for: {relative_path}")
    print(f"  Full path: {full_path}")
    print(f"  Exists: {'✅ YES' if full_path.exists() else '❌ NO'}")
    
    if not full_path.exists():
        # Try to find similar paths
        parent_dir = full_path.parent
        if parent_dir.exists():
            print(f"  Parent directory exists, contents:")
            try:
                for item in parent_dir.iterdir():
                    print(f"    - {item.name}")
            except:
                print(f"    - Could not list contents")
        else:
            print(f"  Parent directory does not exist: {parent_dir}")
    
    return full_path.exists()

def main():
    print("🔍 CHECKING MODEL FILE PATHS")
    print("=" * 50)
    
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Check the paths from config
    paths_to_check = [
        ("models/classification/charpy_shear_regressor.pkl", "Classification Model (Config)"),
        ("models/detection/charpy_3class/charpy_3class_20250729_110009/weights/best.pt", "YOLO Model (Config)"),
        
        # Check alternative paths based on user's description
        ("models/charpy_3class/charpy_3class_data_time/weights/best.pt", "YOLO Model (User Description)"),
        ("models/detection/charpy_3class", "YOLO Detection Directory"),
        ("models/charpy_3class", "Alternative YOLO Directory"),
    ]
    
    results = {}
    for path, desc in paths_to_check:
        results[desc] = check_path(path, desc)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    for desc, exists in results.items():
        status = "✅ FOUND" if exists else "❌ MISSING"
        print(f"  {status}: {desc}")
    
    # Try to find any .pt files in the models directory
    print(f"\n🔍 SEARCHING FOR .pt FILES:")
    models_dir = project_root / "models"
    if models_dir.exists():
        pt_files = list(models_dir.rglob("*.pt"))
        if pt_files:
            print("  Found .pt files:")
            for pt_file in pt_files:
                rel_path = pt_file.relative_to(project_root)
                print(f"    - {rel_path}")
        else:
            print("  No .pt files found (might be hidden by .gitignore)")
    
    # Try to find any .pkl files
    print(f"\n🔍 SEARCHING FOR .pkl FILES:")
    if models_dir.exists():
        pkl_files = list(models_dir.rglob("*.pkl"))
        if pkl_files:
            print("  Found .pkl files:")
            for pkl_file in pkl_files:
                rel_path = pkl_file.relative_to(project_root)
                print(f"    - {rel_path}")
        else:
            print("  No .pkl files found (might be hidden by .gitignore)")

if __name__ == "__main__":
    main()