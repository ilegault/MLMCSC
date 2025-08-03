#!/usr/bin/env python3
"""
Test script to isolate import issues
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

print("Testing individual imports...")

print("1. Testing cv2...")
try:
    import cv2
    print("✅ cv2 imported successfully")
except Exception as e:
    print(f"❌ cv2 failed: {e}")

print("2. Testing numpy...")
try:
    import numpy as np
    print("✅ numpy imported successfully")
except Exception as e:
    print(f"❌ numpy failed: {e}")

print("3. Testing pandas...")
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except Exception as e:
    print(f"❌ pandas failed: {e}")

print("4. Testing scipy...")
try:
    from scipy import stats
    print("✅ scipy imported successfully")
except Exception as e:
    print(f"❌ scipy failed: {e}")

print("5. Testing sklearn...")
try:
    from sklearn.linear_model import RANSACRegressor
    print("✅ sklearn imported successfully")
except Exception as e:
    print(f"❌ sklearn failed: {e}")

print("6. Testing matplotlib...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
except Exception as e:
    print(f"❌ matplotlib failed: {e}")

print("7. Testing postprocessing.charpy_lateral_expansion...")
try:
    from postprocessing.charpy_lateral_expansion import CharpyLateralExpansionMeasurer, CharpyMeasurement
    print("✅ charpy_lateral_expansion imported successfully")
except Exception as e:
    print(f"❌ charpy_lateral_expansion failed: {e}")
    import traceback
    traceback.print_exc()

print("8. Testing postprocessing.charpy_measurement_utils...")
try:
    from postprocessing.charpy_measurement_utils import CalibrationUtility, CharpyDataAnalyzer
    print("✅ charpy_measurement_utils imported successfully")
except Exception as e:
    print(f"❌ charpy_measurement_utils failed: {e}")
    import traceback
    traceback.print_exc()

print("All import tests completed!")