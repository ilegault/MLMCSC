#!/usr/bin/env python3
"""
Test script for Charpy Lateral Expansion Measurement

This script demonstrates and tests the Charpy lateral expansion measurement
functionality with synthetic data and provides examples of how to integrate
with real YOLO detection results.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt

from charpy_lateral_expansion import CharpyLateralExpansionMeasurer, CharpyMeasurement

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_charpy_image(width: int = 400, height: int = 300) -> np.ndarray:
    """
    Create a synthetic Charpy specimen image for testing.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Synthetic image with a Charpy specimen shape
    """
    # Create blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw specimen outline (trapezoid shape to simulate lateral expansion)
    # Bottom edge (wider)
    bottom_left = (50, height - 50)
    bottom_right = (width - 50, height - 50)
    
    # Top edge (narrower due to fracture)
    top_left = (80, 50)
    top_right = (width - 80, 50)
    
    # Draw specimen shape
    pts = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], (100, 100, 100))
    
    # Add some texture
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Draw fracture line at top
    cv2.line(image, top_left, top_right, (200, 200, 200), 3)
    
    return image


def create_synthetic_yolo_detections(image_shape: tuple) -> List[Dict[str, Any]]:
    """
    Create synthetic YOLO detection results for testing.
    
    Args:
        image_shape: Shape of the image (height, width, channels)
        
    Returns:
        List of synthetic YOLO detections
    """
    height, width = image_shape[:2]
    
    # Define corner positions (matching the synthetic image)
    corners = [
        (50, height - 50),   # Bottom left
        (width - 50, height - 50),  # Bottom right
        (80, 50),            # Top left
        (width - 80, 50)     # Top right
    ]
    
    detections = []
    for i, (x, y) in enumerate(corners):
        detection = {
            'class': 'charpy_corner',
            'x': float(x),
            'y': float(y),
            'width': 10.0,
            'height': 10.0,
            'confidence': 0.9 + i * 0.02  # Varying confidence scores
        }
        detections.append(detection)
    
    return detections


def test_basic_measurement():
    """Test basic measurement functionality with synthetic data."""
    logger.info("Testing basic measurement functionality...")
    
    # Create synthetic image and detections
    image = create_synthetic_charpy_image()
    detections = create_synthetic_yolo_detections(image.shape)
    
    # Save synthetic image for testing
    test_image_path = "test_charpy_specimen.jpg"
    cv2.imwrite(test_image_path, image)
    
    # Initialize measurer
    measurer = CharpyLateralExpansionMeasurer(
        calibration_factor=0.1,  # 0.1 mm per pixel
        visualization_enabled=True
    )
    
    try:
        # Process the specimen
        measurements, result_image = measurer.process_charpy_specimen(
            image_path=test_image_path,
            yolo_detections=detections,
            output_path="test_charpy_measurements.jpg"
        )
        
        # Print results
        print("\n=== Measurement Results ===")
        print(f"Left expansion: {measurements.left_expansion_pixels:.2f}px ({measurements.left_expansion_mm:.2f}mm)")
        print(f"Right expansion: {measurements.right_expansion_pixels:.2f}px ({measurements.right_expansion_mm:.2f}mm)")
        print(f"Total expansion: {measurements.total_expansion_pixels:.2f}px ({measurements.total_expansion_mm:.2f}mm)")
        print(f"Calibration factor: {measurements.calibration_factor} mm/px")
        print(f"Number of corners detected: {len(measurements.corners)}")
        
        # Validate results
        assert measurements.total_expansion_pixels > 0, "Total expansion should be positive"
        assert len(measurements.corners) == 4, "Should detect exactly 4 corners"
        assert measurements.total_expansion_mm is not None, "MM measurement should be available"
        
        logger.info("✓ Basic measurement test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic measurement test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing functionality."""
    logger.info("Testing batch processing functionality...")
    
    try:
        # Create multiple synthetic images
        image_paths = []
        detection_results = []
        
        for i in range(3):
            # Create slightly different images
            image = create_synthetic_charpy_image(width=400 + i*20, height=300 + i*10)
            detections = create_synthetic_yolo_detections(image.shape)
            
            # Add some variation to corner positions
            for detection in detections:
                detection['x'] += np.random.normal(0, 2)
                detection['y'] += np.random.normal(0, 2)
            
            image_path = f"test_specimen_{i}.jpg"
            cv2.imwrite(image_path, image)
            
            image_paths.append(image_path)
            detection_results.append(detections)
        
        # Initialize measurer
        measurer = CharpyLateralExpansionMeasurer(calibration_factor=0.1)
        
        # Process batch
        measurements = measurer.batch_process(
            image_paths=image_paths,
            detection_results=detection_results,
            output_dir="batch_output"
        )
        
        # Validate results
        assert len(measurements) == 3, "Should process all 3 specimens"
        
        for i, measurement in enumerate(measurements):
            print(f"\nSpecimen {i+1}:")
            print(f"  Total expansion: {measurement.total_expansion_pixels:.2f}px ({measurement.total_expansion_mm:.2f}mm)")
            assert measurement.total_expansion_pixels > 0, f"Specimen {i+1} should have positive expansion"
        
        # Save results to JSON
        measurer.save_measurements_to_json(measurements, "batch_measurements.json")
        
        logger.info("✓ Batch processing test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Batch processing test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing edge cases and error handling...")
    
    measurer = CharpyLateralExpansionMeasurer()
    
    # Test 1: Insufficient corners
    try:
        insufficient_detections = [
            {'class': 'charpy_corner', 'x': 100, 'y': 50, 'confidence': 0.9},
            {'class': 'charpy_corner', 'x': 200, 'y': 50, 'confidence': 0.9}
        ]
        
        image = create_synthetic_charpy_image()
        cv2.imwrite("test_insufficient.jpg", image)
        
        measurements, _ = measurer.process_charpy_specimen(
            "test_insufficient.jpg", 
            insufficient_detections
        )
        
        logger.error("✗ Should have failed with insufficient corners")
        return False
        
    except ValueError as e:
        logger.info("✓ Correctly handled insufficient corners")
    
    # Test 2: Low confidence detections
    try:
        low_confidence_detections = [
            {'class': 'charpy_corner', 'x': 50, 'y': 50, 'confidence': 0.3},
            {'class': 'charpy_corner', 'x': 200, 'y': 50, 'confidence': 0.3},
            {'class': 'charpy_corner', 'x': 50, 'y': 200, 'confidence': 0.3},
            {'class': 'charpy_corner', 'x': 200, 'y': 200, 'confidence': 0.3}
        ]
        
        corners, confidences = measurer.parse_yolo_output(low_confidence_detections)
        assert len(corners) == 0, "Should filter out low confidence detections"
        logger.info("✓ Correctly filtered low confidence detections")
        
    except Exception as e:
        logger.error(f"✗ Low confidence test failed: {e}")
        return False
    
    # Test 3: Invalid image path
    try:
        valid_detections = create_synthetic_yolo_detections((300, 400, 3))
        measurements, _ = measurer.process_charpy_specimen(
            "nonexistent_image.jpg", 
            valid_detections
        )
        
        logger.error("✗ Should have failed with invalid image path")
        return False
        
    except ValueError as e:
        logger.info("✓ Correctly handled invalid image path")
    
    logger.info("✓ Edge cases test passed")
    return True


def test_calibration_scenarios():
    """Test different calibration scenarios."""
    logger.info("Testing calibration scenarios...")
    
    # Create test image and detections
    image = create_synthetic_charpy_image()
    detections = create_synthetic_yolo_detections(image.shape)
    cv2.imwrite("test_calibration.jpg", image)
    
    # Test 1: No calibration
    measurer_no_cal = CharpyLateralExpansionMeasurer(calibration_factor=None)
    measurements_no_cal, _ = measurer_no_cal.process_charpy_specimen(
        "test_calibration.jpg", detections
    )
    
    assert measurements_no_cal.total_expansion_mm is None, "Should have no mm measurement without calibration"
    logger.info("✓ No calibration scenario handled correctly")
    
    # Test 2: With calibration
    measurer_with_cal = CharpyLateralExpansionMeasurer(calibration_factor=0.05)
    measurements_with_cal, _ = measurer_with_cal.process_charpy_specimen(
        "test_calibration.jpg", detections
    )
    
    assert measurements_with_cal.total_expansion_mm is not None, "Should have mm measurement with calibration"
    expected_mm = measurements_with_cal.total_expansion_pixels * 0.05
    assert abs(measurements_with_cal.total_expansion_mm - expected_mm) < 0.001, "MM conversion should be accurate"
    logger.info("✓ Calibration scenario handled correctly")
    
    # Test 3: Override calibration in conversion
    manual_mm = measurer_no_cal.pixels_to_mm(100, calibration_factor=0.2)
    assert manual_mm == 20.0, "Manual calibration override should work"
    logger.info("✓ Manual calibration override works")
    
    logger.info("✓ Calibration scenarios test passed")
    return True


def demonstrate_real_world_integration():
    """Demonstrate how to integrate with real YOLO model output."""
    logger.info("Demonstrating real-world integration...")
    
    # Example of how you would integrate with actual YOLO model
    print("\n=== Real-World Integration Example ===")
    print("""
    # Example integration with actual YOLO model:
    
    from ultralytics import YOLO
    from charpy_lateral_expansion import CharpyLateralExpansionMeasurer
    
    # Load your trained YOLO model
    model = YOLO('path/to/your/charpy_model.pt')
    
    # Initialize the measurer
    measurer = CharpyLateralExpansionMeasurer(
        calibration_factor=0.1,  # Adjust based on your setup
        min_confidence=0.5
    )
    
    # Process an image
    image_path = 'path/to/charpy_specimen.jpg'
    results = model(image_path)
    
    # Convert YOLO results to expected format
    detections = []
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls)] == 'charpy_corner':
                detection = {
                    'class': 'charpy_corner',
                    'x': float(box.xywh[0][0]),  # Center x
                    'y': float(box.xywh[0][1]),  # Center y
                    'width': float(box.xywh[0][2]),
                    'height': float(box.xywh[0][3]),
                    'confidence': float(box.conf)
                }
                detections.append(detection)
    
    # Measure lateral expansion
    measurements, result_image = measurer.process_charpy_specimen(
        image_path=image_path,
        yolo_detections=detections,
        output_path='measurements_output.jpg'
    )
    
    # Use the results
    print(f"Total lateral expansion: {measurements.total_expansion_mm:.2f}mm")
    """)
    
    return True


def run_all_tests():
    """Run all test functions."""
    logger.info("Starting comprehensive test suite for Charpy lateral expansion measurement...")
    
    tests = [
        ("Basic Measurement", test_basic_measurement),
        ("Batch Processing", test_batch_processing),
        ("Edge Cases", test_edge_cases),
        ("Calibration Scenarios", test_calibration_scenarios),
        ("Real-World Integration", demonstrate_real_world_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! The Charpy lateral expansion measurement system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Clean up test files
    import os
    test_files = [
        "test_charpy_specimen.jpg",
        "test_charpy_measurements.jpg",
        "test_specimen_0.jpg",
        "test_specimen_1.jpg", 
        "test_specimen_2.jpg",
        "test_insufficient.jpg",
        "test_calibration.jpg",
        "batch_measurements.json"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Clean up batch output directory
    import shutil
    if os.path.exists("batch_output"):
        shutil.rmtree("batch_output")
    
    exit(0 if success else 1)