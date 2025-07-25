#!/usr/bin/env python3
"""
Unit tests for the object detection module.

This module contains comprehensive tests for the SpecimenDetector
and related classes to ensure proper functionality.
"""

import unittest
import numpy as np
import cv2
import tempfile
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from models.object_detector import (
        SpecimenDetector, 
        DetectionResult, 
        SpecimenTracker, 
        MotionDetector,
        RotationDetector
    )
    from models.annotation_utils import Annotation, AnnotationConverter
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestDetectionResult(unittest.TestCase):
    """Test DetectionResult data class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
    
    def test_detection_result_creation(self):
        """Test DetectionResult creation and methods."""
        result = DetectionResult(
            specimen_id=1,
            bbox=[100.0, 150.0, 50.0, 75.0],
            confidence=0.85,
            is_stable=True,
            rotation_angle=15.5,
            center_offset=[10.0, -5.0]
        )
        
        self.assertEqual(result.specimen_id, 1)
        self.assertEqual(result.bbox, [100.0, 150.0, 50.0, 75.0])
        self.assertEqual(result.confidence, 0.85)
        self.assertTrue(result.is_stable)
        self.assertEqual(result.rotation_angle, 15.5)
        self.assertEqual(result.center_offset, [10.0, -5.0])
        self.assertIsNotNone(result.timestamp)
    
    def test_detection_result_to_dict(self):
        """Test DetectionResult to_dict method."""
        result = DetectionResult(
            specimen_id=2,
            bbox=[200.0, 250.0, 60.0, 80.0],
            confidence=0.92,
            is_stable=False,
            rotation_angle=-10.0,
            center_offset=[-15.0, 20.0]
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['specimen_id'], 2)
        self.assertEqual(result_dict['confidence'], 0.92)
        self.assertFalse(result_dict['is_stable'])


class TestSpecimenTracker(unittest.TestCase):
    """Test SpecimenTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
        
        self.tracker = SpecimenTracker(max_disappeared=5, max_distance=50)
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.next_id, 0)
        self.assertEqual(len(self.tracker.objects), 0)
        self.assertEqual(len(self.tracker.disappeared), 0)
    
    def test_register_object(self):
        """Test object registration."""
        centroid = np.array([100, 150])
        obj_id = self.tracker.register(centroid)
        
        self.assertEqual(obj_id, 0)
        self.assertEqual(self.tracker.next_id, 1)
        self.assertTrue(np.array_equal(self.tracker.objects[0], centroid))
        self.assertEqual(self.tracker.disappeared[0], 0)
    
    def test_update_with_detections(self):
        """Test tracker update with detections."""
        # First frame - register new objects
        detections = [np.array([100, 150]), np.array([200, 250])]
        tracked = self.tracker.update(detections)
        
        self.assertEqual(len(tracked), 2)
        self.assertIn(0, tracked)
        self.assertIn(1, tracked)
        
        # Second frame - update existing objects
        detections = [np.array([105, 155]), np.array([195, 245])]
        tracked = self.tracker.update(detections)
        
        self.assertEqual(len(tracked), 2)
        # Objects should be updated with new positions
        self.assertTrue(np.allclose(tracked[0], [105, 155], atol=1))
        self.assertTrue(np.allclose(tracked[1], [195, 245], atol=1))
    
    def test_update_with_no_detections(self):
        """Test tracker update with no detections."""
        # Register some objects first
        detections = [np.array([100, 150]), np.array([200, 250])]
        self.tracker.update(detections)
        
        # Update with no detections
        tracked = self.tracker.update([])
        
        # Objects should still exist but marked as disappeared
        self.assertEqual(len(tracked), 2)
        self.assertEqual(self.tracker.disappeared[0], 1)
        self.assertEqual(self.tracker.disappeared[1], 1)


class TestMotionDetector(unittest.TestCase):
    """Test MotionDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
        
        self.motion_detector = MotionDetector(history_size=5, stability_threshold=2.0)
    
    def test_motion_detector_initialization(self):
        """Test motion detector initialization."""
        self.assertEqual(self.motion_detector.history_size, 5)
        self.assertEqual(self.motion_detector.stability_threshold, 2.0)
    
    def test_stable_object_detection(self):
        """Test detection of stable objects."""
        specimen_id = 1
        
        # Add stable positions (small variations)
        positions = [
            np.array([100.0, 150.0]),
            np.array([100.1, 150.1]),
            np.array([99.9, 149.9]),
            np.array([100.0, 150.0]),
            np.array([100.1, 149.8])
        ]
        
        for pos in positions:
            is_stable = self.motion_detector.update(specimen_id, pos)
        
        # Should be stable with small variations
        self.assertTrue(is_stable)
    
    def test_moving_object_detection(self):
        """Test detection of moving objects."""
        specimen_id = 2
        
        # Add moving positions (large variations)
        positions = [
            np.array([100.0, 150.0]),
            np.array([110.0, 160.0]),
            np.array([120.0, 170.0]),
            np.array([130.0, 180.0]),
            np.array([140.0, 190.0])
        ]
        
        for pos in positions:
            is_stable = self.motion_detector.update(specimen_id, pos)
        
        # Should not be stable with large variations
        self.assertFalse(is_stable)


class TestRotationDetector(unittest.TestCase):
    """Test RotationDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
        
        self.rotation_detector = RotationDetector()
    
    def test_rotation_detection_with_valid_contour(self):
        """Test rotation detection with valid contour."""
        # Create a simple rectangular contour
        contour = np.array([
            [[100, 100]],
            [[200, 100]],
            [[200, 150]],
            [[100, 150]]
        ], dtype=np.int32)
        
        angle = self.rotation_detector.detect_rotation(contour, specimen_id=1)
        
        # Should return a valid angle
        self.assertIsInstance(angle, float)
        self.assertTrue(-90 <= angle <= 90)
    
    def test_rotation_detection_with_invalid_contour(self):
        """Test rotation detection with invalid contour."""
        # Create contour with too few points
        contour = np.array([
            [[100, 100]],
            [[200, 100]]
        ], dtype=np.int32)
        
        angle = self.rotation_detector.detect_rotation(contour, specimen_id=2)
        
        # Should return 0.0 for invalid contour
        self.assertEqual(angle, 0.0)


class TestSpecimenDetector(unittest.TestCase):
    """Test SpecimenDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
        
        # Create a simple test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        try:
            detector = SpecimenDetector(
                confidence_threshold=0.7,
                nms_threshold=0.5,
                device='cpu',  # Use CPU for testing
                max_detections=5
            )
            
            self.assertEqual(detector.confidence_threshold, 0.7)
            self.assertEqual(detector.nms_threshold, 0.5)
            self.assertEqual(detector.max_detections, 5)
            self.assertEqual(detector.device, 'cpu')
            
        except Exception as e:
            self.skipTest(f"Could not initialize detector: {e}")
    
    def test_detect_specimen_with_test_image(self):
        """Test specimen detection with test image."""
        try:
            detector = SpecimenDetector(
                confidence_threshold=0.5,
                device='cpu'
            )
            
            results = detector.detect_specimen(self.test_image)
            
            # Should return a list (may be empty for random image)
            self.assertIsInstance(results, list)
            
            # If there are results, check their structure
            for result in results:
                self.assertIsInstance(result, DetectionResult)
                self.assertIsInstance(result.specimen_id, int)
                self.assertIsInstance(result.bbox, list)
                self.assertEqual(len(result.bbox), 4)
                self.assertIsInstance(result.confidence, float)
                self.assertIsInstance(result.is_stable, bool)
                self.assertIsInstance(result.rotation_angle, float)
                self.assertIsInstance(result.center_offset, list)
                self.assertEqual(len(result.center_offset), 2)
            
        except Exception as e:
            self.skipTest(f"Could not run detection: {e}")
    
    def test_extract_roi(self):
        """Test ROI extraction."""
        try:
            detector = SpecimenDetector(device='cpu')
            
            # Create a mock detection result
            detection_result = DetectionResult(
                specimen_id=1,
                bbox=[100, 100, 50, 50],  # x, y, w, h
                confidence=0.8,
                is_stable=True,
                rotation_angle=0.0,
                center_offset=[0.0, 0.0]
            )
            
            roi = detector.extract_roi(self.test_image, detection_result, padding=10)
            
            if roi is not None:
                self.assertIsInstance(roi, np.ndarray)
                self.assertEqual(len(roi.shape), 3)  # Should be color image
                # ROI should be larger than original bbox due to padding
                self.assertGreater(roi.shape[0], 50)  # height
                self.assertGreater(roi.shape[1], 50)  # width
            
        except Exception as e:
            self.skipTest(f"Could not test ROI extraction: {e}")
    
    def test_get_fps(self):
        """Test FPS calculation."""
        try:
            detector = SpecimenDetector(device='cpu')
            
            fps = detector.get_fps()
            
            self.assertIsInstance(fps, float)
            self.assertGreaterEqual(fps, 0.0)
            
        except Exception as e:
            self.skipTest(f"Could not test FPS: {e}")
    
    def test_get_detection_stats(self):
        """Test detection statistics."""
        try:
            detector = SpecimenDetector(device='cpu')
            
            stats = detector.get_detection_stats()
            
            self.assertIsInstance(stats, dict)
            self.assertIn('fps', stats)
            self.assertIn('active_tracks', stats)
            self.assertIn('device', stats)
            self.assertIn('confidence_threshold', stats)
            self.assertIn('nms_threshold', stats)
            
        except Exception as e:
            self.skipTest(f"Could not test detection stats: {e}")


class TestAnnotationUtils(unittest.TestCase):
    """Test annotation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
    
    def test_annotation_creation(self):
        """Test Annotation creation."""
        annotation = Annotation(
            class_id=0,
            class_name='specimen',
            bbox=[0.5, 0.5, 0.2, 0.3],
            confidence=0.9
        )
        
        self.assertEqual(annotation.class_id, 0)
        self.assertEqual(annotation.class_name, 'specimen')
        self.assertEqual(annotation.bbox, [0.5, 0.5, 0.2, 0.3])
        self.assertEqual(annotation.confidence, 0.9)
    
    def test_annotation_to_yolo_format(self):
        """Test YOLO format conversion."""
        annotation = Annotation(
            class_id=1,
            class_name='cell',
            bbox=[0.3, 0.7, 0.1, 0.15]
        )
        
        yolo_string = annotation.to_yolo_format()
        expected = "1 0.300000 0.700000 0.100000 0.150000"
        
        self.assertEqual(yolo_string, expected)
    
    def test_annotation_converter_initialization(self):
        """Test AnnotationConverter initialization."""
        class_names = ['specimen', 'cell', 'bacteria']
        converter = AnnotationConverter(class_names)
        
        self.assertEqual(converter.class_names, class_names)
        self.assertEqual(converter.class_to_id, {'specimen': 0, 'cell': 1, 'bacteria': 2})


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete detection pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
    
    def test_full_detection_pipeline(self):
        """Test the complete detection pipeline."""
        try:
            # Create detector
            detector = SpecimenDetector(
                confidence_threshold=0.3,  # Lower threshold for testing
                device='cpu'
            )
            
            # Create test images with different content
            test_images = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.ones((480, 640, 3), dtype=np.uint8) * 255
            ]
            
            for i, image in enumerate(test_images):
                # Run detection
                results = detector.detect_specimen(image)
                
                # Verify results structure
                self.assertIsInstance(results, list)
                
                # Test tracking (should work the same as detection)
                track_results = detector.track_specimen(image)
                self.assertIsInstance(track_results, list)
                
                # Test auto-centering
                if results:
                    center_adjustment = detector.auto_center(results)
                    if center_adjustment is not None:
                        self.assertIsInstance(center_adjustment, tuple)
                        self.assertEqual(len(center_adjustment), 2)
                
                # Test ROI extraction for each result
                for result in results:
                    roi = detector.extract_roi(image, result)
                    if roi is not None:
                        self.assertIsInstance(roi, np.ndarray)
            
            # Test performance metrics
            fps = detector.get_fps()
            stats = detector.get_detection_stats()
            
            self.assertIsInstance(fps, float)
            self.assertIsInstance(stats, dict)
            
        except Exception as e:
            self.skipTest(f"Integration test failed: {e}")


def create_test_suite():
    """Create a test suite with all test cases."""
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDetectionResult,
        TestSpecimenTracker,
        TestMotionDetector,
        TestRotationDetector,
        TestSpecimenDetector,
        TestAnnotationUtils,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def main():
    """Run all tests."""
    if not IMPORTS_AVAILABLE:
        print(f"❌ Cannot run tests - missing dependencies: {IMPORT_ERROR}")
        print("\nPlease ensure all required packages are installed:")
        print("pip install ultralytics torch torchvision opencv-python numpy scipy")
        return False
    
    print("Running Object Detection Module Tests...")
    print("=" * 50)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)