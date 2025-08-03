#!/usr/bin/env python3
"""
Test Script for Initial Model Training

This script tests the initial model training pipeline with a small subset
of your manually labeled data to ensure everything works correctly.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.mlmcsc.feature_extraction import FractureFeatureExtractor
from src.mlmcsc.regression.online_learning import OnlineLearningSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_yolo_model():
    """Test YOLO model loading and inference."""
    logger.info("Testing YOLO model...")
    
    try:
        model_path = "src/models/detection/charpy_3class/charpy_3class_20250729_110009/weights/best.pt"
        model = YOLO(model_path)
        logger.info("✓ YOLO model loaded successfully")
        
        # Test with a sample image if available
        sample_dir = Path("src/database/samples/shiny_training_data/50_percent")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg"))
            if sample_images:
                test_image = cv2.imread(str(sample_images[0]))
                if test_image is not None:
                    results = model(test_image, verbose=False)
                    logger.info(f"✓ YOLO inference successful on test image")
                    
                    # Check for fracture surface detections
                    fracture_detections = 0
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                if int(box.cls) == 2:  # fracture_surface class
                                    fracture_detections += 1
                    
                    logger.info(f"✓ Found {fracture_detections} fracture surface detections")
                else:
                    logger.warning("Could not load test image")
            else:
                logger.warning("No test images found")
        else:
            logger.warning("Sample directory not found")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ YOLO model test failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction pipeline."""
    logger.info("Testing feature extraction...")
    
    try:
        # Initialize feature extractor
        extractor = FractureFeatureExtractor()
        logger.info("✓ Feature extractor initialized")
        
        # Create a dummy image for testing
        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        test_bbox = [100, 100, 200, 150]  # [x, y, width, height]
        
        # Extract features
        result = extractor.extract_features(
            image=test_image,
            specimen_id=1,
            bbox=test_bbox
        )
        
        logger.info(f"✓ Feature extraction successful")
        logger.info(f"✓ Feature vector length: {len(result.feature_vector)}")
        logger.info(f"✓ Feature names count: {len(result.features)}")
        
        # Test with real image if available
        sample_dir = Path("src/database/samples/shiny_training_data/50_percent")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg"))
            if sample_images:
                real_image = cv2.imread(str(sample_images[0]))
                if real_image is not None:
                    h, w = real_image.shape[:2]
                    real_bbox = [w//4, h//4, w//2, h//2]
                    
                    real_result = extractor.extract_features(
                        image=real_image,
                        specimen_id=2,
                        bbox=real_bbox
                    )
                    
                    logger.info(f"✓ Real image feature extraction successful")
                    logger.info(f"✓ Real image feature vector length: {len(real_result.feature_vector)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature extraction test failed: {e}")
        return False


def test_online_learning():
    """Test online learning system."""
    logger.info("Testing online learning system...")
    
    try:
        # Initialize online learner
        learner = OnlineLearningSystem(
            model_type='sgd',
            target_property='shear_percentage'
        )
        logger.info("✓ Online learning system initialized")
        
        # Create dummy training data
        n_samples = 20
        n_features = 50
        
        dummy_features = []
        dummy_targets = []
        
        for i in range(n_samples):
            # Create dummy feature vector
            feature_vector = np.random.randn(n_features)
            feature_names = [f'feature_{j}' for j in range(n_features)]
            
            dummy_features.append({
                'feature_vector': feature_vector,
                'feature_names': feature_names,
                'specimen_id': i
            })
            
            # Create dummy target (shear percentage)
            dummy_targets.append(np.random.uniform(10, 100))
        
        # Initialize model
        performance = learner.initialize_model(
            feature_data=dummy_features,
            target_values=dummy_targets
        )
        
        logger.info(f"✓ Model initialization successful")
        logger.info(f"✓ Initial performance: R² = {performance.get('r2', 0):.3f}")
        
        # Test prediction
        test_features = [dummy_features[0]]  # Use first sample for prediction
        prediction = learner.predict(test_features)
        
        logger.info(f"✓ Prediction successful: {prediction:.2f}%")
        
        # Test model update
        new_features = dummy_features[15:20]  # Use last 5 samples for update
        new_targets = dummy_targets[15:20]
        
        update_result = learner.update_model(new_features, new_targets)
        
        logger.info(f"✓ Model update successful")
        logger.info(f"✓ Added {update_result.samples_added} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Online learning test failed: {e}")
        return False


def test_data_loading():
    """Test loading of manually labeled training data."""
    logger.info("Testing training data loading...")
    
    try:
        training_data_path = Path("src/database/samples/shiny_training_data")
        
        if not training_data_path.exists():
            logger.error("✗ Training data directory not found")
            return False
        
        # Check directory structure
        shear_dirs = [d for d in training_data_path.iterdir() if d.is_dir()]
        logger.info(f"✓ Found {len(shear_dirs)} shear percentage directories")
        
        total_samples = 0
        for shear_dir in shear_dirs:
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(shear_dir.glob(ext))
            
            shear_name = shear_dir.name
            logger.info(f"✓ {shear_name}: {len(image_files)} images")
            total_samples += len(image_files)
        
        logger.info(f"✓ Total training samples: {total_samples}")
        
        if total_samples < 50:
            logger.warning("⚠ Less than 50 training samples found - consider adding more")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loading test failed: {e}")
        return False


def test_online_learning_integration():
    """Test integration between feature extraction and online learning."""
    logger.info("Testing online learning integration...")
    
    try:
        # Initialize components
        extractor = FractureFeatureExtractor()
        learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='immediate'
        )
        
        logger.info("✓ Components initialized")
        
        # Test with real image if available
        sample_dir = Path("src/database/samples/shiny_training_data/50_percent")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg"))
            if len(sample_images) >= 2:
                # Extract features from multiple images
                feature_data = []
                target_values = []
                
                for i, img_path in enumerate(sample_images[:5]):  # Use first 5 images
                    try:
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            h, w = image.shape[:2]
                            bbox = [w//4, h//4, w//2, h//2]
                            
                            result = extractor.extract_features(
                                image=image,
                                specimen_id=i + 1,
                                bbox=bbox
                            )
                            
                            feature_data.append({
                                'feature_vector': result.feature_vector,
                                'feature_names': list(result.features.keys()),
                                'specimen_id': i + 1
                            })
                            target_values.append(50.0)  # Assume 50% shear for this test
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_path}: {e}")
                        continue
                
                if len(feature_data) >= 3:
                    # Initialize online learner with first 3 samples
                    performance = learner.initialize_model(
                        feature_data[:3], 
                        target_values[:3]
                    )
                    logger.info(f"✓ Online learner initialized with R²: {performance.get('r2', 0):.3f}")
                    
                    # Test online learning pipeline with remaining samples
                    if len(feature_data) > 3:
                        result = learner.process_technician_submission(
                            feature_data=feature_data[3],
                            label=target_values[3],
                            timestamp="2024-01-01T12:00:00",
                            technician_id="test_tech",
                            confidence=0.8
                        )
                        
                        logger.info(f"✓ Online learning pipeline test successful")
                        logger.info(f"✓ Update applied: {result.get('update_applied', False)}")
                        
                        return True
                else:
                    logger.warning("Not enough valid features extracted for integration test")
            else:
                logger.warning("Not enough test images found for integration test")
        
        # Fallback to dummy data test
        logger.info("Using dummy data for integration test")
        
        # Create dummy feature data
        dummy_features = []
        dummy_targets = []
        
        for i in range(5):
            feature_vector = np.random.randn(50)
            dummy_features.append({
                'feature_vector': feature_vector,
                'feature_names': [f'feature_{j}' for j in range(50)],
                'specimen_id': i + 1
            })
            dummy_targets.append(np.random.uniform(10, 90))
        
        # Initialize and test
        performance = learner.initialize_model(dummy_features[:3], dummy_targets[:3])
        logger.info(f"✓ Online learner initialized with dummy data, R²: {performance.get('r2', 0):.3f}")
        
        # Test pipeline
        result = learner.process_technician_submission(
            feature_data=dummy_features[3],
            label=dummy_targets[3],
            timestamp="2024-01-01T12:00:00",
            technician_id="test_tech",
            confidence=0.8
        )
        
        logger.info(f"✓ Integration test successful with dummy data")
        logger.info(f"✓ Update applied: {result.get('update_applied', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Online learning integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RUNNING INITIAL TRAINING SYSTEM TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("YOLO Model", test_yolo_model),
        ("Feature Extraction", test_feature_extraction),
        ("Online Learning", test_online_learning),
        ("Online Learning Integration", test_online_learning_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! You can proceed with full training.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python src/mlmcsc/utils/train_initial_shear_model.py")
        logger.info("2. Check the results in src/models/shear_prediction/")
        logger.info("3. Use the trained model for online learning")
    else:
        logger.error("❌ Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()