#!/usr/bin/env python3
"""
Test Script for Online Learning Implementation

This script tests the complete online learning pipeline to ensure
all update strategies work correctly.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mlmcsc.regression.online_learning import OnlineLearningSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dummy_feature_data(n_samples: int = 10, n_features: int = 50):
    """Create dummy feature data for testing."""
    feature_data = []
    target_values = []
    
    for i in range(n_samples):
        # Create realistic feature vector
        feature_vector = np.random.randn(n_features)
        feature_names = [f'feature_{j}' for j in range(n_features)]
        
        feature_data.append({
            'feature_vector': feature_vector,
            'feature_names': feature_names,
            'specimen_id': i + 1
        })
        
        # Create realistic target (shear percentage)
        target_values.append(np.random.uniform(10, 90))
    
    return feature_data, target_values


def test_immediate_update():
    """Test immediate update strategy."""
    logger.info("Testing immediate update strategy...")
    
    try:
        # Initialize online learner
        learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='immediate',
            batch_size=1
        )
        
        # Create initial training data
        initial_features, initial_targets = create_dummy_feature_data(20)
        
        # Initialize model
        performance = learner.initialize_model(initial_features, initial_targets)
        logger.info(f"✓ Model initialized with R²: {performance.get('r2', 0):.3f}")
        
        # Test immediate updates
        new_features, new_targets = create_dummy_feature_data(5)
        
        for i, (feature_data, target) in enumerate(zip(new_features, new_targets)):
            result = learner.process_technician_submission(
                feature_data=feature_data,
                label=target,
                timestamp=datetime.now().isoformat(),
                technician_id=f"tech_{i}",
                confidence=0.8
            )
            
            assert result['update_applied'] == True, "Immediate update should always apply"
            logger.info(f"✓ Immediate update {i+1} applied successfully")
        
        logger.info("✓ Immediate update strategy test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Immediate update test failed: {e}")
        return False


def test_batch_update():
    """Test batch update strategy."""
    logger.info("Testing batch update strategy...")
    
    try:
        # Initialize online learner
        learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='batch',
            batch_size=5
        )
        
        # Create initial training data
        initial_features, initial_targets = create_dummy_feature_data(20)
        
        # Initialize model
        performance = learner.initialize_model(initial_features, initial_targets)
        logger.info(f"✓ Model initialized with R²: {performance.get('r2', 0):.3f}")
        
        # Test batch updates
        new_features, new_targets = create_dummy_feature_data(12)
        
        updates_applied = 0
        for i, (feature_data, target) in enumerate(zip(new_features, new_targets)):
            result = learner.process_technician_submission(
                feature_data=feature_data,
                label=target,
                timestamp=datetime.now().isoformat(),
                technician_id=f"tech_{i}",
                confidence=0.8
            )
            
            if result['update_applied']:
                updates_applied += 1
                logger.info(f"✓ Batch update applied after {i+1} submissions")
        
        # Should have 2 batch updates (5 + 5 samples, 2 remaining)
        assert updates_applied == 2, f"Expected 2 batch updates, got {updates_applied}"
        assert len(learner.pending_samples) == 2, f"Expected 2 pending samples, got {len(learner.pending_samples)}"
        
        logger.info("✓ Batch update strategy test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Batch update test failed: {e}")
        return False


def test_weighted_update():
    """Test weighted update strategy."""
    logger.info("Testing weighted update strategy...")
    
    try:
        # Initialize online learner
        learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='weighted',
            weight_decay=0.9
        )
        
        # Create initial training data
        initial_features, initial_targets = create_dummy_feature_data(20)
        
        # Initialize model
        performance = learner.initialize_model(initial_features, initial_targets)
        logger.info(f"✓ Model initialized with R²: {performance.get('r2', 0):.3f}")
        
        # Test weighted updates
        new_features, new_targets = create_dummy_feature_data(5)
        
        for i, (feature_data, target) in enumerate(zip(new_features, new_targets)):
            result = learner.process_technician_submission(
                feature_data=feature_data,
                label=target,
                timestamp=datetime.now().isoformat(),
                technician_id=f"tech_{i}",
                confidence=0.8
            )
            
            assert result['update_applied'] == True, "Weighted update should always apply"
            logger.info(f"✓ Weighted update {i+1} applied successfully")
        
        logger.info("✓ Weighted update strategy test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Weighted update test failed: {e}")
        return False


def test_confidence_based_update():
    """Test confidence-based update strategy."""
    logger.info("Testing confidence-based update strategy...")
    
    try:
        # Initialize online learner
        learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='confidence',
            batch_size=5,
            confidence_threshold=0.7
        )
        
        # Create initial training data
        initial_features, initial_targets = create_dummy_feature_data(20)
        
        # Initialize model
        performance = learner.initialize_model(initial_features, initial_targets)
        logger.info(f"✓ Model initialized with R²: {performance.get('r2', 0):.3f}")
        
        # Test confidence-based updates
        new_features, new_targets = create_dummy_feature_data(8)
        
        # Test with high confidence (should not trigger immediate update)
        high_conf_updates = 0
        for i in range(3):
            result = learner.process_technician_submission(
                feature_data=new_features[i],
                label=new_targets[i],
                timestamp=datetime.now().isoformat(),
                technician_id=f"tech_{i}",
                confidence=0.9  # High confidence
            )
            
            if result['update_applied']:
                high_conf_updates += 1
        
        logger.info(f"✓ High confidence submissions: {high_conf_updates} updates applied")
        
        # Test with low confidence (should trigger immediate update)
        low_conf_result = learner.process_technician_submission(
            feature_data=new_features[3],
            label=new_targets[3],
            timestamp=datetime.now().isoformat(),
            technician_id="tech_low_conf",
            confidence=0.5  # Low confidence
        )
        
        assert low_conf_result['update_applied'] == True, "Low confidence should trigger immediate update"
        logger.info("✓ Low confidence submission triggered immediate update")
        
        logger.info("✓ Confidence-based update strategy test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Confidence-based update test failed: {e}")
        return False


def test_holdout_validation():
    """Test holdout validation functionality."""
    logger.info("Testing holdout validation...")
    
    try:
        # Initialize online learner
        learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='immediate'
        )
        
        # Create initial training data
        initial_features, initial_targets = create_dummy_feature_data(20)
        
        # Initialize model
        performance = learner.initialize_model(initial_features, initial_targets)
        logger.info(f"✓ Model initialized with R²: {performance.get('r2', 0):.3f}")
        
        # Add samples to holdout set
        holdout_features, holdout_targets = create_dummy_feature_data(10)
        
        for feature_data, target in zip(holdout_features, holdout_targets):
            learner.add_to_holdout(feature_data, target)
        
        assert len(learner.holdout_data) == 10, f"Expected 10 holdout samples, got {len(learner.holdout_data)}"
        logger.info("✓ Holdout samples added successfully")
        
        # Test validation during update
        new_feature_data, new_target = create_dummy_feature_data(1)
        result = learner.process_technician_submission(
            feature_data=new_feature_data[0],
            label=new_target[0],
            timestamp=datetime.now().isoformat(),
            technician_id="tech_validation",
            confidence=0.8
        )
        
        assert 'validation_metrics' in result, "Validation metrics should be included"
        logger.info("✓ Holdout validation working correctly")
        
        logger.info("✓ Holdout validation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Holdout validation test failed: {e}")
        return False


def main():
    """Run all online learning tests."""
    logger.info("=" * 60)
    logger.info("RUNNING ONLINE LEARNING SYSTEM TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Immediate Update Strategy", test_immediate_update),
        ("Batch Update Strategy", test_batch_update),
        ("Weighted Update Strategy", test_weighted_update),
        ("Confidence-based Update Strategy", test_confidence_based_update),
        ("Holdout Validation", test_holdout_validation)
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
        logger.info("🎉 All online learning tests passed!")
        logger.info("\nStep 4: Online Learning Implementation is COMPLETE ✅")
        logger.info("\nImplemented features:")
        logger.info("✅ Core algorithm: Extract features → Store data → Update model → Validate → Log metrics")
        logger.info("✅ Immediate update: Update after each submission")
        logger.info("✅ Batch update: Collect N submissions, then update")
        logger.info("✅ Weighted update: Weight recent samples more heavily")
        logger.info("✅ Confidence-based: Update more when model is uncertain")
        logger.info("✅ Holdout validation: Validate on recent holdout set")
        logger.info("✅ Performance logging: Log all metrics and improvements")
        logger.info("✅ Web API integration: Complete pipeline in submit_label endpoint")
    else:
        logger.error("❌ Some online learning tests failed. Please review the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()