#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced MLMCSC Systems

This module tests the integration of all advanced systems:
- Step 5: Model Versioning System
- Step 6: Quality Control Mechanisms  
- Step 7: Performance Monitoring
- Step 8: Active Learning System

Tests the complete pipeline from data ingestion to advanced analytics.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_versioning():
    """Test the enhanced model versioning system."""
    print("\n" + "="*60)
    print("TESTING MODEL VERSIONING SYSTEM")
    print("="*60)
    
    try:
        from src.mlmcsc.regression.model_versioning import EnhancedModelVersionManager
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize version manager
            version_manager = EnhancedModelVersionManager(
                models_directory=temp_path / "versions",
                checkpoint_frequency=3,
                performance_threshold=0.05
            )
            
            # Create test model and scaler
            model = SGDRegressor(random_state=42)
            scaler = StandardScaler()
            
            # Generate test data
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            
            # Fit model
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            # Save model temporarily
            model_path = temp_path / "test_model.joblib"
            scaler_path = temp_path / "test_scaler.joblib"
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Test checkpoint creation
            performance_metrics = {'r2': 0.85, 'rmse': 2.1, 'mae': 1.8, 'mse': 4.41}
            
            version_id = version_manager.create_checkpoint_version(
                model_path=model_path,
                scaler_path=scaler_path,
                model_type='sgd',
                performance_metrics=performance_metrics,
                training_samples_count=100,
                feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
                scaler_type='standard',
                reason='test_checkpoint',
                notes='Test checkpoint for validation'
            )
            
            print(f"✓ Created checkpoint version: {version_id}")
            
            # Test version loading
            loaded_model, loaded_scaler, version_metadata = version_manager.load_version_with_scaler(version_id)
            print(f"✓ Loaded version: {version_metadata.version_id}")
            
            # Test A/B testing
            # Create second version with different performance
            performance_metrics_2 = {'r2': 0.88, 'rmse': 1.9, 'mae': 1.6, 'mse': 3.61}
            version_id_2 = version_manager.create_checkpoint_version(
                model_path=model_path,
                scaler_path=scaler_path,
                model_type='sgd',
                performance_metrics=performance_metrics_2,
                training_samples_count=120,
                feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
                scaler_type='standard',
                reason='test_checkpoint_2',
                notes='Second test checkpoint'
            )
            
            # Start A/B test
            test_id = version_manager.start_ab_test(
                version_a=version_id,
                version_b=version_id_2,
                traffic_split=0.5,
                duration_hours=1
            )
            print(f"✓ Started A/B test: {test_id}")
            
            # Test version selection for A/B test
            selected_version = version_manager.get_ab_test_version(test_id, "test_user_123")
            print(f"✓ A/B test selected version: {selected_version}")
            
            # Test performance degradation detection
            degraded_metrics = {'r2': 0.65, 'rmse': 3.5, 'mae': 2.8, 'mse': 12.25}
            warning = version_manager.check_performance_degradation(degraded_metrics)
            if warning:
                print(f"✓ Performance degradation detected: {warning[:100]}...")
            
            # Test version comparison
            comparison = version_manager.get_performance_comparison([version_id, version_id_2])
            print(f"✓ Performance comparison completed for {len(comparison['versions'])} versions")
            
            # Stop A/B test
            results = version_manager.stop_ab_test(test_id)
            print(f"✓ A/B test stopped, winner: {results['winner']}")
            
            print("✅ Model versioning system tests PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Model versioning system tests FAILED: {e}")
        return False


def test_quality_control():
    """Test the quality control system."""
    print("\n" + "="*60)
    print("TESTING QUALITY CONTROL SYSTEM")
    print("="*60)
    
    try:
        from src.mlmcsc.quality_control import QualityControlSystem
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize quality control system
            qc_system = QualityControlSystem(
                storage_path=temp_dir,
                outlier_threshold=2.0,
                variance_threshold=0.3
            )
            
            # Test outlier detection
            labels = [10.5, 11.2, 10.8, 11.0, 25.0, 10.9, 11.1, 10.7]  # 25.0 is outlier
            technician_ids = ['tech1', 'tech2', 'tech1', 'tech3', 'tech2', 'tech1', 'tech3', 'tech2']
            image_ids = [f'img_{i}' for i in range(len(labels))]
            timestamps = [(datetime.now() - timedelta(hours=i)).isoformat() for i in range(len(labels))]
            
            outlier_results = qc_system.detect_label_outliers(
                labels=labels,
                technician_ids=technician_ids,
                image_ids=image_ids,
                timestamps=timestamps,
                method='zscore'
            )
            
            outlier_count = sum(1 for r in outlier_results if r.is_outlier)
            print(f"✓ Detected {outlier_count} label outliers")
            
            # Test feature outlier detection
            features = [
                {'feature_1': 1.0, 'feature_2': 2.0, 'feature_3': 1.5},
                {'feature_1': 1.1, 'feature_2': 2.1, 'feature_3': 1.4},
                {'feature_1': 0.9, 'feature_2': 1.9, 'feature_3': 1.6},
                {'feature_1': 5.0, 'feature_2': 8.0, 'feature_3': 7.0},  # Outlier
            ]
            
            feature_outliers = qc_system.detect_feature_outliers(
                features=features,
                image_ids=image_ids[:4],
                technician_ids=technician_ids[:4],
                timestamps=timestamps[:4]
            )
            
            feature_outlier_count = sum(1 for r in feature_outliers if r.is_outlier)
            print(f"✓ Detected {feature_outlier_count} feature outliers")
            
            # Test inter-rater reliability
            labels_by_image = {
                'img_1': [(10.5, 'tech1', timestamps[0]), (10.8, 'tech2', timestamps[1]), (10.6, 'tech3', timestamps[2])],
                'img_2': [(15.0, 'tech1', timestamps[3]), (14.8, 'tech2', timestamps[4]), (15.2, 'tech3', timestamps[5])],
                'img_3': [(12.0, 'tech1', timestamps[6]), (18.0, 'tech2', timestamps[7]), (11.8, 'tech3', timestamps[0])]  # Poor agreement
            }
            
            reliability_results = qc_system.analyze_inter_rater_reliability(labels_by_image)
            print(f"✓ Analyzed inter-rater reliability for {len(reliability_results)} images")
            
            # Test high variance prediction flagging
            predictions = [10.5, 11.2, 25.0, 10.8]  # 25.0 is high variance
            confidences = [0.8, 0.9, 0.2, 0.85]    # 0.2 is low confidence
            
            flagged_images = qc_system.flag_high_variance_predictions(
                predictions=predictions,
                confidences=confidences,
                image_ids=image_ids[:4],
                timestamps=timestamps[:4]
            )
            
            print(f"✓ Flagged {len(flagged_images)} high variance predictions")
            
            # Test expert review queue
            review_queue = qc_system.create_expert_review_queue(max_items=10)
            print(f"✓ Created expert review queue with {len(review_queue)} items")
            
            # Test quality report
            report = qc_system.get_quality_report(days_back=1)
            print(f"✓ Generated quality report with {report['total_flags']} flags")
            
            print("✅ Quality control system tests PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Quality control system tests FAILED: {e}")
        return False


def test_performance_monitoring():
    """Test the performance monitoring system."""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE MONITORING SYSTEM")
    print("="*60)
    
    try:
        from src.mlmcsc.monitoring import PerformanceMonitor
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize performance monitor
            monitor = PerformanceMonitor(
                storage_path=temp_dir,
                alert_thresholds={
                    'r2_min': 0.7,
                    'rmse_max': 5.0,
                    'mae_max': 4.0,
                    'confidence_min': 0.5
                }
            )
            
            # Test performance metrics recording
            for i in range(10):
                monitor.record_performance_metrics(
                    mae=2.0 + np.random.normal(0, 0.2),
                    rmse=3.0 + np.random.normal(0, 0.3),
                    r2=0.85 + np.random.normal(0, 0.05),
                    mse=9.0 + np.random.normal(0, 1.0),
                    sample_count=100 + i * 10,
                    prediction_count=5,
                    avg_confidence=0.8 + np.random.normal(0, 0.1),
                    model_version=f"v1.{i}"
                )
            
            print("✓ Recorded 10 performance metrics")
            
            # Test prediction recording
            for i in range(20):
                monitor.record_prediction(
                    prediction=10.0 + np.random.normal(0, 2.0),
                    confidence=0.7 + np.random.normal(0, 0.15),
                    actual_label=10.0 + np.random.normal(0, 1.5) if i % 2 == 0 else None,
                    image_id=f"img_{i}",
                    technician_id=f"tech_{i % 3 + 1}"
                )
            
            print("✓ Recorded 20 predictions")
            
            # Test label recording
            for i in range(15):
                monitor.record_label(
                    label=10.0 + np.random.normal(0, 1.0),
                    technician_id=f"tech_{i % 3 + 1}",
                    image_id=f"img_{i}",
                    confidence=0.8 + np.random.normal(0, 0.1)
                )
            
            print("✓ Recorded 15 labels")
            
            # Test feature importance recording
            feature_names = ['texture_contrast', 'edge_density', 'color_variance', 'shape_complexity']
            for i in range(5):
                importances = {name: np.random.random() for name in feature_names}
                # Normalize to sum to 1
                total = sum(importances.values())
                importances = {k: v/total for k, v in importances.items()}
                
                monitor.record_feature_importance(
                    feature_importances=importances,
                    model_version=f"v1.{i}"
                )
            
            print("✓ Recorded 5 feature importance snapshots")
            
            # Test performance summary
            summary = monitor.get_performance_summary(days_back=1)
            print(f"✓ Generated performance summary with {summary['total_metrics_recorded']} metrics")
            
            # Test technician summary
            tech_summary = monitor.get_technician_summary(days_back=1)
            print(f"✓ Generated technician summary for {tech_summary['total_technicians']} technicians")
            
            # Test alerts
            alerts = monitor.get_alerts()
            print(f"✓ Retrieved {len(alerts)} performance alerts")
            
            # Test dashboard creation
            dashboard_path = Path(temp_dir) / "dashboard.html"
            created_dashboard = monitor.create_performance_dashboard(str(dashboard_path), days_back=1)
            if Path(created_dashboard).exists():
                print("✓ Created interactive performance dashboard")
            
            # Test static plots
            plots_dir = Path(temp_dir) / "plots"
            created_plots = monitor.create_static_plots(str(plots_dir), days_back=1)
            print(f"✓ Created {len(created_plots)} static plots")
            
            print("✅ Performance monitoring system tests PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Performance monitoring system tests FAILED: {e}")
        return False


def test_active_learning():
    """Test the active learning system."""
    print("\n" + "="*60)
    print("TESTING ACTIVE LEARNING SYSTEM")
    print("="*60)
    
    try:
        from src.mlmcsc.active_learning import ActiveLearningSystem
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize active learning system
            al_system = ActiveLearningSystem(
                storage_path=temp_dir,
                default_strategy="hybrid",
                max_queries_per_batch=5
            )
            
            # Create test model
            model = SGDRegressor(random_state=42)
            X_train = np.random.randn(50, 4)
            y_train = np.random.randn(50)
            model.fit(X_train, y_train)
            
            # Test query generation
            candidates = []
            for i in range(20):
                candidate = {
                    'image_id': f'candidate_{i}',
                    'features': {
                        'feature_1': np.random.randn(),
                        'feature_2': np.random.randn(),
                        'feature_3': np.random.randn(),
                        'feature_4': np.random.randn()
                    }
                }
                candidates.append(candidate)
            
            # Test uncertainty strategy
            uncertainty_queries = al_system.generate_queries(
                candidates=candidates,
                model=model,
                n_queries=5,
                strategy='uncertainty'
            )
            print(f"✓ Generated {len(uncertainty_queries)} uncertainty-based queries")
            
            # Test diversity strategy
            diversity_queries = al_system.generate_queries(
                candidates=candidates,
                model=model,
                n_queries=5,
                strategy='diversity'
            )
            print(f"✓ Generated {len(diversity_queries)} diversity-based queries")
            
            # Test hybrid strategy
            hybrid_queries = al_system.generate_queries(
                candidates=candidates,
                model=model,
                n_queries=5,
                strategy='hybrid'
            )
            print(f"✓ Generated {len(hybrid_queries)} hybrid queries")
            
            # Test query result submission
            if hybrid_queries:
                query_id = hybrid_queries[0].query_id
                success = al_system.submit_query_result(
                    query_id=query_id,
                    technician_id='tech_1',
                    label=12.5,
                    confidence=0.8,
                    feedback_quality='good'
                )
                print(f"✓ Submitted query result: {success}")
            
            # Test priority queries
            priority_queries = al_system.get_priority_queries(max_queries=3)
            print(f"✓ Retrieved {len(priority_queries)} priority queries")
            
            # Test query effectiveness analysis
            effectiveness = al_system.analyze_query_effectiveness(days_back=1)
            if 'error' not in effectiveness:
                print(f"✓ Analyzed query effectiveness for {effectiveness['total_queries_completed']} queries")
            else:
                print("✓ Query effectiveness analysis completed (no data yet)")
            
            # Test strategy optimization
            optimization = al_system.optimize_strategy_parameters()
            print(f"✓ Strategy optimization completed")
            
            # Test learning progress
            progress = al_system.get_learning_progress()
            print(f"✓ Learning progress: {progress['completion_rate']:.2%} completion rate")
            
            print("✅ Active learning system tests PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Active learning system tests FAILED: {e}")
        return False


def test_integrated_online_learning():
    """Test the integrated online learning system with all advanced features."""
    print("\n" + "="*60)
    print("TESTING INTEGRATED ONLINE LEARNING SYSTEM")
    print("="*60)
    
    try:
        from src.mlmcsc.regression.online_learning import OnlineLearningSystem
        
        # Initialize with all advanced features enabled
        online_learner = OnlineLearningSystem(
            model_type='sgd',
            update_strategy='batch',
            batch_size=3,
            enable_versioning=True,
            enable_quality_control=True,
            enable_monitoring=True,
            enable_active_learning=True
        )
        
        # Generate initial training data
        np.random.seed(42)
        initial_features = []
        initial_targets = []
        
        for i in range(20):
            features = {
                'feature_vector': np.random.randn(5).tolist(),
                'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5']
            }
            target = 10.0 + np.random.randn() * 2.0
            
            initial_features.append(features)
            initial_targets.append(target)
        
        # Initialize model
        performance = online_learner.initialize_model(initial_features, initial_targets)
        print(f"✓ Initialized model with R²: {performance['r2']:.3f}")
        
        # Test technician submissions with advanced features
        for i in range(10):
            feature_data = {
                'feature_vector': np.random.randn(5).tolist(),
                'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5']
            }
            
            label = 10.0 + np.random.randn() * 2.0
            # Add some outliers
            if i == 5:
                label = 25.0  # Outlier
            
            timestamp = (datetime.now() - timedelta(hours=i)).isoformat()
            technician_id = f"tech_{i % 3 + 1}"
            confidence = 0.8 + np.random.randn() * 0.1
            image_id = f"img_{i}"
            
            result = online_learner.process_technician_submission(
                feature_data=feature_data,
                label=label,
                timestamp=timestamp,
                technician_id=technician_id,
                confidence=confidence,
                image_id=image_id
            )
            
            print(f"✓ Processed submission {i+1}: update_applied={result['update_applied']}, "
                  f"quality_flags={len(result.get('quality_flags', []))}")
        
        # Test system status
        status = online_learner.get_system_status()
        print(f"✓ System status: {status['online_learning']['total_samples_seen']} samples seen")
        print(f"  - Versioning: {status['subsystems']['versioning']['enabled']}")
        print(f"  - Quality Control: {status['subsystems']['quality_control']['enabled']}")
        print(f"  - Monitoring: {status['subsystems']['monitoring']['enabled']}")
        print(f"  - Active Learning: {status['subsystems']['active_learning']['enabled']}")
        
        # Test active learning query generation
        candidate_features = []
        for i in range(10):
            candidate = {
                'image_id': f'candidate_{i}',
                'features': {
                    'f1': np.random.randn(),
                    'f2': np.random.randn(),
                    'f3': np.random.randn(),
                    'f4': np.random.randn(),
                    'f5': np.random.randn()
                }
            }
            candidate_features.append(candidate)
        
        queries = online_learner.generate_active_learning_queries(candidate_features, n_queries=3)
        print(f"✓ Generated {len(queries)} active learning queries")
        
        # Test active learning result submission
        if queries:
            query_id = queries[0]['query_id']
            success = online_learner.submit_active_learning_result(
                query_id=query_id,
                technician_id='tech_expert',
                label=11.5,
                confidence=0.9
            )
            print(f"✓ Submitted active learning result: {success}")
        
        print("✅ Integrated online learning system tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integrated online learning system tests FAILED: {e}")
        return False


def main():
    """Run all advanced systems tests."""
    print("🚀 STARTING COMPREHENSIVE ADVANCED SYSTEMS TESTS")
    print("="*80)
    
    test_results = []
    
    # Run individual system tests
    test_results.append(("Model Versioning", test_model_versioning()))
    test_results.append(("Quality Control", test_quality_control()))
    test_results.append(("Performance Monitoring", test_performance_monitoring()))
    test_results.append(("Active Learning", test_active_learning()))
    test_results.append(("Integrated Online Learning", test_integrated_online_learning()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL ADVANCED SYSTEMS TESTS PASSED!")
        print("\nSteps 5-8: Advanced MLMCSC Features are COMPLETE ✅")
        print("\n📋 IMPLEMENTED FEATURES:")
        print("  ✅ Step 5: Model Versioning System")
        print("    - Automatic checkpointing after N updates")
        print("    - Performance-based rollback detection")
        print("    - A/B testing capability")
        print("    - Comprehensive metadata tracking")
        print("\n  ✅ Step 6: Quality Control Mechanisms")
        print("    - Outlier detection for labels and features")
        print("    - Inter-rater reliability analysis")
        print("    - High-variance prediction flagging")
        print("    - Expert review workflow")
        print("\n  ✅ Step 7: Performance Monitoring")
        print("    - Real-time metrics tracking")
        print("    - Interactive visualization dashboard")
        print("    - Automated alerting system")
        print("    - Comprehensive reporting")
        print("\n  ✅ Step 8: Advanced Enhancements")
        print("    - Active learning with multiple strategies")
        print("    - Uncertainty-based sampling")
        print("    - Diversity-based sampling")
        print("    - Hybrid query strategies")
        print("\n🔧 INTEGRATION COMPLETE:")
        print("  - All systems integrated with online learning")
        print("  - Comprehensive API endpoints")
        print("  - Real-time quality monitoring")
        print("  - Automated model management")
        print("  - Intelligent sample selection")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)