#!/usr/bin/env python3
"""
Data Management System Demo

This script demonstrates the comprehensive data management capabilities
of the MLMCSC system including database operations, data versioning,
backup management, and privacy compliance.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import DatabaseManager, DataPipeline, PipelineConfig, load_config
from src.database.models import Image, Label, Prediction

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_database_operations():
    """Demonstrate basic database operations."""
    print("\n" + "="*60)
    print("DATABASE OPERATIONS DEMO")
    print("="*60)
    
    # Load configuration
    config = load_config()
    db_url = config.database.get_connection_url(project_root)
    
    # Initialize database manager
    db_manager = DatabaseManager(db_url)
    
    print(f"✅ Database initialized: {db_url}")
    
    # Register a technician
    try:
        tech_id = db_manager.register_technician(
            technician_id="demo_tech_001",
            name="Demo Technician",
            email="demo@example.com",
            department="Quality Control",
            experience_level="expert"
        )
        print(f"✅ Technician registered with ID: {tech_id}")
    except ValueError as e:
        print(f"ℹ️  Technician already exists: {e}")
        # Get existing technician
        with db_manager.get_session() as session:
            from src.database.models import Technician
            tech = session.query(Technician).filter(
                Technician.technician_id == "demo_tech_001"
            ).first()
            tech_id = tech.id if tech else None
    
    # Register a model version
    try:
        model_id = db_manager.register_model_version(
            version="demo_v1.0",
            name="Demo Model v1.0",
            model_path="models/demo_model.pkl",
            description="Demo model for testing",
            hyperparameters={"learning_rate": 0.001, "epochs": 100},
            performance_metrics={"training_accuracy": 0.95, "validation_accuracy": 0.92}
        )
        print(f"✅ Model version registered with ID: {model_id}")
    except ValueError as e:
        print(f"ℹ️  Model version already exists: {e}")
        with db_manager.get_session() as session:
            from src.database.models import ModelVersion
            model = session.query(ModelVersion).filter(
                ModelVersion.version == "demo_v1.0"
            ).first()
            model_id = model.id if model else None
    
    # Store sample image (create a dummy file for demo)
    demo_image_path = project_root / "demo_images" / "sample_image.jpg"
    demo_image_path.parent.mkdir(exist_ok=True)
    
    if not demo_image_path.exists():
        # Create a dummy image file
        demo_image_path.write_text("dummy image content")
    
    try:
        image_id = db_manager.store_image(
            image_path=demo_image_path,
            specimen_id="SPEC_001",
            magnification=400.0,
            acquisition_settings={"brightness": 0.5, "contrast": 0.7}
        )
        print(f"✅ Image stored with ID: {image_id}")
    except Exception as e:
        print(f"ℹ️  Image already exists or error: {e}")
        # Get existing image
        with db_manager.get_session() as session:
            image = session.query(Image).filter(
                Image.path == str(demo_image_path)
            ).first()
            image_id = image.id if image else None
    
    if image_id and model_id:
        # Store a prediction
        pred_id = db_manager.store_prediction(
            image_id=image_id,
            model_version_id=model_id,
            prediction=0.85,
            confidence=0.92,
            detection_bbox=[100, 100, 200, 200],
            detection_confidence=0.88,
            processing_time=0.15
        )
        print(f"✅ Prediction stored with ID: {pred_id}")
        
        # Store a label (if technician exists)
        if tech_id:
            label_id = db_manager.store_label(
                image_id=image_id,
                technician_id=tech_id,
                label=0.87,
                original_prediction=0.85,
                original_confidence=0.92,
                time_spent=45.0,
                difficulty_rating=3,
                notes="Good quality specimen"
            )
            print(f"✅ Label stored with ID: {label_id}")
    
    # Store feature vector
    if image_id:
        feature_vector = np.random.rand(512)  # Dummy feature vector
        feature_id = db_manager.store_features(
            image_id=image_id,
            feature_type="cnn",
            extractor_version="resnet50_v1",
            feature_vector=feature_vector,
            extraction_time=0.05
        )
        print(f"✅ Features stored with ID: {feature_id}")
    
    # Get system health
    health = db_manager.get_system_health()
    print(f"\n📊 System Health:")
    for category, metrics in health.items():
        if isinstance(metrics, dict):
            print(f"  {category.replace('_', ' ').title()}:")
            for key, value in metrics.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {category}: {metrics}")
    
    return db_manager


def demo_data_versioning(db_manager):
    """Demonstrate data versioning capabilities."""
    print("\n" + "="*60)
    print("DATA VERSIONING DEMO")
    print("="*60)
    
    # Create a data version
    version_name = f"demo_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    version_id = db_manager.create_data_version(
        version=version_name,
        name=f"Demo Version {datetime.now().strftime('%Y-%m-%d')}",
        description="Demonstration data version with sample data",
        user_id="demo_user"
    )
    print(f"✅ Data version created: {version_name} (ID: {version_id})")
    
    # Get training data splits
    training_data = db_manager.get_training_data(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        verified_only=False
    )
    
    print(f"📊 Training Data Splits:")
    print(f"  Train: {len(training_data['train'])} samples")
    print(f"  Validation: {len(training_data['val'])} samples")
    print(f"  Test: {len(training_data['test'])} samples")
    
    return version_id


def demo_backup_management(db_manager):
    """Demonstrate backup management."""
    print("\n" + "="*60)
    print("BACKUP MANAGEMENT DEMO")
    print("="*60)
    
    # Create a backup
    backup_name = f"demo_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = db_manager.backup_database(backup_name)
    
    file_size = backup_path.stat().st_size
    print(f"✅ Backup created: {backup_path}")
    print(f"   Size: {file_size / 1024:.2f} KB")
    
    # List backups
    backup_files = list(db_manager.backup_dir.glob("*.db"))
    print(f"\n📁 Available backups ({len(backup_files)}):")
    for backup in sorted(backup_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
        size = backup.stat().st_size / 1024
        modified = datetime.fromtimestamp(backup.stat().st_mtime)
        print(f"  {backup.name} - {size:.2f} KB - {modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return backup_path


def demo_data_pipeline(db_manager):
    """Demonstrate data pipeline functionality."""
    print("\n" + "="*60)
    print("DATA PIPELINE DEMO")
    print("="*60)
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        backup_enabled=True,
        backup_schedule="daily",
        auto_versioning=True,
        version_trigger_threshold=10,  # Lower threshold for demo
        quality_monitoring=True,
        metrics_collection=True
    )
    
    # Initialize pipeline
    pipeline = DataPipeline(db_manager, pipeline_config)
    
    print("✅ Data pipeline initialized")
    
    # Run quality check
    quality_report = pipeline.check_data_quality()
    print(f"\n🔍 Data Quality Report:")
    
    img_quality = quality_report['image_quality']
    print(f"  Images: {img_quality['total_images']} total, "
          f"{img_quality['low_quality_percent']:.1f}% low quality")
    
    label_quality = quality_report['label_quality']
    print(f"  Labels: {label_quality['total_labels']} total, "
          f"{label_quality['verification_rate']:.1f}% verified")
    
    completeness = quality_report['data_completeness']
    print(f"  Completeness: {completeness['completeness_rate']:.1f}%")
    
    # Collect system metrics
    pipeline.collect_system_metrics()
    print("✅ System metrics collected")
    
    # Generate privacy report
    privacy_report = pipeline.generate_privacy_report()
    print(f"\n🔒 Privacy Compliance:")
    print(f"  GDPR Ready: {privacy_report['compliance_status']['gdpr_ready']}")
    print(f"  Active Technicians: {privacy_report['active_users']['active_technicians']}")
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    print(f"\n⚙️  Pipeline Status:")
    print(f"  Running: {status['is_running']}")
    if status['last_backup']:
        print(f"  Last Backup: {status['last_backup']['created_at']}")
    
    return pipeline


def demo_privacy_compliance(db_manager):
    """Demonstrate privacy compliance features."""
    print("\n" + "="*60)
    print("PRIVACY COMPLIANCE DEMO")
    print("="*60)
    
    # Export user data (GDPR compliance)
    user_data = db_manager.export_user_data("demo_tech_001")
    print(f"✅ User data exported for demo_tech_001")
    print(f"   Technician info: {'Yes' if user_data['technician_info'] else 'No'}")
    print(f"   Labels: {len(user_data['labels'])}")
    print(f"   Audit logs: {len(user_data['audit_logs'])}")
    
    # Demonstrate anonymization (without actually running it)
    print(f"\n🔒 Anonymization capabilities:")
    print(f"   - Anonymize audit logs older than retention period")
    print(f"   - Remove personal identifiers from old records")
    print(f"   - Maintain data utility while protecting privacy")
    
    # Record metrics about privacy operations
    db_manager.record_metric("privacy_export_requests", 1, "compliance", "count")
    print("✅ Privacy metrics recorded")


def demo_efficient_querying(db_manager):
    """Demonstrate efficient querying for training."""
    print("\n" + "="*60)
    print("EFFICIENT QUERYING DEMO")
    print("="*60)
    
    # Get images with pagination
    images = db_manager.get_images(limit=10, offset=0, valid_only=True)
    print(f"✅ Retrieved {len(images)} images (paginated)")
    
    # Get predictions with filtering
    predictions = db_manager.get_predictions(
        confidence_threshold=0.8,
        limit=10
    )
    print(f"✅ Retrieved {len(predictions)} high-confidence predictions")
    
    # Get features for training
    if images:
        image_ids = [img.id for img in images[:5]]
        features = db_manager.get_features_for_training(
            image_ids=image_ids,
            feature_type="cnn"
        )
        print(f"✅ Retrieved feature matrix: {features.shape if len(features) > 0 else 'No features'}")
    
    # Demonstrate batch processing
    training_splits = db_manager.get_training_data(verified_only=False)
    total_samples = sum(len(split) for split in training_splits.values())
    print(f"✅ Training data ready: {total_samples} total samples")


def main():
    """Run the complete data management demo."""
    print("🚀 MLMCSC Data Management System Demo")
    print("="*60)
    
    try:
        # Demo 1: Basic database operations
        db_manager = demo_database_operations()
        
        # Demo 2: Data versioning
        demo_data_versioning(db_manager)
        
        # Demo 3: Backup management
        demo_backup_management(db_manager)
        
        # Demo 4: Data pipeline
        demo_data_pipeline(db_manager)
        
        # Demo 5: Privacy compliance
        demo_privacy_compliance(db_manager)
        
        # Demo 6: Efficient querying
        demo_efficient_querying(db_manager)
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n📋 Summary of capabilities demonstrated:")
        print("  ✓ Database initialization and schema management")
        print("  ✓ Image, prediction, and label storage")
        print("  ✓ Feature vector management")
        print("  ✓ Data versioning and snapshots")
        print("  ✓ Automated backup creation")
        print("  ✓ Data quality monitoring")
        print("  ✓ Privacy compliance (GDPR)")
        print("  ✓ System metrics collection")
        print("  ✓ Efficient querying for ML training")
        print("  ✓ Audit logging and traceability")
        
        print("\n🎯 Next steps:")
        print("  1. Integrate with your ML training pipeline")
        print("  2. Set up automated data pipeline")
        print("  3. Configure backup schedules")
        print("  4. Customize quality monitoring thresholds")
        print("  5. Implement user authentication")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())