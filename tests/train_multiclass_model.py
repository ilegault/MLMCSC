#!/usr/bin/env python3
"""
Multi-Class Charpy Model Training Script
Trains a YOLO model on your multi-class Charpy dataset
"""

import os
from ultralytics import YOLO
from pathlib import Path
import yaml

def setup_training_config():
    """Create optimized training configuration for multi-class Charpy detection"""
    
    import torch
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Adjust parameters based on device
    if device == 'cpu':
        batch_size = 8           # Smaller batch for CPU
        epochs = 50              # Fewer epochs for CPU training
        workers = 2              # Fewer workers for CPU
        imgsz = 416              # Smaller image size for faster CPU training
        print("⚠️  Training on CPU - using optimized CPU settings")
    else:
        batch_size = 16          # Larger batch for GPU
        epochs = 100             # More epochs for GPU
        workers = 4              # More workers for GPU
        imgsz = 640              # Standard size for GPU
        print("✅ Training on GPU - using optimized GPU settings")
    
    # Training parameters optimized for your dataset size and classes
    training_config = {
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'patience': 15,          # Early stopping patience
        'save_period': 10,       # Save checkpoint every 10 epochs
        'workers': workers,      # Data loading workers
        'device': device,        # Auto-detect device
        
        # Learning rate schedule
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.1,             # Final learning rate factor
        'momentum': 0.937,       # SGD momentum
        'weight_decay': 0.0005,  # Weight decay
        
        # Augmentation (important for small dataset)
        'hsv_h': 0.015,         # Hue augmentation
        'hsv_s': 0.7,           # Saturation augmentation  
        'hsv_v': 0.4,           # Value augmentation
        'degrees': 10.0,        # Rotation augmentation
        'translate': 0.1,       # Translation augmentation
        'scale': 0.5,           # Scale augmentation
        'shear': 0.0,           # Shear augmentation
        'perspective': 0.0,     # Perspective augmentation
        'flipud': 0.0,          # Vertical flip probability
        'fliplr': 0.5,          # Horizontal flip probability
        'mosaic': 1.0,          # Mosaic augmentation probability
        'mixup': 0.1,           # Mixup augmentation probability
        
        # Loss weights (balanced for multi-class)
        'box': 7.5,             # Box loss weight
        'cls': 0.5,             # Classification loss weight
        'dfl': 1.5,             # Distribution focal loss weight
        
        # Validation
        'val': True,            # Validate during training
        'plots': True,          # Save training plots
        'save': True,           # Save checkpoints
    }
    
    return training_config

def train_multiclass_model():
    """Train the multi-class Charpy detection model"""
    
    print("🚀 MULTI-CLASS CHARPY MODEL TRAINING")
    print("=" * 50)
    
    # Paths
    dataset_yaml = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/dataset.yaml"
    output_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/models/charpy_multiclass"
    
    # Verify dataset exists
    if not os.path.exists(dataset_yaml):
        print(f"❌ Dataset configuration not found: {dataset_yaml}")
        return
    
    # Load and verify dataset config
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"📊 Dataset Configuration:")
    print(f"   Classes: {dataset_config['nc']}")
    print(f"   Names: {list(dataset_config['names'].values())}")
    print(f"   Path: {dataset_config['path']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model (YOLOv8 nano for faster training, especially on CPU)
    print(f"\n🤖 Initializing YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Start with pre-trained weights (nano version for CPU)
    
    # Get training configuration
    train_config = setup_training_config()
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {train_config['epochs']}")
    print(f"   Batch size: {train_config['batch']}")
    print(f"   Image size: {train_config['imgsz']}")
    print(f"   Learning rate: {train_config['lr0']}")
    print(f"   Augmentation: Enabled")
    
    print(f"\n🎯 Starting training...")
    print(f"   Output directory: {output_dir}")
    if train_config['device'] == 'cpu':
        print(f"   Training on CPU - this may take 60-120 minutes...")
        print(f"   Consider using Google Colab or a GPU system for faster training")
    else:
        print(f"   Training on GPU - this may take 30-60 minutes...")
    
    try:
        # Train the model
        results = model.train(
            data=dataset_yaml,
            project=output_dir,
            name='charpy_multiclass_v1',
            **train_config
        )
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"✅ Model saved to: {output_dir}/charpy_multiclass_v1")
        print(f"✅ Best weights: {output_dir}/charpy_multiclass_v1/weights/best.pt")
        print(f"✅ Training plots: {output_dir}/charpy_multiclass_v1/")
        
        # Print training summary
        if hasattr(results, 'results_dict'):
            final_metrics = results.results_dict
            print(f"\n📈 Final Training Metrics:")
            if 'metrics/mAP50(B)' in final_metrics:
                print(f"   mAP@0.5: {final_metrics['metrics/mAP50(B)']:.3f}")
            if 'metrics/mAP50-95(B)' in final_metrics:
                print(f"   mAP@0.5-0.95: {final_metrics['metrics/mAP50-95(B)']:.3f}")
        
        return f"{output_dir}/charpy_multiclass_v1/weights/best.pt"
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Common issues:")
        print("   - GPU memory insufficient (try reducing batch size)")
        print("   - Dataset path issues")
        print("   - Missing dependencies")
        return None

def validate_trained_model(model_path):
    """Validate the trained model"""
    
    if not model_path or not os.path.exists(model_path):
        print("❌ No model to validate")
        return
    
    print(f"\n🔍 VALIDATING TRAINED MODEL")
    print("=" * 30)
    
    try:
        # Load trained model
        model = YOLO(model_path)
        
        # Run validation
        dataset_yaml = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/dataset.yaml"
        results = model.val(data=dataset_yaml)
        
        print(f"✅ Validation completed")
        print(f"📊 Results saved with model")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")

def main():
    print("🎯 MULTI-CLASS CHARPY MODEL TRAINER")
    print("=" * 50)
    print("This will train a YOLOv8 model on your multi-class dataset.")
    print("Expected training time: 30-60 minutes")
    print()
    print("Your dataset:")
    print("  - 5 classes: specimen, edge, corner, fracture_surface, measurement_point")
    print("  - 1,385 total annotations")
    print("  - 164 images across train/val/test splits")
    print()
    
    response = input("Start training? (y/N): ").lower().strip()
    if response != 'y':
        print("Training cancelled")
        return
    
    # Train the model
    model_path = train_multiclass_model()
    
    # Validate if training succeeded
    if model_path:
        validate_trained_model(model_path)
        
        print(f"\n🎉 TRAINING PIPELINE COMPLETE!")
        print(f"=" * 50)
        print(f"✅ Multi-class model trained successfully")
        print(f"✅ Model location: {model_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. Test the model: python test_multiclass_model.py")
        print(f"   2. Run inference: python run_multiclass_inference.py")
        print(f"   3. Analyze results: Check training plots and metrics")

if __name__ == "__main__":
    main()