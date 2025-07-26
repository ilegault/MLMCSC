#!/usr/bin/env python3
"""
Fast CPU Multi-Class Training Script
Optimized for quick CPU training with reduced parameters
"""

import os
from ultralytics import YOLO
import yaml

def train_fast_cpu_model():
    """Train a multi-class model quickly on CPU with minimal settings"""
    
    print("⚡ FAST CPU MULTI-CLASS TRAINING")
    print("=" * 50)
    print("This is optimized for quick CPU training (15-30 minutes)")
    print("Trade-off: Faster training but potentially lower accuracy")
    print()
    
    # Paths
    dataset_yaml = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/dataset.yaml"
    output_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/models/charpy_multiclass_fast"
    
    # Verify dataset exists
    if not os.path.exists(dataset_yaml):
        print(f"❌ Dataset configuration not found: {dataset_yaml}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fast CPU training configuration
    fast_config = {
        'epochs': 30,            # Fewer epochs for speed
        'batch': 4,              # Small batch size for CPU
        'imgsz': 320,            # Smaller image size for speed
        'patience': 10,          # Early stopping
        'save_period': -1,       # Don't save intermediate checkpoints
        'workers': 1,            # Single worker for stability
        'device': 'cpu',         # Force CPU
        'cache': False,          # Don't cache images (saves RAM)
        'verbose': True,         # Show progress
        
        # Minimal augmentation for speed
        'hsv_h': 0.01,
        'hsv_s': 0.3,
        'hsv_v': 0.2,
        'degrees': 5.0,
        'translate': 0.05,
        'scale': 0.3,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,           # Reduced mosaic
        'mixup': 0.0,            # No mixup for speed
        
        # Simplified loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
    
    print(f"⚙️  Fast Training Configuration:")
    print(f"   Epochs: {fast_config['epochs']}")
    print(f"   Batch size: {fast_config['batch']}")
    print(f"   Image size: {fast_config['imgsz']}")
    print(f"   Device: {fast_config['device']}")
    print(f"   Estimated time: 15-30 minutes")
    
    # Initialize model
    print(f"\n🤖 Loading YOLOv8 nano model...")
    model = YOLO('yolov8n.pt')
    
    print(f"\n🚀 Starting fast training...")
    
    try:
        # Train the model
        results = model.train(
            data=dataset_yaml,
            project=output_dir,
            name='charpy_fast_v1',
            **fast_config
        )
        
        model_path = f"{output_dir}/charpy_fast_v1/weights/best.pt"
        
        print(f"\n🎉 FAST TRAINING COMPLETED!")
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Training time: Approximately 15-30 minutes")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None

def quick_test_fast_model(model_path):
    """Quick test of the fast-trained model"""
    
    if not model_path or not os.path.exists(model_path):
        print("❌ No model to test")
        return
    
    print(f"\n🔍 QUICK MODEL TEST")
    print("=" * 20)
    
    try:
        import cv2
        
        # Load model
        model = YOLO(model_path)
        
        # Test on a sample image
        test_image = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/images/test/charpy_0000.jpg"
        
        if os.path.exists(test_image):
            img = cv2.imread(test_image)
            results = model(img, conf=0.3)
            
            detections = 0
            classes_found = set()
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detections += 1
                        classes_found.add(int(box.cls))
            
            print(f"✅ Test completed:")
            print(f"   Detections: {detections}")
            print(f"   Classes detected: {len(classes_found)}")
            print(f"   Classes: {[model.names[c] for c in classes_found]}")
            
            if len(classes_found) >= 3:
                print("🎉 Multi-class detection working!")
            elif len(classes_found) >= 2:
                print("⚠️  Limited multi-class detection")
            else:
                print("❌ Single-class or no detection")
        else:
            print(f"❌ Test image not found: {test_image}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    print("⚡ FAST CPU MULTI-CLASS TRAINER")
    print("=" * 50)
    print("This script trains a multi-class model quickly on CPU.")
    print("Perfect for:")
    print("  - Quick prototyping")
    print("  - Testing your annotations")
    print("  - Systems without GPU")
    print()
    print("Settings:")
    print("  - 30 epochs (vs 100 in full training)")
    print("  - 320px images (vs 640px)")
    print("  - Minimal augmentation")
    print("  - Expected time: 15-30 minutes")
    print()
    
    response = input("Start fast training? (y/N): ").lower().strip()
    if response != 'y':
        print("Training cancelled")
        return
    
    # Train the model
    model_path = train_fast_cpu_model()
    
    # Quick test
    if model_path:
        quick_test_fast_model(model_path)
        
        print(f"\n🎉 FAST TRAINING COMPLETE!")
        print(f"=" * 30)
        print(f"✅ Fast model trained in ~15-30 minutes")
        print(f"✅ Model location: {model_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. Test thoroughly: python test_multiclass_model.py")
        print(f"   2. If results are good, you're done!")
        print(f"   3. If you need better accuracy, run the full training:")
        print(f"      python train_multiclass_model.py")

if __name__ == "__main__":
    main()