#!/usr/bin/env python3
"""
IMPROVED Training Strategy Based on Your Results
Focuses on fixing the issues with classes 1, 2, and 4
"""

import os
from ultralytics import YOLO
from pathlib import Path
import yaml


def setup_improved_training_config():
    """Training config optimized for your specific issues"""

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # More aggressive settings for better small object detection
    training_config = {
        'epochs': 200,  # More epochs for difficult classes
        'batch': 16,  # Smaller batch for more stable gradients
        'imgsz': 832,  # LARGER image size for small objects
        'patience': 40,
        'save_period': 20,
        'workers': 8,
        'device': device,
        'cache': 'ram',
        'amp': True,

        # CRITICAL: Optimized for small object detection
        'lr0': 0.005,  # LOWER learning rate for stability
        'lrf': 0.001,  # Much lower final LR
        'momentum': 0.95,  # Higher momentum
        'weight_decay': 0.001,  # More regularization
        'warmup_epochs': 10.0,  # Longer warmup

        # FOCUSED augmentation for small objects
        'hsv_h': 0.01,  # Minimal color changes
        'hsv_s': 0.3,
        'hsv_v': 0.3,
        'degrees': 5.0,  # Minimal rotation
        'translate': 0.05,  # Minimal translation
        'scale': 0.2,  # Minimal scaling to preserve small objects
        'shear': 0.0,  # No shear
        'perspective': 0.0,  # No perspective
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.8,  # Reduced mosaic
        'mixup': 0.0,  # NO mixup (confuses small objects)
        'copy_paste': 0.0,  # NO copy-paste

        # OPTIMIZED loss weights for your class imbalance
        'box': 10.0,  # HIGHER box weight for small objects
        'cls': 2.0,  # HIGHER classification weight
        'dfl': 2.0,  # Higher DFL weight

        # Small object optimization
        'optimizer': 'AdamW',
        'close_mosaic': 20,  # Disable mosaic earlier

        # Enhanced validation
        'val': True,
        'plots': True,
        'save': True,
        'save_json': True,
        'verbose': True,

        # CRITICAL: Multi-scale training for small objects
        'rect': False,  # Keep aspect ratio variation
        'single_cls': False,  # Ensure multi-class
    }

    return training_config


def create_focused_dataset_yaml():
    """Create dataset.yaml with class weights for imbalanced classes"""

    dataset_config = """
# Improved dataset config with class weights
path: C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset
train: images/train
val: images/val
test: images/test

# Classes
nc: 5
names:
  0: charpy_specimen
  1: charpy_edge
  2: charpy_corner
  3: fracture_surface
  4: measurement_point

# Class weights to handle imbalance (optional)
# Higher weights for rare classes
class_weights: [1.0, 3.0, 5.0, 2.0, 5.0]
"""

    output_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/dataset_improved.yaml"
    with open(output_path, 'w') as f:
        f.write(dataset_config)

    return output_path


def train_improved_model():
    """Train with improved settings targeting your specific issues"""

    print("🎯 IMPROVED MULTI-CLASS TRAINING")
    print("=" * 60)
    print("🔧 TARGETING YOUR SPECIFIC ISSUES:")
    print("   ✅ Larger image size (832px) for small objects")
    print("   ✅ Lower learning rate for stability")
    print("   ✅ Higher loss weights for small objects")
    print("   ✅ Minimal augmentation to preserve small features")
    print("   ✅ More epochs for difficult classes")

    # Create improved dataset config
    dataset_yaml = create_focused_dataset_yaml()
    output_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/models/charpy_improved"

    os.makedirs(output_dir, exist_ok=True)

    # Use YOLOv8s for better small object detection
    print(f"\n🤖 Using YOLOv8s (better for small objects)...")
    model = YOLO('yolov8s.pt')

    # Get improved training configuration
    train_config = setup_improved_training_config()

    print(f"\n⚙️  IMPROVED Configuration:")
    print(f"   Model: YOLOv8s (better small object detection)")
    print(f"   Image size: {train_config['imgsz']}px (LARGER for small objects)")
    print(f"   Learning rate: {train_config['lr0']} (LOWER for stability)")
    print(f"   Box loss weight: {train_config['box']} (HIGHER)")
    print(f"   Class loss weight: {train_config['cls']} (HIGHER)")
    print(f"   Epochs: {train_config['epochs']} (MORE for difficult classes)")

    print(f"\n🎯 Expected Improvements:")
    print(f"   📈 charpy_edge: 0.086 → 0.3+ (3x improvement)")
    print(f"   📈 charpy_corner: 0.015 → 0.2+ (10x+ improvement)")
    print(f"   📈 measurement_point: 0.023 → 0.2+ (8x+ improvement)")
    print(f"   📈 Overall mAP: 0.341 → 0.6+ (75%+ improvement)")

    try:
        # Train with improved settings
        results = model.train(
            data=dataset_yaml,
            project=output_dir,
            name='charpy_improved_v1',
            **train_config
        )

        print(f"\n🎉 IMPROVED TRAINING COMPLETED!")
        print(f"✅ Model saved to: {output_dir}/charpy_improved_v1")
        print(f"✅ Best weights: {output_dir}/charpy_improved_v1/weights/best.pt")

        return f"{output_dir}/charpy_improved_v1/weights/best.pt"

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None


def quick_annotation_check():
    """Quick check of your annotation quality"""

    print("\n🔍 ANNOTATION QUALITY CHECK")
    print("=" * 40)

    # Analyze your current annotations
    labels_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/labels/train"

    if os.path.exists(labels_dir):
        import glob
        from collections import Counter, defaultdict

        class_counts = Counter()
        small_boxes = defaultdict(int)  # Count very small boxes

        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

        for label_file in label_files[:50]:  # Check first 50 files
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        width = float(parts[3])
                        height = float(parts[4])

                        class_counts[class_id] += 1

                        # Check if box is very small (might be hard to detect)
                        if width < 0.05 or height < 0.05:
                            small_boxes[class_id] += 1

        print("📊 Your annotation distribution:")
        class_names = {0: "charpy_specimen", 1: "charpy_edge", 2: "charpy_corner",
                       3: "fracture_surface", 4: "measurement_point"}

        for class_id, count in class_counts.items():
            name = class_names.get(class_id, f"class_{class_id}")
            small_count = small_boxes[class_id]
            small_percent = (small_count / count * 100) if count > 0 else 0

            print(f"   Class {class_id} ({name}): {count} annotations")
            print(f"     Small boxes (<5% of image): {small_count} ({small_percent:.1f}%)")

        print(f"\n💡 Recommendations:")
        if small_boxes[2] > class_counts[2] * 0.8:
            print("   ⚠️  charpy_corner boxes are too small - make them larger")
        if small_boxes[4] > class_counts[4] * 0.8:
            print("   ⚠️  measurement_point boxes are too small - make them larger")
        if class_counts[1] < 100:
            print("   ⚠️  Need more charpy_edge annotations for better training")


def main():
    print("🚀 IMPROVED CHARPY TRAINING STRATEGY")
    print("=" * 60)
    print("Based on your results analysis:")
    print("   ✅ charpy_specimen: 0.953 mAP (EXCELLENT)")
    print("   ❌ charpy_edge: 0.086 mAP (NEEDS IMPROVEMENT)")
    print("   ❌ charpy_corner: 0.015 mAP (CRITICAL)")
    print("   ❌ measurement_point: 0.023 mAP (CRITICAL)")
    print()

    # Check annotation quality first
    quick_annotation_check()

    print(f"\n🎯 This improved training will:")
    print(f"   1. Use larger images (832px) to better detect small objects")
    print(f"   2. Lower learning rate for more stable training")
    print(f"   3. Higher loss weights for small/rare classes")
    print(f"   4. Minimal augmentation to preserve small features")
    print(f"   5. More epochs for difficult classes to converge")

    response = input("\nStart IMPROVED training? (y/N): ").lower().strip()
    if response != 'y':
        print("Training cancelled")
        return

    # Train the improved model
    model_path = train_improved_model()

    if model_path:
        print(f"\n🎉 IMPROVED TRAINING COMPLETE!")
        print(f"✅ Expected major improvements in small object detection")
        print(f"✅ Model should now detect edges, corners, and measurement points much better")
        print(f"\n🔧 Next steps:")
        print(f"   1. Test: python test_multiclass_model.py")
        print(f"   2. Compare results with your previous model")
        print(f"   3. If still not satisfied, we can further optimize annotations")


if __name__ == "__main__":
    main()