#!/usr/bin/env python3
"""
Training Script for Manually Annotated Charpy Dataset
Use this since you've already annotated measurement edges with angle_aware_annotator
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml


class TrainWithExistingAnnotations:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def step2_verify_annotations(self):
        """Verify and analyze your annotations"""
        print("\n📊 STEP 2: Verifying annotations...")

        class_counts = {}
        total_annotations = 0

        for split in ['train', 'val', 'test']:
            labels_dir = self.dataset_path / 'labels' / split
            if not labels_dir.exists():
                continue

            split_counts = {}
            split_total = 0

            for label_file in labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id not in split_counts:
                                split_counts[class_id] = 0
                            split_counts[class_id] += 1
                            split_total += 1

                            if class_id not in class_counts:
                                class_counts[class_id] = 0
                            class_counts[class_id] += 1
                            total_annotations += 1

            print(f"\n{split.upper()} split:")
            print(f"  Total annotations: {split_total}")
            for class_id, count in sorted(split_counts.items()):
                print(f"  Class {class_id}: {count} annotations")

        print(f"\nTOTAL: {total_annotations} annotations")
        return class_counts

    def step3_create_optimized_config(self):
        """Create dataset configuration optimized for your annotations"""
        print("\n📝 STEP 3: Creating optimized dataset configuration...")

        # Since you have 5 classes based on your confusion matrix
        dataset_yaml = f"""
# Charpy Dataset with Manual Annotations
path: {self.dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes from your dataset
nc: 4
names:
  0: charpy_specimen
  1: charpy_edge         # This should be your measurement edges
  2: charpy_corner
  3: fracture_surface

# Class weights to boost edge detection
# Higher weight for measurement edges since they're thin and harder to detect
class_weights: [1.0, 3.0, 2.0, 1.0, 2.0]
"""

        yaml_path = self.dataset_path / "dataset_corner_measurement.yaml"
        with open(yaml_path, 'w') as f:
            f.write(dataset_yaml)

        print(f"✅ Created dataset config: {yaml_path}")
        return str(yaml_path)

    def step4_train_final_model(self, dataset_yaml):
        """Train the final model with your manual annotations"""
        print("\n🚀 STEP 4: Training final model...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Use YOLOv8m for better detection of thin edges
        model = YOLO('yolov8m.pt')

        # Optimized configuration for edge detection
        train_config = {
            'data': dataset_yaml,
            'epochs': 200,
            'imgsz': 1280,  # Large image size for thin edges
            'batch': 8 if device == 'cpu' else 16,
            'device': device,
            'project': 'models/charpy_final',
            'name': 'with_measurement_edges',

            # Learning rate settings
            'lr0': 0.001,  # Lower initial learning rate
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,

            # Loss weights optimized for edges
            'box': 10.0,  # High box weight for precise edge localization
            'cls': 2.0,  # Classification weight
            'dfl': 2.0,  # Distribution focal loss

            # Minimal augmentation to preserve edge features
            'degrees': 10.0,  # Some rotation since you have angled specimens
            'translate': 0.1,
            'scale': 0.2,
            'shear': 2.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.0,
            'copy_paste': 0.0,

            # Color augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.3,
            'hsv_v': 0.3,

            # Other settings
            'conf': 0.001,
            'iou': 0.5,
            'max_det': 100,
            'patience': 30,
            'save': True,
            'save_period': 20,
            'val': True,
            'plots': True,
            'cache': 'ram' if device != 'cpu' else False,

            # Multi-scale training for better edge detection
            'rect': False,
            'single_cls': False,
            'overlap_mask': True,
            'mask_ratio': 4,

            # Optimizer
            'optimizer': 'AdamW',
            'cos_lr': True,  # Cosine learning rate schedule
        }

        print("Training with settings optimized for edge detection...")
        results = model.train(**train_config)

        print(f"\n✅ Training completed!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")

        # Test the model
        print("\n🧪 Testing the trained model...")
        test_model = YOLO(f"{results.save_dir}/weights/best.pt")

        # Test on a few images
        test_images_dir = self.dataset_path / 'images' / 'test'
        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*.jpg"))[:3]

            for test_img in test_images:
                results = test_model(str(test_img), conf=0.25, save=True)
                print(f"Tested on: {test_img.name}")

        return f"{results.save_dir}/weights/best.pt"


def main():
    print("🎯 TRAINING WITH YOUR MANUAL ANNOTATIONS")
    print("=" * 60)
    print("Since you've already annotated measurement edges using angle_aware_annotator,")
    print("we'll train a model optimized for detecting those thin angled edges.")
    print()

    dataset_path = "data/charpy_dataset"
    trainer = TrainWithExistingAnnotations(dataset_path)

    # Step 2: Verify annotations
    class_counts = trainer.step2_verify_annotations()

    if 1 not in class_counts:
        print("\n⚠️ WARNING: No class 1 (measurement_edge) annotations found!")
        print("Make sure your angle_aware_annotator saved annotations with class ID 1")
        return

    # Step 3: Create config
    dataset_yaml = trainer.step3_create_optimized_config()

    # Step 4: Train
    print("\n" + "=" * 60)
    response = input("Ready to start training? (y/n): ")
    if response.lower() == 'y':
        model_path = trainer.step4_train_final_model(dataset_yaml)

        print("\n🎉 TRAINING COMPLETE!")
        print("\nNext steps:")
        print(f"1. Test your model: {model_path}")
        print("2. Use lower confidence (0.2-0.3) for edge detection")
        print("3. The measurement edges should now be detected!")
    else:
        print("Training cancelled")


if __name__ == "__main__":
    main()