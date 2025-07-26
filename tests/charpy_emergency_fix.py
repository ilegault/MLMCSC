#!/usr/bin/env python3
"""
Emergency Fix for Charpy Model - Quick Annotation Generator
This will help you quickly create better annotations for your images
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


class CharpyQuickFix:
    def __init__(self):
        self.classes = {
            'charpy_specimen': 0,
            'charpy_notch': 1,
            'charpy_edge': 2,
            'charpy_corner': 3,
            'fracture_surface': 4,
            'measurement_point': 5
        }

    def analyze_current_annotations(self, labels_dir):
        """Check what's wrong with current annotations."""
        print("🔍 Analyzing current annotations...")

        labels_path = Path(labels_dir)
        if not labels_path.exists():
            print(f"❌ Labels directory not found: {labels_dir}")
            return

        total_files = 0
        total_annotations = 0
        class_counts = {i: 0 for i in range(6)}

        for label_file in labels_path.glob("*.txt"):
            total_files += 1
            with open(label_file, 'r') as f:
                lines = f.readlines()
                total_annotations += len(lines)
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1

        print(f"\n📊 Current Dataset Analysis:")
        print(f"  Files with annotations: {total_files}")
        print(f"  Total annotations: {total_annotations}")
        print(f"  Avg annotations per image: {total_annotations / max(total_files, 1):.1f}")

        print(f"\n📈 Class Distribution:")
        class_names = ['specimen', 'notch', 'edge', 'corner', 'fracture', 'measurement']
        for i, name in enumerate(class_names):
            count = class_counts[i]
            percentage = (count / max(total_annotations, 1)) * 100
            print(f"  {i}: {name:12} - {count:4d} ({percentage:.1f}%)")

        # Diagnosis
        print(f"\n🩺 Diagnosis:")
        if class_counts[1] == 0:  # No notch annotations
            print("  ❌ CRITICAL: No notch annotations found!")
            print("     This is why your model can't detect notches")
        if total_annotations / max(total_files, 1) < 2:
            print("  ❌ Too few annotations per image")
            print("     Each image needs 3-6 annotations minimum")
        if class_counts[0] == total_annotations:
            print("  ❌ Only annotating whole specimens")
            print("     Need to annotate individual features")

    def create_smart_annotations(self, image_path, output_dir, visualize=False):
        """Create intelligent annotations based on image analysis."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        annotations = []

        # 1. Detect the specimen area (find the non-background region)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold to find specimen
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (should be specimen)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)

            # Convert to YOLO format
            cx = (x + w_box / 2) / w
            cy = (y + h_box / 2) / h
            width = w_box / w
            height = h_box / h

            # 1. Add specimen annotation
            annotations.append(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")

            # 2. Estimate notch location (typically in center of specimen)
            # For Charpy specimens, notch is usually in the middle
            notch_cx = cx
            notch_cy = cy
            notch_width = width * 0.2  # Notch is about 20% of specimen width
            notch_height = height * 0.25  # And 25% of height
            annotations.append(f"1 {notch_cx:.6f} {notch_cy:.6f} {notch_width:.6f} {notch_height:.6f}")

            # 3. Add edge annotations (left and right sides)
            edge_width = 0.05  # Thin boxes for edges
            # Left edge
            left_cx = (x + edge_width * w / 2) / w
            annotations.append(f"2 {left_cx:.6f} {cy:.6f} {edge_width:.6f} {height:.6f}")

            # Right edge
            right_cx = (x + w_box - edge_width * w / 2) / w
            annotations.append(f"2 {right_cx:.6f} {cy:.6f} {edge_width:.6f} {height:.6f}")

            # 4. Add corner annotations
            corner_size = 0.05
            # Top-left corner
            tl_cx = (x + corner_size * w / 2) / w
            tl_cy = (y + corner_size * h / 2) / h
            annotations.append(f"3 {tl_cx:.6f} {tl_cy:.6f} {corner_size:.6f} {corner_size:.6f}")

            # Top-right corner
            tr_cx = (x + w_box - corner_size * w / 2) / w
            tr_cy = (y + corner_size * h / 2) / h
            annotations.append(f"3 {tr_cx:.6f} {tr_cy:.6f} {corner_size:.6f} {corner_size:.6f}")

            if visualize:
                vis_img = img.copy()
                colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

                for ann in annotations:
                    parts = ann.split()
                    class_id = int(parts[0])
                    cx, cy, w_norm, h_norm = map(float, parts[1:])

                    # Convert back to pixel coordinates
                    x1 = int((cx - w_norm / 2) * w)
                    y1 = int((cy - h_norm / 2) * h)
                    x2 = int((cx + w_norm / 2) * w)
                    y2 = int((cy + h_norm / 2) * h)

                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), colors[class_id], 2)

                # Save visualization
                vis_path = Path(output_dir) / f"annotated_{Path(image_path).name}"
                cv2.imwrite(str(vis_path), vis_img)

        return annotations

    def batch_create_annotations(self, images_dir, output_dir, num_images=100):
        """Create annotations for a batch of images."""
        print(f"\n🤖 Creating smart annotations for {num_images} images...")

        images_path = Path(images_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list(images_path.glob("*.jpg"))[:num_images]

        success_count = 0

        for img_file in tqdm(image_files, desc="Annotating"):
            annotations = self.create_smart_annotations(
                img_file,
                output_dir,
                visualize=(success_count < 10)  # Visualize first 10
            )

            if annotations:
                # Save annotations
                label_file = output_path / f"{img_file.stem}.txt"
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(annotations))
                success_count += 1

        print(f"✅ Created annotations for {success_count} images")
        print(f"📁 Saved to: {output_path}")
        print(f"🖼️ Check visualizations in: {output_path}")

    def create_improved_dataset_yaml(self, dataset_dir):
        """Create an improved dataset.yaml file."""
        dataset_dir = Path(dataset_dir)

        yaml_content = f"""# Charpy Specimen Detection Dataset - FIXED
path: {dataset_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 6

# Class names - PROPERLY DEFINED
names:
  0: charpy_specimen
  1: charpy_notch
  2: charpy_edge
  3: charpy_corner
  4: fracture_surface
  5: measurement_point

# Training notes
# - Focus on notch detection (class 1) - most critical
# - Each image should have multiple annotations
# - Use conf=0.25 during training
"""

        yaml_path = dataset_dir / "dataset_fixed.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        print(f"✅ Created improved dataset config: {yaml_path}")
        return str(yaml_path)

    def quick_retrain_script(self, dataset_yaml):
        """Generate a quick retrain script with optimal settings."""
        script = f"""#!/usr/bin/env python3
# Quick Retrain Script for Charpy Model

from ultralytics import YOLO
import torch

print("🚀 Starting Charpy Model Retraining with Fixed Annotations")

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {{device}}")

# Load model (using small model for better feature detection)
model = YOLO('yolov8s.pt')

# Train with optimized settings for multi-class detection
results = model.train(
    data='{dataset_yaml}',
    epochs=100,  # Reduced for faster iteration
    imgsz=640,
    batch=8 if device == 'cpu' else 16,
    device=device,

    # Critical settings for multi-class detection
    conf=0.25,  # Lower confidence threshold
    cls=2.0,    # Higher classification loss weight
    box=7.5,    # High box regression weight

    # Augmentation settings
    augment=True,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.3,

    # Save settings
    save=True,
    save_period=20,
    patience=30,

    # Validation
    val=True,
    plots=True,

    # Output
    project='models/charpy_fixed',
    name='multi_class_v1',
    exist_ok=True
)

# Test the model
print("\\n🧪 Testing trained model...")
test_model = YOLO(results.save_dir + '/weights/best.pt')

# Run on test image with low confidence to see all detections
test_results = test_model('data/charpy_dataset/images/test/charpy_0001.jpg', 
                         conf=0.25, save=True)

print("\\n✅ Training complete!")
print(f"Best model saved at: {{results.save_dir}}/weights/best.pt")
"""

        script_path = Path("quick_retrain_charpy.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        print(f"✅ Created retrain script: {script_path}")
        return script_path


def main():
    """Emergency fix for Charpy model."""
    print("🚨 CHARPY MODEL EMERGENCY FIX")
    print("=" * 60)

    fixer = CharpyQuickFix()

    # Step 1: Analyze current situation
    print("\n📊 STEP 1: Analyzing current annotations...")
    fixer.analyze_current_annotations("data/charpy_dataset/labels/train")

    # Step 2: Create better annotations
    print("\n🔧 STEP 2: Creating better annotations...")
    response = input("\nCreate new annotations? (y/n): ")

    if response.lower() == 'y':
        # Create new annotations
        fixer.batch_create_annotations(
            "data/charpy_dataset/images/train",
            "data/charpy_dataset/labels_fixed/train",
            num_images=100
        )

        # Also for validation
        fixer.batch_create_annotations(
            "data/charpy_dataset/images/val",
            "data/charpy_dataset/labels_fixed/val",
            num_images=30
        )

    # Step 3: Create new dataset config
    print("\n📝 STEP 3: Creating improved dataset configuration...")
    dataset_yaml = fixer.create_improved_dataset_yaml("data/charpy_dataset")

    # Step 4: Generate retrain script
    print("\n🚀 STEP 4: Generating retrain script...")
    script_path = fixer.quick_retrain_script(dataset_yaml)

    print("\n✅ EMERGENCY FIX COMPLETE!")
    print("\n📋 Next steps:")
    print("1. Review the visualized annotations in data/charpy_dataset/labels_fixed/train/")
    print("2. If they look good, copy labels_fixed to labels (backup old ones first!)")
    print("3. Run: python quick_retrain_charpy.py")
    print("4. Test the new model with lower confidence (conf=0.25)")

    print("\n💡 Pro tip: The notch annotations are estimated - refine them manually for best results!")


if __name__ == "__main__":
    main()