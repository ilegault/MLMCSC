#!/usr/bin/env python3
"""
Complete Charpy Specimen Training Pipeline

This script takes your 1500 captured images and trains a custom YOLOv8 model
specifically for Charpy specimen detection with high precision measurement capabilities.

Steps:
1. Organize your captured images
2. Create annotations (manual or semi-automatic)
3. Split into train/val/test sets
4. Train custom YOLOv8 model
5. Validate and export for production use
"""

import os
import shutil
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharpyTrainingPipeline:
    """Complete training pipeline for Charpy specimen detection."""

    def __init__(self,
                 raw_images_dir: str = "data/charpy_training_images",
                 dataset_dir: str = "data/charpy_dataset",
                 output_dir: str = "models/charpy_training"):
        """
        Initialize training pipeline.

        Args:
            raw_images_dir: Directory with your 1500 captured images
            dataset_dir: Directory for organized YOLO dataset
            output_dir: Directory for training outputs
        """
        self.raw_images_dir = Path(raw_images_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)

        # Create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Charpy-specific classes for detection and measurement
        self.charpy_classes = [
            'charpy_specimen',  # 0: Complete specimen outline
            'charpy_notch',  # 1: V-notch area for measurement
            'charpy_edge',  # 2: Specimen edges for length/width
            'charpy_corner',  # 3: Corner points for precise measurement
            'fracture_surface',  # 4: Broken/fractured areas (if applicable)
            'measurement_point'  # 5: Key dimensional reference points
        ]

        logger.info(f"🔬 Charpy Training Pipeline Initialized")
        logger.info(f"📁 Raw images: {self.raw_images_dir}")
        logger.info(f"📁 Dataset: {self.dataset_dir}")
        logger.info(f"📁 Output: {self.output_dir}")

    def step1_organize_images(self) -> Dict[str, int]:
        """Step 1: Organize your 1500 images into proper structure."""
        logger.info("📋 Step 1: Organizing captured images...")

        # Find all captured images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(self.raw_images_dir.glob(ext)))

        logger.info(f"Found {len(image_files)} images")

        # Create organized structure
        organized_dir = self.dataset_dir / "organized"
        organized_dir.mkdir(exist_ok=True)

        # Copy and rename images systematically
        stats = {'copied': 0, 'skipped': 0, 'errors': 0}

        for i, img_file in enumerate(tqdm(image_files, desc="Organizing images")):
            try:
                # Load image to verify it's valid
                img = cv2.imread(str(img_file))
                if img is None:
                    stats['skipped'] += 1
                    continue

                # Create systematic filename
                new_filename = f"charpy_{i:04d}.jpg"
                new_path = organized_dir / new_filename

                # Copy image
                shutil.copy2(img_file, new_path)

                # Copy metadata if exists
                json_file = img_file.with_suffix('.json')
                if json_file.exists():
                    new_json = new_path.with_suffix('.json')
                    shutil.copy2(json_file, new_json)

                stats['copied'] += 1

            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                stats['errors'] += 1

        logger.info(f"✅ Organized {stats['copied']} images")
        logger.info(f"⚠️ Skipped {stats['skipped']} invalid images")
        logger.info(f"❌ Errors: {stats['errors']}")

        return stats

    def step2_create_annotation_tool(self) -> str:
        """Step 2: Create annotation interface for your images."""
        logger.info("🏷️ Step 2: Setting up annotation tool...")

        annotation_script = '''#!/usr/bin/env python3
"""
Charpy Specimen Annotation Tool
Annotate your 1500 images for training.
"""

import sys
from pathlib import Path
sys.path.append('src')

from models.annotation_utils import SimpleAnnotationTool

def main():
    """Run Charpy annotation tool."""
    # Charpy-specific classes
    charpy_classes = [
        'charpy_specimen',      # Complete specimen outline
        'charpy_notch',         # V-notch area  
        'charpy_edge',          # Specimen edges
        'charpy_corner',        # Corner points
        'fracture_surface',     # Broken areas
        'measurement_point'     # Reference points
    ]

    print("Charpy Specimen Annotation Tool")
    print("=" * 50)
    print("Annotation Guidelines:")
    print("1. charpy_specimen - Draw box around entire specimen")
    print("2. charpy_notch - Draw tight box around V-notch")
    print("3. charpy_edge - Mark specimen edges for measurement")
    print("4. charpy_corner - Mark corner points precisely")
    print("5. fracture_surface - Mark broken/fractured areas")
    print("6. measurement_point - Mark key dimensional points")
    print("=" * 50)
    print()

    # Start annotation tool
    tool = SimpleAnnotationTool(charpy_classes)
    tool.run()

if __name__ == "__main__":
    main()
'''

        # Save annotation script
        annotation_file = self.dataset_dir / "annotate_charpy.py"
        with open(annotation_file, 'w', encoding='utf-8') as f:
            f.write(annotation_script)

        logger.info(f"✅ Annotation tool created: {annotation_file}")
        logger.info("📝 To annotate your images:")
        logger.info(f"   cd {self.dataset_dir}")
        logger.info("   python annotate_charpy.py")

        return str(annotation_file)

    def step3_create_quick_annotations(self) -> Dict[str, int]:
        """Step 3: Create basic annotations automatically (to get started faster)."""
        logger.info("🤖 Step 3: Creating basic auto-annotations...")

        organized_dir = self.dataset_dir / "organized"
        annotations_dir = self.dataset_dir / "auto_annotations"
        annotations_dir.mkdir(exist_ok=True)

        image_files = list(organized_dir.glob("*.jpg"))
        stats = {'processed': 0, 'annotations_created': 0}

        for img_file in tqdm(image_files[:100], desc="Auto-annotating (first 100)"):  # Start with 100 for testing
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Create basic annotation - assume specimen takes up most of frame
                # This is just a starting point - you'll refine these manually
                center_x = 0.5
                center_y = 0.5
                width = 0.7  # Assume specimen is ~70% of frame width
                height = 0.8  # Assume specimen is ~80% of frame height

                # Create YOLO format annotation
                annotation_text = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"

                # Save annotation file
                annotation_file = annotations_dir / f"{img_file.stem}.txt"
                with open(annotation_file, 'w') as f:
                    f.write(annotation_text)

                stats['annotations_created'] += 1
                stats['processed'] += 1

            except Exception as e:
                logger.error(f"Error auto-annotating {img_file}: {e}")

        logger.info(f"✅ Created {stats['annotations_created']} basic annotations")
        logger.info("📝 These are starting points - refine them manually for best results")

        return stats

    def step4_prepare_dataset(self,
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.2,
                              test_ratio: float = 0.1) -> Dict[str, int]:
        """Step 4: Split dataset into train/val/test sets."""
        logger.info("📊 Step 4: Preparing dataset splits...")

        # Find annotated images (either manual or auto)
        annotations_dir = self.dataset_dir / "annotations"  # Manual annotations
        auto_annotations_dir = self.dataset_dir / "auto_annotations"  # Auto annotations

        # Use manual annotations if available, otherwise auto
        if annotations_dir.exists() and list(annotations_dir.glob("*.txt")):
            source_annotations = annotations_dir
            logger.info("Using manual annotations")
        else:
            source_annotations = auto_annotations_dir
            logger.info("Using auto-generated annotations")

        # Get all annotated images
        annotation_files = list(source_annotations.glob("*.txt"))
        image_files = []

        organized_dir = self.dataset_dir / "organized"
        for ann_file in annotation_files:
            img_file = organized_dir / f"{ann_file.stem}.jpg"
            if img_file.exists():
                image_files.append(img_file)

        logger.info(f"Found {len(image_files)} annotated images")

        # Shuffle for random split
        random.shuffle(image_files)

        # Calculate split sizes
        total = len(image_files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size

        # Split files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]

        # Create YOLO dataset structure
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            # Create directories
            images_dir = self.dataset_dir / 'images' / split_name
            labels_dir = self.dataset_dir / 'labels' / split_name
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for img_file in tqdm(files, desc=f"Preparing {split_name} set"):
                # Copy image
                shutil.copy2(img_file, images_dir / img_file.name)

                # Copy annotation
                ann_file = source_annotations / f"{img_file.stem}.txt"
                if ann_file.exists():
                    shutil.copy2(ann_file, labels_dir / f"{img_file.stem}.txt")

        # Create dataset.yaml
        dataset_config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.charpy_classes),
            'names': {i: name for i, name in enumerate(self.charpy_classes)}
        }

        dataset_yaml = self.dataset_dir / 'dataset.yaml'
        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        stats = {
            'total': total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }

        logger.info(f"✅ Dataset prepared:")
        logger.info(f"   📚 Train: {stats['train']} images")
        logger.info(f"   🔍 Val: {stats['val']} images")
        logger.info(f"   🧪 Test: {stats['test']} images")
        logger.info(f"   📋 Config: {dataset_yaml}")

        return stats

    def step5_train_model(self,
                          model_size: str = 's',
                          epochs: int = 100,
                          imgsz: int = 1024,
                          batch_size: int = 8) -> Dict[str, str]:
        """Step 5: Train the custom Charpy model."""
        logger.info("🚀 Step 5: Training custom Charpy model...")

        dataset_yaml = self.dataset_dir / 'dataset.yaml'
        if not dataset_yaml.exists():
            raise FileNotFoundError("Dataset not prepared. Run step4_prepare_dataset first.")

        # Initialize model
        model_name = f'yolov8{model_size}.pt'
        model = YOLO(model_name)

        logger.info(f"📦 Using model: {model_name}")
        logger.info(f"🎯 Training parameters:")
        logger.info(f"   📏 Image size: {imgsz}")
        logger.info(f"   📊 Batch size: {batch_size}")
        logger.info(f"   🔄 Epochs: {epochs}")

        # Check if CUDA is available, otherwise use CPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ Using device: {device}")

        # Adjust batch size for CPU training
        if device == 'cpu':
            batch_size = min(batch_size, 4)  # Reduce batch size for CPU
            logger.info(f"📊 Adjusted batch size for CPU: {batch_size}")

        # Training parameters optimized for Charpy specimens
        training_args = {
            'data': str(dataset_yaml),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,  # Use detected device
            'project': str(self.output_dir),
            'name': f'charpy_v8{model_size}',
            'save': True,
            'save_period': 10,
            'cache': True,
            'workers': 4,
            'patience': 50,

            # Optimized for precision measurement
            'augment': True,
            'degrees': 5.0,  # Small rotations (specimens usually aligned)
            'translate': 0.05,  # Minimal translation
            'scale': 0.1,  # Minimal scaling (preserve dimensions)
            'shear': 2.0,  # Small shear
            'perspective': 0.0,  # No perspective (top-down view)
            'mosaic': 0.5,  # Reduced mosaic
            'mixup': 0.0,  # No mixup (preserve specimen integrity)

            # Loss weights for measurement precision
            'box': 7.5,  # Higher box precision
            'cls': 0.5,  # Standard classification
            'dfl': 1.5,  # Distribution focal loss

            # Optimization
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,

            # Validation
            'val': True,
            'plots': True,
            'save_json': True
        }

        # Start training
        logger.info("🏋️ Starting training...")
        results = model.train(**training_args)

        # Validate the model
        logger.info("🔍 Validating model...")
        val_results = model.val()

        # Export for deployment
        logger.info("📦 Exporting model...")
        model.export(format='onnx', dynamic=True, simplify=True)

        # Prepare results
        results_info = {
            'best_model': f"{results.save_dir}/weights/best.pt",
            'last_model': f"{results.save_dir}/weights/last.pt",
            'onnx_model': f"{results.save_dir}/weights/best.onnx",
            'results_dir': str(results.save_dir),
            'map50': float(val_results.box.map50),
            'map50_95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr)
        }

        logger.info("🎉 Training completed!")
        logger.info(f"📊 Results:")
        logger.info(f"   🎯 mAP@50: {results_info['map50']:.3f}")
        logger.info(f"   🎯 mAP@50-95: {results_info['map50_95']:.3f}")
        logger.info(f"   📈 Precision: {results_info['precision']:.3f}")
        logger.info(f"   📉 Recall: {results_info['recall']:.3f}")
        logger.info(f"   💾 Best model: {results_info['best_model']}")

        return results_info

    def step6_test_model(self, model_path: str) -> Dict[str, float]:
        """Step 6: Test the trained model on sample images."""
        logger.info("🧪 Step 6: Testing trained model...")

        # Load trained model
        model = YOLO(model_path)

        # Get test images
        test_images_dir = self.dataset_dir / 'images' / 'test'
        test_images = list(test_images_dir.glob("*.jpg"))[:10]  # Test on 10 images

        logger.info(f"Testing on {len(test_images)} images...")

        results_dir = self.output_dir / "test_results"
        results_dir.mkdir(exist_ok=True)

        total_detections = 0
        total_confidence = 0

        for img_file in test_images:
            # Run inference
            results = model(str(img_file))

            # Load image for visualization
            img = cv2.imread(str(img_file))

            # Draw results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # Draw bounding box
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Draw label
                        label = f"{self.charpy_classes[cls]}: {conf:.2f}"
                        cv2.putText(img, label, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        total_detections += 1
                        total_confidence += conf

            # Save result
            result_file = results_dir / f"test_{img_file.name}"
            cv2.imwrite(str(result_file), img)

        avg_confidence = total_confidence / max(total_detections, 1)

        test_stats = {
            'images_tested': len(test_images),
            'total_detections': total_detections,
            'avg_confidence': avg_confidence,
            'detections_per_image': total_detections / len(test_images)
        }

        logger.info(f"✅ Testing completed:")
        logger.info(f"   📸 Images tested: {test_stats['images_tested']}")
        logger.info(f"   🎯 Total detections: {test_stats['total_detections']}")
        logger.info(f"   📊 Avg confidence: {test_stats['avg_confidence']:.3f}")
        logger.info(f"   📈 Detections/image: {test_stats['detections_per_image']:.1f}")
        logger.info(f"   📁 Results saved to: {results_dir}")

        return test_stats

    def run_complete_pipeline(self) -> Dict[str, any]:
        """Run the complete training pipeline."""
        logger.info("🚀 Starting Complete Charpy Training Pipeline...")
        logger.info("=" * 60)

        results = {}

        try:
            # Step 1: Organize images
            results['step1'] = self.step1_organize_images()

            # Step 2: Create annotation tool
            results['step2'] = self.step2_create_annotation_tool()

            # Step 3: Create basic annotations
            results['step3'] = self.step3_create_quick_annotations()

            # Step 4: Prepare dataset
            results['step4'] = self.step4_prepare_dataset()

            # Step 5: Train model
            results['step5'] = self.step5_train_model()

            # Step 6: Test model
            best_model = results['step5']['best_model']
            results['step6'] = self.step6_test_model(best_model)

            logger.info("🎉 Complete pipeline finished successfully!")
            logger.info("=" * 60)
            logger.info("📋 SUMMARY:")
            logger.info(f"   📁 Dataset: {results['step4']['total']} images")
            logger.info(f"   🎯 Model mAP@50: {results['step5']['map50']:.3f}")
            logger.info(f"   💾 Best model: {results['step5']['best_model']}")
            logger.info(f"   🧪 Test confidence: {results['step6']['avg_confidence']:.3f}")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

        return results


def main():
    """Main function to run training pipeline."""
    print("🔬 Charpy Specimen Training Pipeline")
    print("This will train a custom YOLOv8 model on your 1500 captured images")
    print()

    # Initialize pipeline
    pipeline = CharpyTrainingPipeline(
        raw_images_dir="data/charpy_training_images",  # Your 1500 images
        dataset_dir="data/charpy_dataset",  # Organized dataset
        output_dir="models/charpy_training"  # Training outputs
    )

    # Run complete pipeline
    results = pipeline.run_complete_pipeline()

    print("\n🎉 Training Complete!")
    print("Next steps:")
    print("1. Review test results in models/charpy_training/test_results/")
    print("2. Use the trained model for detection:")
    print(f"   model_path = '{results['step5']['best_model']}'")
    print("3. Integrate with your measurement system")


if __name__ == "__main__":
    main()