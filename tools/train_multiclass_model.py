#!/usr/bin/env python3
"""
Charpy 3-Class Detection Model Training Script
GPU-optimized training with best practices for high success rate

Classes:
0. charpy_specimen - Full specimen bounding box
1. charpy_corner - Corner points (4 per specimen)
2. fracture_surface - The fracture surface area

This script includes:
- GPU optimization and memory management
- Advanced data augmentation for microscopy
- Multi-scale training
- Best hyperparameters for small object detection
- Comprehensive logging and monitoring
- Model validation and export
"""

import os
import sys
import torch
import yaml
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CharpyModelTrainer:
    """Advanced trainer for Charpy 3-class detection model."""

    def __init__(self, dataset_path, output_dir="models/detection"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        self.device = self._setup_gpu()
        self.model_name = f"charpy_3class_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Class information
        self.classes = {
            0: "charpy_specimen",
            1: "charpy_corner",
            2: "fracture_surface"
        }

        # Training parameters optimized for your use case
        self.training_config = {
            'model_size': 'm',  # Medium model for 4GB GPU
            'epochs': 300,  # Extended training for convergence
            'patience': 50,  # Early stopping patience
            'batch_size': 8,  # Reduced for 4GB GPU memory
            'imgsz': 640,  # Standard YOLO size
            'multi_scale': True,
            'workers': 4,  # Reduced workers to save memory
            'device': self.device
        }

        logger.info(f"Initialized CharpyModelTrainer")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    def _setup_gpu(self):
        """Setup and validate GPU availability."""
        if not torch.cuda.is_available():
            logger.warning("GPU not available, training will be slow!")
            return 'cpu'

        # Get GPU info
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use all available GPUs
        if gpu_count > 1:
            logger.info(f"Using all {gpu_count} GPUs for training")
            return '0,1,2,3,4,5,6,7'[:gpu_count * 2 - 1]  # Format: '0,1,2' for multiple GPUs
        else:
            return '0'  # Single GPU as string

    def validate_dataset(self):
        """Validate dataset structure and annotations."""
        logger.info("Validating dataset...")

        splits = ['train', 'val', 'test']
        dataset_stats = {}

        for split in splits:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split

            if not images_dir.exists() or not labels_dir.exists():
                logger.error(f"Missing {split} directory!")
                return False

            # Count files
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            label_files = list(labels_dir.glob('*.txt'))

            # Validate annotations
            class_counts = {0: 0, 1: 0, 2: 0}
            total_boxes = 0

            for label_file in label_files:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id in class_counts:
                                class_counts[class_id] += 1
                                total_boxes += 1

            dataset_stats[split] = {
                'images': len(image_files),
                'labels': len(label_files),
                'total_boxes': total_boxes,
                'class_distribution': class_counts
            }

            logger.info(f"{split.upper()} set:")
            logger.info(f"  Images: {len(image_files)}")
            logger.info(f"  Labels: {len(label_files)}")
            logger.info(f"  Total annotations: {total_boxes}")
            logger.info(f"  Class distribution: {class_counts}")

        # Save dataset statistics
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)

        # Check for balanced classes
        train_classes = dataset_stats['train']['class_distribution']
        if train_classes[1] < train_classes[0] * 3:  # Expect ~4 corners per specimen
            logger.warning("Low corner annotations - consider adding more")

        if train_classes[2] < train_classes[0] * 0.5:  # Expect fracture surface for most specimens
            logger.warning("Low fracture surface annotations - critical for measurement!")

        return True

    def create_optimized_config(self):
        """Create optimized dataset configuration for training."""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 3,
            'names': self.classes,

            # Add dataset-specific parameters
            'overlap_thresh': 0.5,  # For corner detection
            'conf_thresh': 0.25,  # Lower threshold for small objects
        }

        config_file = self.dataset_path / 'dataset_optimized.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_file

    def get_training_hyperparameters(self):
        """Get optimized hyperparameters for Charpy detection."""
        hyperparams = {
            # Model configuration
            'data': str(self.create_optimized_config()),
            'epochs': self.training_config['epochs'],
            'patience': self.training_config['patience'],
            'batch': self.training_config['batch_size'],
            'imgsz': self.training_config['imgsz'],
            'device': self.training_config['device'],
            'workers': self.training_config['workers'],

            # Save and logging
            'project': str(self.output_dir),
            'name': self.model_name,
            'exist_ok': False,
            'save': True,
            'save_period': 10,
            'save_json': True,
            'save_hybrid': False,
            'plots': True,
            'verbose': True,

            # Optimization
            'optimizer': 'AdamW',  # Better for fine-tuning
            'lr0': 0.001,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # Loss weights - adjusted for multi-class
            'box': 7.5,  # Box loss weight
            'cls': 1.5,  # Classification loss weight (increased for 3 classes)
            'dfl': 1.5,  # Distribution focal loss weight

            # Augmentation - optimized for microscopy
            'degrees': 15.0,  # Rotation
            'translate': 0.1,  # Translation
            'scale': 0.3,  # Scaling (reduced for consistent microscope view)
            'shear': 5.0,  # Shear
            'perspective': 0.0,  # No perspective (microscope is orthogonal)
            'flipud': 0.5,  # Vertical flip
            'fliplr': 0.5,  # Horizontal flip
            'bgr': 0.0,  # No BGR augmentation
            'mosaic': 0.5,  # Mosaic augmentation (reduced)
            'mixup': 0.1,  # Mixup augmentation
            'copy_paste': 0.0,  # No copy-paste for microscopy

            # Color augmentation - adjusted for microscope lighting
            'hsv_h': 0.015,  # Hue variation
            'hsv_s': 0.5,  # Saturation variation
            'hsv_v': 0.3,  # Value (brightness) variation

            # Advanced options
            'cos_lr': True,  # Cosine learning rate scheduler
            'close_mosaic': 200,  # Disable mosaic for last epochs
            'resume': False,
            'amp': True,  # Automatic mixed precision for faster training
            'fraction': 1.0,  # Use full dataset
            'profile': False,
            'freeze': None,  # No layer freezing
            'multi_scale': True,  # Multi-scale training
            'single_cls': False,  # Multi-class detection
            'val': True,  # Run validation
            'rect': True,  # Rectangular training

            # NMS parameters
            'iou': 0.5,  # NMS IoU threshold
            'max_det': 100,  # Max detections per image

            # Label options
            'label_smoothing': 0.0,
            'nbs': 64,  # Nominal batch size
        }

        return hyperparams

    def train_model(self):
        """Train the Charpy detection model with GPU optimization."""
        logger.info("=" * 60)
        logger.info("STARTING CHARPY 3-CLASS MODEL TRAINING")
        logger.info("=" * 60)

        # Validate dataset
        if not self.validate_dataset():
            logger.error("Dataset validation failed!")
            return None

        # Select base model
        model_size = self.training_config['model_size']
        base_model = f'yolov8{model_size}.pt'

        logger.info(f"Loading base model: {base_model}")
        model = YOLO(base_model)

        # Get hyperparameters
        hyperparams = self.get_training_hyperparameters()

        # Log hyperparameters
        logger.info("Training hyperparameters:")
        for key, value in hyperparams.items():
            if key not in ['data', 'project', 'name']:
                logger.info(f"  {key}: {value}")

        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")

        # Train the model
        try:
            logger.info("Starting training...")
            results = model.train(**hyperparams)

            logger.info("Training completed successfully!")

            # Save best model path
            best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
            logger.info(f"Best model saved at: {best_model_path}")

            return results, best_model_path

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None, None

    def validate_model(self, model_path):
        """Validate the trained model."""
        logger.info("=" * 60)
        logger.info("VALIDATING TRAINED MODEL")
        logger.info("=" * 60)

        model = YOLO(model_path)

        # Run validation on test set
        results = model.val(
            data=str(self.create_optimized_config()),
            split='test',
            batch=self.training_config['batch_size'],
            device=self.training_config['device'],
            plots=True,
            save_json=True
        )

        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'classes': {}
        }

        # Per-class metrics
        for i, class_name in self.classes.items():
            if i < len(results.box.ap50):
                metrics['classes'][class_name] = {
                    'AP50': float(results.box.ap50[i]),
                    'AP': float(results.box.ap[i]) if i < len(results.box.ap) else 0
                }

        logger.info("Validation Results:")
        logger.info(f"  mAP@50: {metrics['mAP50']:.3f}")
        logger.info(f"  mAP@50-95: {metrics['mAP50-95']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")

        logger.info("Per-class performance:")
        for class_name, class_metrics in metrics['classes'].items():
            logger.info(f"  {class_name}: AP50={class_metrics['AP50']:.3f}")

        # Save metrics
        metrics_file = self.output_dir / f"{self.model_name}_validation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def export_model(self, model_path):
        """Export model in various formats for deployment."""
        logger.info("=" * 60)
        logger.info("EXPORTING MODEL FOR DEPLOYMENT")
        logger.info("=" * 60)

        model = YOLO(model_path)

        # Export formats
        formats = {
            'onnx': {'dynamic': True, 'simplify': True},
            'torchscript': {},
            'saved_model': {},  # TensorFlow SavedModel
        }

        exported_models = {}

        for format_name, kwargs in formats.items():
            try:
                logger.info(f"Exporting to {format_name}...")
                export_path = model.export(format=format_name, **kwargs)
                exported_models[format_name] = export_path
                logger.info(f"  ✅ Exported to: {export_path}")
            except Exception as e:
                logger.error(f"  ❌ Failed to export {format_name}: {e}")

        return exported_models

    def create_inference_example(self, model_path):
        """Create example inference script."""
        inference_script = f'''#!/usr/bin/env python3
"""
Charpy 3-Class Detection Inference Script
Use this script to run inference on new images
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Load trained model
model = YOLO('{model_path}')

# Class names
classes = {{
    0: "charpy_specimen",
    1: "charpy_corner",
    2: "fracture_surface"
}}

def detect_charpy_features(image_path, conf_threshold=0.25):
    """Detect Charpy specimen features in an image."""
    # Load image
    img = cv2.imread(str(image_path))

    # Run inference
    results = model(img, conf=conf_threshold)

    # Process results
    detections = {{
        'specimen': [],
        'corners': [],
        'fracture': None
    }}

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                if cls == 0:  # Specimen
                    detections['specimen'].append({{
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    }})
                elif cls == 1:  # Corner
                    detections['corners'].append({{
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    }})
                elif cls == 2:  # Fracture surface
                    detections['fracture'] = {{
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'measurement_line': y1  # Top edge for measurement
                    }}

    return detections

def measure_fracture(detections, pixels_per_mm=100):
    """Measure fracture surface width."""
    if detections['fracture'] is None:
        return None

    bbox = detections['fracture']['bbox']
    width_pixels = bbox[2] - bbox[0]
    width_mm = width_pixels / pixels_per_mm

    return {{
        'width_pixels': width_pixels,
        'width_mm': width_mm,
        'measurement_y': detections['fracture']['measurement_line']
    }}

# Example usage
if __name__ == "__main__":
    test_image = "path/to/test/image.jpg"

    # Detect features
    detections = detect_charpy_features(test_image)

    # Measure fracture
    measurement = measure_fracture(detections)

    if measurement:
        print(f"Fracture width: {{measurement['width_mm']:.2f}} mm")
'''

        script_file = self.output_dir / f"{self.model_name}_inference.py"
        with open(script_file, 'w') as f:
            f.write(inference_script)

        logger.info(f"Created inference script: {script_file}")

    def run_complete_training(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete training pipeline...")
        start_time = datetime.now()

        # 1. Train model
        results, best_model_path = self.train_model()
        if not results:
            logger.error("Training failed!")
            return False

        # 2. Validate model
        metrics = self.validate_model(best_model_path)

        # 3. Export model
        exported_models = self.export_model(best_model_path)

        # 4. Create inference example
        self.create_inference_example(best_model_path)

        # 5. Create summary report
        duration = datetime.now() - start_time
        summary = {
            'training_completed': datetime.now().isoformat(),
            'duration_minutes': duration.total_seconds() / 60,
            'model_name': self.model_name,
            'best_model_path': str(best_model_path),
            'validation_metrics': metrics,
            'exported_formats': {k: str(v) for k, v in exported_models.items()},
            'device_used': self.training_config['device'],
            'final_epoch': results.epoch[-1] if hasattr(results, 'epoch') else 'N/A'
        }

        summary_file = self.output_dir / f"{self.model_name}_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration.total_seconds() / 60:.1f} minutes")
        logger.info(f"Best model: {best_model_path}")
        logger.info(f"mAP@50: {metrics['mAP50']:.3f}")
        logger.info(f"Summary saved: {summary_file}")

        # Performance recommendations
        if metrics['mAP50'] < 0.5:
            logger.warning("Low mAP detected. Consider:")
            logger.warning("  - Adding more training data")
            logger.warning("  - Increasing training epochs")
            logger.warning("  - Adjusting augmentation parameters")
        elif metrics['mAP50'] > 0.8:
            logger.info("Excellent model performance achieved!")

        return True


def main():
    """Main training function."""
    print("🚀 CHARPY 3-CLASS GPU-OPTIMIZED TRAINING")
    print("=" * 60)

    # Setup paths
    dataset_path = Path(__file__).parent.parent / "data" / "datasets" / "charpy_dataset_v2"

    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("   Please run the annotation script first!")
        return

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - training will be slow!")
        response = input("Continue with CPU training? (y/N): ")
        if response.lower() != 'y':
            return

    # Initialize trainer
    trainer = CharpyModelTrainer(dataset_path)

    # Run training
    success = trainer.run_complete_training()

    if success:
        print("\n✅ Training completed successfully!")
        print("📁 Check the 'runs/charpy_3class' folder for:")
        print("   - Trained model weights")
        print("   - Training plots and metrics")
        print("   - Validation results")
        print("   - Exported models")
        print("   - Inference example script")
    else:
        print("\n❌ Training failed. Check training.log for details.")


if __name__ == "__main__":
    main()