#!/usr/bin/env python3
"""
Training script for custom specimen detection using YOLOv8.

This script helps train a custom YOLO model for microscope specimen detection.
It includes data preparation, training, and evaluation utilities.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
from ultralytics import YOLO
import yaml
import cv2
import numpy as np
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Utility class for preparing YOLO training datasets."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'images'
        self.labels_dir = self.dataset_path / 'labels'
        
    def create_directory_structure(self):
        """Create the required directory structure for YOLO training."""
        dirs_to_create = [
            self.images_dir / 'train',
            self.images_dir / 'val',
            self.images_dir / 'test',
            self.labels_dir / 'train',
            self.labels_dir / 'val',
            self.labels_dir / 'test'
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def split_dataset(
        self,
        source_images_dir: str,
        source_labels_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        """
        Split dataset into train/val/test sets.
        
        Args:
            source_images_dir: Directory containing source images
            source_labels_dir: Directory containing source labels
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        source_images = Path(source_images_dir)
        source_labels = Path(source_labels_dir)
        
        # Get all image files
        image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))
        random.shuffle(image_files)
        
        total_files = len(image_files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        
        # Split files
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Copy files to respective directories
        self._copy_files(train_files, source_labels, 'train')
        self._copy_files(val_files, source_labels, 'val')
        self._copy_files(test_files, source_labels, 'test')
        
        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(train_files)} files")
        logger.info(f"  Val: {len(val_files)} files")
        logger.info(f"  Test: {len(test_files)} files")
    
    def _copy_files(self, image_files: List[Path], source_labels: Path, split: str):
        """Copy image and label files to the appropriate split directory."""
        for image_file in tqdm(image_files, desc=f"Copying {split} files"):
            # Copy image
            dest_image = self.images_dir / split / image_file.name
            shutil.copy2(image_file, dest_image)
            
            # Copy corresponding label file
            label_file = source_labels / f"{image_file.stem}.txt"
            if label_file.exists():
                dest_label = self.labels_dir / split / f"{image_file.stem}.txt"
                shutil.copy2(label_file, dest_label)
    
    def create_dataset_config(self, classes: List[str]) -> str:
        """Create dataset configuration file for YOLO training."""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(classes),
            'names': {i: name for i, name in enumerate(classes)}
        }
        
        config_path = self.dataset_path / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Dataset config created: {config_path}")
        return str(config_path)


class SpecimenTrainer:
    """Main training class for specimen detection models."""
    
    def __init__(self, dataset_path: str, output_dir: str = "training_output"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Default specimen classes (customize as needed)
        self.classes = [
            'specimen',
            'cell',
            'bacteria',
            'particle',
            'debris',
            'crystal',
            'fiber',
            'bubble'
        ]
    
    def prepare_dataset(
        self,
        source_images_dir: str,
        source_labels_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        """Prepare dataset for training."""
        preparer = DatasetPreparer(self.dataset_path)
        preparer.create_directory_structure()
        preparer.split_dataset(
            source_images_dir,
            source_labels_dir,
            train_ratio,
            val_ratio,
            test_ratio
        )
        
        # Create dataset config
        config_path = preparer.create_dataset_config(self.classes)
        return config_path
    
    def train_model(
        self,
        config_path: str,
        model_size: str = 'n',
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        device: str = 'auto',
        resume: bool = False,
        pretrained: bool = True
    ) -> Dict:
        """
        Train the specimen detection model.
        
        Args:
            config_path: Path to dataset configuration file
            model_size: YOLO model size (n, s, m, l, x)
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            device: Device to use ('cpu', 'cuda', 'auto')
            resume: Resume training from last checkpoint
            pretrained: Use pretrained weights
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        # Load model
        if pretrained:
            model_path = f'yolov8{model_size}.pt'
        else:
            model_path = f'yolov8{model_size}.yaml'
        
        model = YOLO(model_path)
        
        # Training parameters
        train_params = {
            'data': config_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'project': str(self.output_dir),
            'name': f'specimen_detector_v8{model_size}',
            'save': True,
            'save_period': 10,
            'cache': True,
            'workers': 8,
            'patience': 50,
            'resume': resume,
            
            # Data augmentation
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            
            # Geometric augmentations
            'degrees': 15.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0,
            'flipud': 0.5,
            'fliplr': 0.5,
            
            # Color augmentations
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            
            # Optimization
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss weights
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Validation
            'val': True,
            'plots': True,
            'save_json': True
        }
        
        # Start training
        results = model.train(**train_params)
        
        # Validate the model
        logger.info("Validating trained model...")
        validation_results = model.val()
        
        # Export model for deployment
        logger.info("Exporting model...")
        model.export(format='onnx', dynamic=True, simplify=True)
        
        # Log results
        logger.info("Training completed!")
        logger.info(f"Best model saved at: {results.save_dir}/weights/best.pt")
        logger.info(f"Validation mAP50: {validation_results.box.map50:.4f}")
        logger.info(f"Validation mAP50-95: {validation_results.box.map:.4f}")
        
        return {
            'training_results': results,
            'validation_results': validation_results,
            'best_model_path': f"{results.save_dir}/weights/best.pt",
            'onnx_model_path': f"{results.save_dir}/weights/best.onnx"
        }
    
    def evaluate_model(self, model_path: str, test_data_path: str) -> Dict:
        """
        Evaluate trained model on test data.
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test dataset
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model...")
        
        model = YOLO(model_path)
        results = model.val(data=test_data_path, split='test')
        
        evaluation_results = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
        }
        
        logger.info("Evaluation completed:")
        for metric, value in evaluation_results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return evaluation_results
    
    def create_synthetic_data(
        self,
        output_dir: str,
        num_images: int = 1000,
        image_size: Tuple[int, int] = (640, 640)
    ):
        """
        Create synthetic training data for specimen detection.
        This is useful when real annotated data is limited.
        
        Args:
            output_dir: Directory to save synthetic data
            num_images: Number of synthetic images to generate
            image_size: Size of generated images
        """
        logger.info(f"Generating {num_images} synthetic images...")
        
        output_path = Path(output_dir)
        images_dir = output_path / 'images'
        labels_dir = output_path / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(range(num_images), desc="Generating synthetic data"):
            # Create synthetic microscope image
            image = self._generate_synthetic_image(image_size)
            
            # Generate random specimens
            annotations = self._generate_synthetic_annotations(image_size)
            
            # Save image
            image_path = images_dir / f"synthetic_{i:06d}.jpg"
            cv2.imwrite(str(image_path), image)
            
            # Save annotations
            label_path = labels_dir / f"synthetic_{i:06d}.txt"
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann['class_id']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n")
        
        logger.info(f"Synthetic data generated in {output_dir}")
    
    def _generate_synthetic_image(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate a synthetic microscope image."""
        h, w = size
        
        # Create base image with microscope-like background
        image = np.random.normal(50, 20, (h, w, 3)).astype(np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Add some texture
        noise = np.random.normal(0, 10, (h, w, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Add circular vignette effect (common in microscopy)
        center = (w // 2, h // 2)
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        max_dist = min(h, w) // 2
        vignette = np.clip(1 - (dist_from_center / max_dist) ** 2, 0.3, 1.0)
        
        for c in range(3):
            image[:, :, c] = (image[:, :, c] * vignette).astype(np.uint8)
        
        return image
    
    def _generate_synthetic_annotations(self, image_size: Tuple[int, int]) -> List[Dict]:
        """Generate synthetic specimen annotations."""
        h, w = image_size
        annotations = []
        
        # Generate 1-5 specimens per image
        num_specimens = random.randint(1, 5)
        
        for _ in range(num_specimens):
            # Random specimen properties
            class_id = random.randint(0, len(self.classes) - 1)
            
            # Random position (avoid edges)
            margin = 0.1
            x_center = random.uniform(margin, 1 - margin)
            y_center = random.uniform(margin, 1 - margin)
            
            # Random size (relative to image)
            width = random.uniform(0.02, 0.15)
            height = random.uniform(0.02, 0.15)
            
            annotations.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
        
        return annotations


def main():
    """Main training pipeline."""
    # Configuration
    dataset_path = "data/specimen_dataset"
    source_images = "data/raw/images"  # Update with your image directory
    source_labels = "data/raw/labels"  # Update with your label directory
    
    # Initialize trainer
    trainer = SpecimenTrainer(dataset_path)
    
    # Option 1: Use real data (if available)
    if os.path.exists(source_images) and os.path.exists(source_labels):
        logger.info("Using real annotated data...")
        config_path = trainer.prepare_dataset(source_images, source_labels)
    else:
        # Option 2: Generate synthetic data
        logger.info("Generating synthetic training data...")
        trainer.create_synthetic_data(
            output_dir=f"{dataset_path}/synthetic",
            num_images=2000,
            image_size=(640, 640)
        )
        
        # Prepare synthetic dataset
        preparer = DatasetPreparer(dataset_path)
        preparer.create_directory_structure()
        preparer.split_dataset(
            f"{dataset_path}/synthetic/images",
            f"{dataset_path}/synthetic/labels"
        )
        config_path = preparer.create_dataset_config(trainer.classes)
    
    # Train model
    results = trainer.train_model(
        config_path=config_path,
        model_size='n',  # Use nano for speed
        epochs=100,
        imgsz=640,
        batch_size=16,
        device='auto'
    )
    
    # Evaluate model
    evaluation = trainer.evaluate_model(
        results['best_model_path'],
        config_path
    )
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model: {results['best_model_path']}")
    logger.info(f"Final mAP50: {evaluation['mAP50']:.4f}")


if __name__ == "__main__":
    main()