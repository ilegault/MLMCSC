#!/usr/bin/env python3
# Quick Retrain Script for Charpy Model

from ultralytics import YOLO
import torch

print("🚀 Starting Charpy Model Retraining with Fixed Annotations")

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model (using small model for better feature detection)
model = YOLO('yolov8s.pt')

# Train with optimized settings for multi-class detection
results = model.train(
    data='data\charpy_dataset\dataset_fixed.yaml',
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
print("\n🧪 Testing trained model...")
test_model = YOLO(str(results.save_dir) + '/weights/best.pt')

# Run on test image with low confidence to see all detections
test_results = test_model('data/charpy_dataset/images/test/charpy_0001.jpg', 
                         conf=0.25, save=True)

print("\n✅ Training complete!")
print(f"Best model saved at: {str(results.save_dir)}/weights/best.pt")
