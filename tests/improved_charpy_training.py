#!/usr/bin/env python3
"""
Improved Charpy Model Training Script
Handles both single-class and multi-class scenarios
"""

from ultralytics import YOLO
import torch
import yaml
import os

def analyze_dataset_classes(yaml_path):
    """Analyze what classes are actually used in the dataset"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get base path and check labels
    base_path = data['path']
    train_labels = os.path.join(base_path, data['train'].replace('images', 'labels'))
    
    used_classes = set()
    total_annotations = 0
    
    if os.path.exists(train_labels):
        for label_file in os.listdir(train_labels):
            if label_file.endswith('.txt'):
                with open(os.path.join(train_labels, label_file), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            used_classes.add(class_id)
                            total_annotations += 1
    
    return used_classes, total_annotations, data['nc'], data['names']

def create_optimized_config(original_yaml, used_classes, output_path):
    """Create optimized dataset config based on actually used classes"""
    with open(original_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    if len(used_classes) == 1 and 0 in used_classes:
        # Single class scenario - optimize for specimen detection
        data['nc'] = 1
        data['names'] = {0: 'charpy_specimen'}
        config_type = "single_class"
    else:
        # Multi-class scenario - keep all classes
        config_type = "multi_class"
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return config_type

def main():
    print("🚀 IMPROVED CHARPY MODEL TRAINING")
    print("=" * 50)
    
    # Analyze current dataset
    yaml_path = 'C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/dataset_fixed.yaml'
    used_classes, total_annotations, defined_classes, class_names = analyze_dataset_classes(yaml_path)
    
    print(f"📊 Dataset Analysis:")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Defined classes: {defined_classes}")
    print(f"   Actually used classes: {len(used_classes)} -> {sorted(used_classes)}")
    
    # Create optimized config
    optimized_yaml = 'C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/dataset_optimized.yaml'
    config_type = create_optimized_config(yaml_path, used_classes, optimized_yaml)
    
    print(f"   Configuration type: {config_type}")
    print(f"   Optimized config saved: {optimized_yaml}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    # Load model - use different sizes based on complexity
    if config_type == "single_class":
        model = YOLO('yolov8n.pt')  # Nano for single class
        print("   Using YOLOv8n (optimized for single-class detection)")
    else:
        model = YOLO('yolov8s.pt')  # Small for multi-class
        print("   Using YOLOv8s (optimized for multi-class detection)")
    
    print("\n🏋️ Starting Training...")
    
    # Training parameters optimized for each scenario
    if config_type == "single_class":
        # Single class - focus on detection accuracy
        results = model.train(
            data=optimized_yaml,
            epochs=50,  # Fewer epochs needed for single class
            imgsz=640,
            batch=16 if device == 'cpu' else 32,
            device=device,
            
            # Single class optimization
            conf=0.1,   # Very low confidence - we want to catch everything
            cls=1.0,    # Standard classification weight
            box=7.5,    # High box regression weight
            
            # Augmentation
            augment=True,
            mosaic=0.8,
            mixup=0.1,
            
            # Training settings
            save=True,
            save_period=10,
            patience=15,
            val=True,
            plots=True,
            
            # Output
            project='models/charpy_optimized',
            name='single_class_v1',
            exist_ok=True
        )
    else:
        # Multi-class - more complex training
        results = model.train(
            data=optimized_yaml,
            epochs=100,  # More epochs for multi-class
            imgsz=640,
            batch=8 if device == 'cpu' else 16,
            device=device,
            
            # Multi-class optimization
            conf=0.25,  # Higher confidence threshold
            cls=2.0,    # Higher classification loss weight
            box=7.5,    # High box regression weight
            
            # Stronger augmentation for multi-class
            augment=True,
            mosaic=1.0,
            mixup=0.3,
            copy_paste=0.3,
            
            # Training settings
            save=True,
            save_period=20,
            patience=30,
            val=True,
            plots=True,
            
            # Output
            project='models/charpy_optimized',
            name='multi_class_v1',
            exist_ok=True
        )
    
    print(f"\n✅ Training Complete!")
    print(f"   Model saved at: {str(results.save_dir)}/weights/best.pt")
    
    # Test the model
    print("\n🧪 Testing trained model...")
    test_model = YOLO(str(results.save_dir) + '/weights/best.pt')
    
    # Test on multiple images with different confidence levels
    test_images = [
        'C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/images/test/charpy_0001.jpg'
    ]
    
    for conf_level in [0.1, 0.25, 0.5]:
        print(f"\n   Testing with confidence {conf_level}:")
        for img_path in test_images:
            if os.path.exists(img_path):
                results = test_model(img_path, conf=conf_level, save=True, 
                                   save_txt=True, save_conf=True)
                
                # Print detection results
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    print(f"     {os.path.basename(img_path)}: {len(boxes)} detections")
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = test_model.names[class_id]
                        print(f"       {i+1}. {class_name}: {confidence:.3f}")
                else:
                    print(f"     {os.path.basename(img_path)}: No detections")
    
    print(f"\n🎯 TRAINING SUMMARY:")
    print(f"   Configuration: {config_type}")
    print(f"   Classes used: {len(used_classes)}")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Best model: {str(results.save_dir)}/weights/best.pt")

if __name__ == "__main__":
    main()