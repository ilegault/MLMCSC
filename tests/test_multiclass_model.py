#!/usr/bin/env python3
"""
Multi-Class Model Testing Script
Tests your trained multi-class Charpy detection model
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import os
import glob

def test_multiclass_model(model_path, test_images_dir, output_dir="multiclass_results"):
    """Test the multi-class model on images"""
    
    print("🔍 MULTI-CLASS MODEL TESTING")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Class colors for visualization
    class_colors = {
        0: (0, 255, 0),      # charpy_specimen - Green
        1: (255, 0, 0),      # charpy_edge - Red  
        2: (0, 0, 255),      # charpy_corner - Blue
        3: (255, 255, 0),    # fracture_surface - Cyan
        4: (255, 0, 255)     # measurement_point - Magenta
    }
    
    # Find test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    if not test_images:
        print(f"❌ No test images found in: {test_images_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test with different confidence thresholds
    confidence_thresholds = [0.5, 0.3, 0.1]
    
    all_results = {}
    
    for conf_thresh in confidence_thresholds:
        print(f"\n📊 Testing with confidence threshold: {conf_thresh}")
        
        threshold_results = {
            'confidence': conf_thresh,
            'total_detections': 0,
            'class_counts': {},
            'images_processed': 0,
            'images_with_detections': 0
        }
        
        for img_path in test_images:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Run inference
            results = model(img, conf=conf_thresh, iou=0.45)
            
            # Process results
            image_detections = 0
            annotated_img = img.copy()
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Count detections
                        image_detections += 1
                        threshold_results['total_detections'] += 1
                        
                        if cls not in threshold_results['class_counts']:
                            threshold_results['class_counts'][cls] = 0
                        threshold_results['class_counts'][cls] += 1
                        
                        # Draw on image
                        color = class_colors.get(cls, (128, 128, 128))
                        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(annotated_img, label, (int(x1), int(y1 - 10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save annotated image (only for best threshold)
            if conf_thresh == 0.3:  # Save images for middle threshold
                basename = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(output_dir, f"{basename}_detected.jpg")
                cv2.imwrite(output_path, annotated_img)
            
            threshold_results['images_processed'] += 1
            if image_detections > 0:
                threshold_results['images_with_detections'] += 1
        
        all_results[conf_thresh] = threshold_results
        
        # Print results for this threshold
        print(f"  Total detections: {threshold_results['total_detections']}")
        print(f"  Images with detections: {threshold_results['images_with_detections']}/{threshold_results['images_processed']}")
        
        if threshold_results['class_counts']:
            print("  Detections by class:")
            for class_id, count in threshold_results['class_counts'].items():
                class_name = model.names[class_id] if class_id < len(model.names) else f"class_{class_id}"
                percentage = (count / threshold_results['total_detections']) * 100
                print(f"    - {class_name}: {count} ({percentage:.1f}%)")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Testing completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Detailed results: {results_file}")
    
    # Analysis and recommendations
    print(f"\n💡 ANALYSIS:")
    best_results = all_results[0.3]  # Use middle threshold for analysis
    
    if best_results['total_detections'] == 0:
        print("❌ No detections found - model may need more training")
    elif len(best_results['class_counts']) == 1:
        print("⚠️  Only detecting one class - check if all classes are learned")
    elif len(best_results['class_counts']) >= 4:
        print("✅ Multi-class detection working - detecting multiple feature types!")
    else:
        print("⚠️  Limited multi-class detection - some classes may need more training data")
    
    return all_results

def compare_with_ground_truth(model_path, test_labels_dir, test_images_dir):
    """Compare model predictions with ground truth labels"""
    
    print(f"\n🎯 GROUND TRUTH COMPARISON")
    print("=" * 30)
    
    if not os.path.exists(test_labels_dir):
        print(f"❌ Test labels directory not found: {test_labels_dir}")
        return
    
    model = YOLO(model_path)
    
    # Get test label files
    label_files = glob.glob(os.path.join(test_labels_dir, "*.txt"))
    
    if not label_files:
        print("❌ No ground truth labels found")
        return
    
    print(f"Comparing against {len(label_files)} ground truth files")
    
    total_gt_annotations = 0
    total_predictions = 0
    class_gt_counts = {}
    class_pred_counts = {}
    
    for label_file in label_files:
        basename = os.path.splitext(os.path.basename(label_file))[0]
        
        # Load ground truth
        gt_annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        gt_annotations.append(class_id)
                        total_gt_annotations += 1
                        
                        if class_id not in class_gt_counts:
                            class_gt_counts[class_id] = 0
                        class_gt_counts[class_id] += 1
        
        # Load corresponding image and run prediction
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(test_images_dir, basename + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    results = model(img, conf=0.3)
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls = int(box.cls)
                                total_predictions += 1
                                
                                if cls not in class_pred_counts:
                                    class_pred_counts[cls] = 0
                                class_pred_counts[cls] += 1
                break
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"Ground Truth Annotations: {total_gt_annotations}")
    print(f"Model Predictions: {total_predictions}")
    
    print(f"\nGround Truth by Class:")
    for class_id, count in class_gt_counts.items():
        class_name = model.names[class_id] if class_id < len(model.names) else f"class_{class_id}"
        print(f"  {class_name}: {count}")
    
    print(f"\nPredictions by Class:")
    for class_id, count in class_pred_counts.items():
        class_name = model.names[class_id] if class_id < len(model.names) else f"class_{class_id}"
        print(f"  {class_name}: {count}")

def main():
    print("🧪 MULTI-CLASS MODEL TESTER")
    print("=" * 50)
    
    # Default paths - update these based on your setup
    model_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/models/charpy_improved/charpy_improved_v1/weights/best.pt"
    test_images_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/images/test"
    test_labels_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset/labels/test"
    
    print(f"Model: {model_path}")
    print(f"Test images: {test_images_dir}")
    print(f"Test labels: {test_labels_dir}")
    print()
    
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print("   Run 'python train_multiclass_model.py' first")
        return
    
    # Test the model
    results = test_multiclass_model(model_path, test_images_dir)
    
    # Compare with ground truth
    compare_with_ground_truth(model_path, test_labels_dir, test_images_dir)
    
    print(f"\n🎉 TESTING COMPLETE!")
    print(f"Check the 'multiclass_results' folder for annotated images")

if __name__ == "__main__":
    main()