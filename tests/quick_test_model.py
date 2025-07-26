#!/usr/bin/env python3
"""
Quick test script to check what your model is actually detecting
Run this NOW to see what's happening with your model
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json


def quick_test_model(model_path, test_image_path):
    """Quick test to see what your model detects."""

    print("🔍 Quick Model Test")
    print("=" * 50)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Load image
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"❌ Cannot load image: {test_image_path}")
        return

    print(f"Testing on: {test_image_path}")
    print(f"Image shape: {img.shape}")

    # Test with different confidence thresholds
    thresholds = [0.5, 0.25, 0.1, 0.05]

    for conf_thresh in thresholds:
        print(f"\n📊 Testing with confidence threshold: {conf_thresh}")

        # Run inference
        results = model(img, conf=conf_thresh, iou=0.45)

        # Count detections
        total_detections = 0
        class_counts = {}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)

                    class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"

                    if class_name not in class_counts:
                        class_counts[class_name] = []

                    class_counts[class_name].append(conf)
                    total_detections += 1

        print(f"  Total detections: {total_detections}")

        if class_counts:
            print("  Detections by class:")
            for class_name, confidences in class_counts.items():
                avg_conf = np.mean(confidences)
                print(f"    - {class_name}: {len(confidences)} detections (avg conf: {avg_conf:.3f})")
        else:
            print("  ❌ No detections at this threshold")

    # Save best visualization (with lowest threshold that has detections)
    best_threshold = 0.1
    results = model(img, conf=best_threshold, iou=0.45)

    annotated_img = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for result in results:
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)

                color = colors[cls % len(colors)]

                # Draw box
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw label
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.putText(annotated_img, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save result
    output_path = "quick_test_result.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"\n✅ Saved annotated image to: {output_path}")

    # Print recommendations
    print("\n💡 Recommendations based on results:")

    if total_detections == 0:
        print("❌ No detections found - your model needs retraining with better annotations")
    elif 'charpy_specimen' in class_counts and len(class_counts) == 1:
        print("⚠️  Only detecting whole specimens - multi-class features not learned:")
        print("   - Model may need more training epochs")
        print("   - Check if all classes have sufficient training data")
        print("   - Verify dataset.yaml configuration")
    elif len(class_counts) < 3:
        print("⚠️  Limited multi-class detection - only detecting few feature types!")
        print("   - Some classes may need more training data")
        print("   - Consider adjusting class weights in training")
    else:
        print("✅ Multi-class detection working - detecting multiple feature types!")


def analyze_predictions_json(predictions_file):
    """Analyze your predictions.json file."""
    print("\n📊 Analyzing predictions.json...")
    print("=" * 50)

    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # Analyze predictions
    category_counts = {}
    confidence_ranges = {
        'high': 0,  # > 0.5
        'medium': 0,  # 0.25 - 0.5
        'low': 0,  # 0.1 - 0.25
        'very_low': 0  # < 0.1
    }

    for pred in predictions:
        cat_id = pred['category_id']
        score = pred['score']

        if cat_id not in category_counts:
            category_counts[cat_id] = 0
        category_counts[cat_id] += 1

        if score > 0.5:
            confidence_ranges['high'] += 1
        elif score > 0.25:
            confidence_ranges['medium'] += 1
        elif score > 0.1:
            confidence_ranges['low'] += 1
        else:
            confidence_ranges['very_low'] += 1

    print(f"Total predictions: {len(predictions)}")
    print(f"\nPredictions by category:")
    for cat_id, count in category_counts.items():
        print(f"  Category {cat_id}: {count} predictions")

    print(f"\nConfidence distribution:")
    for range_name, count in confidence_ranges.items():
        percentage = (count / len(predictions)) * 100
        print(f"  {range_name}: {count} ({percentage:.1f}%)")

    print("\n⚠️  Analysis:")
    if len(category_counts) == 1 and 1 in category_counts:
        print("❌ Only detecting category 1 (charpy_specimen)")
        print("   Your model is not detecting the specific features needed for measurement")

    if confidence_ranges['very_low'] > len(predictions) * 0.5:
        print("❌ Most predictions have very low confidence (<0.1)")
        print("   This indicates the model is uncertain about its detections")

    # Calculate average confidence
    avg_confidence = sum(pred['score'] for pred in predictions) / len(predictions)
    print(f"\nAverage confidence: {avg_confidence:.3f}")

    if avg_confidence < 0.2:
        print("❌ Very low average confidence - model needs better training")


if __name__ == "__main__":
    # Test your current model
    print("🔬 Charpy Model Quick Test")
    print("This will help you understand what your model is detecting\n")

    # Update these paths to match your setup
    model_path = "models/charpy_multiclass/charpy_multiclass_v1/weights/best.pt"  # New multi-class model
    test_image = "data/charpy_dataset/images/test/charpy_0000.jpg"  # Use one of your test images
    predictions_json = "predictions.json"

    # Run quick test
    if Path(model_path).exists():
        quick_test_model(model_path, test_image)
    else:
        print(f"❌ Model not found at: {model_path}")
        print("   Update the model_path variable with your actual model location")

    # Analyze predictions.json if it exists
    if Path(predictions_json).exists():
        analyze_predictions_json(predictions_json)
    else:
        print(f"\n❌ predictions.json not found")
        print("   This file should be created after running inference")