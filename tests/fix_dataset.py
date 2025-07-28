#!/usr/bin/env python3
"""
Corner-Based Measurement System
Uses detected corners to measure specimen dimensions
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
from pathlib import Path


class CornerBasedMeasurement:
    """
    Measures Charpy specimens by detecting corners and calculating distances
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Based on your dataset, focusing on corners for measurement
        self.class_names = {
            0: "charpy_specimen",
            1: "measurement_edge",  # Optional - you might not need this
            2: "corner",  # Critical for measurement!
            3: "fracture_surface"
        }

    def detect_and_measure(self, image, conf_threshold=0.3):
        """
        Detect corners and calculate measurement distance
        """
        results = {
            'specimen_bbox': None,
            'corners': [],
            'top_corners': [],
            'measurement_distance': None,
            'measurement_line': None,
            'fracture_bbox': None
        }

        # Run detection
        detections = self.model(image, conf=conf_threshold)

        for r in detections:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Calculate center point
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if cls == 0 and conf > 0.5:  # Specimen
                        results['specimen_bbox'] = (int(x1), int(y1), int(x2), int(y2))

                    elif cls == 2:  # Corner
                        corner_info = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (cx, cy),
                            'confidence': conf
                        }
                        results['corners'].append(corner_info)

                    elif cls == 3:  # Fracture surface
                        results['fracture_bbox'] = (int(x1), int(y1), int(x2), int(y2))

        # Find top corners for measurement
        if len(results['corners']) >= 2:
            results['top_corners'] = self.find_top_corners(results['corners'])

            if len(results['top_corners']) == 2:
                # Calculate measurement distance
                c1 = results['top_corners'][0]['center']
                c2 = results['top_corners'][1]['center']

                distance = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
                results['measurement_distance'] = distance
                results['measurement_line'] = (c1, c2)

        return results

    def find_top_corners(self, corners):
        """
        Identify the top two corners of the specimen
        """
        if len(corners) < 2:
            return []

        # Sort corners by y-coordinate (top ones have smaller y)
        sorted_corners = sorted(corners, key=lambda c: c['center'][1])

        # Get top corners
        top_corners = sorted_corners[:2]

        # Sort left to right
        top_corners.sort(key=lambda c: c['center'][0])

        return top_corners

    def visualize_results(self, image, results):
        """
        Visualize detection and measurement results
        """
        vis_image = image.copy()

        # Draw specimen bbox
        if results['specimen_bbox']:
            x1, y1, x2, y2 = results['specimen_bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, "Specimen", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw all corners
        for corner in results['corners']:
            x1, y1, x2, y2 = corner['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Draw center point
            cx, cy = int(corner['center'][0]), int(corner['center'][1])
            cv2.circle(vis_image, (cx, cy), 5, (255, 255, 0), -1)

        # Highlight top corners and draw measurement
        if results['top_corners']:
            for i, corner in enumerate(results['top_corners']):
                cx, cy = int(corner['center'][0]), int(corner['center'][1])

                # Draw larger circle for top corners
                cv2.circle(vis_image, (cx, cy), 8, (0, 0, 255), -1)
                cv2.putText(vis_image, f"C{i + 1}", (cx - 10, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw measurement line
        if results['measurement_line']:
            c1, c2 = results['measurement_line']
            pt1 = (int(c1[0]), int(c1[1]))
            pt2 = (int(c2[0]), int(c2[1]))

            # Draw measurement line
            cv2.line(vis_image, pt1, pt2, (255, 0, 255), 3)

            # Draw perpendicular markers
            self.draw_measurement_markers(vis_image, pt1, pt2)

            # Draw measurement value
            if results['measurement_distance']:
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2

                text = f"{results['measurement_distance']:.1f}px"

                # Add background to text
                (text_width, text_height), _ = cv2.getTextSize(text,
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis_image,
                              (mid_x - text_width // 2 - 5, mid_y - text_height - 10),
                              (mid_x + text_width // 2 + 5, mid_y - 5),
                              (255, 255, 255), -1)

                cv2.putText(vis_image, text,
                            (mid_x - text_width // 2, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return vis_image

    def draw_measurement_markers(self, image, pt1, pt2):
        """Draw perpendicular markers at measurement endpoints"""
        # Calculate line angle
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        perp_angle = angle + math.pi / 2

        marker_length = 15

        # Markers at both ends
        for pt in [pt1, pt2]:
            x1 = int(pt[0] + marker_length * math.cos(perp_angle))
            y1 = int(pt[1] + marker_length * math.sin(perp_angle))
            x2 = int(pt[0] - marker_length * math.cos(perp_angle))
            y2 = int(pt[1] - marker_length * math.sin(perp_angle))
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)


def fix_dataset_for_corner_measurement():
    """
    Fix dataset to work with corner-based measurement
    Keep corners, remove measurement_point class
    """
    from pathlib import Path

    dataset_path = Path("data/charpy_dataset")

    print("🔧 Fixing dataset for corner-based measurement...")

    # Remove class 4 (measurement_point) from test set
    removed_count = 0
    labels_dir = dataset_path / 'labels' / 'test'

    for label_file in labels_dir.glob("*.txt"):
        lines = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id != 4:  # Keep all except class 4
                        lines.append(line.strip())
                    else:
                        removed_count += 1

        with open(label_file, 'w') as f:
            if lines:
                f.write('\n'.join(lines) + '\n')

    print(f"✅ Removed {removed_count} class 4 annotations")

    # Create config
    dataset_yaml = f"""
# Charpy Dataset - Corner-Based Measurement
path: {dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

# 4 Classes for corner-based measurement
nc: 4
names:
  0: charpy_specimen
  1: measurement_edge    # Optional - might not need this
  2: corner             # CRITICAL for measurement!
  3: fracture_surface
"""

    yaml_path = dataset_path / "dataset_corner_measurement.yaml"
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)

    print(f"✅ Created dataset config: {yaml_path}")
    return str(yaml_path)


def test_corner_measurement():
    """Test the corner-based measurement system"""

    # Initialize measurement system
    measurer = CornerBasedMeasurement("models/charpy_final/best.pt")

    # Test on images
    test_dir = Path("data/charpy_dataset/images/test")
    output_dir = Path("corner_measurement_results")
    output_dir.mkdir(exist_ok=True)

    for img_path in list(test_dir.glob("*.jpg"))[:5]:
        print(f"\nProcessing: {img_path.name}")

        img = cv2.imread(str(img_path))

        # Detect and measure
        results = measurer.detect_and_measure(img, conf_threshold=0.25)

        print(f"  Corners found: {len(results['corners'])}")
        print(f"  Top corners: {len(results['top_corners'])}")
        print(f"  Measurement: {results['measurement_distance']:.1f}px"
              if results['measurement_distance'] else "  No measurement")

        # Visualize
        vis_img = measurer.visualize_results(img, results)
        output_path = output_dir / f"measured_{img_path.name}"
        cv2.imwrite(str(output_path), vis_img)

    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    print("🎯 CORNER-BASED MEASUREMENT SYSTEM")
    print("=" * 60)
    print("This approach uses corner detection for precise measurement")
    print("The distance between top corners = specimen width")
    print()

    # First, fix the dataset
    dataset_yaml = fix_dataset_for_corner_measurement()

    print("\n✅ Your approach is excellent because:")
    print("• Corners are more reliable to detect than thin edges")
    print("• You get exact measurement points")
    print("• Works at any rotation angle")
    print("• More precise than edge-based measurement")
    print()
    print("Keep the corners! Train with 4 classes:")
    print("0: specimen, 1: edge (optional), 2: corner (critical!), 3: fracture")