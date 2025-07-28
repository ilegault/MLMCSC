#!/usr/bin/env python3
"""
Annotation Helper for Charpy Multi-Class Detection
Helps you add detailed annotations to your existing dataset
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path

class CharpyAnnotationHelper:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.classes = {
            0: "charpy_specimen",
            1: "charpy_notch", 
            2: "charpy_edge",
            3: "charpy_corner",
            4: "fracture_surface",
            5: "measurement_point"
        }
        self.colors = {
            0: (0, 255, 0),    # Green for specimen
            1: (255, 0, 0),    # Red for notch (most important)
            2: (0, 0, 255),    # Blue for edge
            3: (255, 255, 0),  # Cyan for corner
            4: (255, 0, 255),  # Magenta for fracture
            5: (0, 255, 255)   # Yellow for measurement point
        }
        
    def visualize_current_annotations(self, split='train', max_images=5):
        """Visualize current annotations to understand what's missing"""
        images_dir = self.dataset_path / 'images' / split
        labels_dir = self.dataset_path / 'labels' / split
        
        print(f"🖼️  Visualizing {split} annotations...")
        
        image_files = list(images_dir.glob('*.jpg'))[:max_images]
        
        for img_file in image_files:
            label_file = labels_dir / (img_file.stem + '.txt')
            
            if not label_file.exists():
                continue
                
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Load annotations
            with open(label_file, 'r') as f:
                annotations = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
            
            # Draw annotations
            for class_id, x_center, y_center, width, height in annotations:
                # Convert normalized coordinates to pixel coordinates
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Draw bounding box
                color = self.colors.get(class_id, (128, 128, 128))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_id}: {self.classes.get(class_id, 'Unknown')}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
            
            # Show image
            cv2.imshow(f'Current Annotations - {img_file.name}', img)
            print(f"   {img_file.name}: {len(annotations)} annotations")
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
                
        cv2.destroyAllWindows()
    
    def suggest_annotation_strategy(self):
        """Suggest how to add multi-class annotations"""
        print("\n🎯 MULTI-CLASS ANNOTATION STRATEGY")
        print("=" * 50)
        
        print("📝 Recommended annotation approach:")
        print("\n1. 🔴 PRIORITY 1 - Charpy Notch (Class 1):")
        print("   - Most critical feature for Charpy testing")
        print("   - Usually V-shaped notch in the specimen")
        print("   - Should be small, precise bounding box around notch")
        
        print("\n2. 🟡 PRIORITY 2 - Specimen Edges (Class 2):")
        print("   - Main edges of the specimen")
        print("   - Usually 2-4 edge annotations per specimen")
        print("   - Help define specimen boundaries")
        
        print("\n3. 🟢 PRIORITY 3 - Corners (Class 3):")
        print("   - Corner points of the specimen")
        print("   - Usually 4 corner annotations")
        print("   - Small bounding boxes at corner points")
        
        print("\n4. 🟣 PRIORITY 4 - Fracture Surface (Class 4):")
        print("   - Area where fracture occurred (if visible)")
        print("   - Only annotate if fracture is visible")
        print("   - Usually after impact testing")
        
        print("\n5. 🔵 PRIORITY 5 - Measurement Points (Class 5):")
        print("   - Specific points for measurements")
        print("   - Optional - only if you need precise measurement locations")
        
        print("\n🛠️  ANNOTATION TOOLS:")
        print("   - Use labelImg, CVAT, or Roboflow for annotation")
        print("   - Start with just Class 1 (notch) + existing Class 0 (specimen)")
        print("   - Gradually add more classes as needed")
        
        print("\n📊 RECOMMENDED WORKFLOW:")
        print("   1. Keep existing specimen annotations (Class 0)")
        print("   2. Add notch annotations (Class 1) to all images")
        print("   3. Train and test with 2 classes")
        print("   4. Gradually add edge/corner annotations")
        print("   5. Retrain with more classes")
    
    def create_sample_annotations(self, output_dir):
        """Create sample annotation files showing multi-class format"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Sample annotation formats
        samples = {
            'single_class_current.txt': [
                "0 0.500000 0.500000 0.700000 0.800000"
            ],
            'multi_class_example.txt': [
                "0 0.500000 0.500000 0.700000 0.800000",  # Specimen
                "1 0.450000 0.300000 0.050000 0.100000",  # Notch
                "2 0.200000 0.400000 0.050000 0.600000",  # Left edge
                "2 0.800000 0.400000 0.050000 0.600000",  # Right edge
                "3 0.200000 0.200000 0.040000 0.040000",  # Top-left corner
                "3 0.800000 0.200000 0.040000 0.040000",  # Top-right corner
            ],
            'notch_focused_example.txt': [
                "0 0.500000 0.500000 0.700000 0.800000",  # Specimen
                "1 0.450000 0.300000 0.050000 0.100000",  # Notch (main focus)
            ]
        }
        
        for filename, annotations in samples.items():
            with open(output_path / filename, 'w') as f:
                f.write('\n'.join(annotations) + '\n')
        
        print(f"\n📄 Sample annotation files created in: {output_path}")
        print("   - single_class_current.txt: Your current format")
        print("   - multi_class_example.txt: Full multi-class example")
        print("   - notch_focused_example.txt: Recommended starting point")

def main():
    dataset_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    helper = CharpyAnnotationHelper(dataset_path)
    
    print("🔧 CHARPY ANNOTATION HELPER")
    print("=" * 50)
    
    # Show current situation
    print("📊 Current dataset analysis:")
    print("   - All annotations are Class 0 (charpy_specimen)")
    print("   - Need to add Classes 1-5 for detailed detection")
    
    # Suggest strategy
    helper.suggest_annotation_strategy()
    
    # Create sample files
    helper.create_sample_annotations("C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/annotation_samples")
    
    # Ask if user wants to visualize current annotations
    print(f"\n🖼️  Would you like to visualize current annotations?")
    print("   This will show you what your current dataset looks like")
    response = input("   View annotations? (y/n): ").lower().strip()
    
    if response == 'y':
        helper.visualize_current_annotations('train', max_images=3)
    
    print(f"\n✅ Next steps:")
    print("   1. Use an annotation tool (labelImg, CVAT, Roboflow)")
    print("   2. Start by adding Class 1 (notch) annotations")
    print("   4. Gradually add more classes as needed")

if __name__ == "__main__":
    main()