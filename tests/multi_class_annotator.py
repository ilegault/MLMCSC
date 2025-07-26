#!/usr/bin/env python3
"""
Multi-Class Charpy Annotation Tool
Allows you to add detailed annotations for edges, corners, fracture surfaces, and measurement points
"""

import cv2
import os
import json
import glob
from pathlib import Path
import numpy as np

class MultiClassAnnotator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.current_image = None
        self.current_image_path = None
        self.current_annotations = []
        self.image_files = []
        self.current_index = 0
        
        # Class definitions
        self.classes = {
            0: {"name": "charpy_specimen", "color": (0, 255, 0), "key": "s"},
            1: {"name": "charpy_edge", "color": (255, 0, 0), "key": "e"},
            2: {"name": "charpy_corner", "color": (0, 0, 255), "key": "c"},
            3: {"name": "fracture_surface", "color": (255, 255, 0), "key": "f"},
            4: {"name": "measurement_point", "color": (255, 0, 255), "key": "m"}
        }
        
        self.current_class = 1  # Start with charpy_edge
        self.drawing = False
        self.start_point = None
        
        # Load existing images
        self.load_images()
        
    def load_images(self):
        """Load all images from the dataset"""
        for split in ['train', 'val', 'test']:
            images_dir = f"{self.dataset_path}/images/{split}"
            if os.path.exists(images_dir):
                pattern = os.path.join(images_dir, "*.jpg")
                split_images = glob.glob(pattern)
                for img_path in split_images:
                    self.image_files.append({
                        'path': img_path,
                        'split': split,
                        'basename': os.path.splitext(os.path.basename(img_path))[0]
                    })
        
        print(f"Found {len(self.image_files)} images to annotate")
        
    def load_existing_annotations(self, image_info):
        """Load existing annotations for the current image"""
        split = image_info['split']
        basename = image_info['basename']
        label_path = f"{self.dataset_path}/labels/{split}/{basename}.txt"
        
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append({
                                'class': class_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
        
        return annotations
    
    def save_annotations(self, image_info, annotations):
        """Save annotations to label file"""
        split = image_info['split']
        basename = image_info['basename']
        label_path = f"{self.dataset_path}/labels/{split}/{basename}.txt"
        
        # Ensure labels directory exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
    
    def draw_annotations(self, image, annotations):
        """Draw existing annotations on the image"""
        h, w = image.shape[:2]
        display_image = image.copy()
        
        for ann in annotations:
            class_id = ann['class']
            if class_id in self.classes:
                color = self.classes[class_id]['color']
                name = self.classes[class_id]['name']
                
                # Convert normalized coordinates to pixel coordinates
                x_center = int(ann['x_center'] * w)
                y_center = int(ann['y_center'] * h)
                box_width = int(ann['width'] * w)
                box_height = int(ann['height'] * h)
                
                # Calculate box corners
                x1 = x_center - box_width // 2
                y1 = y_center - box_height // 2
                x2 = x_center + box_width // 2
                y2 = y_center + box_height // 2
                
                # Draw rectangle
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_id}: {name}"
                cv2.putText(display_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_image
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                
                # Calculate bounding box
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure x1,y1 is top-left and x2,y2 is bottom-right
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Convert to normalized YOLO format
                h, w = self.current_image.shape[:2]
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Add annotation
                annotation = {
                    'class': self.current_class,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                }
                
                self.current_annotations.append(annotation)
                print(f"Added {self.classes[self.current_class]['name']} annotation")
    
    def display_help(self):
        """Display help information"""
        print("\n" + "="*60)
        print("MULTI-CLASS CHARPY ANNOTATION TOOL")
        print("="*60)
        print("CONTROLS:")
        print("  Mouse: Click and drag to draw bounding boxes")
        print("  s: Select charpy_specimen class")
        print("  e: Select charpy_edge class")
        print("  c: Select charpy_corner class") 
        print("  f: Select fracture_surface class")
        print("  m: Select measurement_point class")
        print("  d: Delete last annotation")
        print("  n: Next image")
        print("  p: Previous image")
        print("  r: Reset all annotations for current image")
        print("  q: Quit and save")
        print("  h: Show this help")
        print("="*60)
        print("CLASSES:")
        for class_id, info in self.classes.items():
            print(f"  {class_id}: {info['name']} (key: {info['key']})")
        print("="*60)
    
    def run(self):
        """Main annotation loop"""
        if not self.image_files:
            print("No images found!")
            return
            
        self.display_help()
        
        cv2.namedWindow('Charpy Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Charpy Annotator', self.mouse_callback)
        
        while self.current_index < len(self.image_files):
            # Load current image
            image_info = self.image_files[self.current_index]
            self.current_image = cv2.imread(image_info['path'])
            self.current_image_path = image_info['path']
            
            if self.current_image is None:
                print(f"Could not load image: {image_info['path']}")
                self.current_index += 1
                continue
            
            # Load existing annotations
            self.current_annotations = self.load_existing_annotations(image_info)
            
            print(f"\nImage {self.current_index + 1}/{len(self.image_files)}: {image_info['basename']}")
            print(f"Split: {image_info['split']}")
            print(f"Current class: {self.classes[self.current_class]['name']}")
            print(f"Existing annotations: {len(self.current_annotations)}")
            
            while True:
                # Draw image with annotations
                display_image = self.draw_annotations(self.current_image, self.current_annotations)
                
                # Show current class
                class_info = self.classes[self.current_class]
                cv2.putText(display_image, f"Current: {class_info['name']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, class_info['color'], 2)
                
                # Show image info
                cv2.putText(display_image, f"Image: {self.current_index + 1}/{len(self.image_files)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Charpy Annotator', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Save and quit
                    self.save_annotations(image_info, self.current_annotations)
                    print("Saved annotations and exiting...")
                    cv2.destroyAllWindows()
                    return
                    
                elif key == ord('n'):
                    # Next image
                    self.save_annotations(image_info, self.current_annotations)
                    print("Saved annotations")
                    self.current_index += 1
                    break
                    
                elif key == ord('p'):
                    # Previous image
                    self.save_annotations(image_info, self.current_annotations)
                    print("Saved annotations")
                    self.current_index = max(0, self.current_index - 1)
                    break
                    
                elif key == ord('d'):
                    # Delete last annotation
                    if self.current_annotations:
                        deleted = self.current_annotations.pop()
                        print(f"Deleted {self.classes[deleted['class']]['name']} annotation")
                    
                elif key == ord('r'):
                    # Reset annotations
                    self.current_annotations = []
                    print("Reset all annotations for this image")
                    
                elif key == ord('h'):
                    # Show help
                    self.display_help()
                    
                # Class selection keys
                elif key == ord('s'):
                    self.current_class = 0
                    print(f"Selected class: {self.classes[0]['name']}")
                elif key == ord('e'):
                    self.current_class = 1
                    print(f"Selected class: {self.classes[1]['name']}")
                elif key == ord('c'):
                    self.current_class = 2
                    print(f"Selected class: {self.classes[2]['name']}")
                elif key == ord('f'):
                    self.current_class = 3
                    print(f"Selected class: {self.classes[3]['name']}")
                elif key == ord('m'):
                    self.current_class = 4
                    print(f"Selected class: {self.classes[4]['name']}")
        
        cv2.destroyAllWindows()
        print("Annotation complete!")

def main():
    dataset_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    print("🎯 MULTI-CLASS CHARPY ANNOTATOR")
    print("=" * 50)
    print("This tool will help you add detailed annotations to your Charpy images.")
    print("You can annotate:")
    print("  - charpy_specimen (overall specimen)")
    print("  - charpy_edge (specimen edges)")
    print("  - charpy_corner (specimen corners)")
    print("  - fracture_surface (fracture areas)")
    print("  - measurement_point (measurement locations)")
    print()
    
    response = input("Start annotation? (y/N): ").lower().strip()
    if response != 'y':
        print("Annotation cancelled")
        return
    
    annotator = MultiClassAnnotator(dataset_path)
    annotator.run()

if __name__ == "__main__":
    main()