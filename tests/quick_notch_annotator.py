#!/usr/bin/env python3
"""
Quick Notch Annotator for Charpy Specimens
Helps you quickly add notch annotations to your existing dataset
"""

import cv2
import os
import numpy as np
from pathlib import Path

class QuickNotchAnnotator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.current_image = None
        self.current_annotations = []
        self.drawing = False
        self.start_point = None
        
    def load_existing_annotations(self, label_file):
        """Load existing annotations from label file"""
        annotations = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
        return annotations
    
    def save_annotations(self, label_file, annotations):
        """Save annotations to label file"""
        with open(label_file, 'w') as f:
            for annotation in annotations:
                class_id, x_center, y_center, width, height = annotation
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def pixel_to_yolo(self, x1, y1, x2, y2, img_width, img_height):
        """Convert pixel coordinates to YOLO format"""
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        return x_center, y_center, width, height
    
    def yolo_to_pixel(self, x_center, y_center, width, height, img_width, img_height):
        """Convert YOLO format to pixel coordinates"""
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        return x1, y1, x2, y2
    
    def draw_annotations(self, img, annotations):
        """Draw existing annotations on image"""
        h, w = img.shape[:2]
        colors = {0: (0, 255, 0), 1: (255, 0, 0)}  # Green for specimen, Red for notch
        labels = {0: "Specimen", 1: "Notch"}
        
        for class_id, x_center, y_center, width, height in annotations:
            x1, y1, x2, y2 = self.yolo_to_pixel(x_center, y_center, width, height, w, h)
            color = colors.get(class_id, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{labels.get(class_id, 'Unknown')}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.current_image.copy()
                img_copy = self.draw_annotations(img_copy, self.current_annotations)
                cv2.rectangle(img_copy, self.start_point, (x, y), (255, 0, 0), 2)
                cv2.putText(img_copy, "Drawing Notch...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow('Annotator', img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Convert to YOLO format and add as notch (class 1)
                h, w = self.current_image.shape[:2]
                x_center, y_center, width, height = self.pixel_to_yolo(
                    self.start_point[0], self.start_point[1], 
                    end_point[0], end_point[1], w, h
                )
                
                # Add notch annotation (class 1)
                self.current_annotations.append((1, x_center, y_center, width, height))
                
                # Redraw image
                img_copy = self.current_image.copy()
                img_copy = self.draw_annotations(img_copy, self.current_annotations)
                cv2.imshow('Annotator', img_copy)
    
    def annotate_split(self, split='train', start_from=0):
        """Annotate images in a specific split"""
        images_dir = self.dataset_path / 'images' / split
        labels_dir = self.dataset_path / 'labels' / split
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return
        
        image_files = sorted(list(images_dir.glob('*.jpg')))[start_from:]
        
        print(f"🖼️  Annotating {len(image_files)} images in {split} set")
        print("📝 Instructions:")
        print("   - Click and drag to draw bounding box around the NOTCH")
        print("   - Press 's' to save and go to next image")
        print("   - Press 'd' to delete last notch annotation")
        print("   - Press 'q' to quit")
        print("   - Press 'n' to skip image without saving")
        
        cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotator', self.mouse_callback)
        
        for i, img_file in enumerate(image_files):
            label_file = labels_dir / (img_file.stem + '.txt')
            
            # Load image
            self.current_image = cv2.imread(str(img_file))
            if self.current_image is None:
                continue
            
            # Load existing annotations
            self.current_annotations = self.load_existing_annotations(label_file)
            
            # Check if notch already exists
            has_notch = any(ann[0] == 1 for ann in self.current_annotations)
            
            print(f"\n📷 Image {start_from + i + 1}/{start_from + len(image_files)}: {img_file.name}")
            if has_notch:
                print("   ✅ Already has notch annotation")
            else:
                print("   ⚠️  No notch annotation - please add one")
            
            # Display image with existing annotations
            img_display = self.current_image.copy()
            img_display = self.draw_annotations(img_display, self.current_annotations)
            cv2.imshow('Annotator', img_display)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # Save and next
                    self.save_annotations(label_file, self.current_annotations)
                    print("   💾 Saved annotations")
                    break
                elif key == ord('d'):  # Delete last notch
                    # Remove last notch annotation (class 1)
                    for j in range(len(self.current_annotations) - 1, -1, -1):
                        if self.current_annotations[j][0] == 1:
                            del self.current_annotations[j]
                            print("   🗑️  Deleted last notch")
                            break
                    # Redraw
                    img_display = self.current_image.copy()
                    img_display = self.draw_annotations(img_display, self.current_annotations)
                    cv2.imshow('Annotator', img_display)
                elif key == ord('n'):  # Skip without saving
                    print("   ⏭️  Skipped")
                    break
                elif key == ord('q'):  # Quit
                    print("   🚪 Quitting...")
                    cv2.destroyAllWindows()
                    return
        
        cv2.destroyAllWindows()
        print(f"\n✅ Finished annotating {split} set!")

def main():
    dataset_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    annotator = QuickNotchAnnotator(dataset_path)
    
    print("🔧 QUICK NOTCH ANNOTATOR")
    print("=" * 50)
    print("This tool helps you add notch annotations to your existing dataset")
    print("You'll keep your existing specimen annotations and add notch annotations")
    
    # Choose split to annotate
    print(f"\nWhich set would you like to annotate?")
    print("1. Train set (recommended - start here)")
    print("2. Validation set")
    print("3. Test set")
    
    choice = input("Enter choice (1-3): ").strip()
    
    split_map = {'1': 'train', '2': 'val', '3': 'test'}
    split = split_map.get(choice, 'train')
    
    # Ask about starting point
    start_from = 0
    start_input = input(f"Start from image number (default 1): ").strip()
    if start_input.isdigit():
        start_from = max(0, int(start_input) - 1)
    
    print(f"\n🚀 Starting annotation of {split} set from image {start_from + 1}")
    annotator.annotate_split(split, start_from)
    
    print(f"\n🎯 Next steps:")
    print("1. Annotate at least 20-30 images with notches")
    print("2. Run: python improved_charpy_training.py")
    print("3. Your model will now detect both specimens AND notches!")

if __name__ == "__main__":
    main()