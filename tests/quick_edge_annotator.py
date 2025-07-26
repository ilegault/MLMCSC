#!/usr/bin/env python3
"""
Quick Edge Annotator
Helps quickly add charpy_edge annotations to all images based on the existing charpy_specimen boxes
"""

import os
import glob
import cv2
import numpy as np

def auto_generate_edge_annotations():
    """
    Automatically generate charpy_edge annotations based on existing charpy_specimen annotations
    This creates edge boxes around the perimeter of the specimen boxes
    """
    
    dataset_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    
    print("🚀 QUICK EDGE ANNOTATION GENERATOR")
    print("=" * 50)
    print("This will automatically generate charpy_edge annotations")
    print("based on your existing charpy_specimen annotations.")
    print()
    
    response = input("Continue? (y/N): ").lower().strip()
    if response != 'y':
        print("Cancelled")
        return
    
    total_processed = 0
    total_edges_added = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = f"{dataset_path}/labels/{split}"
        if not os.path.exists(labels_dir):
            continue
            
        print(f"\n📁 Processing {split.upper()} split:")
        
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for label_file in label_files:
            # Read existing annotations
            annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            annotations.append({
                                'class': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
            
            # Find charpy_specimen annotations (class 0)
            specimen_boxes = [ann for ann in annotations if ann['class'] == 0]
            
            if not specimen_boxes:
                continue
            
            # Generate edge annotations for each specimen
            new_annotations = annotations.copy()
            edges_added = 0
            
            for specimen in specimen_boxes:
                # Generate 4 edge boxes around the specimen
                x_c, y_c = specimen['x_center'], specimen['y_center']
                w, h = specimen['width'], specimen['height']
                
                # Edge thickness (relative to specimen size)
                edge_thickness = min(w, h) * 0.15
                
                # Top edge
                top_edge = {
                    'class': 1,  # charpy_edge
                    'x_center': x_c,
                    'y_center': y_c - h/2 + edge_thickness/2,
                    'width': w * 0.8,
                    'height': edge_thickness
                }
                
                # Bottom edge
                bottom_edge = {
                    'class': 1,
                    'x_center': x_c,
                    'y_center': y_c + h/2 - edge_thickness/2,
                    'width': w * 0.8,
                    'height': edge_thickness
                }
                
                # Left edge
                left_edge = {
                    'class': 1,
                    'x_center': x_c - w/2 + edge_thickness/2,
                    'y_center': y_c,
                    'width': edge_thickness,
                    'height': h * 0.6
                }
                
                # Right edge
                right_edge = {
                    'class': 1,
                    'x_center': x_c + w/2 - edge_thickness/2,
                    'y_center': y_c,
                    'width': edge_thickness,
                    'height': h * 0.6
                }
                
                # Add edges (ensure they're within bounds)
                for edge in [top_edge, bottom_edge, left_edge, right_edge]:
                    if (0 < edge['x_center'] < 1 and 0 < edge['y_center'] < 1 and
                        edge['width'] > 0 and edge['height'] > 0):
                        new_annotations.append(edge)
                        edges_added += 1
            
            # Write updated annotations
            if edges_added > 0:
                with open(label_file, 'w') as f:
                    for ann in new_annotations:
                        f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
                
                basename = os.path.basename(label_file)
                print(f"   ✅ {basename}: Added {edges_added} edge annotations")
                total_edges_added += edges_added
            
            total_processed += 1
    
    print(f"\n🎉 COMPLETED!")
    print(f"✅ Processed {total_processed} files")
    print(f"✅ Added {total_edges_added} edge annotations")
    print("\n💡 Next steps:")
    print("1. Run 'python analyze_dataset.py' to see the results")
    print("2. Use 'python multi_class_annotator.py' to refine and add more classes")

def auto_generate_corner_annotations():
    """Generate corner annotations at specimen corners"""
    
    dataset_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    
    print("\n🔄 ADDING CORNER ANNOTATIONS")
    print("=" * 30)
    
    total_corners_added = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = f"{dataset_path}/labels/{split}"
        if not os.path.exists(labels_dir):
            continue
            
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for label_file in label_files:
            # Read existing annotations
            annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            annotations.append({
                                'class': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
            
            # Find charpy_specimen annotations (class 0)
            specimen_boxes = [ann for ann in annotations if ann['class'] == 0]
            
            if not specimen_boxes:
                continue
            
            new_annotations = annotations.copy()
            corners_added = 0
            
            for specimen in specimen_boxes:
                x_c, y_c = specimen['x_center'], specimen['y_center']
                w, h = specimen['width'], specimen['height']
                
                # Corner size
                corner_size = min(w, h) * 0.1
                
                # Four corners
                corners = [
                    # Top-left
                    {
                        'class': 2,  # charpy_corner
                        'x_center': x_c - w/2 + corner_size/2,
                        'y_center': y_c - h/2 + corner_size/2,
                        'width': corner_size,
                        'height': corner_size
                    },
                    # Top-right
                    {
                        'class': 2,
                        'x_center': x_c + w/2 - corner_size/2,
                        'y_center': y_c - h/2 + corner_size/2,
                        'width': corner_size,
                        'height': corner_size
                    },
                    # Bottom-left
                    {
                        'class': 2,
                        'x_center': x_c - w/2 + corner_size/2,
                        'y_center': y_c + h/2 - corner_size/2,
                        'width': corner_size,
                        'height': corner_size
                    },
                    # Bottom-right
                    {
                        'class': 2,
                        'x_center': x_c + w/2 - corner_size/2,
                        'y_center': y_c + h/2 - corner_size/2,
                        'width': corner_size,
                        'height': corner_size
                    }
                ]
                
                # Add valid corners
                for corner in corners:
                    if (0 < corner['x_center'] < 1 and 0 < corner['y_center'] < 1):
                        new_annotations.append(corner)
                        corners_added += 1
            
            # Write updated annotations
            if corners_added > 0:
                with open(label_file, 'w') as f:
                    for ann in new_annotations:
                        f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
                
                total_corners_added += corners_added
    
    print(f"✅ Added {total_corners_added} corner annotations")

def main():
    print("🎯 QUICK ANNOTATION HELPERS")
    print("=" * 50)
    print("These tools help you quickly generate initial annotations")
    print("for edges and corners based on your existing specimen boxes.")
    print()
    print("Options:")
    print("1. Generate edge annotations")
    print("2. Generate edge + corner annotations")
    print("3. Cancel")
    print()
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        auto_generate_edge_annotations()
    elif choice == "2":
        auto_generate_edge_annotations()
        auto_generate_corner_annotations()
    else:
        print("Cancelled")

if __name__ == "__main__":
    main()