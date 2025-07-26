#!/usr/bin/env python3
"""
Fix Corrupt Annotations Script
Fixes bounding boxes that extend beyond image boundaries
"""

import os
import glob
import shutil

def fix_corrupt_annotations():
    """Fix annotations with coordinates outside [0,1] range"""
    
    dataset_path = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    
    print("🔧 FIXING CORRUPT ANNOTATIONS")
    print("=" * 50)
    print("This will fix bounding boxes that extend beyond image boundaries")
    print("(coordinates > 1.0 or < 0.0)")
    print()
    
    response = input("Continue? (y/N): ").lower().strip()
    if response != 'y':
        print("Cancelled")
        return
    
    total_files_processed = 0
    total_annotations_fixed = 0
    total_annotations_removed = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = f"{dataset_path}/labels/{split}"
        if not os.path.exists(labels_dir):
            continue
            
        print(f"\n📁 Processing {split.upper()} split:")
        
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for label_file in label_files:
            original_annotations = []
            fixed_annotations = []
            
            # Read original annotations
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                original_annotations.append({
                                    'class': class_id,
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height,
                                    'original_line': line
                                })
                            except ValueError:
                                continue
            
            if not original_annotations:
                continue
            
            annotations_fixed_in_file = 0
            annotations_removed_in_file = 0
            
            for ann in original_annotations:
                # Check if annotation is corrupt
                x_min = ann['x_center'] - ann['width'] / 2
                y_min = ann['y_center'] - ann['height'] / 2
                x_max = ann['x_center'] + ann['width'] / 2
                y_max = ann['y_center'] + ann['height'] / 2
                
                is_corrupt = (x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1 or
                             ann['x_center'] < 0 or ann['x_center'] > 1 or
                             ann['y_center'] < 0 or ann['y_center'] > 1 or
                             ann['width'] <= 0 or ann['height'] <= 0)
                
                if is_corrupt:
                    # Try to fix the annotation
                    fixed_ann = fix_annotation(ann)
                    
                    if fixed_ann:
                        fixed_annotations.append(fixed_ann)
                        annotations_fixed_in_file += 1
                    else:
                        # Remove annotation if it can't be fixed
                        annotations_removed_in_file += 1
                else:
                    # Keep good annotations as-is
                    fixed_annotations.append(ann)
            
            # Write fixed annotations if any changes were made
            if annotations_fixed_in_file > 0 or annotations_removed_in_file > 0:
                # Backup original file
                backup_file = label_file + '.backup'
                shutil.copy2(label_file, backup_file)
                
                # Write fixed annotations
                with open(label_file, 'w') as f:
                    for ann in fixed_annotations:
                        f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
                
                basename = os.path.basename(label_file)
                if annotations_fixed_in_file > 0:
                    print(f"   🔧 {basename}: Fixed {annotations_fixed_in_file} annotations")
                if annotations_removed_in_file > 0:
                    print(f"   🗑️  {basename}: Removed {annotations_removed_in_file} unfixable annotations")
                
                total_annotations_fixed += annotations_fixed_in_file
                total_annotations_removed += annotations_removed_in_file
            
            total_files_processed += 1
    
    print(f"\n🎉 FIXING COMPLETED!")
    print(f"✅ Processed {total_files_processed} files")
    print(f"✅ Fixed {total_annotations_fixed} annotations")
    print(f"✅ Removed {total_annotations_removed} unfixable annotations")
    print(f"✅ Original files backed up with .backup extension")
    
    if total_annotations_fixed > 0 or total_annotations_removed > 0:
        print(f"\n💡 Next steps:")
        print(f"1. Run 'python analyze_dataset.py' to verify the fixes")
        print(f"2. Restart your training - the corrupt annotation errors should be gone")

def fix_annotation(ann):
    """Try to fix a corrupt annotation by clipping to valid bounds"""
    
    # Calculate current bounds
    x_min = ann['x_center'] - ann['width'] / 2
    y_min = ann['y_center'] - ann['height'] / 2
    x_max = ann['x_center'] + ann['width'] / 2
    y_max = ann['y_center'] + ann['height'] / 2
    
    # Clip bounds to [0, 1]
    x_min_clipped = max(0.0, min(1.0, x_min))
    y_min_clipped = max(0.0, min(1.0, y_min))
    x_max_clipped = max(0.0, min(1.0, x_max))
    y_max_clipped = max(0.0, min(1.0, y_max))
    
    # Calculate new center and dimensions
    new_width = x_max_clipped - x_min_clipped
    new_height = y_max_clipped - y_min_clipped
    
    # Check if the clipped box is still valid (has area)
    if new_width <= 0.01 or new_height <= 0.01:  # Minimum size threshold
        return None  # Can't fix - too small
    
    new_x_center = (x_min_clipped + x_max_clipped) / 2
    new_y_center = (y_min_clipped + y_max_clipped) / 2
    
    # Ensure center is within bounds
    new_x_center = max(0.0, min(1.0, new_x_center))
    new_y_center = max(0.0, min(1.0, new_y_center))
    
    return {
        'class': ann['class'],
        'x_center': new_x_center,
        'y_center': new_y_center,
        'width': new_width,
        'height': new_height
    }

def main():
    print("🔧 CORRUPT ANNOTATION FIXER")
    print("=" * 50)
    print("Your training detected corrupt annotations with coordinates")
    print("outside the valid [0,1] range. This script will:")
    print()
    print("1. Find all corrupt annotations")
    print("2. Try to fix them by clipping to image boundaries")
    print("3. Remove annotations that can't be fixed")
    print("4. Backup original files")
    print()
    print("This will resolve the training errors you encountered.")
    print()
    
    fix_corrupt_annotations()

if __name__ == "__main__":
    main()