#!/usr/bin/env python3
"""
Dataset Analysis Script for Charpy Detection
Analyzes what classes are actually present in your dataset
"""

import os
from collections import Counter
import glob

def analyze_labels(labels_dir):
    """Analyze label files to see what classes are present"""
    class_counts = Counter()
    total_annotations = 0
    files_analyzed = 0
    
    # Get all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    for label_file in label_files:
        files_analyzed += 1
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 5:  # class_id x y w h
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_annotations += 1
    
    return class_counts, total_annotations, files_analyzed

def main():
    # Define class names from your dataset
    class_names = {
        0: "charpy_specimen",
        1: "charpy_edge",
        2: "charpy_corner",
        3: "fracture_surface",
        4: "measurement_point"
    }
    
    print("🔍 CHARPY DATASET ANALYSIS")
    print("=" * 50)
    
    # Analyze each split
    base_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests"
    for split in ['train', 'val', 'test']:
        labels_dir = f"{base_dir}/data/charpy_dataset/labels/{split}"
        if os.path.exists(labels_dir):
            print(f"\n📁 {split.upper()} SET:")
            class_counts, total_annotations, files_analyzed = analyze_labels(labels_dir)
            
            print(f"   Files analyzed: {files_analyzed}")
            print(f"   Total annotations: {total_annotations}")
            print(f"   Classes found:")
            
            if class_counts:
                for class_id, count in sorted(class_counts.items()):
                    class_name = class_names.get(class_id, f"Unknown_{class_id}")
                    percentage = (count / total_annotations) * 100
                    print(f"     Class {class_id} ({class_name}): {count} annotations ({percentage:.1f}%)")
            else:
                print("     ❌ No annotations found!")
        else:
            print(f"\n📁 {split.upper()} SET: Directory not found")
    
    print("\n" + "=" * 50)
    print("🎯 ANALYSIS SUMMARY:")
    print("=" * 50)
    
    # Check if this is a single-class dataset
    all_classes = set()
    for split in ['train', 'val', 'test']:
        labels_dir = f"{base_dir}/data/charpy_dataset/labels/{split}"
        if os.path.exists(labels_dir):
            class_counts, _, _ = analyze_labels(labels_dir)
            all_classes.update(class_counts.keys())
    
    if len(all_classes) == 1 and 0 in all_classes:
        print("⚠️  ISSUE DETECTED: Single-class dataset!")
        print("   - You have 5 classes defined but only using class 0 (charpy_specimen)")
        print("   - Your model will only learn to detect 'existence' of specimens")
        print("   - It won't learn to identify specific features like notches, edges, etc.")
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Add annotations for other classes (notch, edge, corner, etc.)")
        print("   2. Or reduce your dataset to 1 class if you only need specimen detection")
        print("   3. Consider using a different annotation tool to add detailed labels")
    elif len(all_classes) > 1:
        print("✅ Multi-class dataset detected!")
        print(f"   Using {len(all_classes)} out of 5 defined classes")
    else:
        print("❌ No valid annotations found!")

if __name__ == "__main__":
    main()