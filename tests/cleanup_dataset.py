#!/usr/bin/env python3
"""
Dataset Cleanup Script
Removes finger images (charpy_0001-0005) and updates class configuration
"""

import os
import glob
import shutil
from pathlib import Path

def cleanup_finger_images():
    """Remove finger images from all dataset splits"""
    base_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    
    # Files to remove (finger images)
    finger_files = ['charpy_0001', 'charpy_0002', 'charpy_0003', 'charpy_0004', 'charpy_0005']
    
    print("🧹 CLEANING UP FINGER IMAGES")
    print("=" * 50)
    
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Cleaning {split.upper()} split:")
        
        # Clean labels
        labels_dir = f"{base_dir}/labels/{split}"
        if os.path.exists(labels_dir):
            for finger_file in finger_files:
                label_file = f"{labels_dir}/{finger_file}.txt"
                if os.path.exists(label_file):
                    os.remove(label_file)
                    print(f"   ✅ Removed {finger_file}.txt")
                else:
                    print(f"   ⚠️  {finger_file}.txt not found")
        
        # Clean images (check common image extensions)
        images_dir = f"{base_dir}/images/{split}"
        if os.path.exists(images_dir):
            for finger_file in finger_files:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_file = f"{images_dir}/{finger_file}{ext}"
                    if os.path.exists(image_file):
                        os.remove(image_file)
                        print(f"   ✅ Removed {finger_file}{ext}")

def update_class_labels():
    """Update all label files to remove charpy_notch class and renumber"""
    base_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    
    # Class mapping: old_class -> new_class
    # Remove class 1 (charpy_notch) and shift everything down
    class_mapping = {
        0: 0,  # charpy_specimen stays 0
        # 1: removed (charpy_notch)
        2: 1,  # charpy_edge becomes 1
        3: 2,  # charpy_corner becomes 2
        4: 3,  # fracture_surface becomes 3
        5: 4   # measurement_point becomes 4
    }
    
    print("\n🔄 UPDATING CLASS LABELS")
    print("=" * 50)
    
    total_updated = 0
    total_removed_annotations = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = f"{base_dir}/labels/{split}"
        if not os.path.exists(labels_dir):
            continue
            
        print(f"\n📁 Processing {split.upper()} split:")
        
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for label_file in label_files:
            updated_lines = []
            removed_count = 0
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        old_class = int(parts[0])
                        
                        if old_class == 1:  # Remove charpy_notch annotations
                            removed_count += 1
                            total_removed_annotations += 1
                        elif old_class in class_mapping:
                            # Update class number
                            new_class = class_mapping[old_class]
                            parts[0] = str(new_class)
                            updated_lines.append(' '.join(parts) + '\n')
            
            # Write updated file
            if updated_lines or removed_count > 0:
                with open(label_file, 'w') as f:
                    f.writelines(updated_lines)
                
                if removed_count > 0:
                    print(f"   📝 {os.path.basename(label_file)}: Removed {removed_count} charpy_notch annotations")
                
                total_updated += 1
    
    print(f"\n✅ Updated {total_updated} label files")
    print(f"✅ Removed {total_removed_annotations} charpy_notch annotations")

def create_updated_dataset_config():
    """Create updated dataset.yaml without charpy_notch class"""
    base_dir = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/data/charpy_dataset"
    
    print("\n📝 CREATING UPDATED DATASET CONFIG")
    print("=" * 50)
    
    # New class configuration (without charpy_notch)
    config_content = f"""names:
  0: charpy_specimen
  1: charpy_edge
  2: charpy_corner
  3: fracture_surface
  4: measurement_point
nc: 5
path: {base_dir}
test: images/test
train: images/train
val: images/val
"""
    
    # Backup original config
    original_config = f"{base_dir}/dataset.yaml"
    backup_config = f"{base_dir}/dataset_backup.yaml"
    
    if os.path.exists(original_config):
        shutil.copy2(original_config, backup_config)
        print(f"✅ Backed up original config to dataset_backup.yaml")
    
    # Write new config
    with open(original_config, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created updated dataset.yaml with 5 classes")
    print("   Classes: charpy_specimen, charpy_edge, charpy_corner, fracture_surface, measurement_point")

def update_analysis_script():
    """Update the analysis script to reflect new class structure"""
    analysis_script = "C:/Users/IGLeg/PycharmProjects/MLMCSC/tests/analyze_dataset.py"
    
    print("\n🔧 UPDATING ANALYSIS SCRIPT")
    print("=" * 50)
    
    # Read current script
    with open(analysis_script, 'r') as f:
        content = f.read()
    
    # Update class names dictionary
    old_class_dict = '''class_names = {
        0: "charpy_specimen",
        1: "charpy_notch", 
        2: "charpy_edge",
        3: "charpy_corner",
        4: "fracture_surface",
        5: "measurement_point"
    }'''
    
    new_class_dict = '''class_names = {
        0: "charpy_specimen",
        1: "charpy_edge",
        2: "charpy_corner",
        3: "fracture_surface",
        4: "measurement_point"
    }'''
    
    # Update the content
    updated_content = content.replace(old_class_dict, new_class_dict)
    
    # Update the issue detection logic
    old_issue_text = '''if len(all_classes) == 1 and 0 in all_classes:
        print("⚠️  ISSUE DETECTED: Single-class dataset!")
        print("   - You have 6 classes defined but only using class 0 (charpy_specimen)")'''
    
    new_issue_text = '''if len(all_classes) == 1 and 0 in all_classes:
        print("⚠️  ISSUE DETECTED: Single-class dataset!")
        print("   - You have 5 classes defined but only using class 0 (charpy_specimen)")'''
    
    updated_content = updated_content.replace(old_issue_text, new_issue_text)
    
    # Update the multi-class detection
    old_multi_text = '''print(f"   Using {len(all_classes)} out of 6 defined classes")'''
    new_multi_text = '''print(f"   Using {len(all_classes)} out of 5 defined classes")'''
    
    updated_content = updated_content.replace(old_multi_text, new_multi_text)
    
    # Write updated script
    with open(analysis_script, 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated analyze_dataset.py with new class structure")

def main():
    print("🚀 CHARPY DATASET CLEANUP")
    print("=" * 50)
    print("This script will:")
    print("1. Remove finger images (charpy_0001-0005) from all splits")
    print("2. Remove charpy_notch class (class 1) from all annotations")
    print("3. Renumber remaining classes: 0,1,2,3,4")
    print("4. Update dataset.yaml configuration")
    print("5. Update analysis script")
    print()
    
    response = input("Continue? (y/N): ").lower().strip()
    if response != 'y':
        print("❌ Cleanup cancelled")
        return
    
    try:
        # Step 1: Remove finger images
        cleanup_finger_images()
        
        # Step 2: Update class labels in all annotation files
        update_class_labels()
        
        # Step 3: Create updated dataset configuration
        create_updated_dataset_config()
        
        # Step 4: Update analysis script
        update_analysis_script()
        
        print("\n" + "=" * 50)
        print("🎉 CLEANUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ Removed finger images (charpy_0001-0005)")
        print("✅ Removed charpy_notch class from all annotations")
        print("✅ Renumbered classes: 0→0, 2→1, 3→2, 4→3, 5→4")
        print("✅ Updated dataset.yaml (5 classes)")
        print("✅ Updated analyze_dataset.py")
        print()
        print("📊 Run 'python analyze_dataset.py' to verify the cleanup")
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        print("Please check the error and try again")

if __name__ == "__main__":
    main()