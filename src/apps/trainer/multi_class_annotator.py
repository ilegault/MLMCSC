#!/usr/bin/env python3
"""
Charpy Specimen 3-Class Annotation System
Optimized for fracture surface measurement and specimen detection

Classes:
0. charpy_specimen - Full specimen bounding box
1. charpy_corner - Corner points (4 per specimen)
2. fracture_surface - The fracture surface area

This focused approach is designed for:
- Detecting complete specimens
- Identifying corners for orientation/alignment
- Locating fracture surfaces for measurement
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class Charpy3ClassAnnotator:
    """Annotation tool optimized for 3-class Charpy specimen detection."""

    def __init__(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "data" / "datasets" / "charpy_dataset_v2"
        self.dataset_path = Path(dataset_path)

        # Define only 3 classes for focused detection
        self.classes = {
            0: "charpy_specimen",
            1: "charpy_corner",
            2: "fracture_surface"
        }

        self.class_colors = {
            0: (0, 255, 0),  # Green - Full specimen
            1: (255, 255, 0),  # Cyan - Corners
            2: (255, 0, 0)  # Red - Fracture surface (critical)
        }

        self.class_descriptions = {
            0: "Full specimen bounding box - encompasses entire specimen",
            1: "Corner points - small boxes at 4 corners for orientation",
            2: "Fracture surface - the rough surface area for measurement"
        }

        # Current state
        self.current_image = None
        self.current_image_path = None
        self.current_annotations = []
        self.current_class = 0
        self.image_list = []
        self.current_image_index = 0

        # UI state
        self.drawing = False
        self.start_point = None
        self.temp_rect = None

        # Annotation guidelines
        self.corner_box_size = 0.03  # 3% of image dimension for corner boxes

        self.setup_directories()
        self.setup_gui()

    def setup_directories(self):
        """Create directory structure for annotations."""
        # Create annotation directories
        for split in ['train', 'val', 'test']:
            labels_dir = self.dataset_path / 'labels' / split
            labels_dir.mkdir(parents=True, exist_ok=True)

            images_dir = self.dataset_path / 'images' / split
            images_dir.mkdir(parents=True, exist_ok=True)

        # Check if images are in the root directory and need to be organized
        root_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            root_images.extend(list(self.dataset_path.glob(ext)))

        if root_images:
            print(f"Found {len(root_images)} images in root directory. Organizing into train/val/test splits...")
            
            # Split images: 70% train, 20% val, 10% test
            import random
            random.shuffle(root_images)
            
            n_total = len(root_images)
            n_train = int(0.7 * n_total)
            n_val = int(0.2 * n_total)
            
            train_images = root_images[:n_train]
            val_images = root_images[n_train:n_train + n_val]
            test_images = root_images[n_train + n_val:]
            
            # Move images to appropriate directories
            import shutil
            for images, split in [(train_images, 'train'), (val_images, 'val'), (test_images, 'test')]:
                for img_path in images:
                    dest_path = self.dataset_path / 'images' / split / img_path.name
                    if not dest_path.exists():
                        shutil.move(str(img_path), str(dest_path))
                        print(f"Moved {img_path.name} to {split}")

        # Now collect all organized images and validate them
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / 'images' / split
            if images_dir.exists():
                # Get all image files
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    for img_path in images_dir.glob(ext):
                        # Check if file is not empty and can be loaded
                        if img_path.stat().st_size > 0:
                            try:
                                # Try to load the image to verify it's valid
                                test_img = cv2.imread(str(img_path))
                                if test_img is not None:
                                    self.image_list.append(img_path)
                                else:
                                    print(f"Warning: Could not load image {img_path.name} - skipping")
                            except Exception as e:
                                print(f"Warning: Error loading image {img_path.name}: {e} - skipping")
                        else:
                            print(f"Warning: Empty image file {img_path.name} - removing")
                            img_path.unlink()  # Delete empty file

        print(f"Found {len(self.image_list)} valid images to annotate")

    def setup_gui(self):
        """Setup the annotation GUI."""
        self.root = tk.Tk()
        self.root.title("Charpy 3-Class Annotator - Fracture Surface Focus")
        self.root.geometry("1400x900")

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Class selection
        ttk.Label(control_frame, text="Select Class:", font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        self.class_var = tk.IntVar(value=0)
        for class_id, class_name in self.classes.items():
            color_text = self.get_color_name(self.class_colors[class_id])
            rb = ttk.Radiobutton(
                control_frame,
                text=f"{class_id}: {class_name} ({color_text})",
                variable=self.class_var,
                value=class_id,
                command=self.on_class_changed
            )
            rb.pack(anchor=tk.W, padx=20, pady=5)

        # Class description
        self.desc_label = ttk.Label(
            control_frame,
            text=self.class_descriptions[0],
            wraplength=250,
            font=('Arial', 9),
            foreground='gray'
        )
        self.desc_label.pack(pady=(10, 20), padx=20)

        # Annotation strategy guide
        ttk.Label(control_frame, text="Annotation Strategy:", font=('Arial', 11, 'bold')).pack(pady=(20, 10))

        strategy_text = """1. SPECIMEN (Green):
   - Draw box around entire specimen
   - Include all edges

2. CORNERS (Cyan):
   - Small boxes at 4 corners
   - Use 'Auto-add Corners' button

3. FRACTURE (Red):
   - Box around rough surface
   - Top edge is measurement line
   - Be precise with top edge!"""

        strategy_label = ttk.Label(control_frame, text=strategy_text, font=('Arial', 9))
        strategy_label.pack(padx=20, pady=5)

        # Buttons
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=20)

        ttk.Button(
            control_frame,
            text="Auto-add Corner Boxes",
            command=self.auto_add_corners
        ).pack(fill=tk.X, padx=20, pady=5)

        ttk.Button(
            control_frame,
            text="Clear All Annotations",
            command=self.clear_annotations
        ).pack(fill=tk.X, padx=20, pady=5)

        ttk.Button(
            control_frame,
            text="Delete Last Annotation",
            command=self.delete_last_annotation
        ).pack(fill=tk.X, padx=20, pady=5)

        # Navigation
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=20)

        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, padx=20)

        ttk.Button(nav_frame, text="< Previous", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next >", command=self.next_image).pack(side=tk.RIGHT)

        # Progress
        self.progress_label = ttk.Label(control_frame, text="Progress: 0/0")
        self.progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(control_frame, length=250, mode='determinate')
        self.progress_bar.pack(padx=20, pady=5)

        # Statistics
        self.stats_text = tk.Text(control_frame, height=8, width=35, font=('Courier', 9))
        self.stats_text.pack(pady=(20, 0), padx=20)

        # Right panel - Canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas for image display
        self.canvas = tk.Canvas(canvas_frame, bg='black', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Button-3>', self.on_right_click)  # Right click to delete

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Delete>', lambda e: self.delete_last_annotation())
        self.root.bind('1', lambda e: self.class_var.set(0))
        self.root.bind('2', lambda e: self.class_var.set(1))
        self.root.bind('3', lambda e: self.class_var.set(2))
        self.root.bind('<Control-s>', lambda e: self.save_current_annotations())

        # Load first image
        if self.image_list:
            self.load_image(0)

    def get_color_name(self, bgr_color):
        """Convert BGR color to name."""
        color_map = {
            (0, 255, 0): "Green",
            (255, 255, 0): "Cyan",
            (255, 0, 0): "Red"
        }
        return color_map.get(bgr_color, "Unknown")

    def on_class_changed(self):
        """Handle class selection change."""
        self.current_class = self.class_var.get()
        self.desc_label.config(text=self.class_descriptions[self.current_class])

    def load_image(self, index):
        """Load an image and its annotations."""
        if not 0 <= index < len(self.image_list):
            return

        self.current_image_index = index
        self.current_image_path = self.image_list[index]

        # Load image
        image = cv2.imread(str(self.current_image_path))
        if image is None:
            messagebox.showerror("Error", f"Could not load image: {self.current_image_path}")
            return

        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = self.current_image.shape[:2]

        # Load existing annotations
        self.load_annotations()

        # Display image
        self.display_image()

        # Update progress
        self.update_progress()
        self.update_stats()

    def load_annotations(self):
        """Load existing annotations for current image."""
        self.current_annotations = []

        # Determine split (train/val/test) from image path
        split = self.current_image_path.parent.name
        label_file = self.dataset_path / 'labels' / split / f"{self.current_image_path.stem}.txt"

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

                        self.current_annotations.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height]
                        })

    def save_current_annotations(self):
        """Save annotations for current image."""
        if not self.current_image_path:
            return

        split = self.current_image_path.parent.name
        label_file = self.dataset_path / 'labels' / split / f"{self.current_image_path.stem}.txt"

        with open(label_file, 'w') as f:
            for ann in self.current_annotations:
                class_id = ann['class_id']
                x_center, y_center, width, height = ann['bbox']
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Also save metadata
        metadata = {
            'image': self.current_image_path.name,
            'annotations': len(self.current_annotations),
            'classes': {str(k): sum(1 for a in self.current_annotations if a['class_id'] == k)
                        for k in self.classes.keys()},
            'timestamp': datetime.now().isoformat()
        }

        metadata_file = label_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def display_image(self):
        """Display current image with annotations."""
        if self.current_image is None:
            return

        # Create display image
        display_img = self.current_image.copy()

        # Draw existing annotations
        for ann in self.current_annotations:
            self.draw_annotation(display_img, ann)

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(Image.fromarray(display_img))

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def draw_annotation(self, img, annotation):
        """Draw a single annotation on the image."""
        class_id = annotation['class_id']
        x_center, y_center, width, height = annotation['bbox']

        # Convert normalized to pixel coordinates
        x1 = int((x_center - width / 2) * self.image_width)
        y1 = int((y_center - height / 2) * self.image_height)
        x2 = int((x_center + width / 2) * self.image_width)
        y2 = int((y_center + height / 2) * self.image_height)

        # Draw rectangle
        color = self.class_colors.get(class_id, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{self.classes.get(class_id, 'Unknown')}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        # Special indicator for fracture surface top edge
        if class_id == 2:  # Fracture surface
            # Draw measurement line at top edge
            cv2.line(img, (x1, y1), (x2, y1), (255, 255, 255), 3)
            cv2.putText(img, "MEASURE HERE", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def on_mouse_down(self, event):
        """Handle mouse button press."""
        self.drawing = True
        self.start_point = (event.x, event.y)

    def on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.drawing and self.start_point:
            # Update temporary rectangle
            if self.temp_rect:
                self.canvas.delete(self.temp_rect)

            color = self.get_tk_color(self.class_colors[self.current_class])
            self.temp_rect = self.canvas.create_rectangle(
                self.start_point[0], self.start_point[1],
                event.x, event.y,
                outline=color, width=2
            )

    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if self.drawing and self.start_point:
            self.drawing = False

            # Clear temporary rectangle
            if self.temp_rect:
                self.canvas.delete(self.temp_rect)
                self.temp_rect = None

            # Calculate bounding box
            x1, y1 = self.start_point
            x2, y2 = event.x, event.y

            # Ensure proper order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Minimum box size
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                return

            # Convert to normalized coordinates
            x_center = ((x1 + x2) / 2) / self.image_width
            y_center = ((y1 + y2) / 2) / self.image_height
            width = (x2 - x1) / self.image_width
            height = (y2 - y1) / self.image_height

            # Add annotation
            self.current_annotations.append({
                'class_id': self.current_class,
                'bbox': [x_center, y_center, width, height]
            })

            # Save and refresh
            self.save_current_annotations()
            self.display_image()
            self.update_stats()

    def on_right_click(self, event):
        """Handle right click to delete annotation."""
        # Find closest annotation
        click_x = event.x / self.image_width
        click_y = event.y / self.image_height

        min_dist = float('inf')
        closest_idx = -1

        for i, ann in enumerate(self.current_annotations):
            x_center, y_center = ann['bbox'][0], ann['bbox'][1]
            dist = ((x_center - click_x) ** 2 + (y_center - click_y) ** 2) ** 0.5

            if dist < min_dist and dist < 0.1:  # Within 10% of image
                min_dist = dist
                closest_idx = i

        if closest_idx >= 0:
            del self.current_annotations[closest_idx]
            self.save_current_annotations()
            self.display_image()
            self.update_stats()

    def auto_add_corners(self):
        """Automatically add corner annotations based on specimen bbox."""
        # Find specimen annotation (class 0)
        specimen_ann = None
        for ann in self.current_annotations:
            if ann['class_id'] == 0:
                specimen_ann = ann
                break

        if not specimen_ann:
            messagebox.showwarning("Warning", "Please annotate the specimen first (class 0)")
            return

        # Get specimen bbox
        x_center, y_center, width, height = specimen_ann['bbox']

        # Calculate corner positions
        corners = [
            (x_center - width / 2, y_center - height / 2),  # Top-left
            (x_center + width / 2, y_center - height / 2),  # Top-right
            (x_center - width / 2, y_center + height / 2),  # Bottom-left
            (x_center + width / 2, y_center + height / 2),  # Bottom-right
        ]

        # Remove existing corner annotations
        self.current_annotations = [a for a in self.current_annotations if a['class_id'] != 1]

        # Add corner annotations
        for corner_x, corner_y in corners:
            self.current_annotations.append({
                'class_id': 1,  # Corner class
                'bbox': [corner_x, corner_y, self.corner_box_size, self.corner_box_size]
            })

        self.save_current_annotations()
        self.display_image()
        self.update_stats()
        messagebox.showinfo("Success", "Added 4 corner annotations")

    def clear_annotations(self):
        """Clear all annotations for current image."""
        if messagebox.askyesno("Confirm", "Clear all annotations for this image?"):
            self.current_annotations = []
            self.save_current_annotations()
            self.display_image()
            self.update_stats()

    def delete_last_annotation(self):
        """Delete the last added annotation."""
        if self.current_annotations:
            self.current_annotations.pop()
            self.save_current_annotations()
            self.display_image()
            self.update_stats()

    def prev_image(self):
        """Go to previous image."""
        if self.current_image_index > 0:
            self.load_image(self.current_image_index - 1)

    def next_image(self):
        """Go to next image."""
        if self.current_image_index < len(self.image_list) - 1:
            self.load_image(self.current_image_index + 1)

    def update_progress(self):
        """Update progress display."""
        annotated = sum(1 for img in self.image_list
                        if self.has_annotations(img))
        total = len(self.image_list)

        self.progress_label.config(text=f"Progress: {annotated}/{total} images annotated")
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = annotated

    def has_annotations(self, image_path):
        """Check if image has annotations."""
        split = image_path.parent.parent.name
        label_file = self.dataset_path / 'labels' / split / f"{image_path.stem}.txt"
        return label_file.exists() and label_file.stat().st_size > 0

    def update_stats(self):
        """Update statistics display."""
        stats = []
        stats.append(f"Image: {self.current_image_path.name}")
        stats.append(f"Size: {self.image_width}x{self.image_height}")
        stats.append(f"")
        stats.append(f"Annotations in this image:")

        class_counts = {k: 0 for k in self.classes.keys()}
        for ann in self.current_annotations:
            class_counts[ann['class_id']] += 1

        for class_id, count in class_counts.items():
            stats.append(f"  {self.classes[class_id]}: {count}")

        # Check annotation completeness
        stats.append(f"")
        stats.append(f"Status:")
        if class_counts[0] == 0:
            stats.append("  ⚠️ Missing specimen annotation")
        if class_counts[1] < 4 and class_counts[0] > 0:
            stats.append("  ⚠️ Missing corner annotations")
        if class_counts[2] == 0:
            stats.append("  ⚠️ Missing fracture surface")

        if class_counts[0] > 0 and class_counts[1] >= 4 and class_counts[2] > 0:
            stats.append("  ✅ Complete annotation!")

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, '\n'.join(stats))

    def get_tk_color(self, bgr_color):
        """Convert BGR color to Tk hex color."""
        return f'#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}'

    def run(self):
        """Run the annotation tool."""
        self.root.mainloop()


def create_dataset_yaml(dataset_path):
    """Create dataset.yaml file for YOLO training."""
    dataset_path = Path(dataset_path)

    yaml_content = f"""# Charpy 3-Class Dataset Configuration
# Optimized for fracture surface measurement

path: {dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 3  # Number of classes
names:
  0: charpy_specimen  # Full specimen bounding box
  1: charpy_corner    # Corner points for orientation
  2: fracture_surface # Fracture surface for measurement

# Class descriptions
descriptions:
  charpy_specimen: "Complete Charpy impact test specimen"
  charpy_corner: "Corner points of the specimen (4 per specimen)"
  fracture_surface: "Fracture surface area - top edge used for measurement"

# Training notes
notes: |
  This dataset is optimized for:
  1. Detecting complete Charpy specimens
  2. Finding corner points for orientation/alignment
  3. Locating fracture surfaces for measurement

  The fracture_surface class is critical - its top edge is used
  for measuring fracture characteristics.
"""

    yaml_file = dataset_path / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)

    print(f"Created dataset configuration: {yaml_file}")


def main():
    """Main function to run the annotator."""
    print("🎯 CHARPY 3-CLASS ANNOTATION SYSTEM")
    print("=" * 50)
    print("Optimized for fracture surface detection and measurement")
    print()
    print("Classes:")
    print("  0. charpy_specimen - Full specimen bbox")
    print("  1. charpy_corner - Corner points (4 per specimen)")
    print("  2. fracture_surface - Fracture surface area")
    print()
    print("The fracture surface top edge will be used for measurements!")
    print()

    # Use absolute path to ensure it works regardless of where script is run from
    dataset_path = Path(__file__).parent.parent / "data" / "datasets" / "charpy_dataset_v2"

    # Create dataset.yaml
    create_dataset_yaml(dataset_path)

    # Launch annotator
    annotator = Charpy3ClassAnnotator(dataset_path)

    print("Launching annotation tool...")
    print()
    print("SHORTCUTS:")
    print("  1/2/3 - Select class")
    print("  Left/Right Arrow - Navigate images")
    print("  Right Click - Delete annotation")
    print("  Delete - Delete last annotation")
    print("  Ctrl+S - Save annotations")
    print()

    annotator.run()


if __name__ == "__main__":
    main()