#!/usr/bin/env python3
"""
Annotation utilities for specimen detection training data preparation.

This module provides tools for creating, converting, and managing annotations
for microscope specimen detection training.
"""

import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """Data class for storing annotation information."""
    class_id: int
    class_name: str
    bbox: List[float]  # [x_center, y_center, width, height] in normalized coordinates
    confidence: float = 1.0
    
    def to_yolo_format(self) -> str:
        """Convert annotation to YOLO format string."""
        return f"{self.class_id} {self.bbox[0]:.6f} {self.bbox[1]:.6f} {self.bbox[2]:.6f} {self.bbox[3]:.6f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert annotation to dictionary."""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'bbox': self.bbox,
            'confidence': self.confidence
        }


class AnnotationConverter:
    """Converts between different annotation formats."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
    
    def coco_to_yolo(self, coco_file: str, output_dir: str) -> None:
        """
        Convert COCO format annotations to YOLO format.
        
        Args:
            coco_file: Path to COCO JSON file
            output_dir: Directory to save YOLO format files
        """
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create image_id to filename mapping
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Convert each image's annotations
        for image_id, annotations in annotations_by_image.items():
            image_info = images[image_id]
            image_width = image_info['width']
            image_height = image_info['height']
            
            # Create YOLO format file
            filename = Path(image_info['file_name']).stem + '.txt'
            output_file = output_path / filename
            
            with open(output_file, 'w') as f:
                for ann in annotations:
                    # Convert COCO bbox [x, y, width, height] to YOLO format
                    x, y, w, h = ann['bbox']
                    
                    # Convert to center coordinates and normalize
                    x_center = (x + w / 2) / image_width
                    y_center = (y + h / 2) / image_height
                    width = w / image_width
                    height = h / image_height
                    
                    class_id = ann['category_id']
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"Converted {len(annotations_by_image)} images from COCO to YOLO format")
    
    def pascal_voc_to_yolo(self, voc_dir: str, output_dir: str) -> None:
        """
        Convert Pascal VOC format annotations to YOLO format.
        
        Args:
            voc_dir: Directory containing Pascal VOC XML files
            output_dir: Directory to save YOLO format files
        """
        voc_path = Path(voc_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        xml_files = list(voc_path.glob('*.xml'))
        
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            image_width = int(size.find('width').text)
            image_height = int(size.find('height').text)
            
            # Create YOLO format file
            output_file = output_path / f"{xml_file.stem}.txt"
            
            with open(output_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in self.class_to_id:
                        logger.warning(f"Unknown class: {class_name}")
                        continue
                    
                    class_id = self.class_to_id[class_name]
                    
                    # Get bounding box
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    x_center = (xmin + xmax) / 2 / image_width
                    y_center = (ymin + ymax) / 2 / image_height
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"Converted {len(xml_files)} files from Pascal VOC to YOLO format")
    
    def yolo_to_coco(self, yolo_dir: str, images_dir: str, output_file: str) -> None:
        """
        Convert YOLO format annotations to COCO format.
        
        Args:
            yolo_dir: Directory containing YOLO format txt files
            images_dir: Directory containing corresponding images
            output_file: Output COCO JSON file path
        """
        yolo_path = Path(yolo_dir)
        images_path = Path(images_dir)
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for i, class_name in enumerate(self.class_names):
            coco_data['categories'].append({
                'id': i,
                'name': class_name,
                'supercategory': 'specimen'
            })
        
        annotation_id = 1
        
        # Process each annotation file
        txt_files = list(yolo_path.glob('*.txt'))
        
        for i, txt_file in enumerate(txt_files):
            # Find corresponding image
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_image = images_path / f"{txt_file.stem}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            if image_file is None:
                logger.warning(f"No image found for {txt_file}")
                continue
            
            # Get image dimensions
            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(f"Could not load image: {image_file}")
                continue
            
            height, width = image.shape[:2]
            
            # Add image info
            coco_data['images'].append({
                'id': i,
                'file_name': image_file.name,
                'width': width,
                'height': height
            })
            
            # Process annotations
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height
                    
                    # Convert to COCO format (top-left corner)
                    x = x_center - bbox_width / 2
                    y = y_center - bbox_height / 2
                    
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': i,
                        'category_id': class_id,
                        'bbox': [x, y, bbox_width, bbox_height],
                        'area': bbox_width * bbox_height,
                        'iscrowd': 0
                    })
                    
                    annotation_id += 1
        
        # Save COCO format file
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Converted {len(txt_files)} files from YOLO to COCO format")


class SimpleAnnotationTool:
    """Simple GUI tool for creating annotations."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.current_class = 0
        self.annotations = []
        self.current_image = None
        self.image_path = None
        self.drawing = False
        self.start_point = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title("Simple Annotation Tool")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        ttk.Button(control_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Clear All", command=self.clear_annotations).pack(side=tk.LEFT, padx=(0, 5))
        
        # Class selection
        ttk.Label(control_frame, text="Class:").pack(side=tk.LEFT, padx=(20, 5))
        self.class_var = tk.StringVar(value=self.class_names[0])
        class_combo = ttk.Combobox(control_frame, textvariable=self.class_var, values=self.class_names, state="readonly")
        class_combo.pack(side=tk.LEFT, padx=(0, 5))
        class_combo.bind('<<ComboboxSelected>>', self.on_class_changed)
        
        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, bg='white', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def open_image(self):
        """Open an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_image()
    
    def load_image(self):
        """Load and display the current image."""
        if not self.image_path:
            return
        
        # Load image
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Error", "Could not load image")
            return
        
        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = self.current_image.shape[:2]
        
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale = min(canvas_width / self.image_width, canvas_height / self.image_height)
            new_width = int(self.image_width * scale)
            new_height = int(self.image_height * scale)
            
            resized_image = cv2.resize(self.current_image, (new_width, new_height))
            self.display_image = Image.fromarray(resized_image)
            self.photo = ImageTk.PhotoImage(self.display_image)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
            
            self.scale_factor = scale
            self.display_width = new_width
            self.display_height = new_height
            
            # Load existing annotations if available
            self.load_existing_annotations()
            self.draw_annotations()
            
            self.status_var.set(f"Loaded: {Path(self.image_path).name}")
    
    def load_existing_annotations(self):
        """Load existing annotations for the current image."""
        if not self.image_path:
            return
        
        annotation_file = Path(self.image_path).with_suffix('.txt')
        if annotation_file.exists():
            self.annotations = []
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        if class_id < len(self.class_names):
                            annotation = Annotation(
                                class_id=class_id,
                                class_name=self.class_names[class_id],
                                bbox=[x_center, y_center, width, height]
                            )
                            self.annotations.append(annotation)
    
    def draw_annotations(self):
        """Draw existing annotations on the canvas."""
        for i, annotation in enumerate(self.annotations):
            x_center, y_center, width, height = annotation.bbox
            
            # Convert normalized coordinates to display coordinates
            x_center_display = x_center * self.display_width
            y_center_display = y_center * self.display_height
            width_display = width * self.display_width
            height_display = height * self.display_height
            
            x1 = x_center_display - width_display / 2
            y1 = y_center_display - height_display / 2
            x2 = x_center_display + width_display / 2
            y2 = y_center_display + height_display / 2
            
            # Adjust for canvas center
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            offset_x = (canvas_width - self.display_width) // 2
            offset_y = (canvas_height - self.display_height) // 2
            
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            
            # Draw rectangle
            color = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta'][annotation.class_id % 8]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=f"annotation_{i}")
            self.canvas.create_text(x1, y1-10, text=annotation.class_name, fill=color, anchor=tk.SW, tags=f"annotation_{i}")
    
    def on_class_changed(self, event):
        """Handle class selection change."""
        self.current_class = self.class_names.index(self.class_var.get())
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        self.drawing = True
        self.start_point = (event.x, event.y)
    
    def on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.drawing and self.start_point:
            # Clear previous temporary rectangle
            self.canvas.delete("temp_rect")
            
            # Draw temporary rectangle
            self.canvas.create_rectangle(
                self.start_point[0], self.start_point[1],
                event.x, event.y,
                outline='red', width=2, tags="temp_rect"
            )
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if self.drawing and self.start_point:
            self.drawing = False
            
            # Clear temporary rectangle
            self.canvas.delete("temp_rect")
            
            # Calculate bounding box
            x1, y1 = self.start_point
            x2, y2 = event.x, event.y
            
            # Ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Convert to image coordinates
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            offset_x = (canvas_width - self.display_width) // 2
            offset_y = (canvas_height - self.display_height) // 2
            
            x1 -= offset_x
            y1 -= offset_y
            x2 -= offset_x
            y2 -= offset_y
            
            # Check if rectangle is within image bounds
            if x1 >= 0 and y1 >= 0 and x2 <= self.display_width and y2 <= self.display_height:
                # Convert to normalized coordinates
                x_center = (x1 + x2) / 2 / self.display_width
                y_center = (y1 + y2) / 2 / self.display_height
                width = (x2 - x1) / self.display_width
                height = (y2 - y1) / self.display_height
                
                # Create annotation
                annotation = Annotation(
                    class_id=self.current_class,
                    class_name=self.class_names[self.current_class],
                    bbox=[x_center, y_center, width, height]
                )
                
                self.annotations.append(annotation)
                self.draw_annotations()
                
                self.status_var.set(f"Added {annotation.class_name} annotation")
            
            self.start_point = None
    
    def save_annotations(self):
        """Save annotations to file."""
        if not self.image_path or not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return
        
        annotation_file = Path(self.image_path).with_suffix('.txt')
        
        with open(annotation_file, 'w') as f:
            for annotation in self.annotations:
                f.write(annotation.to_yolo_format() + '\n')
        
        messagebox.showinfo("Success", f"Saved {len(self.annotations)} annotations to {annotation_file}")
        self.status_var.set(f"Saved {len(self.annotations)} annotations")
    
    def clear_annotations(self):
        """Clear all annotations."""
        if messagebox.askyesno("Confirm", "Clear all annotations?"):
            self.annotations = []
            self.canvas.delete("annotation")
            self.status_var.set("Cleared all annotations")
    
    def run(self):
        """Run the annotation tool."""
        self.root.mainloop()


def create_class_names_file(output_path: str, class_names: List[str]) -> None:
    """Create a class names file for YOLO training."""
    with open(output_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    logger.info(f"Created class names file: {output_path}")


def validate_annotations(annotations_dir: str, images_dir: str) -> Dict[str, Any]:
    """
    Validate annotation files and provide statistics.
    
    Args:
        annotations_dir: Directory containing annotation files
        images_dir: Directory containing image files
        
    Returns:
        Validation statistics
    """
    annotations_path = Path(annotations_dir)
    images_path = Path(images_dir)
    
    annotation_files = list(annotations_path.glob('*.txt'))
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(images_path.glob(ext)))
    
    stats = {
        'total_annotation_files': len(annotation_files),
        'total_image_files': len(image_files),
        'matched_pairs': 0,
        'orphaned_annotations': 0,
        'orphaned_images': 0,
        'total_annotations': 0,
        'class_distribution': {},
        'bbox_size_stats': {'min': float('inf'), 'max': 0, 'avg': 0},
        'errors': []
    }
    
    # Create filename mappings
    annotation_stems = {f.stem for f in annotation_files}
    image_stems = {f.stem for f in image_files}
    
    # Count matched pairs
    matched_stems = annotation_stems.intersection(image_stems)
    stats['matched_pairs'] = len(matched_stems)
    stats['orphaned_annotations'] = len(annotation_stems - image_stems)
    stats['orphaned_images'] = len(image_stems - annotation_stems)
    
    # Analyze annotations
    bbox_sizes = []
    
    for annotation_file in annotation_files:
        try:
            with open(annotation_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        stats['errors'].append(f"{annotation_file}:{line_num} - Invalid format")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                            stats['errors'].append(f"{annotation_file}:{line_num} - Invalid coordinates")
                            continue
                        
                        stats['total_annotations'] += 1
                        
                        # Update class distribution
                        if class_id not in stats['class_distribution']:
                            stats['class_distribution'][class_id] = 0
                        stats['class_distribution'][class_id] += 1
                        
                        # Update bbox size stats
                        bbox_area = width * height
                        bbox_sizes.append(bbox_area)
                        stats['bbox_size_stats']['min'] = min(stats['bbox_size_stats']['min'], bbox_area)
                        stats['bbox_size_stats']['max'] = max(stats['bbox_size_stats']['max'], bbox_area)
                        
                    except ValueError:
                        stats['errors'].append(f"{annotation_file}:{line_num} - Invalid number format")
                        
        except Exception as e:
            stats['errors'].append(f"{annotation_file} - Could not read file: {e}")
    
    # Calculate average bbox size
    if bbox_sizes:
        stats['bbox_size_stats']['avg'] = sum(bbox_sizes) / len(bbox_sizes)
    else:
        stats['bbox_size_stats']['min'] = 0
    
    return stats


if __name__ == "__main__":
    # Example usage
    class_names = ['specimen', 'cell', 'bacteria', 'particle', 'debris']
    
    # Create annotation tool
    tool = SimpleAnnotationTool(class_names)
    tool.run()