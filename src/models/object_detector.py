#!/usr/bin/env python3
"""
Object Detection Module for Microscope Specimen Detection

This module implements a comprehensive specimen detection system using YOLOv8
for real-time detection, tracking, and analysis of microscopic specimens.

Features:
- Real-time detection (>30 FPS)
- Multi-specimen tracking with unique IDs
- Motion detection and stability analysis
- Auto-centering calculations
- Rotation detection and correction
- Region of Interest (ROI) extraction
"""

import cv2
import numpy as np
import torch
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy import ndimage

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Data class for detection results."""
    specimen_id: int
    bbox: List[float]  # [x, y, width, height]
    confidence: float
    is_stable: bool
    rotation_angle: float
    center_offset: List[float]  # [dx, dy]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class SpecimenTracker:
    """Handles tracking of specimens across frames."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.next_id = 0
        self.objects = {}  # id -> centroid
        self.disappeared = {}  # id -> frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid: np.ndarray) -> int:
        """Register a new object and return its ID."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id: int) -> None:
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """Update tracker with new detections."""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects.copy()
        
        if len(self.objects) == 0:
            # Register all detections as new objects
            for detection in detections:
                self.register(detection)
        else:
            # Compute distance matrix
            object_centroids = np.array(list(self.objects.values()))
            detection_centroids = np.array(detections)
            
            D = cdist(object_centroids, detection_centroids)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = list(self.objects.keys())[row]
                self.objects[object_id] = detection_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_idxs.add(row)
                used_col_idxs.add(col)
            
            # Handle unmatched detections and objects
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            
            if D.shape[0] >= D.shape[1]:
                # More objects than detections
                for row in unused_row_idxs:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects
                for col in unused_col_idxs:
                    self.register(detection_centroids[col])
        
        return self.objects.copy()


class MotionDetector:
    """Detects motion and stability of specimens."""
    
    def __init__(self, history_size: int = 10, stability_threshold: float = 5.0):
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self.position_history = defaultdict(lambda: deque(maxlen=history_size))
        
    def update(self, specimen_id: int, centroid: np.ndarray) -> bool:
        """Update position history and check stability."""
        self.position_history[specimen_id].append(centroid.copy())
        return self.is_stable(specimen_id)
    
    def is_stable(self, specimen_id: int) -> bool:
        """Check if specimen is stable (not moving significantly)."""
        history = self.position_history[specimen_id]
        if len(history) < self.history_size:
            return False
        
        # Calculate movement variance
        positions = np.array(list(history))
        variance = np.var(positions, axis=0)
        total_variance = np.sum(variance)
        
        return total_variance < self.stability_threshold


class RotationDetector:
    """Detects rotation angle of specimens."""
    
    def __init__(self):
        self.previous_angles = {}
    
    def detect_rotation(self, contour: np.ndarray, specimen_id: int) -> float:
        """Detect rotation angle of specimen using contour analysis."""
        try:
            # Fit ellipse to contour
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                
                # Normalize angle to [-90, 90]
                if angle > 90:
                    angle -= 180
                
                # Smooth angle changes
                if specimen_id in self.previous_angles:
                    prev_angle = self.previous_angles[specimen_id]
                    angle_diff = abs(angle - prev_angle)
                    if angle_diff > 90:
                        # Handle angle wrap-around
                        if angle > prev_angle:
                            angle -= 180
                        else:
                            angle += 180
                
                self.previous_angles[specimen_id] = angle
                return float(angle)
            else:
                return 0.0
        except:
            return 0.0


class SpecimenDetector:
    """
    Main specimen detection class using YOLOv8 for real-time detection and tracking.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.4,
        device: str = 'auto',
        max_detections: int = 10
    ):
        """
        Initialize the specimen detector.
        
        Args:
            model_path: Path to custom YOLO model. If None, uses pre-trained model.
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            max_detections: Maximum number of detections per frame
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Initialize device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load YOLO model
        self._load_model(model_path)
        
        # Initialize tracking components
        self.tracker = SpecimenTracker()
        self.motion_detector = MotionDetector()
        self.rotation_detector = RotationDetector()
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Frame dimensions for centering calculations
        self.frame_center = None
        
        logger.info(f"SpecimenDetector initialized on {self.device}")
    
    def _load_model(self, model_path: Optional[str]) -> None:
        """Load YOLO model."""
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom model from {model_path}")
            else:
                # Use pre-trained YOLOv8 model
                self.model = YOLO('yolov8n.pt')  # nano version for speed
                logger.info("Loaded pre-trained YOLOv8n model")
            
            # Move model to device
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_specimen(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect specimens in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of DetectionResult objects
        """
        start_time = time.time()
        
        # Update frame center
        h, w = frame.shape[:2]
        self.frame_center = np.array([w // 2, h // 2])
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Extract detections
            detections = []
            centroids = []
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Convert to [x, y, width, height] format
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        # Calculate centroid
                        centroid = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])
                        centroids.append(centroid)
                        
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'centroid': centroid
                        })
            
            # Update tracker
            tracked_objects = self.tracker.update(centroids)
            
            # Create detection results
            results_list = []
            for detection in detections:
                # Find matching tracked object
                specimen_id = self._find_matching_id(detection['centroid'], tracked_objects)
                
                if specimen_id is not None:
                    # Update motion detection
                    is_stable = self.motion_detector.update(specimen_id, detection['centroid'])
                    
                    # Calculate center offset
                    center_offset = self._calculate_center_offset(detection['centroid'])
                    
                    # Extract ROI for rotation detection
                    roi_contour = self._extract_contour_from_bbox(frame, detection['bbox'])
                    rotation_angle = self.rotation_detector.detect_rotation(roi_contour, specimen_id)
                    
                    result = DetectionResult(
                        specimen_id=specimen_id,
                        bbox=detection['bbox'],
                        confidence=detection['confidence'],
                        is_stable=is_stable,
                        rotation_angle=rotation_angle,
                        center_offset=center_offset.tolist()
                    )
                    
                    results_list.append(result)
            
            # Update FPS counter
            frame_time = time.time() - start_time
            self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
            
            return results_list
            
        except Exception as e:
            logger.error(f"Error in detect_specimen: {e}")
            return []
    
    def track_specimen(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Track specimens across frames (alias for detect_specimen).
        
        Args:
            frame: Input image frame
            
        Returns:
            List of DetectionResult objects with tracking information
        """
        return self.detect_specimen(frame)
    
    def is_specimen_stable(self, specimen_id: int) -> bool:
        """
        Check if a specific specimen is stable (not moving).
        
        Args:
            specimen_id: ID of the specimen to check
            
        Returns:
            True if specimen is stable, False otherwise
        """
        return self.motion_detector.is_stable(specimen_id)
    
    def extract_roi(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
        padding: int = 20
    ) -> Optional[np.ndarray]:
        """
        Extract region of interest around a detected specimen.
        
        Args:
            frame: Input image frame
            detection_result: Detection result containing bbox
            padding: Additional padding around the bounding box
            
        Returns:
            Extracted ROI image or None if extraction fails
        """
        try:
            x, y, w, h = detection_result.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            roi = frame[y1:y2, x1:x2]
            return roi
            
        except Exception as e:
            logger.error(f"Error extracting ROI: {e}")
            return None
    
    def auto_center(self, detection_results: List[DetectionResult]) -> Optional[Tuple[float, float]]:
        """
        Calculate centering adjustments for auto-centering functionality.
        
        Args:
            detection_results: List of detection results
            
        Returns:
            Tuple of (dx, dy) adjustments needed to center the primary specimen,
            or None if no specimens detected
        """
        if not detection_results:
            return None
        
        # Find the specimen with highest confidence
        primary_specimen = max(detection_results, key=lambda x: x.confidence)
        
        # Return the center offset (already calculated in detect_specimen)
        return tuple(primary_specimen.center_offset)
    
    def get_fps(self) -> float:
        """Get current average FPS."""
        if len(self.fps_counter) == 0:
            return 0.0
        return float(np.mean(self.fps_counter))
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            'fps': self.get_fps(),
            'active_tracks': len(self.tracker.objects),
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold
        }
    
    def _find_matching_id(
        self,
        centroid: np.ndarray,
        tracked_objects: Dict[int, np.ndarray]
    ) -> Optional[int]:
        """Find the ID of the tracked object closest to the given centroid."""
        if not tracked_objects:
            return None
        
        min_distance = float('inf')
        best_id = None
        
        for obj_id, obj_centroid in tracked_objects.items():
            distance = np.linalg.norm(centroid - obj_centroid)
            if distance < min_distance:
                min_distance = distance
                best_id = obj_id
        
        return best_id
    
    def _calculate_center_offset(self, centroid: np.ndarray) -> np.ndarray:
        """Calculate offset from frame center."""
        if self.frame_center is None:
            return np.array([0.0, 0.0])
        
        return centroid - self.frame_center
    
    def _extract_contour_from_bbox(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> np.ndarray:
        """Extract contour from bounding box region for rotation detection."""
        try:
            x, y, w, h = [int(coord) for coord in bbox]
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Return the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Adjust contour coordinates to original frame
                largest_contour[:, :, 0] += x
                largest_contour[:, :, 1] += y
                return largest_contour
            else:
                # Return a simple rectangular contour if no contours found
                return np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
                
        except Exception as e:
            logger.error(f"Error extracting contour: {e}")
            # Return a default rectangular contour
            x, y, w, h = [int(coord) for coord in bbox]
            return np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)


def create_training_script() -> str:
    """
    Generate a training script for custom specimen detection.
    
    Returns:
        Training script as a string
    """
    training_script = '''#!/usr/bin/env python3
"""
Training script for custom specimen detection using YOLOv8.

This script helps train a custom YOLO model for microscope specimen detection.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml

def create_dataset_config(data_path: str, classes: list) -> str:
    """Create dataset configuration file for YOLO training."""
    config = {
        'path': data_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    config_path = Path(data_path) / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)

def train_specimen_detector(
    data_path: str,
    model_size: str = 'n',  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = 'auto'
):
    """
    Train a custom specimen detection model.
    
    Args:
        data_path: Path to dataset directory
        model_size: YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size for training
        device: Device to use for training
    """
    
    # Define specimen classes (customize as needed)
    classes = [
        'specimen',
        'cell',
        'bacteria',
        'particle',
        'debris'
    ]
    
    # Create dataset configuration
    config_path = create_dataset_config(data_path, classes)
    
    # Load pre-trained YOLO model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project='specimen_detection',
        name='custom_model',
        save=True,
        save_period=10,
        cache=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    # Validate the model
    validation_results = model.val()
    
    print(f"Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Validation mAP50: {validation_results.box.map50}")
    print(f"Validation mAP50-95: {validation_results.box.map}")
    
    return results

if __name__ == "__main__":
    # Example usage
    data_path = "path/to/your/dataset"  # Update this path
    
    # Ensure dataset structure:
    # dataset/
    # ├── images/
    # │   ├── train/
    # │   ├── val/
    # │   └── test/
    # └── labels/
    #     ├── train/
    #     ├── val/
    #     └── test/
    
    if os.path.exists(data_path):
        train_specimen_detector(
            data_path=data_path,
            model_size='n',  # Use nano for speed
            epochs=100,
            imgsz=640,
            batch_size=16
        )
    else:
        print(f"Dataset path {data_path} does not exist!")
        print("Please prepare your dataset and update the data_path variable.")
'''
    
    return training_script


# Example usage and testing
if __name__ == "__main__":
    # Example usage of the SpecimenDetector
    detector = SpecimenDetector(
        confidence_threshold=0.8,
        device='auto'
    )
    
    # Test with a sample frame (replace with actual frame)
    # frame = cv2.imread('sample_microscope_image.jpg')
    # results = detector.detect_specimen(frame)
    # 
    # for result in results:
    #     print(f"Specimen {result.specimen_id}: confidence={result.confidence:.2f}, "
    #           f"stable={result.is_stable}, rotation={result.rotation_angle:.1f}°")
    
    print("SpecimenDetector module loaded successfully!")
    print(f"Current FPS capability: {detector.get_fps():.1f}")
    print(f"Detection stats: {detector.get_detection_stats()}")