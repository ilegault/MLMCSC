#!/usr/bin/env python3
"""
Charpy Specimen Lateral Expansion Measurement

This module implements post-processing algorithms to measure the lateral expansion
of Charpy impact test specimens by comparing distances between corners at the
fracture surface versus the bottom edge.

The algorithm processes YOLO model detection output with 4 corner points labeled
as 'charpy_corner' and optionally 'fracture_surface' detection boxes.
"""

import cv2
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class LineSegment:
    """Represents a line segment with start and end points."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    
    def length(self) -> float:
        """Calculate the length of the line segment."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx**2 + dy**2)
    
    def direction_vector(self) -> Tuple[float, float]:
        """Get the normalized direction vector of the line."""
        length = self.length()
        if length == 0:
            return (0, 0)
        dx = (self.end[0] - self.start[0]) / length
        dy = (self.end[1] - self.start[1]) / length
        return (dx, dy)


@dataclass
class CharpyMeasurement:
    """Container for Charpy lateral expansion measurements."""
    left_expansion_pixels: float
    right_expansion_pixels: float
    total_expansion_pixels: float
    left_expansion_mm: Optional[float] = None
    right_expansion_mm: Optional[float] = None
    total_expansion_mm: Optional[float] = None
    calibration_factor: Optional[float] = None
    corners: List[Tuple[float, float]] = None
    confidence_scores: List[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'left_expansion_pixels': self.left_expansion_pixels,
            'right_expansion_pixels': self.right_expansion_pixels,
            'total_expansion_pixels': self.total_expansion_pixels,
            'left_expansion_mm': self.left_expansion_mm,
            'right_expansion_mm': self.right_expansion_mm,
            'total_expansion_mm': self.total_expansion_mm,
            'calibration_factor': self.calibration_factor,
            'corners': self.corners,
            'confidence_scores': self.confidence_scores
        }


class CharpyLateralExpansionMeasurer:
    """
    Measures lateral expansion of Charpy impact test specimens.
    
    This class processes YOLO detection output to identify corner points
    and calculate the lateral expansion by comparing distances between
    corners at the fracture surface versus the bottom edge.
    """
    
    def __init__(self, 
                 calibration_factor: Optional[float] = None,
                 min_confidence: float = 0.5,
                 visualization_enabled: bool = True):
        """
        Initialize the Charpy lateral expansion measurer.
        
        Args:
            calibration_factor: Conversion factor from pixels to mm (mm per pixel)
            min_confidence: Minimum confidence threshold for corner detections
            visualization_enabled: Whether to generate visualization images
        """
        self.calibration_factor = calibration_factor
        self.min_confidence = min_confidence
        self.visualization_enabled = visualization_enabled
        
        logger.info(f"CharpyLateralExpansionMeasurer initialized with calibration: {calibration_factor}")
    
    def parse_yolo_output(self, detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Extract corner coordinates from YOLO output.
        
        Args:
            detections: List of YOLO detection dictionaries
            
        Returns:
            Tuple of (corner coordinates, confidence scores)
        """
        corners = []
        confidences = []
        
        for detection in detections:
            if (detection.get('class') == 'charpy_corner' and 
                detection.get('confidence', 0) >= self.min_confidence):
                
                # Extract center point of bounding box
                x_center = detection['x']
                y_center = detection['y']
                confidence = detection.get('confidence', 1.0)
                
                corners.append((float(x_center), float(y_center)))
                confidences.append(float(confidence))
        
        logger.info(f"Parsed {len(corners)} corner detections from YOLO output")
        return corners, confidences
    
    def identify_bottom_corners(self, corners: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Identify bottom corners (impact end) and top corners (fracture end).
        Bottom corners have higher y-values (further from fracture surface).
        """
        # Sort corners by y-coordinate (bottom corners have higher y-values)
        sorted_corners = sorted(corners, key=lambda p: p[1])
        
        # Bottom two corners (impact end - more deformed)
        bottom_corners = sorted_corners[2:]  # Last 2 (highest y)
        
        # Sort horizontally to get left and right
        bottom_left = min(bottom_corners, key=lambda p: p[0])
        bottom_right = max(bottom_corners, key=lambda p: p[0])
        
        logger.debug(f"Identified bottom corners - Left: {bottom_left}, Right: {bottom_right}")
        return bottom_left, bottom_right
    
    def identify_top_corners(self, corners: List[Tuple[float, float]], 
                        bottom_corners: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Identify top corners (fracture surface end).
        These are the corners NOT in the bottom_corners list.
        """
        # Find corners that are not bottom corners
        top_corners = [c for c in corners if c not in bottom_corners]
        
        if len(top_corners) != 2:
            raise ValueError(f"Expected 2 top corners, found {len(top_corners)}")
        
        # Sort horizontally
        top_left = min(top_corners, key=lambda p: p[0])
        top_right = max(top_corners, key=lambda p: p[0])
        
        logger.debug(f"Identified top corners - Left: {top_left}, Right: {top_right}")
        return top_left, top_right
    
    def create_reference_line(self, 
                            bottom_left: Tuple[float, float], 
                            bottom_right: Tuple[float, float]) -> LineSegment:
        """
        Draw a line between the two bottom corners.
        
        Args:
            bottom_left: Left bottom corner coordinates
            bottom_right: Right bottom corner coordinates
            
        Returns:
            LineSegment representing the baseline reference
        """
        return LineSegment(bottom_left, bottom_right)
    
    def create_perpendicular_lines(self, 
                                 bottom_left: Tuple[float, float], 
                                 bottom_right: Tuple[float, float], 
                                 line_length: float = 500) -> Tuple[LineSegment, LineSegment]:
        """
        Create perpendicular lines at each bottom corner.
        
        Args:
            bottom_left: Left bottom corner coordinates
            bottom_right: Right bottom corner coordinates
            line_length: Length of perpendicular lines
            
        Returns:
            Tuple of (left_perpendicular, right_perpendicular) line segments
        """
        # Calculate baseline vector
        dx = bottom_right[0] - bottom_left[0]
        dy = bottom_right[1] - bottom_left[1]
        
        # Normalize and rotate 90 degrees for perpendicular
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            raise ValueError("Bottom corners are identical")
        
        perp_dx = -dy / length
        perp_dy = dx / length
        
        # Create perpendicular lines extending upward
        left_perp_end = (bottom_left[0] + perp_dx * line_length, 
                        bottom_left[1] + perp_dy * line_length)
        right_perp_end = (bottom_right[0] + perp_dx * line_length, 
                         bottom_right[1] + perp_dy * line_length)
        
        left_perp = LineSegment(bottom_left, left_perp_end)
        right_perp = LineSegment(bottom_right, right_perp_end)
        
        return left_perp, right_perp
    
    def point_to_line_distance(self, 
                             point: Tuple[float, float], 
                             line_start: Tuple[float, float], 
                             line_end: Tuple[float, float]) -> float:
        """
        Calculate perpendicular distance from point to line.
        
        Args:
            point: Point coordinates
            line_start: Line start coordinates
            line_end: Line end coordinates
            
        Returns:
            Perpendicular distance from point to line
        """
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Using cross product formula
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def find_perpendicular_projection(self, 
                                    point: Tuple[float, float], 
                                    line: LineSegment) -> Tuple[float, float]:
        """
        Find the perpendicular projection of a point onto a line.
        
        Args:
            point: Point coordinates
            line: Line segment
            
        Returns:
            Coordinates of the projection point
        """
        x0, y0 = point
        x1, y1 = line.start
        x2, y2 = line.end
        
        # Vector from line start to point
        dx_point = x0 - x1
        dy_point = y0 - y1
        
        # Line direction vector
        dx_line = x2 - x1
        dy_line = y2 - y1
        
        # Length squared of line
        line_length_sq = dx_line**2 + dy_line**2
        
        if line_length_sq == 0:
            return line.start
        
        # Parameter t for projection
        t = (dx_point * dx_line + dy_point * dy_line) / line_length_sq
        
        # Projection point
        proj_x = x1 + t * dx_line
        proj_y = y1 + t * dy_line
        
        return (proj_x, proj_y)
    
    def measure_lateral_expansion(self, 
                            top_left: Tuple[float, float], 
                            top_right: Tuple[float, float], 
                            left_perp: LineSegment, 
                            right_perp: LineSegment,
                            bottom_left: Tuple[float, float],
                            bottom_right: Tuple[float, float]) -> Dict[str, float]:
        """
        Measure ACTUAL lateral expansion by calculating how much the specimen
        has bulged outward from its original straight edges.
        
        Original method was WRONG - it measured specimen width, not expansion.
        This corrected method measures how much material has bulged OUT.
        """
        
        # CORRECTED APPROACH:
        # 1. The fracture surface (top) is less deformed - use it as reference
        # 2. Original edges would be straight lines from top corners downward
        # 3. Measure how far bottom corners have bulged from these original edges
        
        # Calculate where the original straight edges would be
        # (vertical lines from top corners)
        original_left_edge_x = top_left[0]
        original_right_edge_x = top_right[0]
        
        # Measure how far bottom corners have bulged outward
        left_expansion = abs(bottom_left[0] - original_left_edge_x)
        right_expansion = abs(bottom_right[0] - original_right_edge_x)
        
        # Total lateral expansion
        total_expansion = left_expansion + right_expansion
        
        logger.debug(f"CORRECTED Lateral expansion measurements:")
        logger.debug(f"  Original left edge X: {original_left_edge_x:.2f}")
        logger.debug(f"  Original right edge X: {original_right_edge_x:.2f}")
        logger.debug(f"  Bottom left actual X: {bottom_left[0]:.2f}")
        logger.debug(f"  Bottom right actual X: {bottom_right[0]:.2f}")
        logger.debug(f"  Left bulging: {left_expansion:.2f}px")
        logger.debug(f"  Right bulging: {right_expansion:.2f}px")
        logger.debug(f"  Total expansion: {total_expansion:.2f}px")
        
        return {
            'left_expansion_pixels': left_expansion,
            'right_expansion_pixels': right_expansion,
            'total_expansion_pixels': total_expansion,
            'original_left_edge_x': original_left_edge_x,
            'original_right_edge_x': original_right_edge_x
        }
    
    def pixels_to_mm(self, pixel_distance: float, calibration_factor: Optional[float] = None) -> Optional[float]:
        """
        Convert pixel measurements to millimeters.
        
        Args:
            pixel_distance: Distance in pixels
            calibration_factor: mm per pixel (overrides instance calibration)
            
        Returns:
            Distance in millimeters, or None if no calibration factor available
        """
        factor = calibration_factor or self.calibration_factor
        if factor is None:
            return None
        return pixel_distance * factor
    
    def visualize_measurements(self, 
                             image: np.ndarray, 
                             corners: List[Tuple[float, float]], 
                             measurements: CharpyMeasurement,
                             bottom_left: Tuple[float, float],
                             bottom_right: Tuple[float, float],
                             top_left: Tuple[float, float],
                             top_right: Tuple[float, float],
                             left_perp: LineSegment,
                             right_perp: LineSegment) -> np.ndarray:
        """Enhanced visualization showing actual lateral expansion."""
        
        result_image = image.copy()
        
        # Draw original straight edges (green lines)
        original_left_x = top_left[0]
        original_right_x = top_right[0]
        
        # Draw original edges as vertical lines
        cv2.line(result_image,
                 (int(original_left_x), int(top_left[1])),
                 (int(original_left_x), int(bottom_left[1])),
                 (0, 255, 0), 2)  # Green for original edges
        
        cv2.line(result_image,
                 (int(original_right_x), int(top_right[1])),
                 (int(original_right_x), int(bottom_right[1])),
                 (0, 255, 0), 2)
        
        # Draw expansion measurement lines (yellow)
        cv2.line(result_image,
                 (int(original_left_x), int(bottom_left[1])),
                 (int(bottom_left[0]), int(bottom_left[1])),
                 (0, 255, 255), 3)  # Yellow for left expansion
        
        cv2.line(result_image,
                 (int(original_right_x), int(bottom_right[1])),
                 (int(bottom_right[0]), int(bottom_right[1])),
                 (0, 255, 255), 3)  # Yellow for right expansion
        
        # Draw corners
        for corner in corners:
            cv2.circle(result_image, (int(corner[0]), int(corner[1])), 5, (255, 0, 0), -1)
        
        # Add measurement text
        cv2.putText(result_image, f"Left: {measurements.left_expansion_pixels:.1f}px", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_image, f"Right: {measurements.right_expansion_pixels:.1f}px", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_image, f"Total: {measurements.total_expansion_pixels:.1f}px", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if measurements.total_expansion_mm:
            cv2.putText(result_image, f"Total: {measurements.total_expansion_mm:.2f}mm", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Legend
        cv2.putText(result_image, "Green: Original edges", 
                   (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_image, "Yellow: Expansion", 
                   (10, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(result_image, "Blue: Corners", 
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return result_image
    
    def validate_corners(self, corners: List[Tuple[float, float]]) -> bool:
        """
        Validate that corner positions make geometric sense.
        
        Args:
            corners: List of corner coordinates
            
        Returns:
            True if corners are valid, False otherwise
        """
        if len(corners) != 4:
            logger.warning(f"Expected 4 corners, got {len(corners)}")
            return False
        
        # Check if corners form a reasonable quadrilateral
        # Calculate centroid
        cx = sum(c[0] for c in corners) / 4
        cy = sum(c[1] for c in corners) / 4
        
        # Check if all corners are reasonably distributed around centroid
        distances = [math.sqrt((c[0] - cx)**2 + (c[1] - cy)**2) for c in corners]
        min_dist = min(distances)
        max_dist = max(distances)
        
        # Ratio should not be too extreme
        if max_dist / min_dist > 5.0:
            logger.warning("Corner distribution appears invalid")
            return False
        
        return True
    
    def process_charpy_specimen(self, 
                              image_path: str, 
                              yolo_detections: List[Dict[str, Any]],
                              output_path: Optional[str] = None) -> Tuple[CharpyMeasurement, Optional[np.ndarray]]:
        """
        Complete pipeline for measuring Charpy specimen lateral expansion.
        
        Args:
            image_path: Path to input image
            yolo_detections: List of YOLO detection dictionaries
            output_path: Optional path to save visualization image
            
        Returns:
            Tuple of (measurements, result_image)
        """
        try:
            # 1. Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            logger.info(f"Processing Charpy specimen image: {image_path}")
            
            # 2. Parse YOLO output
            corners, confidences = self.parse_yolo_output(yolo_detections)
            
            # 3. Validate corners
            if not self.validate_corners(corners):
                raise ValueError("Invalid corner configuration detected")
            
            # 4. Identify corner positions
            bottom_left, bottom_right = self.identify_bottom_corners(corners)
            top_left, top_right = self.identify_top_corners(corners, [bottom_left, bottom_right])
            
            # 5. Create reference lines
            baseline = self.create_reference_line(bottom_left, bottom_right)
            left_perp, right_perp = self.create_perpendicular_lines(bottom_left, bottom_right)
            
            # 6. Measure lateral expansion
            expansion_data = self.measure_lateral_expansion(top_left, top_right, 
                                                          left_perp, right_perp,
                                                          bottom_left, bottom_right)
            
            # 7. Convert to mm if calibration available
            left_mm = self.pixels_to_mm(expansion_data['left_expansion_pixels'])
            right_mm = self.pixels_to_mm(expansion_data['right_expansion_pixels'])
            total_mm = self.pixels_to_mm(expansion_data['total_expansion_pixels'])
            
            # 8. Create measurement object
            measurements = CharpyMeasurement(
                left_expansion_pixels=expansion_data['left_expansion_pixels'],
                right_expansion_pixels=expansion_data['right_expansion_pixels'],
                total_expansion_pixels=expansion_data['total_expansion_pixels'],
                left_expansion_mm=left_mm,
                right_expansion_mm=right_mm,
                total_expansion_mm=total_mm,
                calibration_factor=self.calibration_factor,
                corners=corners,
                confidence_scores=confidences
            )
            
            # 9. Generate visualization if enabled
            result_image = None
            if self.visualization_enabled:
                result_image = self.visualize_measurements(
                    image, corners, measurements,
                    bottom_left, bottom_right, top_left, top_right,
                    left_perp, right_perp
                )
                
                # Save visualization if output path provided
                if output_path:
                    cv2.imwrite(output_path, result_image)
                    logger.info(f"Visualization saved to {output_path}")
            
            logger.info(f"Measurement completed - Total expansion: {measurements.total_expansion_pixels:.2f}px"
                       f"{f' ({measurements.total_expansion_mm:.2f}mm)' if total_mm else ''}")
            
            return measurements, result_image
            
        except Exception as e:
            logger.error(f"Charpy specimen processing failed: {e}")
            raise
    
    def batch_process(self, 
                     image_paths: List[str], 
                     detection_results: List[List[Dict[str, Any]]],
                     output_dir: Optional[str] = None) -> List[CharpyMeasurement]:
        """
        Process multiple Charpy specimens in batch.
        
        Args:
            image_paths: List of image file paths
            detection_results: List of YOLO detection results for each image
            output_dir: Optional directory to save visualization images
            
        Returns:
            List of measurement results
        """
        if len(image_paths) != len(detection_results):
            raise ValueError("Number of images and detection results must match")
        
        results = []
        output_path_obj = Path(output_dir) if output_dir else None
        
        if output_path_obj:
            output_path_obj.mkdir(parents=True, exist_ok=True)
        
        for i, (image_path, detections) in enumerate(zip(image_paths, detection_results)):
            try:
                output_file = None
                if output_path_obj:
                    image_name = Path(image_path).stem
                    output_file = str(output_path_obj / f"{image_name}_measurements.jpg")
                
                measurement, _ = self.process_charpy_specimen(image_path, detections, output_file)
                results.append(measurement)
                
                logger.info(f"Processed {i+1}/{len(image_paths)}: {image_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                # Create empty measurement for failed cases
                empty_measurement = CharpyMeasurement(
                    left_expansion_pixels=0.0,
                    right_expansion_pixels=0.0,
                    total_expansion_pixels=0.0
                )
                results.append(empty_measurement)
        
        logger.info(f"Batch processing completed: {len(results)} specimens processed")
        return results
    
    def save_measurements_to_json(self, 
                                measurements: List[CharpyMeasurement], 
                                output_path: str) -> None:
        """
        Save measurement results to JSON file.
        
        Args:
            measurements: List of measurement results
            output_path: Path to save JSON file
        """
        from datetime import datetime
        
        data = {
            'measurements': [m.to_dict() for m in measurements],
            'calibration_factor': self.calibration_factor,
            'processing_timestamp': datetime.now().isoformat(),
            'total_specimens': len(measurements)
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Measurements saved to {output_path}")
    
    def save_measurements_to_json(self, 
                                measurements: List[CharpyMeasurement], 
                                output_path: str) -> None:
        """
        Save measurement results to JSON file.
        
        Args:
            measurements: List of measurement results
            output_path: Path to save JSON file
        """
        data = {
            'measurements': [m.to_dict() for m in measurements],
            'calibration_factor': self.calibration_factor,
            'processing_timestamp': str(pd.Timestamp.now()),
            'total_specimens': len(measurements)
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Measurements saved to {output_path}")


def main():
    """Example usage of the Charpy lateral expansion measurer."""
    # Example YOLO detections (normally would come from your YOLO model)
    example_detections = [
        {'class': 'charpy_corner', 'x': 100, 'y': 50, 'confidence': 0.95},
        {'class': 'charpy_corner', 'x': 300, 'y': 55, 'confidence': 0.92},
        {'class': 'charpy_corner', 'x': 110, 'y': 200, 'confidence': 0.88},
        {'class': 'charpy_corner', 'x': 290, 'y': 205, 'confidence': 0.91}
    ]
    
    # Initialize measurer with calibration factor (example: 0.1 mm per pixel)
    measurer = CharpyLateralExpansionMeasurer(calibration_factor=0.1)
    
    # Process specimen (replace with actual image path)
    try:
        measurements, result_image = measurer.process_charpy_specimen(
            image_path="path/to/charpy_specimen.jpg",
            yolo_detections=example_detections,
            output_path="charpy_measurements.jpg"
        )
        
        print(f"Lateral expansion measurements:")
        print(f"Left: {measurements.left_expansion_pixels:.2f}px ({measurements.left_expansion_mm:.2f}mm)")
        print(f"Right: {measurements.right_expansion_pixels:.2f}px ({measurements.right_expansion_mm:.2f}mm)")
        print(f"Total: {measurements.total_expansion_pixels:.2f}px ({measurements.total_expansion_mm:.2f}mm)")
        
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    main()