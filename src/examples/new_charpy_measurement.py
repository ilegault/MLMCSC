#!/usr/bin/env python3
"""
CORRECTED Charpy Lateral Expansion Measurement

This fixes the fundamental issue: measuring actual lateral expansion
(bulging outward) rather than specimen width.
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class CharpyMeasurement:
    """Results from Charpy lateral expansion measurement."""
    left_expansion_pixels: float
    right_expansion_pixels: float
    total_expansion_pixels: float
    left_expansion_mm: Optional[float] = None
    right_expansion_mm: Optional[float] = None
    total_expansion_mm: Optional[float] = None
    calibration_factor: Optional[float] = None
    corners: List[Tuple[float, float]] = None
    confidence_scores: List[float] = None


class CorrectedCharpyMeasurer:
    """
    CORRECTED implementation that measures actual lateral expansion.

    Lateral expansion = how much the specimen sides have bulged outward
    from the original straight edges, NOT the width of the specimen.
    """

    def __init__(self, calibration_factor: Optional[float] = None):
        self.calibration_factor = calibration_factor

    def find_specimen_edges(self, image: np.ndarray, corners: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Find the actual edges of the specimen using edge detection.
        This is crucial for measuring how much material has bulged OUT from the original edges.
        """
        # Sort corners to identify specimen orientation
        sorted_corners = sorted(corners, key=lambda p: p[1])  # Sort by y-coordinate

        # Assume specimen is roughly rectangular with fracture at top
        top_corners = sorted_corners[:2]  # Fracture end (narrower)
        bottom_corners = sorted_corners[2:]  # Impact end (wider due to expansion)

        # Sort horizontally
        top_left = min(top_corners, key=lambda p: p[0])
        top_right = max(top_corners, key=lambda p: p[0])
        bottom_left = min(bottom_corners, key=lambda p: p[0])
        bottom_right = max(bottom_corners, key=lambda p: p[0])

        return {
            'top_left': top_left,
            'top_right': top_right,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right
        }

    def calculate_original_edges(self, corners: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Calculate where the original straight edges would be.
        Method: Draw lines from fracture surface (top) toward impact end.
        """
        top_left = corners['top_left']
        top_right = corners['top_right']
        bottom_left = corners['bottom_left']
        bottom_right = corners['bottom_right']

        # The original edges would be straight lines from the fracture surface
        # extending downward. The fracture surface is less deformed.

        # Left original edge: line from top_left extending downward
        # Direction: vertical (since specimen should be oriented vertically)
        left_edge_start = top_left
        left_edge_end = (top_left[0], bottom_left[1])  # Same x, but at bottom y-level

        # Right original edge: line from top_right extending downward
        right_edge_start = top_right
        right_edge_end = (top_right[0], bottom_right[1])  # Same x, but at bottom y-level

        return {
            'left_edge': (left_edge_start, left_edge_end),
            'right_edge': (right_edge_start, right_edge_end),
            'left_original_x': top_left[0],
            'right_original_x': top_right[0]
        }

    def measure_bulging_distance(self, actual_corner: Tuple[float, float],
                                 original_x: float) -> float:
        """
        Measure how far a corner has bulged out from the original edge.

        Args:
            actual_corner: Where the corner actually is
            original_x: Where the original straight edge would be

        Returns:
            Distance the corner has bulged outward (always positive)
        """
        actual_x = actual_corner[0]
        return abs(actual_x - original_x)

    def measure_lateral_expansion(self, corners: List[Tuple[float, float]]) -> CharpyMeasurement:
        """
        Measure actual lateral expansion by calculating how much the specimen
        has bulged outward from its original straight edges.
        """
        # Organize corners
        corner_positions = self.find_specimen_edges(None, corners)

        # Calculate original straight edges
        original_edges = self.calculate_original_edges(corner_positions)

        # Measure how much each bottom corner has bulged outward
        left_expansion = self.measure_bulging_distance(
            corner_positions['bottom_left'],
            original_edges['left_original_x']
        )

        right_expansion = self.measure_bulging_distance(
            corner_positions['bottom_right'],
            original_edges['right_original_x']
        )

        total_expansion = left_expansion + right_expansion

        # Convert to mm if calibration available
        left_mm = None
        right_mm = None
        total_mm = None

        if self.calibration_factor:
            left_mm = left_expansion * self.calibration_factor
            right_mm = right_expansion * self.calibration_factor
            total_mm = total_expansion * self.calibration_factor

        return CharpyMeasurement(
            left_expansion_pixels=left_expansion,
            right_expansion_pixels=right_expansion,
            total_expansion_pixels=total_expansion,
            left_expansion_mm=left_mm,
            right_expansion_mm=right_mm,
            total_expansion_mm=total_mm,
            calibration_factor=self.calibration_factor,
            corners=corners
        )

    def visualize_measurement(self, image: np.ndarray,
                              corners: List[Tuple[float, float]],
                              measurements: CharpyMeasurement) -> np.ndarray:
        """
        Create visualization showing the lateral expansion measurement.
        """
        result_image = image.copy()

        # Organize corners
        corner_positions = self.find_specimen_edges(None, corners)
        original_edges = self.calculate_original_edges(corner_positions)

        # Draw corners
        for corner in corners:
            cv2.circle(result_image, (int(corner[0]), int(corner[1])), 5, (0, 255, 255), -1)

        # Draw original straight edges (what the edges SHOULD be)
        left_start, left_end = original_edges['left_edge']
        right_start, right_end = original_edges['right_edge']

        cv2.line(result_image,
                 (int(left_start[0]), int(left_start[1])),
                 (int(left_end[0]), int(left_end[1])),
                 (0, 255, 0), 2)  # Green for original edges

        cv2.line(result_image,
                 (int(right_start[0]), int(right_start[1])),
                 (int(right_end[0]), int(right_end[1])),
                 (0, 255, 0), 2)

        # Draw actual specimen outline
        pts = np.array([
            corner_positions['bottom_left'],
            corner_positions['bottom_right'],
            corner_positions['top_right'],
            corner_positions['top_left']
        ], dtype=np.int32)
        cv2.polylines(result_image, [pts], True, (255, 0, 0), 2)  # Blue for actual outline

        # Draw expansion measurement lines
        bottom_left = corner_positions['bottom_left']
        bottom_right = corner_positions['bottom_right']

        # Left expansion line
        cv2.line(result_image,
                 (int(original_edges['left_original_x']), int(bottom_left[1])),
                 (int(bottom_left[0]), int(bottom_left[1])),
                 (255, 255, 0), 3)  # Yellow for measurements

        # Right expansion line
        cv2.line(result_image,
                 (int(original_edges['right_original_x']), int(bottom_right[1])),
                 (int(bottom_right[0]), int(bottom_right[1])),
                 (255, 255, 0), 3)

        # Add text annotations
        cv2.putText(result_image, f"Left: {measurements.left_expansion_pixels:.1f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(result_image, f"Right: {measurements.right_expansion_pixels:.1f}px",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(result_image, f"Total: {measurements.total_expansion_pixels:.1f}px",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if measurements.total_expansion_mm:
            cv2.putText(result_image, f"Total: {measurements.total_expansion_mm:.2f}mm",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Legend
        cv2.putText(result_image, "Green: Original edges",
                    (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_image, "Blue: Actual outline",
                    (10, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(result_image, "Yellow: Expansion",
                    (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return result_image


def test_corrected_measurement():
    """Test the corrected measurement approach."""

    # Example corners from a Charpy specimen
    # These represent: top-left, top-right, bottom-left, bottom-right
    test_corners = [
        (100, 50),  # Top-left (fracture surface)
        (300, 50),  # Top-right (fracture surface)
        (80, 200),  # Bottom-left (bulged outward by 20px)
        (320, 200)  # Bottom-right (bulged outward by 20px)
    ]

    # Initialize corrected measurer
    measurer = CorrectedCharpyMeasurer(calibration_factor=0.1)  # 0.1 mm/pixel

    # Measure lateral expansion
    measurements = measurer.measure_lateral_expansion(test_corners)

    print("=== CORRECTED Lateral Expansion Measurement ===")
    print(f"Left bulging: {measurements.left_expansion_pixels:.1f}px ({measurements.left_expansion_mm:.2f}mm)")
    print(f"Right bulging: {measurements.right_expansion_pixels:.1f}px ({measurements.right_expansion_mm:.2f}mm)")
    print(f"Total expansion: {measurements.total_expansion_pixels:.1f}px ({measurements.total_expansion_mm:.2f}mm)")
    print()
    print("This measures how much the specimen sides have BULGED OUTWARD")
    print("from where the original straight edges would be.")

    return measurements


if __name__ == "__main__":
    test_corrected_measurement()