#!/usr/bin/env python3
"""
Charpy Measurement Utilities

This module provides additional utilities and alternative measurement methods
for Charpy specimen lateral expansion analysis, including:
- Alternative measurement approaches
- Calibration utilities
- Data export functions
- Statistical analysis tools
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import math

# Import optional dependencies with fallbacks
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

try:
    from sklearn.linear_model import RANSACRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    RANSACRegressor = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Import pandas with fallback
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .charpy_lateral_expansion import CharpyLateralExpansionMeasurer, CharpyMeasurement

logger = logging.getLogger(__name__)


class AlternativeCharpyMeasurer:
    """
    Alternative measurement approaches for Charpy lateral expansion.
    
    This class implements different measurement strategies that might be
    more suitable for certain specimen types or imaging conditions.
    """
    
    def __init__(self, calibration_factor: Optional[float] = None):
        """Initialize the alternative measurer."""
        self.calibration_factor = calibration_factor
    
    def horizontal_distance_method(self, 
                                 corners: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Alternative method: Measure horizontal distance between top and bottom corners.
        
        This method directly measures the horizontal distance difference between
        the top corners and bottom corners, which might better match mechanical
        measurement methods.
        
        Args:
            corners: List of corner coordinates
            
        Returns:
            Dictionary with measurement results
        """
        if len(corners) != 4:
            raise ValueError(f"Need exactly 4 corners, got {len(corners)}")
        
        # Sort corners by y-coordinate
        sorted_corners = sorted(corners, key=lambda p: p[1])
        
        # Top two corners (lowest y values)
        top_corners = sorted_corners[:2]
        top_left = min(top_corners, key=lambda p: p[0])
        top_right = max(top_corners, key=lambda p: p[0])
        
        # Bottom two corners (highest y values)
        bottom_corners = sorted_corners[2:]
        bottom_left = min(bottom_corners, key=lambda p: p[0])
        bottom_right = max(bottom_corners, key=lambda p: p[0])
        
        # Calculate distances
        top_width = top_right[0] - top_left[0]
        bottom_width = bottom_right[0] - bottom_left[0]
        
        # Lateral expansion is the difference
        total_expansion = bottom_width - top_width
        left_expansion = (bottom_left[0] - top_left[0])
        right_expansion = (top_right[0] - bottom_right[0])
        
        return {
            'left_expansion_pixels': abs(left_expansion),
            'right_expansion_pixels': abs(right_expansion),
            'total_expansion_pixels': total_expansion,
            'top_width_pixels': top_width,
            'bottom_width_pixels': bottom_width
        }
    
    def edge_detection_refinement(self, 
                                image: np.ndarray, 
                                corners: List[Tuple[float, float]],
                                search_radius: int = 20) -> List[Tuple[float, float]]:
        """
        Refine corner positions using edge detection.
        
        Args:
            image: Input image
            corners: Initial corner positions
            search_radius: Radius to search for edges around each corner
            
        Returns:
            Refined corner positions
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        refined_corners = []
        
        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            
            # Define search region
            x_min = max(0, x - search_radius)
            x_max = min(gray.shape[1], x + search_radius)
            y_min = max(0, y - search_radius)
            y_max = min(gray.shape[0], y + search_radius)
            
            # Extract region of interest
            roi = edges[y_min:y_max, x_min:x_max]
            
            # Find edge points in ROI
            edge_points = np.where(roi > 0)
            
            if len(edge_points[0]) > 0:
                # Calculate centroid of edge points
                edge_y = edge_points[0] + y_min
                edge_x = edge_points[1] + x_min
                
                refined_x = np.mean(edge_x)
                refined_y = np.mean(edge_y)
                
                refined_corners.append((refined_x, refined_y))
            else:
                # Keep original corner if no edges found
                refined_corners.append(corner)
        
        return refined_corners
    
    def robust_line_fitting(self, 
                          points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Fit a robust line through points using RANSAC (if available) or least squares.
        
        Args:
            points: List of point coordinates
            
        Returns:
            Tuple of (slope, intercept) for the fitted line
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points for line fitting")
        
        # Convert to numpy arrays
        X = np.array([[p[0]] for p in points])
        y = np.array([p[1] for p in points])
        
        if HAS_SKLEARN and RANSACRegressor is not None:
            # Use RANSAC for robust fitting
            try:
                ransac = RANSACRegressor(random_state=42)
                ransac.fit(X, y)
                
                slope = ransac.estimator_.coef_[0]
                intercept = ransac.estimator_.intercept_
                
                return slope, intercept
            except Exception as e:
                logger.warning(f"RANSAC fitting failed, falling back to least squares: {e}")
        
        # Fallback to simple least squares fitting
        if len(points) == 2:
            # Simple two-point line
            p1, p2 = points
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')
            intercept = p1[1] - slope * p1[0] if slope != float('inf') else p1[0]
        else:
            # Least squares fitting using numpy
            x_vals = X.flatten()
            coeffs = np.polyfit(x_vals, y, 1)
            slope, intercept = coeffs[0], coeffs[1]
        
        return slope, intercept


class CalibrationUtility:
    """
    Utility class for camera calibration and pixel-to-mm conversion.
    """
    
    @staticmethod
    def calibrate_from_known_distance(image_path: str, 
                                    point1: Tuple[float, float], 
                                    point2: Tuple[float, float], 
                                    known_distance_mm: float) -> float:
        """
        Calculate calibration factor from a known distance in the image.
        
        Args:
            image_path: Path to calibration image
            point1: First point coordinates
            point2: Second point coordinates
            known_distance_mm: Known distance between points in mm
            
        Returns:
            Calibration factor (mm per pixel)
        """
        # Calculate pixel distance
        pixel_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        # Calculate calibration factor
        calibration_factor = known_distance_mm / pixel_distance
        
        logger.info(f"Calibration: {pixel_distance:.2f}px = {known_distance_mm}mm")
        logger.info(f"Calibration factor: {calibration_factor:.6f} mm/px")
        
        return calibration_factor
    
    @staticmethod
    def calibrate_from_charpy_dimensions(corners: List[Tuple[float, float]], 
                                       specimen_width_mm: float = 10.0) -> float:
        """
        Calibrate using standard Charpy specimen dimensions.
        
        Standard Charpy specimen: 10mm x 10mm x 55mm
        
        Args:
            corners: Corner coordinates of the specimen
            specimen_width_mm: Known specimen width in mm
            
        Returns:
            Calibration factor (mm per pixel)
        """
        if len(corners) != 4:
            raise ValueError("Need exactly 4 corners for calibration")
        
        # Find bottom corners (should represent the full width)
        sorted_corners = sorted(corners, key=lambda p: p[1], reverse=True)
        bottom_corners = sorted_corners[:2]
        bottom_left = min(bottom_corners, key=lambda p: p[0])
        bottom_right = max(bottom_corners, key=lambda p: p[0])
        
        # Calculate pixel width
        pixel_width = bottom_right[0] - bottom_left[0]
        
        # Calculate calibration factor
        calibration_factor = specimen_width_mm / pixel_width
        
        logger.info(f"Specimen width: {pixel_width:.2f}px = {specimen_width_mm}mm")
        logger.info(f"Calibration factor: {calibration_factor:.6f} mm/px")
        
        return calibration_factor


class CharpyDataAnalyzer:
    """
    Statistical analysis tools for Charpy measurement data.
    """
    
    @staticmethod
    def analyze_measurement_batch(measurements: List[CharpyMeasurement]) -> Dict[str, Any]:
        """
        Perform statistical analysis on a batch of measurements.
        
        Args:
            measurements: List of measurement results
            
        Returns:
            Dictionary with statistical analysis results
        """
        # Extract measurement values
        total_expansions = [m.total_expansion_pixels for m in measurements if m.total_expansion_pixels > 0]
        left_expansions = [m.left_expansion_pixels for m in measurements if m.left_expansion_pixels > 0]
        right_expansions = [m.right_expansion_pixels for m in measurements if m.right_expansion_pixels > 0]
        
        if not total_expansions:
            return {"error": "No valid measurements found"}
        
        # Calculate statistics
        analysis = {
            'total_specimens': len(measurements),
            'valid_measurements': len(total_expansions),
            'total_expansion_stats': {
                'mean': np.mean(total_expansions),
                'std': np.std(total_expansions),
                'min': np.min(total_expansions),
                'max': np.max(total_expansions),
                'median': np.median(total_expansions),
                'q25': np.percentile(total_expansions, 25),
                'q75': np.percentile(total_expansions, 75)
            },
            'left_expansion_stats': {
                'mean': np.mean(left_expansions),
                'std': np.std(left_expansions)
            } if left_expansions else None,
            'right_expansion_stats': {
                'mean': np.mean(right_expansions),
                'std': np.std(right_expansions)
            } if right_expansions else None
        }
        
        # Detect outliers using IQR method
        q1 = analysis['total_expansion_stats']['q25']
        q3 = analysis['total_expansion_stats']['q75']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in total_expansions if x < lower_bound or x > upper_bound]
        analysis['outliers'] = {
            'count': len(outliers),
            'values': outliers,
            'percentage': len(outliers) / len(total_expansions) * 100
        }
        
        return analysis
    
    @staticmethod
    def export_to_csv(measurements: List[CharpyMeasurement], 
                     output_path: str, 
                     include_corners: bool = False) -> None:
        """
        Export measurements to CSV format.
        
        Args:
            measurements: List of measurement results
            output_path: Path to save CSV file
            include_corners: Whether to include corner coordinates
        """
        if not HAS_PANDAS:
            logger.warning("Pandas not available, using basic CSV export")
            # Basic CSV export without pandas
            import csv
            
            # Prepare headers
            headers = [
                'specimen_id', 'left_expansion_pixels', 'right_expansion_pixels', 
                'total_expansion_pixels', 'left_expansion_mm', 'right_expansion_mm',
                'total_expansion_mm', 'calibration_factor'
            ]
            
            if include_corners:
                headers.extend([f'corner_{j+1}_x' for j in range(4)])
                headers.extend([f'corner_{j+1}_y' for j in range(4)])
            
            headers.extend(['avg_confidence', 'min_confidence'])
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                for i, measurement in enumerate(measurements):
                    row = [
                        i + 1,
                        measurement.left_expansion_pixels,
                        measurement.right_expansion_pixels,
                        measurement.total_expansion_pixels,
                        measurement.left_expansion_mm or '',
                        measurement.right_expansion_mm or '',
                        measurement.total_expansion_mm or '',
                        measurement.calibration_factor or ''
                    ]
                    
                    if include_corners:
                        if measurement.corners and len(measurement.corners) >= 4:
                            for j in range(4):
                                row.extend([measurement.corners[j][0], measurement.corners[j][1]])
                        else:
                            row.extend([''] * 8)  # Empty values for missing corners
                    
                    if measurement.confidence_scores:
                        row.extend([np.mean(measurement.confidence_scores), np.min(measurement.confidence_scores)])
                    else:
                        row.extend(['', ''])
                    
                    writer.writerow(row)
        else:
            # Use pandas for more sophisticated export
            data = []
            
            for i, measurement in enumerate(measurements):
                row = {
                    'specimen_id': i + 1,
                    'left_expansion_pixels': measurement.left_expansion_pixels,
                    'right_expansion_pixels': measurement.right_expansion_pixels,
                    'total_expansion_pixels': measurement.total_expansion_pixels,
                    'left_expansion_mm': measurement.left_expansion_mm,
                    'right_expansion_mm': measurement.right_expansion_mm,
                    'total_expansion_mm': measurement.total_expansion_mm,
                    'calibration_factor': measurement.calibration_factor
                }
                
                if include_corners and measurement.corners:
                    for j, corner in enumerate(measurement.corners):
                        row[f'corner_{j+1}_x'] = corner[0]
                        row[f'corner_{j+1}_y'] = corner[1]
                
                if measurement.confidence_scores:
                    row['avg_confidence'] = np.mean(measurement.confidence_scores)
                    row['min_confidence'] = np.min(measurement.confidence_scores)
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Measurements exported to {output_path}")
    
    @staticmethod
    def create_measurement_report(measurements: List[CharpyMeasurement], 
                                output_path: str) -> None:
        """
        Create a comprehensive measurement report with visualizations.
        
        Args:
            measurements: List of measurement results
            output_path: Path to save report (HTML format)
        """
        # Analyze data
        analysis = CharpyDataAnalyzer.analyze_measurement_batch(measurements)
        
        # Create visualizations if matplotlib is available
        plot_path = None
        if HAS_MATPLOTLIB and plt is not None:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Extract valid measurements
                total_expansions = [m.total_expansion_pixels for m in measurements if m.total_expansion_pixels > 0]
                left_expansions = [m.left_expansion_pixels for m in measurements if m.left_expansion_pixels > 0]
                right_expansions = [m.right_expansion_pixels for m in measurements if m.right_expansion_pixels > 0]
                
                # Histogram of total expansions
                axes[0, 0].hist(total_expansions, bins=20, alpha=0.7, color='blue')
                axes[0, 0].set_title('Distribution of Total Lateral Expansion')
                axes[0, 0].set_xlabel('Expansion (pixels)')
                axes[0, 0].set_ylabel('Frequency')
                
                # Box plot
                axes[0, 1].boxplot([total_expansions], labels=['Total Expansion'])
                axes[0, 1].set_title('Box Plot of Total Expansion')
                axes[0, 1].set_ylabel('Expansion (pixels)')
                
                # Left vs Right expansion scatter
                if left_expansions and right_expansions:
                    axes[1, 0].scatter(left_expansions, right_expansions, alpha=0.6)
                    axes[1, 0].set_xlabel('Left Expansion (pixels)')
                    axes[1, 0].set_ylabel('Right Expansion (pixels)')
                    axes[1, 0].set_title('Left vs Right Expansion')
                    
                    # Add diagonal line for reference
                    max_val = max(max(left_expansions), max(right_expansions))
                    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
                
                # Time series (specimen order)
                specimen_ids = list(range(1, len(total_expansions) + 1))
                axes[1, 1].plot(specimen_ids, total_expansions, 'o-', alpha=0.7)
                axes[1, 1].set_xlabel('Specimen ID')
                axes[1, 1].set_ylabel('Total Expansion (pixels)')
                axes[1, 1].set_title('Expansion by Specimen Order')
                
                plt.tight_layout()
                
                # Save plot
                plot_path = output_path.replace('.html', '_plots.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Visualization plots saved to {plot_path}")
            except Exception as e:
                logger.warning(f"Failed to create visualizations: {e}")
                plot_path = None
        else:
            logger.warning("Matplotlib not available - skipping visualizations")
        
        # Generate HTML report
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Charpy Lateral Expansion Measurement Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .outliers {{ background-color: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Charpy Lateral Expansion Measurement Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <h2>Summary Statistics</h2>
            <div class="stats">
                <p><strong>Total Specimens:</strong> {analysis['total_specimens']}</p>
                <p><strong>Valid Measurements:</strong> {analysis['valid_measurements']}</p>
                <p><strong>Mean Total Expansion:</strong> {analysis['total_expansion_stats']['mean']:.2f} pixels</p>
                <p><strong>Standard Deviation:</strong> {analysis['total_expansion_stats']['std']:.2f} pixels</p>
                <p><strong>Range:</strong> {analysis['total_expansion_stats']['min']:.2f} - {analysis['total_expansion_stats']['max']:.2f} pixels</p>
                <p><strong>Median:</strong> {analysis['total_expansion_stats']['median']:.2f} pixels</p>
            </div>
            
            <h2>Outlier Analysis</h2>
            <div class="outliers">
                <p><strong>Outliers Detected:</strong> {analysis['outliers']['count']} ({analysis['outliers']['percentage']:.1f}%)</p>
                {f"<p><strong>Outlier Values:</strong> {', '.join([f'{x:.2f}' for x in analysis['outliers']['values']])}</p>" if analysis['outliers']['values'] else ""}
            </div>
            
            <h2>Visualizations</h2>
            {f'<img src="{Path(plot_path).name}" alt="Measurement Analysis Plots" style="max-width: 100%; height: auto;">' if plot_path else '<p>Visualizations not available (matplotlib not installed)</p>'}
            
            <h2>Individual Measurements</h2>
            <table>
                <tr>
                    <th>Specimen ID</th>
                    <th>Left Expansion (px)</th>
                    <th>Right Expansion (px)</th>
                    <th>Total Expansion (px)</th>
                    <th>Total Expansion (mm)</th>
                </tr>
        """
        
        for i, measurement in enumerate(measurements):
            html_content += f"""
                <tr>
                    <td>{i + 1}</td>
                    <td>{measurement.left_expansion_pixels:.2f}</td>
                    <td>{measurement.right_expansion_pixels:.2f}</td>
                    <td>{measurement.total_expansion_pixels:.2f}</td>
                    <td>{measurement.total_expansion_mm:.2f if measurement.total_expansion_mm else 'N/A'}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report saved to {output_path}")


def demonstrate_alternative_methods():
    """Demonstrate alternative measurement methods."""
    print("\n=== Alternative Measurement Methods Demo ===")
    
    # Create test corners
    corners = [(50, 200), (350, 200), (80, 50), (320, 50)]  # bottom_left, bottom_right, top_left, top_right
    
    # Standard method
    standard_measurer = CharpyLateralExpansionMeasurer()
    bottom_left, bottom_right = standard_measurer.identify_bottom_corners(corners)
    top_left, top_right = standard_measurer.identify_top_corners(corners, [bottom_left, bottom_right])
    left_perp, right_perp = standard_measurer.create_perpendicular_lines(bottom_left, bottom_right)
    standard_result = standard_measurer.measure_lateral_expansion(top_left, top_right, left_perp, right_perp)
    
    # Alternative method
    alt_measurer = AlternativeCharpyMeasurer()
    alt_result = alt_measurer.horizontal_distance_method(corners)
    
    print(f"Standard Method - Total Expansion: {standard_result['total_expansion_pixels']:.2f}px")
    print(f"Alternative Method - Total Expansion: {alt_result['total_expansion_pixels']:.2f}px")
    print(f"Alternative Method - Top Width: {alt_result['top_width_pixels']:.2f}px")
    print(f"Alternative Method - Bottom Width: {alt_result['bottom_width_pixels']:.2f}px")


if __name__ == "__main__":
    # Demonstrate alternative methods
    demonstrate_alternative_methods()
    
    # Example calibration
    print("\n=== Calibration Example ===")
    test_corners = [(50, 200), (350, 200), (80, 50), (320, 50)]
    calibration_factor = CalibrationUtility.calibrate_from_charpy_dimensions(test_corners, 10.0)
    print(f"Calculated calibration factor: {calibration_factor:.6f} mm/px")