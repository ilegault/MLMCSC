#!/usr/bin/env python3
"""
Charpy Lateral Expansion Measurement Integration Example

This script demonstrates how to integrate the Charpy lateral expansion measurement
system with the existing MLMCSC framework, including:
- Integration with YOLO detection models
- Quality control integration
- Database storage
- Batch processing workflows
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
current_file = Path(__file__)
print(f"DEBUG: Current file: {current_file}")
print(f"DEBUG: Current file parent: {current_file.parent}")
print(f"DEBUG: Current file parent.parent: {current_file.parent.parent}")

project_root = Path(__file__).parent.parent.parent  # Go up to project root
src_dir = project_root / "src"
print(f"DEBUG: Project root: {project_root}")
print(f"DEBUG: Src dir: {src_dir}")
print(f"DEBUG: Src dir exists: {src_dir.exists()}")

if src_dir.exists():
    print(f"DEBUG: Contents of src dir: {list(src_dir.iterdir())}")

sys.path.insert(0, str(src_dir))
print(f"DEBUG: Python path after insert: {sys.path[:3]}")  # Show first 3 entries

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# MLMCSC imports
print("DEBUG: Attempting to import postprocessing...")
try:
    from postprocessing import CharpyLateralExpansionMeasurer, CharpyMeasurement
    print("DEBUG: Successfully imported from postprocessing")
except ImportError as e:
    print(f"DEBUG: Failed to import from postprocessing: {e}")
    # Try alternative import
    try:
        import postprocessing
        print(f"DEBUG: postprocessing module location: {postprocessing.__file__}")
        from postprocessing import CharpyLateralExpansionMeasurer, CharpyMeasurement
    except Exception as e2:
        print(f"DEBUG: Alternative import also failed: {e2}")
        raise

print("DEBUG: Attempting to import charpy_measurement_utils...")
try:
    from postprocessing.charpy_measurement_utils import CalibrationUtility, CharpyDataAnalyzer
    print("DEBUG: Successfully imported charpy_measurement_utils")
except ImportError as e:
    print(f"DEBUG: Failed to import charpy_measurement_utils: {e}")
    raise

print("DEBUG: Attempting to import quality_control...")
try:
    from mlmcsc.quality_control import QualityControlSystem
    print("DEBUG: Successfully imported quality_control")
except ImportError as e:
    print(f"DEBUG: Failed to import quality_control: {e}")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharpyMeasurementPipeline:
    """
    Complete pipeline for Charpy specimen measurement integrated with MLMCSC.
    """
    
    def __init__(self, 
                 calibration_factor: Optional[float] = None,
                 quality_control_enabled: bool = True,
                 output_dir: str = "charpy_results"):
        """
        Initialize the measurement pipeline.
        
        Args:
            calibration_factor: Pixel to mm conversion factor
            quality_control_enabled: Whether to enable quality control checks
            output_dir: Directory for output files
        """
        self.calibration_factor = calibration_factor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.measurer = CharpyLateralExpansionMeasurer(
            calibration_factor=calibration_factor,
            visualization_enabled=True
        )
        
        # Initialize quality control if enabled
        self.quality_control = None
        if quality_control_enabled:
            self.quality_control = QualityControlSystem(
                storage_path=str(self.output_dir / "quality_control")
            )
        
        logger.info(f"CharpyMeasurementPipeline initialized with output dir: {self.output_dir}")
    
    def simulate_yolo_detection(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Simulate YOLO detection results for demonstration.
        
        In a real implementation, this would be replaced with actual YOLO model inference.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of simulated YOLO detections
        """
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        
        # Simulate corner detections based on image dimensions
        # In practice, these would come from your trained YOLO model
        detections = [
            {
                'class': 'charpy_corner',
                'x': width * 0.15,  # Left side, bottom
                'y': height * 0.85,
                'width': 20,
                'height': 20,
                'confidence': 0.92
            },
            {
                'class': 'charpy_corner', 
                'x': width * 0.85,  # Right side, bottom
                'y': height * 0.85,
                'width': 20,
                'height': 20,
                'confidence': 0.89
            },
            {
                'class': 'charpy_corner',
                'x': width * 0.25,  # Left side, top (narrower due to fracture)
                'y': height * 0.15,
                'width': 20,
                'height': 20,
                'confidence': 0.87
            },
            {
                'class': 'charpy_corner',
                'x': width * 0.75,  # Right side, top
                'y': height * 0.15,
                'width': 20,
                'height': 20,
                'confidence': 0.91
            }
        ]
        
        logger.info(f"Simulated {len(detections)} corner detections for {image_path}")
        return detections
    
    def process_single_specimen(self, 
                              image_path: str, 
                              specimen_id: str,
                              yolo_detections: Optional[List[Dict[str, Any]]] = None) -> CharpyMeasurement:
        """
        Process a single Charpy specimen.
        
        Args:
            image_path: Path to specimen image
            specimen_id: Unique identifier for the specimen
            yolo_detections: YOLO detection results (if None, will simulate)
            
        Returns:
            Measurement results
        """
        try:
            # Get YOLO detections
            if yolo_detections is None:
                yolo_detections = self.simulate_yolo_detection(image_path)
            
            # Generate output paths
            visualization_path = self.output_dir / f"{specimen_id}_measurement.jpg"
            
            # Perform measurement
            measurement, result_image = self.measurer.process_charpy_specimen(
                image_path=image_path,
                yolo_detections=yolo_detections,
                output_path=str(visualization_path)
            )
            
            # Quality control checks
            if self.quality_control:
                self._perform_quality_checks(measurement, specimen_id, yolo_detections)
            
            logger.info(f"Processed specimen {specimen_id}: "
                       f"Total expansion = {measurement.total_expansion_pixels:.2f}px"
                       f"{f' ({measurement.total_expansion_mm:.2f}mm)' if measurement.total_expansion_mm else ''}")
            
            return measurement
            
        except Exception as e:
            logger.error(f"Failed to process specimen {specimen_id}: {e}")
            raise
    
    def _perform_quality_checks(self, 
                              measurement: CharpyMeasurement, 
                              specimen_id: str,
                              yolo_detections: List[Dict[str, Any]]) -> None:
        """
        Perform quality control checks on the measurement.
        
        Args:
            measurement: Measurement results
            specimen_id: Specimen identifier
            yolo_detections: YOLO detection results
        """
        if not self.quality_control:
            return
        
        # Check detection confidence
        confidences = measurement.confidence_scores or []
        if confidences:
            min_confidence = min(confidences)
            avg_confidence = np.mean(confidences)
            
            if min_confidence < 0.7:
                self.quality_control._create_quality_flag(
                    flag_type='low_confidence',
                    severity='medium',
                    image_id=specimen_id,
                    description=f"Low detection confidence: min={min_confidence:.2f}, avg={avg_confidence:.2f}"
                )
        
        # Check for unusual expansion values
        if measurement.total_expansion_pixels > 100:  # Threshold for unusually high expansion
            self.quality_control._create_quality_flag(
                flag_type='high_variance',
                severity='high',
                image_id=specimen_id,
                description=f"Unusually high lateral expansion: {measurement.total_expansion_pixels:.2f}px"
            )
        
        # Check for asymmetric expansion
        if (measurement.left_expansion_pixels > 0 and measurement.right_expansion_pixels > 0):
            asymmetry_ratio = abs(measurement.left_expansion_pixels - measurement.right_expansion_pixels) / \
                            max(measurement.left_expansion_pixels, measurement.right_expansion_pixels)
            
            if asymmetry_ratio > 0.5:  # More than 50% asymmetry
                self.quality_control._create_quality_flag(
                    flag_type='asymmetric_expansion',
                    severity='medium',
                    image_id=specimen_id,
                    description=f"Asymmetric expansion detected: ratio={asymmetry_ratio:.2f}"
                )
    
    def process_batch(self, 
                     image_paths: List[str], 
                     specimen_ids: Optional[List[str]] = None) -> List[CharpyMeasurement]:
        """
        Process a batch of Charpy specimens.
        
        Args:
            image_paths: List of image file paths
            specimen_ids: List of specimen identifiers (auto-generated if None)
            
        Returns:
            List of measurement results
        """
        if specimen_ids is None:
            specimen_ids = [f"specimen_{i+1:03d}" for i in range(len(image_paths))]
        
        if len(image_paths) != len(specimen_ids):
            raise ValueError("Number of images and specimen IDs must match")
        
        measurements = []
        
        for image_path, specimen_id in zip(image_paths, specimen_ids):
            try:
                measurement = self.process_single_specimen(image_path, specimen_id)
                measurements.append(measurement)
            except Exception as e:
                logger.error(f"Failed to process {specimen_id}: {e}")
                # Create empty measurement for failed cases
                empty_measurement = CharpyMeasurement(
                    left_expansion_pixels=0.0,
                    right_expansion_pixels=0.0,
                    total_expansion_pixels=0.0
                )
                measurements.append(empty_measurement)
        
        # Generate batch report
        self._generate_batch_report(measurements, specimen_ids)
        
        return measurements
    
    def _generate_batch_report(self, 
                             measurements: List[CharpyMeasurement], 
                             specimen_ids: List[str]) -> None:
        """
        Generate comprehensive batch processing report.
        
        Args:
            measurements: List of measurement results
            specimen_ids: List of specimen identifiers
        """
        # Save measurements to JSON
        json_path = self.output_dir / "batch_measurements.json"
        measurement_data = {
            'measurements': [
                {**m.to_dict(), 'specimen_id': sid} 
                for m, sid in zip(measurements, specimen_ids)
            ],
            'processing_timestamp': datetime.now().isoformat(),
            'calibration_factor': self.calibration_factor,
            'total_specimens': len(measurements)
        }
        
        with open(json_path, 'w') as f:
            json.dump(measurement_data, f, indent=2)
        
        # Export to CSV
        csv_path = self.output_dir / "batch_measurements.csv"
        CharpyDataAnalyzer.export_to_csv(measurements, str(csv_path))
        
        # Generate comprehensive report
        report_path = self.output_dir / "measurement_report.html"
        CharpyDataAnalyzer.create_measurement_report(measurements, str(report_path))
        
        # Statistical analysis
        analysis = CharpyDataAnalyzer.analyze_measurement_batch(measurements)
        
        # Save analysis results
        analysis_path = self.output_dir / "statistical_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Batch report generated:")
        logger.info(f"  - JSON data: {json_path}")
        logger.info(f"  - CSV export: {csv_path}")
        logger.info(f"  - HTML report: {report_path}")
        logger.info(f"  - Statistical analysis: {analysis_path}")
    
    def calibrate_system(self, 
                        calibration_image_path: str, 
                        method: str = 'charpy_dimensions') -> float:
        """
        Calibrate the measurement system.
        
        Args:
            calibration_image_path: Path to calibration image
            method: Calibration method ('charpy_dimensions' or 'known_distance')
            
        Returns:
            Calculated calibration factor
        """
        if method == 'charpy_dimensions':
            # Use YOLO to detect corners in calibration image
            detections = self.simulate_yolo_detection(calibration_image_path)
            corners, _ = self.measurer.parse_yolo_output(detections)
            
            if len(corners) != 4:
                raise ValueError(f"Need exactly 4 corners for calibration, got {len(corners)}")
            
            calibration_factor = CalibrationUtility.calibrate_from_charpy_dimensions(
                corners, specimen_width_mm=10.0
            )
            
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
        
        # Update measurer with new calibration
        self.calibration_factor = calibration_factor
        self.measurer.calibration_factor = calibration_factor
        
        # Save calibration info
        calibration_info = {
            'calibration_factor': calibration_factor,
            'method': method,
            'calibration_image': calibration_image_path,
            'timestamp': datetime.now().isoformat()
        }
        
        calibration_path = self.output_dir / "calibration_info.json"
        with open(calibration_path, 'w') as f:
            json.dump(calibration_info, f, indent=2)
        
        logger.info(f"System calibrated: {calibration_factor:.6f} mm/px")
        return calibration_factor


def create_demo_images(output_dir: str = "demo_images") -> List[str]:
    """
    Create synthetic demo images for testing.
    
    Args:
        output_dir: Directory to save demo images
        
    Returns:
        List of created image paths
    """
    demo_dir = Path(output_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    # Create 5 demo specimens with varying expansion
    for i in range(5):
        # Create image with different expansion levels
        width, height = 400, 300
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Vary expansion amount
        expansion_factor = 0.8 + i * 0.05  # 0.8 to 1.0
        
        # Bottom edge (reference)
        bottom_left = (50, height - 50)
        bottom_right = (width - 50, height - 50)
        
        # Top edge (with expansion)
        top_width = (width - 100) * expansion_factor
        margin = (width - 100 - top_width) / 2
        top_left = (50 + margin, 50)
        top_right = (width - 50 - margin, 50)
        
        # Draw specimen
        pts = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], (120, 120, 120))
        
        # Add texture and noise
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Draw fracture line
        cv2.line(image, tuple(map(int, top_left)), tuple(map(int, top_right)), (200, 200, 200), 2)
        
        # Add specimen label
        cv2.putText(image, f"Specimen {i+1}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save image
        image_path = demo_dir / f"charpy_specimen_{i+1:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        image_paths.append(str(image_path))
    
    logger.info(f"Created {len(image_paths)} demo images in {demo_dir}")
    return image_paths


def main():
    """Main demonstration function."""
    print("=== Charpy Lateral Expansion Measurement Demo ===\n")
    
    # Create demo images
    demo_images = create_demo_images()
    
    # Initialize pipeline
    pipeline = CharpyMeasurementPipeline(
        calibration_factor=None,  # Will calibrate using first image
        quality_control_enabled=True,
        output_dir="charpy_demo_results"
    )
    
    # Calibrate system using first image
    print("1. Calibrating measurement system...")
    try:
        calibration_factor = pipeline.calibrate_system(demo_images[0])
        print(f"   ✓ Calibration completed: {calibration_factor:.6f} mm/px\n")
    except Exception as e:
        print(f"   ✗ Calibration failed: {e}")
        print("   Continuing without calibration...\n")
    
    # Process single specimen
    print("2. Processing single specimen...")
    try:
        measurement = pipeline.process_single_specimen(
            demo_images[0], 
            "demo_specimen_001"
        )
        print(f"   ✓ Single specimen processed successfully")
        print(f"   Total expansion: {measurement.total_expansion_pixels:.2f}px"
              f"{f' ({measurement.total_expansion_mm:.2f}mm)' if measurement.total_expansion_mm else ''}\n")
    except Exception as e:
        print(f"   ✗ Single specimen processing failed: {e}\n")
    
    # Process batch
    print("3. Processing batch of specimens...")
    try:
        measurements = pipeline.process_batch(demo_images)
        print(f"   ✓ Batch processing completed: {len(measurements)} specimens")
        
        # Print summary
        valid_measurements = [m for m in measurements if m.total_expansion_pixels > 0]
        if valid_measurements:
            avg_expansion = np.mean([m.total_expansion_pixels for m in valid_measurements])
            print(f"   Average expansion: {avg_expansion:.2f}px")
        print()
    except Exception as e:
        print(f"   ✗ Batch processing failed: {e}\n")
    
    # Quality control summary
    if pipeline.quality_control:
        print("4. Quality control summary...")
        flags = list(pipeline.quality_control.quality_flags.values())
        if flags:
            print(f"   Quality flags raised: {len(flags)}")
            for flag in flags[:3]:  # Show first 3 flags
                print(f"   - {flag.flag_type}: {flag.description}")
        else:
            print("   ✓ No quality issues detected")
        print()
    
    print("5. Output files generated:")
    output_dir = Path("charpy_demo_results")
    if output_dir.exists():
        for file_path in sorted(output_dir.glob("*")):
            if file_path.is_file():
                print(f"   - {file_path.name}")
    
    print(f"\n=== Demo completed successfully! ===")
    print(f"Check the 'charpy_demo_results' directory for all output files.")


if __name__ == "__main__":
    main()