#!/usr/bin/env python3
"""
Training Data Extractor

Extracts individual training samples from your 10-sample reference image
and creates proper training dataset with conservative augmentation.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from improved_fracture_detector import ImprovedFractureSurfaceDetector
from shiny_region_analyzer import ShinyRegionAnalyzer
from shiny_region_classifier import ShinyRegionBasedClassifier

logger = logging.getLogger(__name__)


class TrainingDataExtractor:
    """Extract and prepare training data from reference images."""

    def __init__(self):
        self.classifier = ShinyRegionBasedClassifier()
        logger.info("Training Data Extractor initialized")

    def extract_samples_from_reference(self,
                                       reference_image_path: str,
                                       output_dir: str,
                                       shear_percentages: Optional[List[float]] = None,
                                       layout: str = "auto") -> bool:
        """
        Extract individual training samples from your reference image.

        Args:
            reference_image_path: Path to your 10-sample reference image
            output_dir: Directory to save extracted samples
            shear_percentages: List of shear percentages (default: [10,20,...,100])
            layout: Grid layout - "auto", "horizontal", "vertical", or "custom"

        Returns:
            True if successful, False otherwise
        """
        try:
            # Default shear percentages
            if shear_percentages is None:
                shear_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

            # Load reference image
            ref_image = cv2.imread(reference_image_path)
            if ref_image is None:
                logger.error(f"Could not load reference image: {reference_image_path}")
                return False

            logger.info(f"Processing reference image: {reference_image_path}")
            logger.info(f"Image size: {ref_image.shape[1]}x{ref_image.shape[0]}")

            # Extract grid cells
            cells = self._extract_grid_cells(ref_image, len(shear_percentages), layout)

            if len(cells) != len(shear_percentages):
                logger.error(f"Expected {len(shear_percentages)} cells, got {len(cells)}")
                return False

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Process each cell
            for i, (cell, shear_percent) in enumerate(zip(cells, shear_percentages)):
                logger.info(f"Processing {shear_percent}% shear sample...")

                # Create directory for this shear percentage
                shear_dir = output_path / f"{shear_percent}_percent"
                shear_dir.mkdir(exist_ok=True)

                # Save original
                cv2.imwrite(str(shear_dir / "original.jpg"), cell)

                # Generate conservative augmentations
                for aug_idx in range(15):  # 15 augmented versions per original
                    augmented = self._conservative_augment(cell)
                    cv2.imwrite(str(shear_dir / f"aug_{aug_idx:02d}.jpg"), augmented)

                logger.info(f"  Saved 16 samples (1 original + 15 augmented)")

            logger.info(f"✅ Successfully extracted training data to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract training data: {e}")
            return False

    def _extract_grid_cells(self, image: np.ndarray, num_cells: int, layout: str) -> List[np.ndarray]:
        """Extract individual cells from grid layout."""

        h, w = image.shape[:2]

        if layout == "auto":
            # Auto-detect layout based on aspect ratio
            if w > h:  # Wider than tall
                if num_cells == 10:
                    rows, cols = 2, 5  # 2x5 grid
                elif num_cells == 6:
                    rows, cols = 2, 3  # 2x3 grid
                else:
                    # Try to make roughly rectangular
                    cols = int(np.ceil(np.sqrt(num_cells * w / h)))
                    rows = int(np.ceil(num_cells / cols))
            else:  # Taller than wide
                if num_cells == 10:
                    rows, cols = 5, 2  # 5x2 grid
                elif num_cells == 6:
                    rows, cols = 3, 2  # 3x2 grid
                else:
                    # Try to make roughly rectangular
                    rows = int(np.ceil(np.sqrt(num_cells * h / w)))
                    cols = int(np.ceil(num_cells / rows))

        elif layout == "horizontal":
            rows = 1
            cols = num_cells
        elif layout == "vertical":
            rows = num_cells
            cols = 1
        else:  # Custom layout
            # For custom, assume square-ish grid
            cols = int(np.ceil(np.sqrt(num_cells)))
            rows = int(np.ceil(num_cells / cols))

        logger.info(f"Using {rows}x{cols} grid layout")

        # Calculate cell dimensions
        cell_h = h // rows
        cell_w = w // cols

        # Extract cells
        cells = []
        for row in range(rows):
            for col in range(cols):
                if len(cells) >= num_cells:
                    break

                # Calculate cell boundaries
                y1 = row * cell_h
                y2 = min((row + 1) * cell_h, h)
                x1 = col * cell_w
                x2 = min((col + 1) * cell_w, w)

                # Extract cell with some padding removal to avoid borders
                pad = min(10, cell_w // 20, cell_h // 20)  # 5% padding
                y1_pad = y1 + pad
                y2_pad = y2 - pad
                x1_pad = x1 + pad
                x2_pad = x2 - pad

                if y2_pad > y1_pad and x2_pad > x1_pad:
                    cell = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    cells.append(cell)
                else:
                    # Fallback without padding
                    cell = image[y1:y2, x1:x2]
                    cells.append(cell)

        return cells

    def _conservative_augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply conservative augmentation that preserves shiny region characteristics.

        Key principle: Preserve the relative brightness and spatial relationships
        that are critical for shear classification.
        """
        result = image.copy()

        # 1. Very small rotation (±3 degrees max)
        # Larger rotations can change the apparent shape of shiny regions
        angle = np.random.uniform(-3, 3)
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 2. Minimal brightness adjustment (±5% max)
        # Large brightness changes can make rough regions look shiny or vice versa
        brightness = np.random.uniform(0.95, 1.05)
        result = np.clip(result.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # 3. Very small contrast adjustment (±2% max)
        # Contrast changes affect the shiny/rough region segmentation
        contrast = np.random.uniform(0.98, 1.02)
        mean_val = np.mean(result)
        result = np.clip((result - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

        # 4. Minimal noise (std ≤ 3)
        # High noise can create false "rough" texture in shiny regions
        noise_std = np.random.uniform(0, 3)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, result.shape)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Small random crop and resize (90-100% of original)
        # This simulates slight variations in framing
        if np.random.random() < 0.3:  # Only apply to 30% of samples
            crop_factor = np.random.uniform(0.90, 1.0)
            result = self._random_crop_resize(result, crop_factor)

        return result

    def _random_crop_resize(self, image: np.ndarray, crop_factor: float) -> np.ndarray:
        """Apply random crop and resize back to original size."""
        h, w = image.shape[:2]

        # Calculate crop size
        crop_h = int(h * crop_factor)
        crop_w = int(w * crop_factor)

        # Random crop position
        start_h = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        start_w = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0

        # Crop
        cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # Resize back to original size
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

        return resized

    def validate_training_data(self, training_data_dir: str) -> Dict[str, Any]:
        """
        Validate the quality of extracted training data.

        Args:
            training_data_dir: Directory containing training data

        Returns:
            Validation report
        """
        training_dir = Path(training_data_dir)

        if not training_dir.exists():
            return {'error': f"Training directory not found: {training_data_dir}"}

        # Collect training data
        training_data = {}
        total_samples = 0

        for shear_dir in training_dir.iterdir():
            if shear_dir.is_dir() and "_percent" in shear_dir.name:
                try:
                    shear_val = float(shear_dir.name.replace("_percent", ""))
                    image_paths = list(shear_dir.glob("*.jpg")) + list(shear_dir.glob("*.png"))
                    training_data[shear_val] = [str(p) for p in image_paths]
                    total_samples += len(image_paths)
                    logger.info(f"Found {len(image_paths)} samples for {shear_val}% shear")
                except ValueError:
                    logger.warning(f"Could not parse shear value from: {shear_dir.name}")

        if not training_data:
            return {'error': "No valid training data found"}

        # Test feature extraction on a few samples
        feature_extraction_success = 0
        feature_extraction_total = 0

        for shear_percent, paths in training_data.items():
            # Test first 3 samples from each class
            test_paths = paths[:3]

            for path in test_paths:
                feature_extraction_total += 1
                try:
                    image = cv2.imread(path)
                    if image is not None:
                        features = self.classifier.process_image(image)
                        if features is not None:
                            feature_extraction_success += 1
                except Exception as e:
                    logger.debug(f"Feature extraction failed for {path}: {e}")

        # Calculate statistics
        shear_values = sorted(training_data.keys())
        sample_counts = [len(training_data[sv]) for sv in shear_values]

        report = {
            'total_samples': total_samples,
            'num_classes': len(training_data),
            'shear_range': (min(shear_values), max(shear_values)),
            'samples_per_class': {
                'min': min(sample_counts),
                'max': max(sample_counts),
                'mean': np.mean(sample_counts),
                'std': np.std(sample_counts)
            },
            'feature_extraction_success_rate': feature_extraction_success / feature_extraction_total if feature_extraction_total > 0 else 0.0,
            'class_distribution': dict(zip(shear_values, sample_counts))
        }

        return report

    def visualize_extracted_samples(self, training_data_dir: str, save_path: Optional[str] = None):
        """Create a visualization of extracted training samples."""

        training_dir = Path(training_data_dir)

        # Collect one sample from each class
        sample_images = []
        sample_labels = []

        for shear_dir in sorted(training_dir.iterdir()):
            if shear_dir.is_dir() and "_percent" in shear_dir.name:
                try:
                    shear_val = float(shear_dir.name.replace("_percent", ""))
                    image_files = list(shear_dir.glob("original.jpg"))

                    if not image_files:
                        image_files = list(shear_dir.glob("*.jpg"))[:1]

                    if image_files:
                        image = cv2.imread(str(image_files[0]))
                        if image is not None:
                            # Convert BGR to RGB for matplotlib
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            sample_images.append(image_rgb)
                            sample_labels.append(f"{shear_val}%")

                except ValueError:
                    continue

        if not sample_images:
            logger.error("No sample images found for visualization")
            return

        # Create visualization
        n_samples = len(sample_images)
        cols = min(5, n_samples)
        rows = int(np.ceil(n_samples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
            if i < len(axes):
                axes[i].imshow(image)
                axes[i].set_title(f"Shear: {label}")
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(sample_images), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        plt.show()


def main():
    """Example usage of the training data extractor."""

    print("🔬 TRAINING DATA EXTRACTOR FOR CHARPY FRACTURE SURFACES")
    print("=" * 60)
    print("Extract individual samples from your 10-sample reference image")
    print()

    extractor = TrainingDataExtractor()

    # Example usage
    print("📝 Example Usage:")
    print("1. extractor.extract_samples_from_reference('ref_image.jpg', 'training_data/')")
    print("2. report = extractor.validate_training_data('training_data/')")
    print("3. extractor.visualize_extracted_samples('training_data/')")
    print()

    print("🎯 Conservative Augmentation Strategy:")
    print("   • Rotation: ±3° (vs your current ±15°)")
    print("   • Brightness: ±5% (vs your current ±20%)")
    print("   • Contrast: ±2% (vs your current ±20%)")
    print("   • Noise: std ≤ 3 (vs your current ≤ 15)")
    print("   • Goal: Preserve shiny region characteristics")

    return extractor


if __name__ == "__main__":
    extractor = main()