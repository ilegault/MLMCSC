#!/usr/bin/env python3
"""
Charpy Reference Image Processor

This tool processes the reference screenshot showing 10 shear percentage examples
and automatically creates a properly organized training dataset with augmented variations.

Features:
- Automatic detection and cropping of 10 reference images
- Creates proper directory structure for training
- Generates 10 augmented variations per reference image
- Handles both grid layouts and any arrangement
- Quality validation for extracted images
- Progress tracking and error handling

Usage:
    python charpy_image_processor.py reference_screenshot.jpg output_directory/
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharpyReferenceProcessor:
    """Processes reference screenshot into training dataset."""

    def __init__(self, output_dir: str = "charpy_training_data"):
        """
        Initialize the processor.

        Args:
            output_dir: Directory to save processed training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Expected shear percentages (modify if your screenshot is different)
        self.shear_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Augmentation parameters
        self.augmentations_per_image = 10
        self.rotation_range = (-15, 15)  # degrees
        self.brightness_range = (0.7, 1.3)
        self.contrast_range = (0.8, 1.2)
        self.noise_std_range = (0, 15)
        self.crop_scale_range = (0.85, 1.0)

        # Quality thresholds
        self.min_image_size = (50, 50)
        self.min_std_threshold = 10  # Minimum standard deviation for texture

        logger.info(f"CharpyReferenceProcessor initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Expected {len(self.shear_percentages)} shear percentage classes")

    def process_reference_screenshot(self, screenshot_path: str,
                                     layout: str = "auto",
                                     custom_percentages: Optional[List[float]] = None) -> bool:
        """
        Process the reference screenshot and create training dataset.

        Args:
            screenshot_path: Path to the reference screenshot
            layout: Layout type ("auto", "grid_2x5", "grid_5x2", "manual")
            custom_percentages: Custom shear percentages if different from default

        Returns:
            True if processing successful
        """
        logger.info(f"Processing reference screenshot: {screenshot_path}")

        # Load screenshot
        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            logger.error(f"Could not load screenshot: {screenshot_path}")
            return False

        # Use custom percentages if provided
        if custom_percentages:
            self.shear_percentages = custom_percentages

        logger.info(f"Screenshot loaded: {screenshot.shape}")

        # Extract individual specimen images
        specimen_images = self._extract_specimen_images(screenshot, layout)

        if len(specimen_images) != len(self.shear_percentages):
            logger.warning(f"Expected {len(self.shear_percentages)} images, found {len(specimen_images)}")
            logger.warning("You may need to manually adjust extraction or verify the layout")

        # Process each specimen image
        total_created = 0
        for i, (specimen_img, shear_percent) in enumerate(zip(specimen_images, self.shear_percentages)):
            logger.info(f"Processing {shear_percent}% shear specimen ({i + 1}/{len(specimen_images)})")

            # Create directory for this shear percentage
            if shear_percent == int(shear_percent):
                # Integer percentage
                dir_name = f"{int(shear_percent)}_percent"
            else:
                # Decimal percentage (e.g., 12.5% -> 12p5_percent)
                dir_name = f"{str(shear_percent).replace('.', 'p')}_percent"

            shear_dir = self.output_dir / dir_name
            shear_dir.mkdir(exist_ok=True)

            # Save original image
            original_path = shear_dir / f"original_{shear_percent}pct.jpg"
            cv2.imwrite(str(original_path), specimen_img)

            # Generate augmented versions
            augmented_images = self._create_augmented_versions(specimen_img)

            # Save augmented images
            for j, aug_img in enumerate(augmented_images):
                aug_path = shear_dir / f"aug_{j:02d}_{shear_percent}pct.jpg"
                cv2.imwrite(str(aug_path), aug_img)
                total_created += 1

            logger.info(f"Created {len(augmented_images)} augmented images for {shear_percent}% shear")

        # Create dataset summary
        self._create_dataset_summary(total_created)

        logger.info(f"✅ Processing completed! Created {total_created} training images")
        logger.info(f"📁 Training data saved to: {self.output_dir}")

        return True

    def _extract_specimen_images(self, screenshot: np.ndarray, layout: str) -> List[np.ndarray]:
        """Extract individual specimen images from screenshot."""
        h, w = screenshot.shape[:2]

        if layout == "auto":
            # Try to automatically detect the layout
            return self._auto_detect_specimens(screenshot)
        elif layout == "grid_2x5":
            # 2 rows, 5 columns
            return self._extract_grid_layout(screenshot, rows=2, cols=5)
        elif layout == "grid_5x2":
            # 5 rows, 2 columns
            return self._extract_grid_layout(screenshot, rows=5, cols=2)
        elif layout == "manual":
            # Manual selection with GUI
            return self._manual_selection(screenshot)
        else:
            logger.error(f"Unknown layout: {layout}")
            return []

    def _auto_detect_specimens(self, screenshot: np.ndarray) -> List[np.ndarray]:
        """Automatically detect specimen regions in screenshot."""
        logger.info("Attempting automatic specimen detection...")

        # Convert to grayscale for processing
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to find specimen regions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and aspect ratio
        min_area = (screenshot.shape[0] * screenshot.shape[1]) * 0.005  # At least 0.5% of image
        max_area = (screenshot.shape[0] * screenshot.shape[1]) * 0.3  # At most 30% of image

        specimen_rects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                    specimen_rects.append((x, y, w, h))

        # Sort rectangles by position (top-to-bottom, left-to-right)
        specimen_rects.sort(key=lambda rect: (rect[1], rect[0]))

        # Extract images from rectangles
        specimen_images = []
        for x, y, w, h in specimen_rects:
            # Add some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(screenshot.shape[1], x + w + padding)
            y2 = min(screenshot.shape[0], y + h + padding)

            specimen_img = screenshot[y1:y2, x1:x2]
            if self._validate_specimen_image(specimen_img):
                specimen_images.append(specimen_img)

        logger.info(f"Auto-detected {len(specimen_images)} specimen regions")

        # If we didn't find enough, try grid extraction as fallback
        if len(specimen_images) < len(self.shear_percentages):
            logger.warning("Auto-detection found insufficient specimens, trying grid layout...")
            return self._extract_grid_layout(screenshot, rows=2, cols=5)

        return specimen_images[:len(self.shear_percentages)]  # Take only what we need

    def _extract_grid_layout(self, screenshot: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
        """Extract specimens assuming a regular grid layout."""
        logger.info(f"Extracting grid layout: {rows}x{cols}")

        h, w = screenshot.shape[:2]

        # Calculate cell dimensions
        cell_h = h // rows
        cell_w = w // cols

        # Add margins to avoid borders/text
        margin_h = cell_h * 0.1  # 10% margin
        margin_w = cell_w * 0.1  # 10% margin

        specimen_images = []

        for row in range(rows):
            for col in range(cols):
                if len(specimen_images) >= len(self.shear_percentages):
                    break

                # Calculate cell boundaries
                y1 = int(row * cell_h + margin_h)
                y2 = int((row + 1) * cell_h - margin_h)
                x1 = int(col * cell_w + margin_w)
                x2 = int((col + 1) * cell_w - margin_w)

                # Extract cell image
                cell_img = screenshot[y1:y2, x1:x2]

                if self._validate_specimen_image(cell_img):
                    specimen_images.append(cell_img)
                else:
                    logger.warning(f"Cell ({row},{col}) failed validation")

        logger.info(f"Extracted {len(specimen_images)} specimens from grid")
        return specimen_images

    def _manual_selection(self, screenshot: np.ndarray) -> List[np.ndarray]:
        """Manual selection of specimen regions using mouse clicks."""
        logger.info("Manual selection mode - click on specimen centers")

        specimen_images = []
        specimen_centers = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                specimen_centers.append((x, y))
                print(f"Selected point {len(specimen_centers)}: ({x}, {y})")

                # Draw marker
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_img, str(len(specimen_centers)),
                            (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Manual Selection', display_img)

        # Setup window
        display_img = screenshot.copy()
        cv2.namedWindow('Manual Selection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Manual Selection', mouse_callback)

        # Instructions
        print(f"Click on the center of each specimen ({len(self.shear_percentages)} total)")
        print("Press SPACE when done, ESC to cancel")

        cv2.imshow('Manual Selection', display_img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or len(specimen_centers) >= len(self.shear_percentages):
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return []

        cv2.destroyAllWindows()

        # Extract regions around selected points
        for i, (cx, cy) in enumerate(specimen_centers):
            # Estimate specimen size (you may need to adjust this)
            region_size = 200  # pixels

            x1 = max(0, cx - region_size // 2)
            y1 = max(0, cy - region_size // 2)
            x2 = min(screenshot.shape[1], cx + region_size // 2)
            y2 = min(screenshot.shape[0], cy + region_size // 2)

            specimen_img = screenshot[y1:y2, x1:x2]

            if self._validate_specimen_image(specimen_img):
                specimen_images.append(specimen_img)

        logger.info(f"Manually selected {len(specimen_images)} specimens")
        return specimen_images

    def _validate_specimen_image(self, image: np.ndarray) -> bool:
        """Validate that an extracted image looks like a specimen."""
        if image.size == 0:
            return False

        h, w = image.shape[:2]
        if h < self.min_image_size[0] or w < self.min_image_size[1]:
            return False

        # Check for sufficient texture/detail
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        std_dev = np.std(gray)

        if std_dev < self.min_std_threshold:
            return False

        return True

    def _create_augmented_versions(self, original_image: np.ndarray) -> List[np.ndarray]:
        """Create augmented versions of the original specimen image."""
        augmented_images = [original_image.copy()]  # Include original

        for i in range(self.augmentations_per_image - 1):  # -1 because we include original
            aug_img = original_image.copy()

            # Apply random transformations
            aug_img = self._apply_rotation(aug_img)
            aug_img = self._apply_brightness_contrast(aug_img)
            aug_img = self._apply_noise(aug_img)
            aug_img = self._apply_random_crop(aug_img)

            augmented_images.append(aug_img)

        return augmented_images

    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation within specified range."""
        angle = random.uniform(*self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)

        return rotated

    def _apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast adjustments."""
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)

        # Apply brightness
        bright_img = image.astype(np.float32) * brightness_factor

        # Apply contrast around mean
        mean_val = np.mean(bright_img)
        contrast_img = mean_val + contrast_factor * (bright_img - mean_val)

        # Clip to valid range
        result = np.clip(contrast_img, 0, 255).astype(np.uint8)

        return result

    def _apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply random Gaussian noise."""
        noise_std = random.uniform(*self.noise_std_range)

        if noise_std > 0:
            noise = np.random.normal(0, noise_std, image.shape)
            noisy_img = image.astype(np.float32) + noise
            return np.clip(noisy_img, 0, 255).astype(np.uint8)

        return image

    def _apply_random_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply random crop and resize back to original size."""
        scale = random.uniform(*self.crop_scale_range)

        if scale < 1.0:
            h, w = image.shape[:2]
            new_h = int(h * scale)
            new_w = int(w * scale)

            # Random crop position
            start_h = random.randint(0, h - new_h)
            start_w = random.randint(0, w - new_w)

            cropped = image[start_h:start_h + new_h, start_w:start_w + new_w]

            # Resize back to original size
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

            return resized

        return image

    def _create_dataset_summary(self, total_images: int) -> None:
        """Create a summary of the generated dataset."""
        summary = {
            'created_at': datetime.now().isoformat(),
            'total_classes': len(self.shear_percentages),
            'shear_percentages': self.shear_percentages,
            'augmentations_per_image': self.augmentations_per_image,
            'total_images_created': total_images,
            'output_directory': str(self.output_dir),
            'directory_structure': {}
        }

        # Document directory structure
        for shear_percent in self.shear_percentages:
            if shear_percent == int(shear_percent):
                dir_name = f"{int(shear_percent)}_percent"
            else:
                dir_name = f"{str(shear_percent).replace('.', 'p')}_percent"

            shear_dir = self.output_dir / dir_name
            if shear_dir.exists():
                image_count = len(list(shear_dir.glob("*.jpg")))
                summary['directory_structure'][dir_name] = {
                    'shear_percentage': shear_percent,
                    'image_count': image_count
                }

        # Save summary
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset summary saved: {summary_path}")

        # Print summary
        print(f"\n📊 DATASET SUMMARY")
        print(f"=" * 50)
        print(f"Total classes: {len(self.shear_percentages)}")
        print(f"Total images: {total_images}")
        print(f"Images per class: ~{total_images // len(self.shear_percentages)}")
        print(f"Directory structure:")

        for dir_name, info in summary['directory_structure'].items():
            print(f"  📁 {dir_name}: {info['image_count']} images ({info['shear_percentage']}% shear)")

    def preview_extraction(self, screenshot_path: str, layout: str = "auto") -> bool:
        """Preview the specimen extraction without saving images."""
        logger.info("Previewing specimen extraction...")

        # Load screenshot
        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            logger.error(f"Could not load screenshot: {screenshot_path}")
            return False

        # Extract specimen images
        specimen_images = self._extract_specimen_images(screenshot, layout)

        if not specimen_images:
            logger.error("No specimens detected")
            return False

        # Create preview
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, (specimen_img, shear_percent) in enumerate(zip(specimen_images, self.shear_percentages)):
            if i < len(axes):
                # Convert BGR to RGB for matplotlib
                rgb_img = cv2.cvtColor(specimen_img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(rgb_img)
                axes[i].set_title(f"{shear_percent}% Shear")
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(specimen_images), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Process Charpy reference screenshot into training dataset")
    parser.add_argument("screenshot", help="Path to reference screenshot")
    parser.add_argument("-o", "--output", default="charpy_training_data",
                        help="Output directory for training data")
    parser.add_argument("-l", "--layout", choices=["auto", "grid_2x5", "grid_5x2", "manual"],
                        default="auto", help="Layout detection method")
    parser.add_argument("-p", "--preview", action="store_true",
                        help="Preview extraction without saving")
    parser.add_argument("--percentages", nargs="+", type=float,
                        help="Custom shear percentages (e.g., --percentages 0 10 20 30 40 50 60 70 80 90 100)")
    parser.add_argument("--augmentations", type=int, default=10,
                        help="Number of augmentations per image")

    args = parser.parse_args()

    print("🔬 CHARPY REFERENCE IMAGE PROCESSOR")
    print("=" * 50)
    print(f"Screenshot: {args.screenshot}")
    print(f"Output: {args.output}")
    print(f"Layout: {args.layout}")
    print(f"Augmentations: {args.augmentations}")

    if args.percentages:
        print(f"Custom percentages: {args.percentages}")

    # Initialize processor
    processor = CharpyReferenceProcessor(args.output)
    processor.augmentations_per_image = args.augmentations

    if args.preview:
        # Preview mode
        print("\n👀 PREVIEW MODE")
        print("-" * 20)
        success = processor.preview_extraction(args.screenshot, args.layout)
        if success:
            print("✅ Preview completed! Check the displayed images.")
            print("💡 If the extraction looks good, run without --preview to create the dataset.")
        else:
            print("❌ Preview failed. Check the screenshot path and try a different layout.")
    else:
        # Full processing mode
        print("\n🚀 PROCESSING MODE")
        print("-" * 20)
        success = processor.process_reference_screenshot(
            args.screenshot,
            args.layout,
            args.percentages
        )

        if success:
            print("\n✅ SUCCESS!")
            print(f"📁 Training data created in: {args.output}")
            print("\n📝 Next steps:")
            print("1. Review the generated images for quality")
            print("2. Train your shear classification model:")
            print(f"   python charpy_shear_classifier.py")
            print(f"   # Select option 1 and use '{args.output}' as reference directory")
        else:
            print("\n❌ Processing failed. Try a different layout or manual selection.")


if __name__ == "__main__":
    main()