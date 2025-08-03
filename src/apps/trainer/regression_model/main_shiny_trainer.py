#!/usr/bin/env python3
"""
Main Shiny Region Trainer

This is the main file that runs everything. Much simpler approach:
1. You manually crop your 10-sample image into individual files
2. This script augments them and trains the model
3. Tests the results

MANUAL CROPPING APPROACH (RECOMMENDED):
- Crop your reference image into 10 separate files
- Name them: 10_percent.jpg, 20_percent.jpg, ..., 100_percent.jpg
- Put them in a folder called "manual_samples/"
- Run this script
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

# Import our new modules
from improved_fracture_detector import ImprovedFractureSurfaceDetector
from shiny_region_analyzer import ShinyRegionAnalyzer, ShinyRegionFeatures
from shiny_region_classifier import ShinyRegionBasedClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MainShinyTrainer:
    """Main class that orchestrates the entire training process."""

    def __init__(self):
        self.classifier = ShinyRegionBasedClassifier()
        logger.info("Main Shiny Trainer initialized")

    def create_training_data_from_manual_crops(self,
                                               manual_samples_dir: str,
                                               output_dir: str,
                                               samples_per_class: int = 20) -> bool:
        """
        Create training dataset from manually cropped samples.

        Expected structure in manual_samples_dir:
        - 10_percent.jpg (or .png)
        - 20_percent.jpg
        - 30_percent.jpg
        - ...
        - 100_percent.jpg

        Args:
            manual_samples_dir: Directory with your 10 manually cropped images
            output_dir: Where to save the augmented training data
            samples_per_class: How many total samples to generate per class
        """

        manual_dir = Path(manual_samples_dir)
        output_path = Path(output_dir)

        if not manual_dir.exists():
            logger.error(f"Manual samples directory not found: {manual_samples_dir}")
            logger.info("Please create this directory and put your 10 cropped images in it:")
            logger.info("  10_percent.jpg, 20_percent.jpg, 30_percent.jpg, ..., 100_percent.jpg")
            return False

        # Find all manual sample files
        shear_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        found_samples = {}

        for shear in shear_percentages:
            # Look for files with this shear percentage
            possible_names = [
                f"{shear}_percent.jpg",
                f"{shear}_percent.png",
                f"{shear}.jpg",
                f"{shear}.png",
                f"shear_{shear}.jpg",
                f"shear_{shear}.png"
            ]

            found = False
            for name in possible_names:
                file_path = manual_dir / name
                if file_path.exists():
                    found_samples[shear] = str(file_path)
                    found = True
                    break

            if not found:
                logger.warning(f"Could not find sample for {shear}% shear")
                logger.info(f"Looking for: {possible_names}")

        if len(found_samples) == 0:
            logger.error("No manual samples found!")
            return False

        logger.info(f"Found {len(found_samples)} manual samples: {list(found_samples.keys())}")

        # Create output directory structure
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each manual sample
        total_generated = 0

        for shear_percent, sample_path in found_samples.items():
            logger.info(f"Processing {shear_percent}% shear sample...")

            # Load original image
            original = cv2.imread(sample_path)
            if original is None:
                logger.error(f"Could not load {sample_path}")
                continue

            # Create directory for this class
            class_dir = output_path / f"{shear_percent}_percent"
            class_dir.mkdir(exist_ok=True)

            # Save original
            cv2.imwrite(str(class_dir / "original.jpg"), original)
            generated_count = 1

            # Generate augmented samples
            augmentations_needed = samples_per_class - 1  # -1 for original

            for aug_idx in range(augmentations_needed):
                try:
                    augmented = self._conservative_augment(original)
                    aug_filename = f"aug_{aug_idx:03d}.jpg"
                    cv2.imwrite(str(class_dir / aug_filename), augmented)
                    generated_count += 1

                except Exception as e:
                    logger.warning(f"Failed to generate augmentation {aug_idx}: {e}")

            logger.info(f"  Generated {generated_count} samples for {shear_percent}% shear")
            total_generated += generated_count

        logger.info(f"✅ Successfully generated {total_generated} training samples in {output_dir}")
        return True

    def _conservative_augment(self, image: np.ndarray) -> np.ndarray:
        """Conservative augmentation that preserves shiny region characteristics."""

        result = image.copy()

        # 1. Small rotation (±4 degrees max)
        angle = np.random.uniform(-4, 4)
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 2. Minimal brightness (±7%)
        brightness = np.random.uniform(0.93, 1.07)
        result = np.clip(result.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # 3. Small contrast adjustment (±3%)
        contrast = np.random.uniform(0.97, 1.03)
        mean_val = np.mean(result)
        result = np.clip((result - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

        # 4. Very low noise
        if np.random.random() < 0.4:  # Only 40% of samples get noise
            noise_std = np.random.uniform(1, 4)
            noise = np.random.normal(0, noise_std, result.shape)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Small random crop (95-100%)
        if np.random.random() < 0.3:  # Only 30% get cropped
            crop_factor = np.random.uniform(0.95, 1.0)
            result = self._random_crop_resize(result, crop_factor)

        return result

    def _random_crop_resize(self, image: np.ndarray, crop_factor: float) -> np.ndarray:
        """Random crop and resize."""
        h, w = image.shape[:2]
        crop_h = int(h * crop_factor)
        crop_w = int(w * crop_factor)

        start_h = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        start_w = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0

        cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

        return resized

    def load_training_data(self, training_dir: str) -> Dict[float, List[str]]:
        """Load training data from directory structure."""

        training_path = Path(training_dir)
        training_data = {}

        if not training_path.exists():
            logger.error(f"Training directory not found: {training_dir}")
            return {}

        for class_dir in training_path.iterdir():
            if class_dir.is_dir() and "_percent" in class_dir.name:
                try:
                    # Extract shear percentage from directory name
                    shear_str = class_dir.name.replace("_percent", "")
                    shear_percent = float(shear_str)

                    # Get all image files
                    image_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        image_files.extend(list(class_dir.glob(ext)))

                    if image_files:
                        training_data[shear_percent] = [str(f) for f in image_files]
                        logger.info(f"Loaded {len(image_files)} samples for {shear_percent}% shear")

                except ValueError:
                    logger.warning(f"Could not parse shear percentage from: {class_dir.name}")

        return training_data

    def train_and_evaluate(self, training_data: Dict[float, List[str]]) -> Dict:
        """Train the model and evaluate its performance."""

        logger.info("Creating feature dataset...")
        X, y = self.classifier.create_training_dataset(training_data)

        logger.info("Training shiny region-based model...")
        results = self.classifier.train_model(X, y)

        # Print results
        print("\n🎉 TRAINING RESULTS:")
        print("=" * 40)
        print(f"Training MAE: {results['train_mae']:.2f}%")
        print(f"Training R²: {results['train_r2']:.3f}")
        print(f"Cross-validation MAE: {results['cv_mae_mean']:.2f} ± {results['cv_mae_std']:.2f}%")
        print(f"Number of samples: {results['n_samples']}")
        print(f"Number of features: {results['n_features']}")

        print(f"\n🔍 TOP 10 IMPORTANT FEATURES:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
            print(f"   {i + 1:2d}. {feature:<25}: {importance:.4f}")

        return results

    def test_on_samples(self, test_images_dir: str, max_samples: int = 5):
        """Test the trained model on some sample images."""

        if not self.classifier.is_trained:
            logger.error("Model must be trained before testing")
            return

        test_path = Path(test_images_dir)
        if not test_path.exists():
            logger.warning(f"Test directory not found: {test_images_dir}")
            return

        # Find test images
        test_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_files.extend(list(test_path.glob(ext)))

        test_files = test_files[:max_samples]

        if not test_files:
            logger.warning(f"No test images found in {test_images_dir}")
            return

        print(f"\n🧪 TESTING ON {len(test_files)} SAMPLE IMAGES:")
        print("=" * 50)

        for test_file in test_files:
            image = cv2.imread(str(test_file))
            if image is None:
                continue

            result = self.classifier.predict_shear_percentage(image)

            if result['success']:
                print(f"\n📷 {test_file.name}:")
                print(f"   Prediction: {result['prediction']:.1f}% shear")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Category: {result['prediction_category']}")

                # Show key features
                features = result['features']
                print(f"   Key metrics:")
                print(f"     Shiny area ratio: {features['shiny_area_ratio']:.3f}")
                print(f"     Largest shiny region: {features['largest_shiny_ratio']:.3f}")
                print(f"     Shiny region count: {features['shiny_region_count']:.0f}")
            else:
                print(f"\n❌ {test_file.name}: {result['error']}")

    def save_model(self, model_path: str = "shiny_region_model.pkl"):
        """Save the trained model."""
        if self.classifier.save_model(model_path):
            logger.info(f"✅ Model saved to {model_path}")
        else:
            logger.error("❌ Failed to save model")


def main():
    """Main function - this is what you run!"""

    print("🔬 SHINY REGION-BASED CHARPY FRACTURE SURFACE TRAINER")
    print("=" * 60)
    print("This script trains a model based on shiny region characteristics")
    print()

    trainer = MainShinyTrainer()

    print("📋 SETUP INSTRUCTIONS:")
    print("1. Manually crop your 10-sample reference image into individual files")
    print("2. Name them: 10_percent.jpg, 20_percent.jpg, ..., 100_percent.jpg")
    print("3. Put them in a folder called 'manual_samples/'")
    print("4. Run this script")
    print()

    # Check if manual samples exist
    manual_dir = "../../src/postprocessing/manual_samples"
    if not Path(manual_dir).exists():
        print(f"❌ Please create '{manual_dir}/' directory with your cropped samples first!")
        print()
        print("Expected files:")
        for shear in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            print(f"   {manual_dir}/{shear}_percent.jpg")
        return

    # Step 1: Create augmented training data
    print("🔄 Step 1: Creating augmented training data...")
    success = trainer.create_training_data_from_manual_crops(
        manual_samples_dir=manual_dir,
        output_dir="../../../database/samples/shiny_training_data",
        samples_per_class=25  # 1 original + 24 augmented = 25 per class
    )

    if not success:
        print("❌ Failed to create training data")
        return

    # Step 2: Load training data
    print("\n🔄 Step 2: Loading training data...")
    training_data = trainer.load_training_data("shiny_training_data")

    if not training_data:
        print("❌ No training data found")
        return

    # Step 3: Train model
    print("\n🔄 Step 3: Training model...")
    results = trainer.train_and_evaluate(training_data)

    # Step 4: Save model
    print("\n🔄 Step 4: Saving model...")
    trainer.save_model("charpy_shear_regressor.pkl")

    # Step 5: Test on any available samples
    print("\n🔄 Step 5: Testing model...")
    trainer.test_on_samples("manual_samples")  # Test on original manual samples

    print("\n✅ TRAINING COMPLETE!")
    print(f"Your new model should have MUCH better accuracy than {results['train_mae']:.1f}% MAE")
    print("Key features should be related to shiny region characteristics")


if __name__ == "__main__":
    main()