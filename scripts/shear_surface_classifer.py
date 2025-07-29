#!/usr/bin/env python3
"""
Charpy Fracture Surface Shear Classification System

This system classifies fracture surfaces into shear percentages (0-100% in 10% increments)
using traditional computer vision features with minimal training data.

Features:
- Surface detection and extraction
- Multi-scale texture analysis
- GLCM (Gray-Level Co-occurrence Matrix) features
- LBP (Local Binary Pattern) features
- Morphological analysis
- Data augmentation for small datasets
- Random Forest classification
- Real-time classification with microscope integration
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Image processing
from skimage import feature, measure, morphology, segmentation, filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, closing, opening
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage, stats
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ShearFeatures:
    """Data class for extracted fracture surface features."""
    # GLCM Features
    contrast: float
    dissimilarity: float
    homogeneity: float
    energy: float
    correlation: float
    asm: float  # Angular Second Moment

    # LBP Features
    lbp_uniformity: float
    lbp_contrast: float
    lbp_dissimilarity: float

    # Morphological Features
    roughness: float
    surface_area_ratio: float
    fractal_dimension: float

    # Intensity Features
    mean_intensity: float
    std_intensity: float
    skewness: float
    kurtosis: float

    # Gradient Features
    gradient_magnitude: float
    gradient_direction_std: float

    # Edge Features
    edge_density: float
    edge_strength: float

    # Regional Features
    smooth_regions_ratio: float
    rough_regions_ratio: float

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array."""
        return np.array([
            self.contrast, self.dissimilarity, self.homogeneity,
            self.energy, self.correlation, self.asm,
            self.lbp_uniformity, self.lbp_contrast, self.lbp_dissimilarity,
            self.roughness, self.surface_area_ratio, self.fractal_dimension,
            self.mean_intensity, self.std_intensity, self.skewness, self.kurtosis,
            self.gradient_magnitude, self.gradient_direction_std,
            self.edge_density, self.edge_strength,
            self.smooth_regions_ratio, self.rough_regions_ratio
        ])

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names."""
        return [
            'contrast', 'dissimilarity', 'homogeneity',
            'energy', 'correlation', 'asm',
            'lbp_uniformity', 'lbp_contrast', 'lbp_dissimilarity',
            'roughness', 'surface_area_ratio', 'fractal_dimension',
            'mean_intensity', 'std_intensity', 'skewness', 'kurtosis',
            'gradient_magnitude', 'gradient_direction_std',
            'edge_density', 'edge_strength',
            'smooth_regions_ratio', 'rough_regions_ratio'
        ]


class FractureSurfaceDetector:
    """Detects and extracts fracture surfaces from Charpy specimen images."""

    def __init__(self):
        self.debug = False

    def detect_fracture_surface(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract the fracture surface from a Charpy specimen image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Extracted fracture surface region or None if not found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Method 1: Try to detect specimen boundaries first
        surface = self._detect_by_specimen_shape(gray)
        if surface is not None:
            return surface

        # Method 2: Use texture-based detection
        surface = self._detect_by_texture(gray)
        if surface is not None:
            return surface

        # Method 3: Use edge-based detection
        surface = self._detect_by_edges(gray)
        if surface is not None:
            return surface

        # Method 4: Use intensity-based detection
        return self._detect_by_intensity(gray)

    def _detect_by_specimen_shape(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface by identifying specimen shape."""
        # Apply Gaussian blur to reduce noise
        blurred = gaussian(gray, sigma=2)

        # Use Otsu thresholding to separate specimen from background
        threshold = threshold_otsu(blurred)
        binary = blurred > threshold

        # Remove small objects and fill holes
        binary = remove_small_objects(binary, min_size=1000)
        binary = ndimage.binary_fill_holes(binary)

        # Find the largest connected component (specimen)
        labeled = label(binary)
        regions = regionprops(labeled)

        if not regions:
            return None

        # Get the largest region
        largest_region = max(regions, key=lambda r: r.area)

        # Extract bounding box with some padding
        minr, minc, maxr, maxc = largest_region.bbox

        # The fracture surface is typically in the middle-upper area
        # Adjust these ratios based on your specimen orientation
        surface_minr = minr + int(0.1 * (maxr - minr))  # Start 10% from top
        surface_maxr = minr + int(0.6 * (maxr - minr))  # End 60% from top
        surface_minc = minc + int(0.2 * (maxc - minc))  # Start 20% from left
        surface_maxc = maxc - int(0.2 * (maxc - minc))  # End 20% from right

        # Extract fracture surface region
        surface = gray[surface_minr:surface_maxr, surface_minc:surface_maxc]

        # Validate the extracted surface
        if surface.size > 0 and self._validate_surface(surface):
            return surface

        return None

    def _detect_by_texture(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface based on texture analysis."""
        # Calculate local standard deviation (texture measure)
        kernel = np.ones((15, 15))
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel / kernel.size)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel / kernel.size)
        texture_map = np.sqrt(local_var)

        # Find regions with high texture (rough fracture surface)
        texture_threshold = np.percentile(texture_map, 75)  # Top 25% texture
        rough_regions = texture_map > texture_threshold

        # Clean up the regions
        rough_regions = morphology.opening(rough_regions, morphology.disk(5))
        rough_regions = morphology.closing(rough_regions, morphology.disk(10))

        # Find the largest rough region
        labeled = label(rough_regions)
        regions = regionprops(labeled)

        if not regions:
            return None

        largest_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest_region.bbox

        # Extract with some padding
        pad = 20
        minr = max(0, minr - pad)
        minc = max(0, minc - pad)
        maxr = min(gray.shape[0], maxr + pad)
        maxc = min(gray.shape[1], maxc + pad)

        surface = gray[minr:maxr, minc:maxc]

        if surface.size > 0 and self._validate_surface(surface):
            return surface

        return None

    def _detect_by_edges(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface based on edge density."""
        # Apply Canny edge detection
        edges = feature.canny(gray, sigma=2, low_threshold=0.1, high_threshold=0.2)

        # Calculate local edge density
        kernel_size = 31
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)

        # Find regions with high edge density
        edge_threshold = np.percentile(edge_density, 80)  # Top 20% edge density
        high_edge_regions = edge_density > edge_threshold

        # Clean up
        high_edge_regions = morphology.opening(high_edge_regions, morphology.disk(3))
        high_edge_regions = morphology.closing(high_edge_regions, morphology.disk(7))

        # Find largest region
        labeled = label(high_edge_regions)
        regions = regionprops(labeled)

        if not regions:
            return None

        largest_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest_region.bbox

        # Extract with padding
        pad = 15
        minr = max(0, minr - pad)
        minc = max(0, minc - pad)
        maxr = min(gray.shape[0], maxr + pad)
        maxc = min(gray.shape[1], maxc + pad)

        surface = gray[minr:maxr, minc:maxc]

        if surface.size > 0 and self._validate_surface(surface):
            return surface

        return None

    def _detect_by_intensity(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface based on intensity patterns."""
        # Fracture surfaces often have specific intensity characteristics
        # Apply histogram equalization to enhance contrast
        equalized = cv2.equalizeHist(gray)

        # Use adaptive thresholding to find regions
        adaptive_thresh = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 2
        )

        # Find regions with specific characteristics
        # Fracture surfaces typically have intermediate intensity values
        mask = (gray > np.percentile(gray, 20)) & (gray < np.percentile(gray, 80))

        # Combine with adaptive threshold
        combined = mask & (adaptive_thresh > 0)

        # Clean up
        combined = morphology.opening(combined, morphology.disk(3))
        combined = morphology.closing(combined, morphology.disk(10))

        # Find largest region
        labeled = label(combined)
        regions = regionprops(labeled)

        if not regions:
            # Fallback: return center portion of image
            h, w = gray.shape
            return gray[h // 4:3 * h // 4, w // 4:3 * w // 4]

        largest_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest_region.bbox

        surface = gray[minr:maxr, minc:maxc]

        if surface.size > 0:
            return surface

        return None

    def _validate_surface(self, surface: np.ndarray) -> bool:
        """Validate that the extracted region looks like a fracture surface."""
        if surface.size < 100:  # Too small
            return False

        h, w = surface.shape
        if h < 20 or w < 20:  # Too small dimensions
            return False

        # Check for reasonable texture (not too uniform)
        std_dev = np.std(surface)
        if std_dev < 10:  # Too uniform
            return False

        # Check for reasonable intensity range
        intensity_range = np.max(surface) - np.min(surface)
        if intensity_range < 50:  # Too narrow range
            return False

        return True


class ShearFeatureExtractor:
    """Extracts comprehensive features from fracture surfaces for shear classification."""

    def __init__(self):
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, 45, 90, 135]
        self.lbp_radius = 3
        self.lbp_n_points = 24

    def extract_features(self, surface: np.ndarray) -> ShearFeatures:
        """
        Extract all features from a fracture surface.

        Args:
            surface: Grayscale fracture surface image

        Returns:
            ShearFeatures object containing all extracted features
        """
        # Normalize surface for consistent feature extraction
        surface_norm = cv2.equalizeHist(surface.astype(np.uint8))

        # Extract different types of features
        glcm_features = self._extract_glcm_features(surface_norm)
        lbp_features = self._extract_lbp_features(surface_norm)
        morph_features = self._extract_morphological_features(surface_norm)
        intensity_features = self._extract_intensity_features(surface_norm)
        gradient_features = self._extract_gradient_features(surface_norm)
        edge_features = self._extract_edge_features(surface_norm)
        regional_features = self._extract_regional_features(surface_norm)

        return ShearFeatures(
            # GLCM features
            contrast=glcm_features['contrast'],
            dissimilarity=glcm_features['dissimilarity'],
            homogeneity=glcm_features['homogeneity'],
            energy=glcm_features['energy'],
            correlation=glcm_features['correlation'],
            asm=glcm_features['asm'],

            # LBP features
            lbp_uniformity=lbp_features['uniformity'],
            lbp_contrast=lbp_features['contrast'],
            lbp_dissimilarity=lbp_features['dissimilarity'],

            # Morphological features
            roughness=morph_features['roughness'],
            surface_area_ratio=morph_features['surface_area_ratio'],
            fractal_dimension=morph_features['fractal_dimension'],

            # Intensity features
            mean_intensity=intensity_features['mean'],
            std_intensity=intensity_features['std'],
            skewness=intensity_features['skewness'],
            kurtosis=intensity_features['kurtosis'],

            # Gradient features
            gradient_magnitude=gradient_features['magnitude'],
            gradient_direction_std=gradient_features['direction_std'],

            # Edge features
            edge_density=edge_features['density'],
            edge_strength=edge_features['strength'],

            # Regional features
            smooth_regions_ratio=regional_features['smooth_ratio'],
            rough_regions_ratio=regional_features['rough_ratio']
        )

    def _extract_glcm_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix features."""
        # Reduce gray levels for GLCM computation
        surface_glcm = (surface // 32).astype(np.uint8)  # 8 gray levels

        # Compute GLCM for multiple distances and angles
        glcms = graycomatrix(
            surface_glcm,
            distances=self.glcm_distances,
            angles=np.radians(self.glcm_angles),
            levels=8,
            symmetric=True,
            normed=True
        )

        # Extract properties and average over distances and angles
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = {}

        for prop in properties:
            values = graycoprops(glcms, prop)
            features[prop.lower()] = np.mean(values)

        # Rename ASM to asm for consistency
        features['asm'] = features.pop('asm')

        return features

    def _extract_lbp_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract Local Binary Pattern features."""
        # Compute LBP
        lbp = local_binary_pattern(
            surface,
            self.lbp_n_points,
            self.lbp_radius,
            method='uniform'
        )

        # Calculate LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_n_points + 2,
                               range=(0, self.lbp_n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize

        # Extract features from histogram
        uniformity = np.sum(hist ** 2)  # Energy/Uniformity

        # Calculate contrast and dissimilarity from LBP
        contrast = 0
        dissimilarity = 0
        for i in range(len(hist)):
            for j in range(len(hist)):
                contrast += hist[i] * hist[j] * (i - j) ** 2
                dissimilarity += hist[i] * hist[j] * abs(i - j)

        return {
            'uniformity': uniformity,
            'contrast': contrast,
            'dissimilarity': dissimilarity
        }

    def _extract_morphological_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract morphological and structural features."""
        # Calculate surface roughness using standard deviation of gradients
        grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        roughness = np.mean(gradient_magnitude)

        # Surface area ratio (3D surface area vs projected area)
        surface_area_3d = np.sum(np.sqrt(1 + grad_x ** 2 + grad_y ** 2))
        projected_area = surface.size
        surface_area_ratio = surface_area_3d / projected_area

        # Fractal dimension using box-counting method
        fractal_dimension = self._calculate_fractal_dimension(surface)

        return {
            'roughness': roughness,
            'surface_area_ratio': surface_area_ratio,
            'fractal_dimension': fractal_dimension
        }

    def _extract_intensity_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract intensity-based statistical features."""
        flat_surface = surface.flatten().astype(np.float32)

        return {
            'mean': np.mean(flat_surface),
            'std': np.std(flat_surface),
            'skewness': stats.skew(flat_surface),
            'kurtosis': stats.kurtosis(flat_surface)
        }

    def _extract_gradient_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract gradient-based features."""
        # Calculate gradients
        grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Gradient direction
        direction = np.arctan2(grad_y, grad_x)

        return {
            'magnitude': np.mean(magnitude),
            'direction_std': np.std(direction)
        }

    def _extract_edge_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract edge-based features."""
        # Apply Canny edge detection
        edges = feature.canny(surface, sigma=1.0)

        # Edge density
        edge_density = np.sum(edges) / edges.size

        # Edge strength (average gradient magnitude at edge pixels)
        grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        if np.sum(edges) > 0:
            edge_strength = np.mean(gradient_magnitude[edges])
        else:
            edge_strength = 0

        return {
            'density': edge_density,
            'strength': edge_strength
        }

    def _extract_regional_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract features based on smooth vs rough regions."""
        # Calculate local variance to identify smooth vs rough regions
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

        # Local mean
        local_mean = cv2.filter2D(surface.astype(np.float32), -1, kernel)

        # Local variance
        local_var = cv2.filter2D(
            (surface.astype(np.float32) - local_mean) ** 2, -1, kernel
        )

        # Threshold to separate smooth and rough regions
        var_threshold = np.percentile(local_var, 50)  # Median threshold

        smooth_regions = local_var < var_threshold
        rough_regions = local_var >= var_threshold

        smooth_ratio = np.sum(smooth_regions) / surface.size
        rough_ratio = np.sum(rough_regions) / surface.size

        return {
            'smooth_ratio': smooth_ratio,
            'rough_ratio': rough_ratio
        }

    def _calculate_fractal_dimension(self, surface: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        # Convert to binary image for box counting
        threshold = threshold_otsu(surface)
        binary = surface > threshold

        # Box counting
        sizes = np.logspace(0.5, 3, num=10, base=2).astype(int)
        counts = []

        for size in sizes:
            if size >= min(binary.shape):
                continue

            # Count boxes that contain at least one True pixel
            h, w = binary.shape
            count = 0

            for i in range(0, h, size):
                for j in range(0, w, size):
                    box = binary[i:i + size, j:j + size]
                    if box.size > 0 and np.any(box):
                        count += 1

            counts.append(count)

        # Fit line to log-log plot
        if len(counts) > 2:
            sizes_valid = sizes[:len(counts)]
            # Remove zeros to avoid log(0)
            valid_idx = np.array(counts) > 0
            if np.sum(valid_idx) > 2:
                log_sizes = np.log(sizes_valid[valid_idx])
                log_counts = np.log(np.array(counts)[valid_idx])

                # Linear regression
                coeffs = np.polyfit(log_sizes, log_counts, 1)
                fractal_dim = -coeffs[0]  # Negative slope

                return max(1.0, min(3.0, fractal_dim))  # Constrain to reasonable range

        return 2.0  # Default value


class DataAugmentor:
    """Augments training data for improved model performance with limited samples."""

    def __init__(self):
        self.rotation_angles = [-15, -10, -5, 5, 10, 15]
        self.brightness_factors = [0.8, 0.9, 1.1, 1.2]
        self.contrast_factors = [0.8, 0.9, 1.1, 1.2]
        self.noise_levels = [5, 10, 15]

    def augment_surface(self, surface: np.ndarray, num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Create augmented versions of a fracture surface.

        Args:
            surface: Original fracture surface
            num_augmentations: Number of augmented versions to create

        Returns:
            List of augmented surfaces including the original
        """
        augmented = [surface.copy()]  # Include original

        for _ in range(num_augmentations):
            augmented_surface = surface.copy()

            # Random rotation
            if np.random.random() < 0.6:
                angle = np.random.choice(self.rotation_angles)
                augmented_surface = self._rotate_surface(augmented_surface, angle)

            # Random brightness adjustment
            if np.random.random() < 0.4:
                factor = np.random.choice(self.brightness_factors)
                augmented_surface = self._adjust_brightness(augmented_surface, factor)

            # Random contrast adjustment
            if np.random.random() < 0.4:
                factor = np.random.choice(self.contrast_factors)
                augmented_surface = self._adjust_contrast(augmented_surface, factor)

            # Random noise
            if np.random.random() < 0.3:
                noise_level = np.random.choice(self.noise_levels)
                augmented_surface = self._add_noise(augmented_surface, noise_level)

            # Random crop and resize (if large enough)
            if np.random.random() < 0.3 and min(surface.shape) > 100:
                augmented_surface = self._random_crop_resize(augmented_surface)

            augmented.append(augmented_surface)

        return augmented

    def _rotate_surface(self, surface: np.ndarray, angle: float) -> np.ndarray:
        """Rotate surface by given angle."""
        h, w = surface.shape
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(surface, rotation_matrix, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)

        return rotated

    def _adjust_brightness(self, surface: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness by multiplying by factor."""
        adjusted = surface.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, surface: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast around mean value."""
        mean_val = np.mean(surface)
        adjusted = mean_val + factor * (surface.astype(np.float32) - mean_val)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _add_noise(self, surface: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to surface."""
        noise = np.random.normal(0, noise_level, surface.shape)
        noisy = surface.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _random_crop_resize(self, surface: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """Randomly crop and resize back to original size."""
        h, w = surface.shape
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)

        # Random crop position
        start_h = np.random.randint(0, h - new_h + 1)
        start_w = np.random.randint(0, w - new_w + 1)

        cropped = surface[start_h:start_h + new_h, start_w:start_w + new_w]

        # Resize back to original size
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return resized


class ShearClassifier:
    """Main classifier for Charpy fracture surface shear percentage."""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier.

        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'ensemble')
        """
        self.model_type = model_type
        self.detector = FractureSurfaceDetector()
        self.feature_extractor = ShearFeatureExtractor()
        self.augmentor = DataAugmentor()

        # Initialize models
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = ShearFeatures.get_feature_names()

        # Training data storage
        self.X_train = None
        self.y_train = None
        self.is_trained = False

        # Model performance
        self.training_accuracy = 0.0
        self.cross_val_scores = []

        logger.info(f"ShearClassifier initialized with {model_type} model")

    def _create_model(self) -> Pipeline:
        """Create machine learning model pipeline."""
        if self.model_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'svm':
            classifier = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42,
                class_weight='balanced',
                probability=True
            )
        else:  # ensemble
            from sklearn.ensemble import VotingClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            svm = SVC(kernel='rbf', C=10, probability=True, random_state=42, class_weight='balanced')
            classifier = VotingClassifier([('rf', rf), ('svm', svm)], voting='soft')

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])

        return pipeline

    def train_from_images(self, training_data: Dict[int, List[str]], augment: bool = True) -> Dict[str, Any]:
        """
        Train classifier from image files.

        Args:
            training_data: Dict mapping shear percentages to lists of image paths
                          e.g., {0: ['0_shear_1.jpg', '0_shear_2.jpg'],
                                10: ['10_shear_1.jpg'], ...}
            augment: Whether to apply data augmentation

        Returns:
            Training results dictionary
        """
        logger.info("Starting training from images...")

        # Prepare training data
        X_features = []
        y_labels = []

        total_images = sum(len(paths) for paths in training_data.values())
        processed = 0

        for shear_percent, image_paths in training_data.items():
            logger.info(f"Processing {len(image_paths)} images for {shear_percent}% shear...")

            for image_path in image_paths:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue

                # Detect fracture surface
                surface = self.detector.detect_fracture_surface(image)
                if surface is None:
                    logger.warning(f"Could not detect fracture surface in: {image_path}")
                    continue

                # Generate augmented versions if requested
                if augment:
                    surfaces = self.augmentor.augment_surface(surface, num_augmentations=8)
                else:
                    surfaces = [surface]

                # Extract features from all surfaces
                for surf in surfaces:
                    features = self.feature_extractor.extract_features(surf)
                    X_features.append(features.to_array())
                    y_labels.append(shear_percent)

                processed += 1
                if processed % 5 == 0:
                    logger.info(f"Processed {processed}/{total_images} images...")

        # Convert to numpy arrays
        self.X_train = np.array(X_features)
        self.y_train = np.array(y_labels)

        logger.info(f"Training data prepared: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        logger.info(f"Class distribution: {dict(zip(*np.unique(self.y_train, return_counts=True)))}")

        # Train model
        return self._train_model()

    def _train_model(self) -> Dict[str, Any]:
        """Train the machine learning model."""
        logger.info(f"Training {self.model_type} model...")

        # Create model
        self.model = self._create_model()

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        self.cross_val_scores = cv_scores

        # Train on full dataset
        self.model.fit(self.X_train, self.y_train)

        # Calculate training accuracy
        y_pred_train = self.model.predict(self.X_train)
        self.training_accuracy = accuracy_score(self.y_train, y_pred_train)

        self.is_trained = True

        results = {
            'training_accuracy': self.training_accuracy,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores.tolist(),
            'n_samples': self.X_train.shape[0],
            'n_features': self.X_train.shape[1],
            'class_distribution': dict(zip(*np.unique(self.y_train, return_counts=True)))
        }

        logger.info(f"Training completed!")
        logger.info(f"Training accuracy: {self.training_accuracy:.3f}")
        logger.info(f"Cross-validation: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

        return results

    def predict_shear_percentage(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict shear percentage from an image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Detect fracture surface
        surface = self.detector.detect_fracture_surface(image)
        if surface is None:
            return {
                'success': False,
                'error': 'Could not detect fracture surface',
                'prediction': None,
                'confidence': 0.0,
                'probabilities': {}
            }

        # Extract features
        features = self.feature_extractor.extract_features(surface)
        feature_vector = features.to_array().reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]

        # Get class labels
        classes = self.model.classes_
        prob_dict = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}

        # Calculate confidence as maximum probability
        confidence = float(np.max(probabilities))

        return {
            'success': True,
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': prob_dict,
            'surface_shape': surface.shape,
            'features': features.__dict__
        }

    def analyze_feature_importance(self) -> Dict[str, float]:
        """Analyze feature importance for Random Forest models."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if self.model_type != 'random_forest':
            logger.warning("Feature importance only available for Random Forest models")
            return {}

        # Get feature importance from the classifier in the pipeline
        classifier = self.model.named_steps['classifier']
        importances = classifier.feature_importances_

        # Create feature importance dictionary
        feature_importance = {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }

        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(),
                                         key=lambda x: x[1], reverse=True))

        return feature_importance

    def save_model(self, filepath: str) -> bool:
        """Save trained model to file."""
        if not self.is_trained:
            logger.error("No trained model to save")
            return False

        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'training_accuracy': self.training_accuracy,
                'cross_val_scores': self.cross_val_scores,
                'timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.training_accuracy = model_data['training_accuracy']
            self.cross_val_scores = model_data['cross_val_scores']
            self.is_trained = True

            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Model type: {self.model_type}")
            logger.info(f"Training accuracy: {self.training_accuracy:.3f}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class CharpyShearAnalyzer:
    """Complete system for Charpy shear analysis with microscope integration."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the analyzer.

        Args:
            model_path: Path to saved model file (optional)
        """
        self.classifier = ShearClassifier(model_type='random_forest')
        self.results_history = []

        # Load model if provided
        if model_path and Path(model_path).exists():
            self.classifier.load_model(model_path)
            logger.info("Pre-trained model loaded successfully")
        else:
            logger.info("No pre-trained model loaded. Train model before analysis.")

    def setup_training_data(self, reference_images_dir: str) -> Dict[int, List[str]]:
        """
        Setup training data from reference images directory.

        Expected directory structure:
        reference_images_dir/
        ├── 0_percent/
        │   ├── shear_0_1.jpg
        │   └── shear_0_2.jpg
        ├── 10_percent/
        │   ├── shear_10_1.jpg
        │   └── shear_10_2.jpg
        └── ...

        Args:
            reference_images_dir: Path to directory containing reference images

        Returns:
            Dictionary mapping shear percentages to image paths
        """
        reference_dir = Path(reference_images_dir)
        training_data = {}

        # Look for percentage directories
        for percent_dir in reference_dir.glob('*_percent'):
            # Extract percentage from directory name
            try:
                percent_str = percent_dir.name.split('_')[0]
                percent = int(percent_str)

                # Get all image files in this directory
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
                image_paths = []

                for ext in image_extensions:
                    image_paths.extend(list(percent_dir.glob(ext)))

                if image_paths:
                    training_data[percent] = [str(path) for path in image_paths]
                    logger.info(f"Found {len(image_paths)} images for {percent}% shear")

            except ValueError:
                logger.warning(f"Could not parse percentage from directory: {percent_dir.name}")

        logger.info(f"Setup training data for {len(training_data)} shear percentages")
        return training_data

    def train_classifier(self, training_data: Dict[int, List[str]],
                         model_save_path: str = "charpy_shear_model.pkl") -> Dict[str, Any]:
        """
        Train the shear classifier.

        Args:
            training_data: Dictionary mapping shear percentages to image paths
            model_save_path: Path to save trained model

        Returns:
            Training results
        """
        logger.info("Starting classifier training...")

        # Train model
        results = self.classifier.train_from_images(training_data, augment=True)

        # Save model
        if self.classifier.save_model(model_save_path):
            results['model_saved'] = model_save_path

        # Analyze feature importance
        if self.classifier.model_type == 'random_forest':
            feature_importance = self.classifier.analyze_feature_importance()
            results['feature_importance'] = feature_importance

            logger.info("Top 5 most important features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                logger.info(f"  {i + 1}. {feature}: {importance:.4f}")

        return results

    def analyze_image(self, image: np.ndarray, save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze a single image for shear percentage.

        Args:
            image: Input image
            save_results: Whether to save results to history

        Returns:
            Analysis results
        """
        start_time = datetime.now()

        # Get prediction
        prediction_result = self.classifier.predict_shear_percentage(image)

        # Add timing information
        analysis_time = (datetime.now() - start_time).total_seconds()
        prediction_result['analysis_time_seconds'] = analysis_time
        prediction_result['timestamp'] = start_time.isoformat()

        # Save to history if requested
        if save_results:
            self.results_history.append(prediction_result)

        # Log results
        if prediction_result['success']:
            pred = prediction_result['prediction']
            conf = prediction_result['confidence']
            logger.info(f"Prediction: {pred}% shear (confidence: {conf:.3f})")
        else:
            logger.error(f"Analysis failed: {prediction_result['error']}")

        return prediction_result

    def analyze_microscope_stream(self, device_id: int = 1, display: bool = True) -> None:
        """
        Analyze live microscope stream.

        Args:
            device_id: Camera device ID
            display: Whether to display results
        """
        # Try to import microscope interface
        try:
            import sys
            from pathlib import Path

            # Add src to path
            src_path = Path(__file__).parent.parent / "src"
            if src_path.exists():
                sys.path.append(str(src_path))
                from camera.microscope_interface import MicroscopeCapture
            else:
                # Fallback to OpenCV
                MicroscopeCapture = None
        except ImportError:
            MicroscopeCapture = None

        # Use microscope interface if available, otherwise use OpenCV
        if MicroscopeCapture:
            self._analyze_with_microscope_interface(device_id, display)
        else:
            self._analyze_with_opencv(device_id, display)

    def _analyze_with_microscope_interface(self, device_id: int, display: bool) -> None:
        """Analyze using microscope interface."""
        from camera.microscope_interface import MicroscopeCapture

        microscope = MicroscopeCapture(device_id=device_id)

        try:
            if not microscope.connect():
                logger.error("Failed to connect to microscope")
                return

            logger.info("Connected to microscope. Press 'q' to quit, 'a' to analyze current frame")

            while True:
                frame = microscope.get_frame()
                if frame is None:
                    continue

                if display:
                    display_frame = frame.copy()

                    # Add instructions
                    cv2.putText(display_frame, "Press 'A' to analyze, 'Q' to quit",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow('Charpy Shear Analysis', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    # Analyze current frame
                    result = self.analyze_image(frame)

                    if result['success']:
                        print(f"\n🔍 ANALYSIS RESULT:")
                        print(f"   Shear Percentage: {result['prediction']}%")
                        print(f"   Confidence: {result['confidence']:.3f}")
                        print(f"   Analysis Time: {result['analysis_time_seconds']:.2f}s")

                        # Show top 3 probabilities
                        sorted_probs = sorted(result['probabilities'].items(),
                                              key=lambda x: x[1], reverse=True)[:3]
                        print(f"   Top predictions:")
                        for percent, prob in sorted_probs:
                            print(f"     {percent}%: {prob:.3f}")
                    else:
                        print(f"\n❌ Analysis failed: {result['error']}")

        finally:
            microscope.disconnect()
            if display:
                cv2.destroyAllWindows()

    def _analyze_with_opencv(self, device_id: int, display: bool) -> None:
        """Analyze using OpenCV camera interface."""
        cap = cv2.VideoCapture(device_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {device_id}")
            return

        try:
            logger.info("Camera opened. Press 'q' to quit, 'a' to analyze current frame")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                if display:
                    display_frame = frame.copy()

                    # Add instructions
                    cv2.putText(display_frame, "Press 'A' to analyze, 'Q' to quit",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow('Charpy Shear Analysis', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    # Analyze current frame
                    result = self.analyze_image(frame)

                    if result['success']:
                        print(f"\n🔍 ANALYSIS RESULT:")
                        print(f"   Shear Percentage: {result['prediction']}%")
                        print(f"   Confidence: {result['confidence']:.3f}")
                        print(f"   Analysis Time: {result['analysis_time_seconds']:.2f}s")

                        # Show top 3 probabilities
                        sorted_probs = sorted(result['probabilities'].items(),
                                              key=lambda x: x[1], reverse=True)[:3]
                        print(f"   Top predictions:")
                        for percent, prob in sorted_probs:
                            print(f"     {percent}%: {prob:.3f}")
                    else:
                        print(f"\n❌ Analysis failed: {result['error']}")

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    def create_analysis_report(self, output_path: str = "shear_analysis_report.json") -> bool:
        """Create analysis report from results history."""
        if not self.results_history:
            logger.warning("No analysis results to report")
            return False

        # Calculate statistics
        successful_analyses = [r for r in self.results_history if r['success']]

        if not successful_analyses:
            logger.warning("No successful analyses to report")
            return False

        predictions = [r['prediction'] for r in successful_analyses]
        confidences = [r['confidence'] for r in successful_analyses]
        analysis_times = [r['analysis_time_seconds'] for r in successful_analyses]

        report = {
            'summary': {
                'total_analyses': len(self.results_history),
                'successful_analyses': len(successful_analyses),
                'success_rate': len(successful_analyses) / len(self.results_history),
                'average_confidence': np.mean(confidences),
                'average_analysis_time': np.mean(analysis_times)
            },
            'predictions': {
                'mean_shear_percentage': np.mean(predictions),
                'std_shear_percentage': np.std(predictions),
                'min_shear_percentage': np.min(predictions),
                'max_shear_percentage': np.max(predictions),
                'prediction_distribution': dict(zip(*np.unique(predictions, return_counts=True)))
            },
            'performance': {
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                },
                'timing_stats': {
                    'mean_seconds': np.mean(analysis_times),
                    'std_seconds': np.std(analysis_times),
                    'min_seconds': np.min(analysis_times),
                    'max_seconds': np.max(analysis_times)
                }
            },
            'detailed_results': self.results_history,
            'report_generated': datetime.now().isoformat()
        }

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Analysis report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return False


def main():
    """Main function demonstrating the shear classification system."""
    print("🔬 CHARPY FRACTURE SURFACE SHEAR CLASSIFICATION SYSTEM")
    print("=" * 70)
    print("This system classifies fracture surfaces into shear percentages")
    print("using traditional computer vision features and machine learning.")
    print()

    # Initialize analyzer
    analyzer = CharpyShearAnalyzer()

    print("🎯 SETUP OPTIONS:")
    print("1. Train new model from reference images")
    print("2. Load existing model and analyze images")
    print("3. Live analysis with microscope")
    print()

    choice = input("Select option (1-3): ").strip()

    if choice == "1":
        # Train new model
        print("\n📚 TRAINING NEW MODEL")
        print("-" * 30)

        reference_dir = input("Enter path to reference images directory: ").strip()
        if not Path(reference_dir).exists():
            print(f"❌ Directory not found: {reference_dir}")
            return

        # Setup training data
        training_data = analyzer.setup_training_data(reference_dir)

        if not training_data:
            print("❌ No training data found. Check directory structure.")
            print("Expected: reference_dir/0_percent/, reference_dir/10_percent/, etc.")
            return

        print(f"\n📊 Training data summary:")
        total_images = 0
        for percent, paths in training_data.items():
            print(f"   {percent}% shear: {len(paths)} images")
            total_images += len(paths)
        print(f"   Total: {total_images} images")

        if total_images < 11:
            print("⚠️  Warning: Very few images. Consider adding more for better accuracy.")

        # Train model
        print(f"\n🚀 Starting training with augmentation...")
        results = analyzer.train_classifier(training_data)

        print(f"\n✅ Training completed!")
        print(f"   Training accuracy: {results['training_accuracy']:.3f}")
        print(f"   Cross-validation: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print(f"   Model saved: {results.get('model_saved', 'Not saved')}")

        if 'feature_importance' in results:
            print(f"\n🔍 Top 5 important features:")
            for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5]):
                print(f"   {i + 1}. {feature}: {importance:.4f}")

    elif choice == "2":
        # Load existing model
        print("\n📁 LOADING EXISTING MODEL")
        print("-" * 30)

        model_path = input("Enter path to model file (.pkl): ").strip()
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return

        if not analyzer.classifier.load_model(model_path):
            print("❌ Failed to load model")
            return

        # Analyze images
        while True:
            image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
            if image_path.lower() == 'quit':
                break

            if not Path(image_path).exists():
                print(f"❌ Image not found: {image_path}")
                continue

            # Load and analyze image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                continue

            print(f"\n🔍 Analyzing {Path(image_path).name}...")
            result = analyzer.analyze_image(image)

            if result['success']:
                print(f"✅ Result: {result['prediction']}% shear")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Analysis time: {result['analysis_time_seconds']:.2f}s")

                # Show top 3 predictions
                sorted_probs = sorted(result['probabilities'].items(),
                                      key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top predictions:")
                for percent, prob in sorted_probs:
                    print(f"     {percent}%: {prob:.3f}")
            else:
                print(f"❌ Analysis failed: {result['error']}")

    elif choice == "3":
        # Live analysis
        print("\n📹 LIVE MICROSCOPE ANALYSIS")
        print("-" * 30)

        model_path = input("Enter path to trained model (.pkl): ").strip()
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return

        if not analyzer.classifier.load_model(model_path):
            print("❌ Failed to load model")
            return

        device_id = input("Enter camera device ID (default: 1): ").strip()
        device_id = int(device_id) if device_id.isdigit() else 1

        print(f"\n🎥 Starting live analysis with camera {device_id}...")
        print("Controls:")
        print("  - Press 'A' to analyze current frame")
        print("  - Press 'Q' to quit")

        analyzer.analyze_microscope_stream(device_id=device_id, display=True)

    else:
        print("❌ Invalid option selected")
        return

    # Generate report if there are results
    if analyzer.results_history:
        print(f"\n📊 Generating analysis report...")
        if analyzer.create_analysis_report():
            print("✅ Report saved as 'shear_analysis_report.json'")

    print(f"\n🎉 Session completed!")


if __name__ == "__main__":
    main()