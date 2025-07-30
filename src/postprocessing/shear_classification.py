#!/usr/bin/env python3
"""
Charpy Fracture Surface Continuous Shear Prediction System

This system predicts continuous shear percentages (e.g., 23.7%, 45.2%) from fracture
surface images using traditional computer vision features and machine learning regression.

Features:
- Automatic fracture surface detection
- 22 texture and morphological features
- Data augmentation for small datasets
- Continuous percentage prediction with confidence intervals
- Real-time microscope integration
- Comprehensive analysis reporting
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
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
    # GLCM Features (Gray-Level Co-occurrence Matrix)
    contrast: float
    dissimilarity: float
    homogeneity: float
    energy: float
    correlation: float
    asm: float  # Angular Second Moment

    # LBP Features (Local Binary Pattern)
    lbp_uniformity: float
    lbp_contrast: float
    lbp_dissimilarity: float

    # Morphological Features
    roughness: float
    surface_area_ratio: float
    fractal_dimension: float

    # Intensity Statistical Features
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
        """Convert features to numpy array for ML models."""
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
        """Get list of feature names for model interpretation."""
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
        logger.info("FractureSurfaceDetector initialized")

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

        # Try multiple detection methods in order of reliability
        methods = [
            self._detect_by_specimen_shape,
            self._detect_by_texture,
            self._detect_by_edges,
            self._detect_by_intensity
        ]

        for method in methods:
            surface = method(gray)
            if surface is not None and self._validate_surface(surface):
                if self.debug:
                    logger.info(f"Surface detected using {method.__name__}")
                return surface

        logger.warning("Could not detect fracture surface using any method")
        return None

    def _detect_by_specimen_shape(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface by identifying specimen shape."""
        try:
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
            minr, minc, maxr, maxc = largest_region.bbox

            # The fracture surface is typically in the middle-upper area
            surface_minr = minr + int(0.1 * (maxr - minr))  # Start 10% from top
            surface_maxr = minr + int(0.6 * (maxr - minr))  # End 60% from top
            surface_minc = minc + int(0.2 * (maxc - minc))  # Start 20% from left
            surface_maxc = maxc - int(0.2 * (maxc - minc))  # End 20% from right

            # Extract fracture surface region
            surface = gray[surface_minr:surface_maxr, surface_minc:surface_maxc]

            if surface.size > 0:
                return surface

        except Exception as e:
            if self.debug:
                logger.debug(f"Shape detection failed: {e}")

        return None

    def _detect_by_texture(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface based on texture analysis."""
        try:
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

            # Extract with padding
            pad = 20
            minr = max(0, minr - pad)
            minc = max(0, minc - pad)
            maxr = min(gray.shape[0], maxr + pad)
            maxc = min(gray.shape[1], maxc + pad)

            surface = gray[minr:maxr, minc:maxc]

            if surface.size > 0:
                return surface

        except Exception as e:
            if self.debug:
                logger.debug(f"Texture detection failed: {e}")

        return None

    def _detect_by_edges(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface based on edge density."""
        try:
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

            if surface.size > 0:
                return surface

        except Exception as e:
            if self.debug:
                logger.debug(f"Edge detection failed: {e}")

        return None

    def _detect_by_intensity(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect fracture surface based on intensity patterns."""
        try:
            # Apply histogram equalization to enhance contrast
            equalized = cv2.equalizeHist(gray)

            # Use adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 2
            )

            # Find regions with intermediate intensity values
            mask = (gray > np.percentile(gray, 20)) & (gray < np.percentile(gray, 80))

            # Combine with adaptive threshold
            combined = mask & (adaptive_thresh > 0)

            # Clean up
            combined = morphology.opening(combined, morphology.disk(3))
            combined = morphology.closing(combined, morphology.disk(10))

            # Find largest region
            labeled = label(combined)
            regions = regionprops(labeled)

            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                minr, minc, maxr, maxc = largest_region.bbox
                surface = gray[minr:maxr, minc:maxc]

                if surface.size > 0:
                    return surface

            # Fallback: return center portion of image
            h, w = gray.shape
            return gray[h // 4:3 * h // 4, w // 4:3 * w // 4]

        except Exception as e:
            if self.debug:
                logger.debug(f"Intensity detection failed: {e}")

            # Ultimate fallback
            h, w = gray.shape
            return gray[h // 4:3 * h // 4, w // 4:3 * w // 4]

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
    """Extracts comprehensive features from fracture surfaces for shear prediction."""

    def __init__(self):
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, 45, 90, 135]
        self.lbp_radius = 3
        self.lbp_n_points = 24
        logger.info("ShearFeatureExtractor initialized")

    def _validate_feature_value(self, value: float, default: float = 0.0) -> float:
        """Validate and clean a feature value."""
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)

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
            contrast=self._validate_feature_value(glcm_features['contrast'], 0.0),
            dissimilarity=self._validate_feature_value(glcm_features['dissimilarity'], 0.0),
            homogeneity=self._validate_feature_value(glcm_features['homogeneity'], 0.5),
            energy=self._validate_feature_value(glcm_features['energy'], 0.5),
            correlation=self._validate_feature_value(glcm_features['correlation'], 0.0),
            asm=self._validate_feature_value(glcm_features['asm'], 0.5),

            # LBP features
            lbp_uniformity=self._validate_feature_value(lbp_features['uniformity'], 0.5),
            lbp_contrast=self._validate_feature_value(lbp_features['contrast'], 0.0),
            lbp_dissimilarity=self._validate_feature_value(lbp_features['dissimilarity'], 0.0),

            # Morphological features
            roughness=self._validate_feature_value(morph_features['roughness'], 1.0),
            surface_area_ratio=self._validate_feature_value(morph_features['surface_area_ratio'], 1.0),
            fractal_dimension=self._validate_feature_value(morph_features['fractal_dimension'], 2.0),

            # Intensity features
            mean_intensity=self._validate_feature_value(intensity_features['mean'], 128.0),
            std_intensity=self._validate_feature_value(intensity_features['std'], 50.0),
            skewness=self._validate_feature_value(intensity_features['skewness'], 0.0),
            kurtosis=self._validate_feature_value(intensity_features['kurtosis'], 3.0),

            # Gradient features
            gradient_magnitude=self._validate_feature_value(gradient_features['magnitude'], 10.0),
            gradient_direction_std=self._validate_feature_value(gradient_features['direction_std'], 1.0),

            # Edge features
            edge_density=self._validate_feature_value(edge_features['density'], 0.1),
            edge_strength=self._validate_feature_value(edge_features['strength'], 10.0),

            # Regional features
            smooth_regions_ratio=self._validate_feature_value(regional_features['smooth_ratio'], 0.5),
            rough_regions_ratio=self._validate_feature_value(regional_features['rough_ratio'], 0.5)
        )

    def _extract_glcm_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix features."""
        try:
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
                features[prop.lower()] = float(np.mean(values))

            return features

        except Exception as e:
            logger.warning(f"GLCM extraction failed: {e}")
            return {
                'contrast': 0.0, 'dissimilarity': 0.0, 'homogeneity': 0.5,
                'energy': 0.5, 'correlation': 0.0, 'asm': 0.5
            }

    def _extract_lbp_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract Local Binary Pattern features."""
        try:
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
                'uniformity': float(uniformity),
                'contrast': float(contrast),
                'dissimilarity': float(dissimilarity)
            }

        except Exception as e:
            logger.warning(f"LBP extraction failed: {e}")
            return {'uniformity': 0.5, 'contrast': 0.0, 'dissimilarity': 0.0}

    def _extract_morphological_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract morphological and structural features."""
        try:
            # Calculate surface roughness using gradients
            grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            roughness = float(np.mean(gradient_magnitude))

            # Surface area ratio (3D surface area vs projected area)
            surface_area_3d = np.sum(np.sqrt(1 + grad_x ** 2 + grad_y ** 2))
            projected_area = surface.size
            surface_area_ratio = float(surface_area_3d / projected_area)

            # Fractal dimension using box-counting method
            fractal_dimension = self._calculate_fractal_dimension(surface)

            return {
                'roughness': roughness,
                'surface_area_ratio': surface_area_ratio,
                'fractal_dimension': fractal_dimension
            }

        except Exception as e:
            logger.warning(f"Morphological extraction failed: {e}")
            return {'roughness': 0.0, 'surface_area_ratio': 1.0, 'fractal_dimension': 2.0}

    def _extract_intensity_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract intensity-based statistical features."""
        try:
            flat_surface = surface.flatten().astype(np.float32)

            return {
                'mean': float(np.mean(flat_surface)),
                'std': float(np.std(flat_surface)),
                'skewness': float(stats.skew(flat_surface)),
                'kurtosis': float(stats.kurtosis(flat_surface))
            }

        except Exception as e:
            logger.warning(f"Intensity extraction failed: {e}")
            return {'mean': 128.0, 'std': 30.0, 'skewness': 0.0, 'kurtosis': 0.0}

    def _extract_gradient_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract gradient-based features."""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)

            # Gradient magnitude
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Gradient direction
            direction = np.arctan2(grad_y, grad_x)

            return {
                'magnitude': float(np.mean(magnitude)),
                'direction_std': float(np.std(direction))
            }

        except Exception as e:
            logger.warning(f"Gradient extraction failed: {e}")
            return {'magnitude': 0.0, 'direction_std': 0.0}

    def _extract_edge_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract edge-based features."""
        try:
            # Apply Canny edge detection
            edges = feature.canny(surface, sigma=1.0)

            # Edge density
            edge_density = float(np.sum(edges) / edges.size)

            # Edge strength (average gradient magnitude at edge pixels)
            grad_x = cv2.Sobel(surface, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(surface, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            if np.sum(edges) > 0:
                edge_strength = float(np.mean(gradient_magnitude[edges]))
            else:
                edge_strength = 0.0

            return {
                'density': edge_density,
                'strength': edge_strength
            }

        except Exception as e:
            logger.warning(f"Edge extraction failed: {e}")
            return {'density': 0.0, 'strength': 0.0}

    def _extract_regional_features(self, surface: np.ndarray) -> Dict[str, float]:
        """Extract features based on smooth vs rough regions."""
        try:
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

            smooth_ratio = float(np.sum(smooth_regions) / surface.size)
            rough_ratio = float(np.sum(rough_regions) / surface.size)

            return {
                'smooth_ratio': smooth_ratio,
                'rough_ratio': rough_ratio
            }

        except Exception as e:
            logger.warning(f"Regional extraction failed: {e}")
            return {'smooth_ratio': 0.5, 'rough_ratio': 0.5}

    def _calculate_fractal_dimension(self, surface: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        try:
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

                    return float(max(1.0, min(3.0, fractal_dim)))  # Constrain to reasonable range

            return 2.0  # Default value

        except Exception as e:
            logger.warning(f"Fractal dimension calculation failed: {e}")
            return 2.0


class DataAugmentor:
    """Augments training data for improved model performance with limited samples."""

    def __init__(self):
        self.rotation_angles = [-15, -10, -5, 5, 10, 15]
        self.brightness_factors = [0.8, 0.9, 1.1, 1.2]
        self.contrast_factors = [0.8, 0.9, 1.1, 1.2]
        self.noise_levels = [5, 10, 15]
        logger.info("DataAugmentor initialized")

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


class ShearRegressor:
    """Regression-based predictor for continuous shear percentage values."""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize regressor for continuous shear prediction.

        Args:
            model_type: Type of regressor ('random_forest', 'svr', 'ensemble')
        """
        self.model_type = model_type
        self.detector = FractureSurfaceDetector()
        self.feature_extractor = ShearFeatureExtractor()
        self.augmentor = DataAugmentor()

        # Initialize models
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ShearFeatures.get_feature_names()

        # Training data storage
        self.X_train = None
        self.y_train = None
        self.is_trained = False

        # Model performance
        self.training_mae = 0.0
        self.training_r2 = 0.0
        self.cross_val_scores = []

        logger.info(f"ShearRegressor initialized with {model_type} model")

    def _create_model(self) -> Pipeline:
        """Create machine learning regression pipeline."""
        if self.model_type == 'random_forest':
            regressor = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svr':
            regressor = SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            )
        else:  # ensemble
            from sklearn.ensemble import VotingRegressor
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
            regressor = VotingRegressor([('rf', rf), ('svr', svr)])

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])

        return pipeline

    def train_from_images(self, training_data: Dict[float, List[str]], augment: bool = True) -> Dict[str, Any]:
        """
        Train regressor from image files with continuous shear values.

        Args:
            training_data: Dict mapping shear percentages to lists of image paths
                          e.g., {0.0: ['0_shear_1.jpg'], 12.5: ['12p5_shear_1.jpg'],
                                25.0: ['25_shear_1.jpg'], ...}
            augment: Whether to apply data augmentation

        Returns:
            Training results dictionary
        """
        logger.info("Starting regression training from images...")

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
                    # Add slight noise to shear values for augmented samples
                    shear_values = [shear_percent]  # Original
                    for _ in range(8):  # Augmented samples get slight variation
                        noise = np.random.normal(0, 0.5)  # Small noise
                        shear_values.append(max(0, min(100, shear_percent + noise)))
                else:
                    surfaces = [surface]
                    shear_values = [shear_percent]

                # Extract features from all surfaces
                for surf, shear_val in zip(surfaces, shear_values):
                    try:
                        features = self.feature_extractor.extract_features(surf)
                        X_features.append(features.to_array())
                        y_labels.append(shear_val)
                    except Exception as e:
                        logger.warning(f"Feature extraction failed for {image_path}: {e}")
                        continue

                processed += 1
                if processed % 5 == 0:
                    logger.info(f"Processed {processed}/{total_images} images...")

        if not X_features:
            raise ValueError("No valid features extracted from training data")

        # Convert to numpy arrays
        self.X_train = np.array(X_features)
        self.y_train = np.array(y_labels)

        # Check for NaN or infinite values
        nan_mask = np.isnan(self.X_train).any(axis=1)
        inf_mask = np.isinf(self.X_train).any(axis=1)
        invalid_mask = nan_mask | inf_mask
        
        # Ensure invalid_mask is a numpy array
        invalid_mask = np.asarray(invalid_mask)
        
        if np.any(invalid_mask):
            logger.warning(f"Found {np.sum(invalid_mask)} samples with NaN or infinite values, removing them...")
            valid_mask = ~invalid_mask
            self.X_train = self.X_train[valid_mask]
            self.y_train = self.y_train[valid_mask]
        
        # Check for NaN in labels
        label_nan_mask = np.isnan(self.y_train)
        # Ensure label_nan_mask is a numpy array
        label_nan_mask = np.asarray(label_nan_mask)
        
        if np.any(label_nan_mask):
            logger.warning(f"Found {np.sum(label_nan_mask)} samples with NaN labels, removing them...")
            valid_label_mask = ~label_nan_mask
            self.X_train = self.X_train[valid_label_mask]
            self.y_train = self.y_train[valid_label_mask]

        if self.X_train.shape[0] == 0:
            raise ValueError("No valid samples remaining after removing NaN/infinite values")

        logger.info(f"Training data prepared: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        logger.info(f"Shear range: {np.min(self.y_train):.1f}% - {np.max(self.y_train):.1f}%")

        # Train model
        return self._train_model()

    def _train_model(self) -> Dict[str, Any]:
        """Train the regression model."""
        logger.info(f"Training {self.model_type} regression model...")

        # Create model
        self.model = self._create_model()

        # Perform cross-validation with MAE scoring
        try:
            # Ensure we have enough samples for cross-validation
            n_samples = self.X_train.shape[0]
            cv_folds = min(5, max(2, n_samples // 10))  # At least 2 folds, at most 5
            
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train,
                                        cv=cv_folds, scoring='neg_mean_absolute_error')
            self.cross_val_scores = -cv_scores  # Convert back to positive MAE
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            self.cross_val_scores = np.array([])

        # Train on full dataset
        self.model.fit(self.X_train, self.y_train)

        # Calculate training metrics
        y_pred_train = self.model.predict(self.X_train)
        self.training_mae = mean_absolute_error(self.y_train, y_pred_train)
        self.training_r2 = r2_score(self.y_train, y_pred_train)

        self.is_trained = True

        results = {
            'training_mae': self.training_mae,
            'training_r2': self.training_r2,
            'cv_mae_mean': np.mean(self.cross_val_scores) if len(self.cross_val_scores) > 0 else 0.0,
            'cv_mae_std': np.std(self.cross_val_scores) if len(self.cross_val_scores) > 0 else 0.0,
            'cv_scores': self.cross_val_scores.tolist() if len(self.cross_val_scores) > 0 else [],
            'n_samples': self.X_train.shape[0],
            'n_features': self.X_train.shape[1],
            'shear_range': {
                'min': float(np.min(self.y_train)),
                'max': float(np.max(self.y_train)),
                'mean': float(np.mean(self.y_train)),
                'std': float(np.std(self.y_train))
            }
        }

        logger.info(f"Training completed!")
        logger.info(f"Training MAE: {self.training_mae:.2f}%")
        logger.info(f"Training R²: {self.training_r2:.3f}")
        if len(self.cross_val_scores) > 0:
            logger.info(
                f"Cross-validation MAE: {np.mean(self.cross_val_scores):.2f} ± {np.std(self.cross_val_scores):.2f}%")

        return results

    def predict_shear_percentage(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict continuous shear percentage from an image.

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
                'confidence_interval': None,
                'features': {}
            }

        # Extract features
        try:
            features = self.feature_extractor.extract_features(surface)
            feature_vector = features.to_array().reshape(1, -1)
        except Exception as e:
            return {
                'success': False,
                'error': f'Feature extraction failed: {e}',
                'prediction': None,
                'confidence_interval': None,
                'features': {}
            }

        # Make prediction
        try:
            prediction = self.model.predict(feature_vector)[0]

            # Estimate prediction uncertainty
            confidence_interval = self._estimate_prediction_interval(feature_vector, prediction)

            # Constrain prediction to valid range
            prediction = max(0.0, min(100.0, prediction))

            return {
                'success': True,
                'prediction': float(prediction),
                'confidence_interval': confidence_interval,
                'surface_shape': surface.shape,
                'features': features.__dict__,
                'prediction_category': self._categorize_prediction(prediction)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {e}',
                'prediction': None,
                'confidence_interval': None,
                'features': {}
            }

    def _estimate_prediction_interval(self, feature_vector: np.ndarray, prediction: float) -> Dict[str, float]:
        """Estimate prediction confidence interval."""
        try:
            if self.model_type == 'random_forest':
                # Use individual tree predictions for uncertainty estimation
                regressor = self.model.named_steps['regressor']
                scaled_features = self.model.named_steps['scaler'].transform(feature_vector)

                # Get predictions from individual trees
                tree_predictions = []
                for tree in regressor.estimators_:
                    tree_pred = tree.predict(scaled_features)[0]
                    tree_predictions.append(tree_pred)

                tree_predictions = np.array(tree_predictions)

                # Calculate confidence interval (±2 standard deviations ≈ 95% CI)
                std_pred = np.std(tree_predictions)

                return {
                    'lower_bound': max(0.0, prediction - 2 * std_pred),
                    'upper_bound': min(100.0, prediction + 2 * std_pred),
                    'std_deviation': float(std_pred),
                    'confidence_width': float(4 * std_pred)
                }
            else:
                # For other models, use training MAE as rough uncertainty estimate
                return {
                    'lower_bound': max(0.0, prediction - self.training_mae),
                    'upper_bound': min(100.0, prediction + self.training_mae),
                    'std_deviation': float(self.training_mae / 2),
                    'confidence_width': float(2 * self.training_mae)
                }
        except Exception as e:
            logger.warning(f"Confidence interval estimation failed: {e}")
            return {
                'lower_bound': max(0.0, prediction - 5.0),
                'upper_bound': min(100.0, prediction + 5.0),
                'std_deviation': 5.0,
                'confidence_width': 10.0
            }

    def _categorize_prediction(self, prediction: float) -> str:
        """Categorize prediction into quality ranges."""
        if prediction <= 15:
            return "Low Shear (Brittle Fracture)"
        elif prediction <= 35:
            return "Low-Medium Shear"
        elif prediction <= 65:
            return "Medium Shear (Mixed Mode)"
        elif prediction <= 85:
            return "High Shear"
        else:
            return "Very High Shear (Ductile Fracture)"

    def analyze_feature_importance(self) -> Dict[str, float]:
        """Analyze feature importance for Random Forest models."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if self.model_type != 'random_forest':
            logger.warning("Feature importance only available for Random Forest models")
            return {}

        try:
            # Get feature importance from the regressor in the pipeline
            regressor = self.model.named_steps['regressor']
            importances = regressor.feature_importances_

            # Create feature importance dictionary
            feature_importance = {
                name: float(importance)
                for name, importance in zip(self.feature_names, importances)
            }

            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(),
                                             key=lambda x: x[1], reverse=True))

            return feature_importance
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {}

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
                'training_mae': self.training_mae,
                'training_r2': self.training_r2,
                'cross_val_scores': self.cross_val_scores,
                'timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Regression model saved to {filepath}")
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
            self.training_mae = model_data.get('training_mae', 0.0)
            self.training_r2 = model_data.get('training_r2', 0.0)
            self.cross_val_scores = model_data.get('cross_val_scores', [])
            self.is_trained = True

            logger.info(f"Regression model loaded from {filepath}")
            logger.info(f"Model type: {self.model_type}")
            logger.info(f"Training MAE: {self.training_mae:.2f}%")
            logger.info(f"Training R²: {self.training_r2:.3f}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class CharpyShearAnalyzer:
    """Complete system for continuous Charpy shear analysis with microscope integration."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the analyzer for continuous shear prediction.

        Args:
            model_path: Path to saved model file (optional)
        """
        self.regressor = ShearRegressor(model_type='random_forest')
        self.results_history = []

        # Load model if provided
        if model_path and Path(model_path).exists():
            if self.regressor.load_model(model_path):
                logger.info("Pre-trained regression model loaded successfully")
            else:
                logger.warning("Failed to load pre-trained model")
        else:
            logger.info("No pre-trained model loaded. Train model before analysis.")

    def setup_training_data(self, reference_images_dir: str) -> Dict[float, List[str]]:
        """
        Setup training data from reference images directory for continuous values.

        Expected directory structure:
        reference_images_dir/
        ├── 0_percent/          # 0.0% shear
        ├── 12p5_percent/       # 12.5% shear
        ├── 25_percent/         # 25.0% shear
        ├── 37p5_percent/       # 37.5% shear
        └── ...

        Also supports:
        ├── shear_0.0/
        ├── shear_15.5/
        └── ...

        Args:
            reference_images_dir: Path to directory containing reference images

        Returns:
            Dictionary mapping shear percentages to image paths
        """
        reference_dir = Path(reference_images_dir)
        training_data = {}

        if not reference_dir.exists():
            logger.error(f"Reference directory does not exist: {reference_dir}")
            return training_data

        # Look for percentage directories (multiple naming formats)
        for subdir in reference_dir.iterdir():
            if not subdir.is_dir():
                continue

            dir_name = subdir.name.lower()
            shear_percent = None

            # Try different naming patterns
            try:
                if '_percent' in dir_name:
                    # Format: "25_percent" or "37p5_percent"
                    percent_part = dir_name.split('_percent')[0]
                    if 'p' in percent_part:
                        # Handle "37p5" -> 37.5
                        parts = percent_part.split('p')
                        shear_percent = float(parts[0]) + float(parts[1]) / 10
                    else:
                        shear_percent = float(percent_part)

                elif dir_name.startswith('shear_'):
                    # Format: "shear_25.5"
                    percent_str = dir_name.replace('shear_', '')
                    shear_percent = float(percent_str)

                elif dir_name.replace('.', '').replace('_', '').isdigit():
                    # Format: "25" or "25.5" or "25_5"
                    percent_str = dir_name.replace('_', '.')
                    shear_percent = float(percent_str)

                else:
                    # Try to extract number from directory name
                    import re
                    numbers = re.findall(r'\d+\.?\d*', dir_name)
                    if numbers:
                        shear_percent = float(numbers[0])
                        # Handle cases like "37_5" meaning 37.5
                        if len(numbers) > 1 and len(numbers[1]) == 1:
                            shear_percent += float(numbers[1]) / 10

                if shear_percent is not None and 0 <= shear_percent <= 100:
                    # Get all image files in this directory
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
                    image_paths = []

                    for ext in image_extensions:
                        image_paths.extend(list(subdir.glob(ext)))

                    if image_paths:
                        training_data[shear_percent] = [str(path) for path in image_paths]
                        logger.info(f"Found {len(image_paths)} images for {shear_percent}% shear")

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse shear percentage from directory '{dir_name}': {e}")

        logger.info(f"Setup training data for {len(training_data)} shear values")

        # Sort by shear percentage for display
        if training_data:
            sorted_data = dict(sorted(training_data.items()))
            logger.info("Training data summary:")
            for shear, paths in sorted_data.items():
                logger.info(f"  {shear:5.1f}%: {len(paths)} images")

        return training_data

    def train_regressor(self, training_data: Dict[float, List[str]],
                        model_save_path: str = "charpy_shear_regressor.pkl") -> Dict[str, Any]:
        """
        Train the continuous shear regressor.

        Args:
            training_data: Dictionary mapping shear percentages to image paths
            model_save_path: Path to save trained model

        Returns:
            Training results
        """
        logger.info("Starting regressor training for continuous shear prediction...")

        if not training_data:
            raise ValueError("No training data provided")

        # Train model
        results = self.regressor.train_from_images(training_data, augment=True)

        # Save model
        if self.regressor.save_model(model_save_path):
            results['model_saved'] = model_save_path

        # Analyze feature importance
        if self.regressor.model_type == 'random_forest':
            feature_importance = self.regressor.analyze_feature_importance()
            if feature_importance:
                results['feature_importance'] = feature_importance

                logger.info("Top 5 most important features:")
                for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                    logger.info(f"  {i + 1}. {feature}: {importance:.4f}")

        return results

    def analyze_image(self, image: np.ndarray, save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze a single image for continuous shear percentage.

        Args:
            image: Input image
            save_results: Whether to save results to history

        Returns:
            Analysis results with continuous shear value
        """
        start_time = datetime.now()

        # Get prediction
        prediction_result = self.regressor.predict_shear_percentage(image)

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
            ci = prediction_result['confidence_interval']
            category = prediction_result['prediction_category']

            logger.info(f"Prediction: {pred:.1f}% shear ({category})")
            logger.info(f"Confidence interval: [{ci['lower_bound']:.1f}%, {ci['upper_bound']:.1f}%]")
            logger.info(f"Uncertainty: ±{ci['std_deviation']:.1f}%")
        else:
            logger.error(f"Analysis failed: {prediction_result['error']}")

        return prediction_result

    def create_analysis_report(self, output_path: str = "shear_analysis_report.json") -> bool:
        """Create comprehensive analysis report from results history."""
        if not self.results_history:
            logger.warning("No analysis results to report")
            return False

        # Calculate statistics
        successful_analyses = [r for r in self.results_history if r['success']]

        if not successful_analyses:
            logger.warning("No successful analyses to report")
            return False

        predictions = [r['prediction'] for r in successful_analyses]
        uncertainties = [r['confidence_interval']['std_deviation'] for r in successful_analyses]
        analysis_times = [r['analysis_time_seconds'] for r in successful_analyses]

        report = {
            'summary': {
                'total_analyses': len(self.results_history),
                'successful_analyses': len(successful_analyses),
                'success_rate': len(successful_analyses) / len(self.results_history),
                'average_uncertainty': np.mean(uncertainties),
                'average_analysis_time': np.mean(analysis_times)
            },
            'predictions': {
                'mean_shear_percentage': np.mean(predictions),
                'std_shear_percentage': np.std(predictions),
                'min_shear_percentage': np.min(predictions),
                'max_shear_percentage': np.max(predictions),
                'median_shear_percentage': np.median(predictions)
            },
            'performance': {
                'uncertainty_stats': {
                    'mean': np.mean(uncertainties),
                    'std': np.std(uncertainties),
                    'min': np.min(uncertainties),
                    'max': np.max(uncertainties)
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
    """Main function demonstrating the continuous shear prediction system."""
    print("🔬 CHARPY FRACTURE SURFACE CONTINUOUS SHEAR PREDICTION SYSTEM")
    print("=" * 80)
    print("This system predicts continuous shear percentages (e.g., 23.7%, 45.2%)")
    print("using advanced computer vision features and regression models.")
    print()

    # Initialize analyzer
    analyzer = CharpyShearAnalyzer()

    print("🎯 SETUP OPTIONS:")
    print("1. Train new regression model from reference images")
    print("2. Load existing model and analyze images")
    print("3. Live continuous analysis with microscope")
    print()

    choice = input("Select option (1-3): ").strip()

    if choice == "1":
        # Train new model
        print("\n📚 TRAINING NEW REGRESSION MODEL")
        print("-" * 40)

        reference_dir = input("Enter path to reference images directory: ").strip()
        if not Path(reference_dir).exists():
            print(f"❌ Directory not found: {reference_dir}")
            return

        print("\n📁 Expected directory structure:")
        print("   reference_images/")
        print("   ├── 0_percent/        # 0.0% shear")
        print("   ├── 12p5_percent/     # 12.5% shear")
        print("   ├── 25_percent/       # 25.0% shear")
        print("   ├── 37p5_percent/     # 37.5% shear")
        print("   └── ... (any percentage values)")
        print()

        # Setup training data
        training_data = analyzer.setup_training_data(reference_dir)

        if not training_data:
            print("❌ No training data found. Check directory structure.")
            return

        print(f"\n📊 Training data summary:")
        total_images = 0
        shear_values = sorted(training_data.keys())
        for shear_val in shear_values:
            paths = training_data[shear_val]
            print(f"   {shear_val:5.1f}% shear: {len(paths)} images")
            total_images += len(paths)
        print(f"   Total: {total_images} images")
        print(f"   Shear range: {min(shear_values):.1f}% - {max(shear_values):.1f}%")

        if total_images < 5:
            print("❌ Too few images for training. Need at least 5 images.")
            return
        elif total_images < 20:
            print("⚠️  Warning: Very few images. Consider adding more for better accuracy.")

        # Train model
        print(f"\n🚀 Starting regression training with augmentation...")
        try:
            results = analyzer.train_regressor(training_data)

            print(f"\n✅ Training completed!")
            print(f"   Training MAE: {results['training_mae']:.2f}%")
            print(f"   Training R²: {results['training_r2']:.3f}")
            if results['cv_mae_mean'] > 0:
                print(f"   Cross-validation MAE: {results['cv_mae_mean']:.2f} ± {results['cv_mae_std']:.2f}%")
            print(f"   Model saved: {results.get('model_saved', 'Not saved')}")

            if 'feature_importance' in results:
                print(f"\n🔍 Top 5 important features:")
                for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5]):
                    print(f"   {i + 1}. {feature}: {importance:.4f}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return

    elif choice == "2":
        # Load existing model
        print("\n📁 LOADING EXISTING REGRESSION MODEL")
        print("-" * 40)

        model_path = input("Enter path to model file (.pkl): ").strip()
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return

        if not analyzer.regressor.load_model(model_path):
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
                pred = result['prediction']
                ci = result['confidence_interval']
                category = result['prediction_category']

                print(f"✅ Result: {pred:.1f}% shear")
                print(f"   Category: {category}")
                print(f"   Confidence interval: [{ci['lower_bound']:.1f}%, {ci['upper_bound']:.1f}%]")
                print(f"   Uncertainty: ±{ci['std_deviation']:.1f}%")
                print(f"   Analysis time: {result['analysis_time_seconds']:.2f}s")
            else:
                print(f"❌ Analysis failed: {result['error']}")

    elif choice == "3":
        # Live analysis
        print("\n📹 LIVE MICROSCOPE CONTINUOUS ANALYSIS")
        print("-" * 45)

        model_path = input("Enter path to trained model (.pkl): ").strip()
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return

        if not analyzer.regressor.load_model(model_path):
            print("❌ Failed to load model")
            return

        device_id = input("Enter camera device ID (default: 1): ").strip()
        device_id = int(device_id) if device_id.isdigit() else 1

        print(f"\n🎥 Starting live continuous analysis with camera {device_id}...")
        print("Controls:")
        print("  - Press 'A' to analyze current frame")
        print("  - Press 'Q' to quit")
        print("Results will show:")
        print("  - Exact shear percentage (e.g., 23.7%)")
        print("  - Confidence interval (e.g., [20.1%, 27.3%])")
        print("  - Uncertainty estimate (±3.6%)")
        print("  - Fracture category")

        # Simple OpenCV camera interface for live analysis
        cap = cv2.VideoCapture(device_id)

        if not cap.isOpened():
            print(f"❌ Failed to open camera {device_id}")
            return

        try:
            print("\n📷 Camera opened. Press 'A' to analyze, 'Q' to quit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Add instructions to frame
                display_frame = frame.copy()
                cv2.putText(display_frame, "Press 'A' to analyze, 'Q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Charpy Continuous Shear Analysis', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    # Analyze current frame
                    print("\n🔍 Analyzing current frame...")
                    result = analyzer.analyze_image(frame)

                    if result['success']:
                        pred = result['prediction']
                        ci = result['confidence_interval']
                        category = result['prediction_category']

                        print(f"✅ ANALYSIS RESULT:")
                        print(f"   Shear Percentage: {pred:.1f}%")
                        print(f"   Category: {category}")
                        print(f"   Confidence Interval: [{ci['lower_bound']:.1f}%, {ci['upper_bound']:.1f}%]")
                        print(f"   Uncertainty: ±{ci['std_deviation']:.1f}%")
                        print(f"   Analysis Time: {result['analysis_time_seconds']:.2f}s")
                    else:
                        print(f"❌ Analysis failed: {result['error']}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    else:
        print("❌ Invalid option selected")
        return

    # Generate report if there are results
    if analyzer.results_history:
        print(f"\n📊 Generating analysis report...")
        if analyzer.create_analysis_report():
            print("✅ Report saved as 'shear_analysis_report.json'")

    print(f"\n🎉 Session completed!")
    print("\n💡 Key Benefits of Continuous Prediction:")
    print("   ✓ Precise values like 23.7% instead of just '20%' or '30%'")
    print("   ✓ Confidence intervals show prediction reliability")
    print("   ✓ Works with any number of reference images")
    print("   ✓ Interpolates between reference values")
    print("   ✓ Better for quality control and research")


if __name__ == "__main__":
    main()