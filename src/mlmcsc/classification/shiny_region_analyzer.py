#!/usr/bin/env python3
"""
Shiny Region Analyzer

Analyzes shiny/reflective regions within fracture surfaces.
This focuses on the KEY FEATURES that actually correlate with shear percentage.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from skimage import morphology, filters
from skimage.measure import regionprops, label
from skimage.morphology import disk

logger = logging.getLogger(__name__)


@dataclass
class ShinyRegionFeatures:
    """Features specifically designed for shiny region analysis."""

    # Core shiny region features
    shiny_area_ratio: float  # Ratio of shiny area to total fracture surface
    shiny_region_count: float  # Number of distinct shiny regions
    largest_shiny_ratio: float  # Size of largest shiny region vs total
    shiny_region_compactness: float  # How compact/round are shiny regions

    # Spatial distribution features
    shiny_center_x: float  # X-coordinate of shiny region centroid
    shiny_center_y: float  # Y-coordinate of shiny region centroid
    shiny_spread_x: float  # Horizontal spread of shiny regions
    shiny_spread_y: float  # Vertical spread of shiny regions

    # Intensity features of shiny regions
    mean_shiny_intensity: float  # Average intensity of shiny regions
    max_shiny_intensity: float  # Peak intensity in shiny regions
    shiny_intensity_std: float  # Intensity variation in shiny regions

    # Shape features of shiny regions
    shiny_aspect_ratio: float  # Width/height ratio of shiny regions
    shiny_elongation: float  # How elongated are shiny regions
    shiny_solidity: float  # How solid (vs hole-filled) are shiny regions

    # Boundary features
    shiny_perimeter_ratio: float  # Perimeter of shiny regions vs total
    shiny_edge_density: float  # Edge density around shiny regions

    # Rough region features (complement to shiny)
    rough_area_ratio: float  # Ratio of rough/dark area to total
    mean_rough_intensity: float  # Average intensity of rough regions

    # Overall fracture surface features
    total_surface_area: float  # Total fracture surface area (pixels)
    surface_aspect_ratio: float  # Overall fracture surface shape
    intensity_contrast: float  # Contrast between shiny and rough

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models."""
        return np.array([
            self.shiny_area_ratio, self.shiny_region_count, self.largest_shiny_ratio,
            self.shiny_region_compactness, self.shiny_center_x, self.shiny_center_y,
            self.shiny_spread_x, self.shiny_spread_y, self.mean_shiny_intensity,
            self.max_shiny_intensity, self.shiny_intensity_std, self.shiny_aspect_ratio,
            self.shiny_elongation, self.shiny_solidity, self.shiny_perimeter_ratio,
            self.shiny_edge_density, self.rough_area_ratio, self.mean_rough_intensity,
            self.total_surface_area, self.surface_aspect_ratio, self.intensity_contrast
        ])

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names."""
        return [
            'shiny_area_ratio', 'shiny_region_count', 'largest_shiny_ratio',
            'shiny_region_compactness', 'shiny_center_x', 'shiny_center_y',
            'shiny_spread_x', 'shiny_spread_y', 'mean_shiny_intensity',
            'max_shiny_intensity', 'shiny_intensity_std', 'shiny_aspect_ratio',
            'shiny_elongation', 'shiny_solidity', 'shiny_perimeter_ratio',
            'shiny_edge_density', 'rough_area_ratio', 'mean_rough_intensity',
            'total_surface_area', 'surface_aspect_ratio', 'intensity_contrast'
        ]


class ShinyRegionAnalyzer:
    """Analyzes shiny/reflective regions within fracture surfaces."""

    def __init__(self):
        self.debug = True
        logger.info("Shiny Region Analyzer initialized")

    def extract_shiny_region_features(self,
                                      surface_mask: np.ndarray,
                                      surface_image: np.ndarray) -> ShinyRegionFeatures:
        """
        Extract features focused on shiny region characteristics.

        Args:
            surface_mask: Binary mask of fracture surface area
            surface_image: Grayscale image of fracture surface region

        Returns:
            ShinyRegionFeatures object with all extracted features
        """
        try:
            # Mask the surface image to focus only on fracture surface
            masked_surface = np.where(surface_mask, surface_image, 0)

            # Segment shiny vs rough regions
            shiny_mask, rough_mask = self._segment_shiny_rough_regions(masked_surface, surface_mask)

            # Extract shiny region features
            shiny_features = self._analyze_shiny_regions(shiny_mask, masked_surface, surface_mask)

            # Extract rough region features
            rough_features = self._analyze_rough_regions(rough_mask, masked_surface)

            # Extract overall surface features
            surface_features = self._analyze_overall_surface(surface_mask, masked_surface)

            # Combine all features
            return ShinyRegionFeatures(
                # Shiny region features
                shiny_area_ratio=shiny_features['area_ratio'],
                shiny_region_count=shiny_features['region_count'],
                largest_shiny_ratio=shiny_features['largest_ratio'],
                shiny_region_compactness=shiny_features['compactness'],
                shiny_center_x=shiny_features['center_x'],
                shiny_center_y=shiny_features['center_y'],
                shiny_spread_x=shiny_features['spread_x'],
                shiny_spread_y=shiny_features['spread_y'],
                mean_shiny_intensity=shiny_features['mean_intensity'],
                max_shiny_intensity=shiny_features['max_intensity'],
                shiny_intensity_std=shiny_features['intensity_std'],
                shiny_aspect_ratio=shiny_features['aspect_ratio'],
                shiny_elongation=shiny_features['elongation'],
                shiny_solidity=shiny_features['solidity'],
                shiny_perimeter_ratio=shiny_features['perimeter_ratio'],
                shiny_edge_density=shiny_features['edge_density'],

                # Rough region features
                rough_area_ratio=rough_features['area_ratio'],
                mean_rough_intensity=rough_features['mean_intensity'],

                # Overall surface features
                total_surface_area=surface_features['total_area'],
                surface_aspect_ratio=surface_features['aspect_ratio'],
                intensity_contrast=surface_features['intensity_contrast']
            )

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            return self._get_default_features()

    def _segment_shiny_rough_regions(self,
                                     surface_image: np.ndarray,
                                     surface_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment surface into shiny (bright) and rough (dark) regions."""

        # Only consider pixels within the fracture surface
        surface_pixels = surface_image[surface_mask]

        if len(surface_pixels) == 0:
            return np.zeros_like(surface_mask), np.zeros_like(surface_mask)

        # Use adaptive thresholding to separate bright (shiny) from dark (rough) regions
        # The threshold is based on the distribution of intensities within the surface

        # Method 1: Use percentile-based thresholding
        bright_threshold = np.percentile(surface_pixels[surface_pixels > 0], 70)  # Top 30% brightest

        # Create shiny region mask (bright regions within fracture surface)
        shiny_mask = (surface_image >= bright_threshold) & surface_mask

        # Create rough region mask (darker regions within fracture surface)
        rough_mask = (surface_image < bright_threshold) & surface_mask & (surface_image > 0)

        # Clean up masks to remove noise
        shiny_mask = morphology.remove_small_objects(shiny_mask, min_size=50)
        shiny_mask = morphology.binary_closing(shiny_mask, disk(3))

        rough_mask = morphology.remove_small_objects(rough_mask, min_size=50)

        return shiny_mask, rough_mask

    def _analyze_shiny_regions(self,
                               shiny_mask: np.ndarray,
                               surface_image: np.ndarray,
                               total_surface_mask: np.ndarray) -> Dict[str, float]:
        """Analyze properties of shiny regions."""

        total_surface_area = np.sum(total_surface_mask)
        shiny_area = np.sum(shiny_mask)

        if shiny_area == 0:
            return self._get_default_shiny_features()

        # Basic area metrics
        area_ratio = shiny_area / total_surface_area if total_surface_area > 0 else 0.0

        # Analyze individual shiny regions
        labeled_shiny = label(shiny_mask.astype(int))
        shiny_regions = regionprops(labeled_shiny, intensity_image=surface_image)

        region_count = len(shiny_regions)

        if region_count == 0:
            return self._get_default_shiny_features()

        # Find largest shiny region
        largest_region = max(shiny_regions, key=lambda r: r.area)
        largest_ratio = largest_region.area / total_surface_area if total_surface_area > 0 else 0.0

        # Calculate compactness (average circularity of shiny regions)
        compactness_values = []
        for region in shiny_regions:
            if region.perimeter > 0:
                compactness = 4 * np.pi * region.area / (region.perimeter ** 2)
                compactness_values.append(compactness)

        compactness = np.mean(compactness_values) if compactness_values else 0.0

        # Spatial distribution (centroid and spread)
        if shiny_regions:
            centroids = np.array([[region.centroid[1], region.centroid[0]] for region in shiny_regions])
            weighted_centroids = np.array([[region.centroid[1] * region.area, region.centroid[0] * region.area]
                                           for region in shiny_regions])

            total_weight = sum(region.area for region in shiny_regions)
            center_x = np.sum(weighted_centroids[:, 0]) / total_weight if total_weight > 0 else 0.5
            center_y = np.sum(weighted_centroids[:, 1]) / total_weight if total_weight > 0 else 0.5

            # Normalize to image dimensions
            center_x = center_x / surface_image.shape[1] if surface_image.shape[1] > 0 else 0.5
            center_y = center_y / surface_image.shape[0] if surface_image.shape[0] > 0 else 0.5

            # Calculate spread
            spread_x = np.std(centroids[:, 0]) / surface_image.shape[1] if surface_image.shape[1] > 0 else 0.0
            spread_y = np.std(centroids[:, 1]) / surface_image.shape[0] if surface_image.shape[0] > 0 else 0.0
        else:
            center_x = center_y = 0.5
            spread_x = spread_y = 0.0

        # Intensity features
        shiny_pixels = surface_image[shiny_mask]
        mean_intensity = np.mean(shiny_pixels) if len(shiny_pixels) > 0 else 0.0
        max_intensity = np.max(shiny_pixels) if len(shiny_pixels) > 0 else 0.0
        intensity_std = np.std(shiny_pixels) if len(shiny_pixels) > 0 else 0.0

        # Shape features (using largest region as representative)
        if largest_region.bbox[2] - largest_region.bbox[0] > 0:
            aspect_ratio = (largest_region.bbox[3] - largest_region.bbox[1]) / (
                        largest_region.bbox[2] - largest_region.bbox[0])
        else:
            aspect_ratio = 1.0

        elongation = 1.0 - largest_region.minor_axis_length / largest_region.major_axis_length if largest_region.major_axis_length > 0 else 0.0
        solidity = largest_region.solidity

        # Perimeter features
        total_perimeter = sum(region.perimeter for region in shiny_regions)
        perimeter_ratio = total_perimeter / (2 * np.sqrt(np.pi * total_surface_area)) if total_surface_area > 0 else 0.0

        # Edge density around shiny regions
        edge_density = self._calculate_edge_density_around_regions(shiny_mask, surface_image)

        return {
            'area_ratio': float(area_ratio),
            'region_count': float(region_count),
            'largest_ratio': float(largest_ratio),
            'compactness': float(compactness),
            'center_x': float(center_x),
            'center_y': float(center_y),
            'spread_x': float(spread_x),
            'spread_y': float(spread_y),
            'mean_intensity': float(mean_intensity),
            'max_intensity': float(max_intensity),
            'intensity_std': float(intensity_std),
            'aspect_ratio': float(aspect_ratio),
            'elongation': float(elongation),
            'solidity': float(solidity),
            'perimeter_ratio': float(perimeter_ratio),
            'edge_density': float(edge_density)
        }

    def _analyze_rough_regions(self, rough_mask: np.ndarray, surface_image: np.ndarray) -> Dict[str, float]:
        """Analyze properties of rough regions."""

        total_pixels = np.sum(surface_image > 0)
        rough_area = np.sum(rough_mask)

        area_ratio = rough_area / total_pixels if total_pixels > 0 else 0.0

        rough_pixels = surface_image[rough_mask]
        mean_intensity = np.mean(rough_pixels) if len(rough_pixels) > 0 else 0.0

        return {
            'area_ratio': float(area_ratio),
            'mean_intensity': float(mean_intensity)
        }

    def _analyze_overall_surface(self, surface_mask: np.ndarray, surface_image: np.ndarray) -> Dict[str, float]:
        """Analyze overall fracture surface properties."""

        total_area = float(np.sum(surface_mask))

        # Surface aspect ratio
        labeled = label(surface_mask.astype(int))
        regions = regionprops(labeled)

        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            bbox_height = largest_region.bbox[2] - largest_region.bbox[0]
            bbox_width = largest_region.bbox[3] - largest_region.bbox[1]
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        else:
            aspect_ratio = 1.0

        # Intensity contrast within surface
        surface_pixels = surface_image[surface_mask]
        if len(surface_pixels) > 0:
            intensity_contrast = np.std(surface_pixels) / np.mean(surface_pixels) if np.mean(
                surface_pixels) > 0 else 0.0
        else:
            intensity_contrast = 0.0

        return {
            'total_area': total_area,
            'aspect_ratio': float(aspect_ratio),
            'intensity_contrast': float(intensity_contrast)
        }

    def _calculate_edge_density_around_regions(self, region_mask: np.ndarray, image: np.ndarray) -> float:
        """Calculate edge density around specified regions."""
        try:
            # Calculate edges using Sobel
            edges = filters.sobel(image)

            # Dilate region mask to get boundary area
            dilated = morphology.dilation(region_mask, disk(3))
            boundary = dilated & ~region_mask

            if np.sum(boundary) == 0:
                return 0.0

            # Calculate edge density in boundary area
            edge_density = np.mean(edges[boundary])
            return float(edge_density)

        except Exception:
            return 0.0

    def _get_default_shiny_features(self) -> Dict[str, float]:
        """Return default shiny features when no shiny regions found."""
        return {
            'area_ratio': 0.0, 'region_count': 0.0, 'largest_ratio': 0.0,
            'compactness': 0.0, 'center_x': 0.5, 'center_y': 0.5,
            'spread_x': 0.0, 'spread_y': 0.0, 'mean_intensity': 0.0,
            'max_intensity': 0.0, 'intensity_std': 0.0, 'aspect_ratio': 1.0,
            'elongation': 0.0, 'solidity': 0.0, 'perimeter_ratio': 0.0,
            'edge_density': 0.0
        }

    def _get_default_features(self) -> ShinyRegionFeatures:
        """Return default features when extraction fails."""
        return ShinyRegionFeatures(
            shiny_area_ratio=0.0, shiny_region_count=0.0, largest_shiny_ratio=0.0,
            shiny_region_compactness=0.0, shiny_center_x=0.5, shiny_center_y=0.5,
            shiny_spread_x=0.0, shiny_spread_y=0.0, mean_shiny_intensity=0.0,
            max_shiny_intensity=0.0, shiny_intensity_std=0.0, shiny_aspect_ratio=1.0,
            shiny_elongation=0.0, shiny_solidity=0.0, shiny_perimeter_ratio=0.0,
            shiny_edge_density=0.0, rough_area_ratio=0.0, mean_rough_intensity=0.0,
            total_surface_area=0.0, surface_aspect_ratio=1.0, intensity_contrast=0.0
        )