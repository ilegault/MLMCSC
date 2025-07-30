#!/usr/bin/env python3
"""
Improved Fracture Surface Detector

Detects FULL fracture surfaces preserving shiny/rough region context.
Key improvement: doesn't crop too tightly, maintains spatial relationships.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from skimage import morphology, filters
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, disk
from skimage.filters import gaussian
from scipy import ndimage

logger = logging.getLogger(__name__)


class ImprovedFractureSurfaceDetector:
    """Detects FULL fracture surfaces preserving shiny/rough region context."""

    def __init__(self):
        self.debug = True
        logger.info("Improved Fracture Surface Detector initialized")

    def detect_full_fracture_surface(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect the complete fracture surface, preserving context for shiny region analysis.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Tuple of (fracture_surface_mask, original_image_roi) or None
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Try multiple detection strategies
        strategies = [
            self._detect_by_brightness_contrast,
            self._detect_by_specimen_outline,
            self._detect_by_texture_variance
        ]

        for strategy in strategies:
            result = strategy(gray)
            if result is not None:
                surface_mask, roi = result
                if self._validate_fracture_surface(surface_mask):
                    if self.debug:
                        logger.info(f"Surface detected using {strategy.__name__}")
                    return surface_mask, roi

        logger.warning("Could not detect complete fracture surface")
        return None

    def _detect_by_brightness_contrast(self, gray: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect fracture surface using brightness contrast (shiny vs rough regions)."""
        try:
            # Apply gentle smoothing to reduce noise while preserving major regions
            smoothed = gaussian(gray, sigma=1)

            # Use multi-level thresholding to separate background, rough, and shiny regions
            # This preserves the contrast that's critical for shear analysis
            thresh_low = np.percentile(smoothed, 15)  # Dark background
            thresh_high = np.percentile(smoothed, 85)  # Bright reflective regions

            # Create mask for fracture surface (intermediate + bright regions)
            fracture_mask = smoothed > thresh_low

            # Clean up the mask - remove small noise but preserve structure
            fracture_mask = morphology.remove_small_objects(fracture_mask, min_size=500)
            fracture_mask = morphology.binary_closing(fracture_mask, disk(5))
            fracture_mask = ndimage.binary_fill_holes(fracture_mask)

            # Find the largest connected component (main fracture surface)
            labeled = label(fracture_mask)
            regions = regionprops(labeled)

            if not regions:
                return None

            # Get the largest region
            largest_region = max(regions, key=lambda r: r.area)

            # Create clean mask for just the fracture surface
            clean_mask = (labeled == largest_region.label)

            # Get bounding box with generous padding to preserve context
            minr, minc, maxr, maxc = largest_region.bbox
            pad = 30  # Generous padding
            minr = max(0, minr - pad)
            minc = max(0, minc - pad)
            maxr = min(gray.shape[0], maxr + pad)
            maxc = min(gray.shape[1], maxc + pad)

            # Extract ROI preserving the full context
            roi = gray[minr:maxr, minc:maxc]
            mask_roi = clean_mask[minr:maxr, minc:maxc]

            return mask_roi, roi

        except Exception as e:
            if self.debug:
                logger.debug(f"Brightness contrast detection failed: {e}")
            return None

    def _detect_by_specimen_outline(self, gray: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect fracture surface by identifying specimen boundaries."""
        try:
            # Use edge detection to find specimen outline
            edges = cv2.Canny(gray, 30, 100)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Find the largest contour (specimen outline)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create mask from contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            mask = mask > 0

            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # The fracture surface is typically in the upper-middle portion
            # Adjust based on your specimen orientation
            frac_y_start = y + int(0.1 * h)  # Start 10% from top
            frac_y_end = y + int(0.7 * h)  # End 70% from top
            frac_x_start = x + int(0.1 * w)  # Start 10% from left
            frac_x_end = x + int(0.9 * w)  # End 90% from left

            # Extract fracture surface region
            roi = gray[frac_y_start:frac_y_end, frac_x_start:frac_x_end]
            mask_roi = mask[frac_y_start:frac_y_end, frac_x_start:frac_x_end]

            return mask_roi, roi

        except Exception as e:
            if self.debug:
                logger.debug(f"Specimen outline detection failed: {e}")
            return None

    def _detect_by_texture_variance(self, gray: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect fracture surface using texture variance."""
        try:
            # Calculate local variance to find textured regions
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)

            # Fracture surfaces have moderate to high variance
            variance_thresh = np.percentile(local_var, 60)
            texture_mask = local_var > variance_thresh

            # Clean up
            texture_mask = morphology.remove_small_objects(texture_mask, min_size=1000)
            texture_mask = morphology.binary_closing(texture_mask, disk(10))

            # Find largest region
            labeled = label(texture_mask)
            regions = regionprops(labeled)

            if not regions:
                return None

            largest_region = max(regions, key=lambda r: r.area)
            minr, minc, maxr, maxc = largest_region.bbox

            # Extract with padding
            pad = 25
            minr = max(0, minr - pad)
            minc = max(0, minc - pad)
            maxr = min(gray.shape[0], maxr + pad)
            maxc = min(gray.shape[1], maxc + pad)

            roi = gray[minr:maxr, minc:maxc]
            mask_roi = (labeled == largest_region.label)[minr:maxr, minc:maxc]

            return mask_roi, roi

        except Exception as e:
            if self.debug:
                logger.debug(f"Texture variance detection failed: {e}")
            return None

    def _validate_fracture_surface(self, surface_mask: np.ndarray) -> bool:
        """Validate that detected region looks like a fracture surface."""
        if surface_mask is None or surface_mask.size == 0:
            return False

        # Check minimum size
        if np.sum(surface_mask) < 1000:  # Minimum 1000 pixels
            return False

        # Check aspect ratio (fracture surfaces are usually wider than tall)
        labeled = label(surface_mask.astype(int))
        regions = regionprops(labeled)

        if not regions:
            return False

        largest_region = max(regions, key=lambda r: r.area)

        # Check aspect ratio
        bbox_height = largest_region.bbox[2] - largest_region.bbox[0]
        bbox_width = largest_region.bbox[3] - largest_region.bbox[1]

        if bbox_width == 0 or bbox_height == 0:
            return False

        aspect_ratio = bbox_width / bbox_height

        # Fracture surfaces should be reasonably wide (not extremely thin or tall)
        if aspect_ratio < 0.3 or aspect_ratio > 5.0:
            return False

        return True