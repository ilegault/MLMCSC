#!/usr/bin/env python3
"""
Geometric Feature Extraction for Fracture Surface Analysis

This module implements geometric and morphological feature extraction including:
- Shape descriptors (area, perimeter, compactness)
- Contour analysis (convexity, solidity)
- Crack density estimation
- Edge detection metrics
- Morphological features
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from skimage import measure, morphology, segmentation
from skimage.feature import canny
from skimage.morphology import skeletonize, disk
from scipy import ndimage
from scipy.spatial.distance import pdist
import warnings

logger = logging.getLogger(__name__)


class GeometricFeatureExtractor:
    """Extracts geometric and morphological features from fracture surfaces."""
    
    def __init__(self):
        """Initialize geometric feature extractor."""
        self.canny_sigma = 1.0
        self.canny_low_threshold = 0.1
        self.canny_high_threshold = 0.2
        self.min_crack_length = 10
        self.skeleton_disk_size = 2
        
        logger.debug("GeometricFeatureExtractor initialized")
    
    def extract_basic_shape_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract basic shape descriptors.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of basic shape features
        """
        features = {}
        
        try:
            # Create binary mask for shape analysis
            if mask is not None:
                binary_mask = mask > 0
            else:
                # Use Otsu thresholding to create binary mask
                threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
                binary_mask = image > threshold
            
            # Find contours
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                # Return zero features if no contours found
                shape_keys = ['area', 'perimeter', 'compactness', 'aspect_ratio', 'extent', 
                            'solidity', 'convexity', 'circularity', 'rectangularity']
                for key in shape_keys:
                    features[f'shape_{key}'] = 0.0
                return features
            
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Basic measurements
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['shape_area'] = float(area)
            features['shape_perimeter'] = float(perimeter)
            
            # Compactness (isoperimetric quotient)
            if perimeter > 0:
                features['shape_compactness'] = float(4 * np.pi * area / (perimeter ** 2))
            else:
                features['shape_compactness'] = 0.0
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            bounding_area = w * h
            
            # Aspect ratio
            if h > 0:
                features['shape_aspect_ratio'] = float(w / h)
            else:
                features['shape_aspect_ratio'] = 0.0
            
            # Extent (ratio of contour area to bounding rectangle area)
            if bounding_area > 0:
                features['shape_extent'] = float(area / bounding_area)
            else:
                features['shape_extent'] = 0.0
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            hull_perimeter = cv2.arcLength(hull, True)
            
            # Solidity (ratio of contour area to convex hull area)
            if hull_area > 0:
                features['shape_solidity'] = float(area / hull_area)
            else:
                features['shape_solidity'] = 0.0
            
            # Convexity (ratio of convex hull perimeter to contour perimeter)
            if perimeter > 0:
                features['shape_convexity'] = float(hull_perimeter / perimeter)
            else:
                features['shape_convexity'] = 0.0
            
            # Circularity
            if area > 0:
                features['shape_circularity'] = float(perimeter ** 2 / (4 * np.pi * area))
            else:
                features['shape_circularity'] = 0.0
            
            # Rectangularity (how well the shape fits its bounding rectangle)
            if bounding_area > 0:
                features['shape_rectangularity'] = float(area / bounding_area)
            else:
                features['shape_rectangularity'] = 0.0
                
        except Exception as e:
            logger.error(f"Basic shape feature extraction failed: {e}")
            # Return zero features
            shape_keys = ['area', 'perimeter', 'compactness', 'aspect_ratio', 'extent', 
                        'solidity', 'convexity', 'circularity', 'rectangularity']
            for key in shape_keys:
                features[f'shape_{key}'] = 0.0
        
        return features
    
    def extract_contour_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract advanced contour analysis features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of contour features
        """
        features = {}
        
        try:
            # Create binary mask
            if mask is not None:
                binary_mask = mask > 0
            else:
                threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
                binary_mask = image > threshold
            
            # Find contours
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                contour_keys = ['roughness', 'complexity', 'fractal_dimension', 'curvature_mean', 
                              'curvature_std', 'corner_count', 'inflection_points']
                for key in contour_keys:
                    features[f'contour_{key}'] = 0.0
                return features
            
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour roughness (deviation from convex hull)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            
            if hull_area > 0:
                features['contour_roughness'] = float(1 - (contour_area / hull_area))
            else:
                features['contour_roughness'] = 0.0
            
            # Contour complexity (number of vertices relative to perimeter)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                features['contour_complexity'] = float(len(largest_contour) / perimeter)
            else:
                features['contour_complexity'] = 0.0
            
            # Approximate contour fractal dimension using box-counting on contour
            try:
                contour_points = largest_contour.reshape(-1, 2)
                if len(contour_points) > 10:
                    # Simple fractal dimension estimation
                    distances = pdist(contour_points[::max(1, len(contour_points)//100)])  # Sample points
                    if len(distances) > 0:
                        log_distances = np.log(distances + 1e-10)
                        features['contour_fractal_dimension'] = float(np.std(log_distances))
                    else:
                        features['contour_fractal_dimension'] = 0.0
                else:
                    features['contour_fractal_dimension'] = 0.0
            except Exception as e:
                logger.warning(f"Contour fractal dimension computation failed: {e}")
                features['contour_fractal_dimension'] = 0.0
            
            # Curvature analysis
            try:
                if len(largest_contour) > 5:
                    # Approximate curvature using angle changes
                    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
                    curvatures = []
                    
                    for i in range(2, len(contour_points) - 2):
                        # Vectors to adjacent points
                        v1 = contour_points[i] - contour_points[i-2]
                        v2 = contour_points[i+2] - contour_points[i]
                        
                        # Normalize vectors
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)
                        
                        if v1_norm > 0 and v2_norm > 0:
                            v1 = v1 / v1_norm
                            v2 = v2 / v2_norm
                            
                            # Compute angle between vectors
                            dot_product = np.clip(np.dot(v1, v2), -1, 1)
                            angle = np.arccos(dot_product)
                            curvatures.append(angle)
                    
                    if curvatures:
                        features['contour_curvature_mean'] = float(np.mean(curvatures))
                        features['contour_curvature_std'] = float(np.std(curvatures))
                    else:
                        features['contour_curvature_mean'] = 0.0
                        features['contour_curvature_std'] = 0.0
                else:
                    features['contour_curvature_mean'] = 0.0
                    features['contour_curvature_std'] = 0.0
            except Exception as e:
                logger.warning(f"Curvature analysis failed: {e}")
                features['contour_curvature_mean'] = 0.0
                features['contour_curvature_std'] = 0.0
            
            # Corner detection using Harris corner detector
            try:
                # Create image from contour
                contour_img = np.zeros(image.shape, dtype=np.uint8)
                cv2.drawContours(contour_img, [largest_contour], -1, 255, 2)
                
                # Harris corner detection
                corners = cv2.cornerHarris(contour_img, 2, 3, 0.04)
                corner_count = np.sum(corners > 0.01 * corners.max())
                features['contour_corner_count'] = float(corner_count)
                
            except Exception as e:
                logger.warning(f"Corner detection failed: {e}")
                features['contour_corner_count'] = 0.0
            
            # Inflection points (points where curvature changes sign)
            try:
                if 'contour_curvature_mean' in features and len(largest_contour) > 10:
                    # Simplified inflection point detection
                    contour_smooth = cv2.approxPolyDP(largest_contour, 2, True)
                    features['contour_inflection_points'] = float(len(contour_smooth))
                else:
                    features['contour_inflection_points'] = 0.0
            except Exception as e:
                logger.warning(f"Inflection point detection failed: {e}")
                features['contour_inflection_points'] = 0.0
                
        except Exception as e:
            logger.error(f"Contour feature extraction failed: {e}")
            # Return zero features
            contour_keys = ['roughness', 'complexity', 'fractal_dimension', 'curvature_mean', 
                          'curvature_std', 'corner_count', 'inflection_points']
            for key in contour_keys:
                features[f'contour_{key}'] = 0.0
        
        return features
    
    def extract_crack_density_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract crack density and distribution features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of crack density features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                masked_image = image.copy()
                masked_image[mask == 0] = 0
            else:
                masked_image = image
            
            # Edge detection using Canny
            edges = canny(
                masked_image, 
                sigma=self.canny_sigma,
                low_threshold=self.canny_low_threshold,
                high_threshold=self.canny_high_threshold
            )
            
            # Skeletonize to get crack centerlines
            skeleton = skeletonize(edges)
            
            # Calculate crack density metrics
            total_pixels = np.sum(mask > 0) if mask is not None else image.size
            crack_pixels = np.sum(skeleton)
            
            if total_pixels > 0:
                features['crack_density'] = float(crack_pixels / total_pixels)
            else:
                features['crack_density'] = 0.0
            
            # Crack length estimation
            features['crack_total_length'] = float(crack_pixels)  # Approximation
            
            # Find connected crack segments
            labeled_skeleton = measure.label(skeleton)
            crack_segments = measure.regionprops(labeled_skeleton)
            
            features['crack_segment_count'] = float(len(crack_segments))
            
            if crack_segments:
                # Crack segment statistics
                segment_areas = [region.area for region in crack_segments]
                features['crack_segment_mean_length'] = float(np.mean(segment_areas))
                features['crack_segment_std_length'] = float(np.std(segment_areas))
                features['crack_segment_max_length'] = float(np.max(segment_areas))
                
                # Crack orientation analysis
                orientations = []
                for region in crack_segments:
                    if region.area > self.min_crack_length:  # Only consider significant cracks
                        orientations.append(region.orientation)
                
                if orientations:
                    features['crack_orientation_mean'] = float(np.mean(orientations))
                    features['crack_orientation_std'] = float(np.std(orientations))
                    
                    # Crack alignment (how aligned are the cracks)
                    orientation_variance = np.var(orientations)
                    features['crack_alignment'] = float(1.0 / (1.0 + orientation_variance))
                else:
                    features['crack_orientation_mean'] = 0.0
                    features['crack_orientation_std'] = 0.0
                    features['crack_alignment'] = 0.0
            else:
                features['crack_segment_mean_length'] = 0.0
                features['crack_segment_std_length'] = 0.0
                features['crack_segment_max_length'] = 0.0
                features['crack_orientation_mean'] = 0.0
                features['crack_orientation_std'] = 0.0
                features['crack_alignment'] = 0.0
            
            # Crack connectivity (branching analysis)
            # Use morphological operations to find junctions
            try:
                # Dilate skeleton slightly to find junctions
                dilated = morphology.dilation(skeleton, disk(2))
                junctions = dilated & ~skeleton
                junction_count = np.sum(junctions)
                
                features['crack_junction_count'] = float(junction_count)
                
                if crack_pixels > 0:
                    features['crack_branching_ratio'] = float(junction_count / crack_pixels)
                else:
                    features['crack_branching_ratio'] = 0.0
                    
            except Exception as e:
                logger.warning(f"Crack connectivity analysis failed: {e}")
                features['crack_junction_count'] = 0.0
                features['crack_branching_ratio'] = 0.0
                
        except Exception as e:
            logger.error(f"Crack density feature extraction failed: {e}")
            # Return zero features
            crack_keys = ['density', 'total_length', 'segment_count', 'segment_mean_length',
                         'segment_std_length', 'segment_max_length', 'orientation_mean',
                         'orientation_std', 'alignment', 'junction_count', 'branching_ratio']
            for key in crack_keys:
                features[f'crack_{key}'] = 0.0
        
        return features
    
    def extract_edge_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract edge detection and gradient-based features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of edge features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                masked_image = image.copy()
                masked_image[mask == 0] = 0
            else:
                masked_image = image
            
            # Sobel edge detection
            sobel_x = cv2.Sobel(masked_image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(masked_image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_direction = np.arctan2(sobel_y, sobel_x)
            
            # Apply mask to gradients
            if mask is not None:
                sobel_magnitude_masked = sobel_magnitude[mask > 0]
                sobel_direction_masked = sobel_direction[mask > 0]
            else:
                sobel_magnitude_masked = sobel_magnitude.flatten()
                sobel_direction_masked = sobel_direction.flatten()
            
            # Edge magnitude statistics
            features['edge_magnitude_mean'] = float(np.mean(sobel_magnitude_masked))
            features['edge_magnitude_std'] = float(np.std(sobel_magnitude_masked))
            features['edge_magnitude_max'] = float(np.max(sobel_magnitude_masked))
            features['edge_magnitude_energy'] = float(np.sum(sobel_magnitude_masked**2))
            
            # Edge direction statistics
            features['edge_direction_mean'] = float(np.mean(sobel_direction_masked))
            features['edge_direction_std'] = float(np.std(sobel_direction_masked))
            
            # Edge density (percentage of pixels with significant edges)
            edge_threshold = np.percentile(sobel_magnitude_masked, 75)  # Top 25% of gradients
            edge_pixels = np.sum(sobel_magnitude_masked > edge_threshold)
            total_pixels = len(sobel_magnitude_masked)
            
            if total_pixels > 0:
                features['edge_density'] = float(edge_pixels / total_pixels)
            else:
                features['edge_density'] = 0.0
            
            # Canny edge detection
            canny_edges = canny(
                masked_image,
                sigma=self.canny_sigma,
                low_threshold=self.canny_low_threshold,
                high_threshold=self.canny_high_threshold
            )
            
            # Apply mask to Canny edges
            if mask is not None:
                canny_edges_masked = canny_edges[mask > 0]
            else:
                canny_edges_masked = canny_edges.flatten()
            
            # Canny edge statistics
            canny_edge_pixels = np.sum(canny_edges_masked)
            if total_pixels > 0:
                features['canny_edge_density'] = float(canny_edge_pixels / total_pixels)
            else:
                features['canny_edge_density'] = 0.0
            
            features['canny_edge_count'] = float(canny_edge_pixels)
            
            # Laplacian edge detection (second derivative)
            laplacian = cv2.Laplacian(masked_image.astype(np.float32), cv2.CV_32F)
            
            if mask is not None:
                laplacian_masked = laplacian[mask > 0]
            else:
                laplacian_masked = laplacian.flatten()
            
            features['laplacian_mean'] = float(np.mean(np.abs(laplacian_masked)))
            features['laplacian_std'] = float(np.std(laplacian_masked))
            features['laplacian_energy'] = float(np.sum(laplacian_masked**2))
            
        except Exception as e:
            logger.error(f"Edge feature extraction failed: {e}")
            # Return zero features
            edge_keys = ['magnitude_mean', 'magnitude_std', 'magnitude_max', 'magnitude_energy',
                        'direction_mean', 'direction_std', 'density', 'canny_edge_density',
                        'canny_edge_count', 'laplacian_mean', 'laplacian_std', 'laplacian_energy']
            for key in edge_keys:
                features[f'edge_{key}'] = 0.0
        
        return features
    
    def extract_morphological_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract morphological features using mathematical morphology.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                binary_mask = mask > 0
            else:
                threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
                binary_mask = image > threshold
            
            # Morphological operations
            kernel_sizes = [3, 5, 7]
            
            for kernel_size in kernel_sizes:
                kernel = disk(kernel_size // 2)
                
                # Opening (erosion followed by dilation)
                opened = morphology.opening(binary_mask, kernel)
                opening_diff = np.sum(binary_mask) - np.sum(opened)
                features[f'morph_opening_diff_k{kernel_size}'] = float(opening_diff)
                
                # Closing (dilation followed by erosion)
                closed = morphology.closing(binary_mask, kernel)
                closing_diff = np.sum(closed) - np.sum(binary_mask)
                features[f'morph_closing_diff_k{kernel_size}'] = float(closing_diff)
                
                # Top-hat (original - opening)
                tophat = binary_mask.astype(int) - opened.astype(int)
                features[f'morph_tophat_sum_k{kernel_size}'] = float(np.sum(tophat))
                
                # Black-hat (closing - original)
                blackhat = closed.astype(int) - binary_mask.astype(int)
                features[f'morph_blackhat_sum_k{kernel_size}'] = float(np.sum(blackhat))
            
            # Granulometry (pattern spectrum)
            try:
                granulometry_sizes = [1, 2, 3, 5, 7]
                granulometry_values = []
                
                for size in granulometry_sizes:
                    kernel = disk(size)
                    opened = morphology.opening(binary_mask, kernel)
                    granulometry_values.append(np.sum(opened))
                
                # Compute granulometry features
                if len(granulometry_values) > 1:
                    # Pattern spectrum (derivative of granulometry)
                    pattern_spectrum = np.diff(granulometry_values)
                    features['morph_pattern_spectrum_mean'] = float(np.mean(pattern_spectrum))
                    features['morph_pattern_spectrum_std'] = float(np.std(pattern_spectrum))
                    features['morph_pattern_spectrum_max'] = float(np.max(pattern_spectrum))
                else:
                    features['morph_pattern_spectrum_mean'] = 0.0
                    features['morph_pattern_spectrum_std'] = 0.0
                    features['morph_pattern_spectrum_max'] = 0.0
                    
            except Exception as e:
                logger.warning(f"Granulometry computation failed: {e}")
                features['morph_pattern_spectrum_mean'] = 0.0
                features['morph_pattern_spectrum_std'] = 0.0
                features['morph_pattern_spectrum_max'] = 0.0
            
            # Euler number (topological feature)
            try:
                euler_number = measure.euler_number(binary_mask)
                features['morph_euler_number'] = float(euler_number)
            except Exception as e:
                logger.warning(f"Euler number computation failed: {e}")
                features['morph_euler_number'] = 0.0
                
        except Exception as e:
            logger.error(f"Morphological feature extraction failed: {e}")
            # Return zero features
            morph_keys = []
            for kernel_size in [3, 5, 7]:
                morph_keys.extend([
                    f'opening_diff_k{kernel_size}', f'closing_diff_k{kernel_size}',
                    f'tophat_sum_k{kernel_size}', f'blackhat_sum_k{kernel_size}'
                ])
            morph_keys.extend(['pattern_spectrum_mean', 'pattern_spectrum_std', 
                             'pattern_spectrum_max', 'euler_number'])
            
            for key in morph_keys:
                features[f'morph_{key}'] = 0.0
        
        return features
    
    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Extract all geometric features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary containing all geometric features and feature vector
        """
        all_features = {}
        
        # Extract different types of geometric features
        shape_features = self.extract_basic_shape_features(image, mask)
        contour_features = self.extract_contour_features(image, mask)
        crack_features = self.extract_crack_density_features(image, mask)
        edge_features = self.extract_edge_features(image, mask)
        morph_features = self.extract_morphological_features(image, mask)
        
        # Combine all features
        all_features.update(shape_features)
        all_features.update(contour_features)
        all_features.update(crack_features)
        all_features.update(edge_features)
        all_features.update(morph_features)
        
        # Create feature vector
        feature_vector = np.array(list(all_features.values()))
        
        return {
            'individual_features': all_features,
            'feature_vector': feature_vector,
            'feature_names': list(all_features.keys())
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all geometric feature names."""
        feature_names = []
        
        # Shape features
        shape_keys = ['area', 'perimeter', 'compactness', 'aspect_ratio', 'extent', 
                     'solidity', 'convexity', 'circularity', 'rectangularity']
        feature_names.extend([f'shape_{key}' for key in shape_keys])
        
        # Contour features
        contour_keys = ['roughness', 'complexity', 'fractal_dimension', 'curvature_mean', 
                       'curvature_std', 'corner_count', 'inflection_points']
        feature_names.extend([f'contour_{key}' for key in contour_keys])
        
        # Crack features
        crack_keys = ['density', 'total_length', 'segment_count', 'segment_mean_length',
                     'segment_std_length', 'segment_max_length', 'orientation_mean',
                     'orientation_std', 'alignment', 'junction_count', 'branching_ratio']
        feature_names.extend([f'crack_{key}' for key in crack_keys])
        
        # Edge features
        edge_keys = ['magnitude_mean', 'magnitude_std', 'magnitude_max', 'magnitude_energy',
                    'direction_mean', 'direction_std', 'density', 'canny_edge_density',
                    'canny_edge_count', 'laplacian_mean', 'laplacian_std', 'laplacian_energy']
        feature_names.extend([f'edge_{key}' for key in edge_keys])
        
        # Morphological features
        for kernel_size in [3, 5, 7]:
            feature_names.extend([
                f'morph_opening_diff_k{kernel_size}', f'morph_closing_diff_k{kernel_size}',
                f'morph_tophat_sum_k{kernel_size}', f'morph_blackhat_sum_k{kernel_size}'
            ])
        feature_names.extend([
            'morph_pattern_spectrum_mean', 'morph_pattern_spectrum_std', 
            'morph_pattern_spectrum_max', 'morph_euler_number'
        ])
        
        return feature_names