#!/usr/bin/env python3
"""
Texture Feature Extraction for Fracture Surface Analysis

This module implements comprehensive texture analysis including:
- Gray Level Co-occurrence Matrix (GLCM) features
- Local Binary Patterns (LBP)
- Gabor filter responses
- Fractal dimension analysis
- Surface roughness metrics
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from skimage import feature, filters, measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import ndimage, fft
from scipy.stats import entropy
import warnings

logger = logging.getLogger(__name__)


class TextureFeatureExtractor:
    """Extracts comprehensive texture features from fracture surfaces."""
    
    def __init__(self):
        """Initialize texture feature extractor."""
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, 45, 90, 135]  # degrees
        self.lbp_radius = 3
        self.lbp_n_points = 24
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.gabor_angles = [0, 45, 90, 135]  # degrees
        
        logger.debug("TextureFeatureExtractor initialized")
    
    def extract_glcm_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract Gray Level Co-occurrence Matrix (GLCM) features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of GLCM features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                masked_image = image.copy()
                masked_image[mask == 0] = 0
            else:
                masked_image = image
            
            # Reduce gray levels for GLCM computation (improves speed and reduces noise)
            gray_levels = 32
            normalized = cv2.normalize(masked_image, None, 0, gray_levels-1, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
            
            # Convert angles to radians
            angles_rad = [np.deg2rad(angle) for angle in self.glcm_angles]
            
            # Compute GLCM
            glcm = graycomatrix(
                normalized,
                distances=self.glcm_distances,
                angles=angles_rad,
                levels=gray_levels,
                symmetric=True,
                normed=True
            )
            
            # Extract GLCM properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            
            for prop in properties:
                try:
                    values = graycoprops(glcm, prop)
                    features[f'glcm_{prop}_mean'] = float(np.mean(values))
                    features[f'glcm_{prop}_std'] = float(np.std(values))
                    features[f'glcm_{prop}_max'] = float(np.max(values))
                    features[f'glcm_{prop}_min'] = float(np.min(values))
                except Exception as e:
                    logger.warning(f"Failed to compute GLCM {prop}: {e}")
                    features[f'glcm_{prop}_mean'] = 0.0
                    features[f'glcm_{prop}_std'] = 0.0
                    features[f'glcm_{prop}_max'] = 0.0
                    features[f'glcm_{prop}_min'] = 0.0
            
        except Exception as e:
            logger.error(f"GLCM feature extraction failed: {e}")
            # Return zero features
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in properties:
                features[f'glcm_{prop}_mean'] = 0.0
                features[f'glcm_{prop}_std'] = 0.0
                features[f'glcm_{prop}_max'] = 0.0
                features[f'glcm_{prop}_min'] = 0.0
        
        return features
    
    def extract_lbp_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract Local Binary Pattern (LBP) features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of LBP features
        """
        features = {}
        
        try:
            # Compute LBP
            lbp = local_binary_pattern(
                image, 
                self.lbp_n_points, 
                self.lbp_radius, 
                method='uniform'
            )
            
            # Apply mask if provided
            if mask is not None:
                lbp_masked = lbp[mask > 0]
            else:
                lbp_masked = lbp.flatten()
            
            # Compute histogram
            n_bins = self.lbp_n_points + 2  # +2 for uniform patterns
            hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float)
            
            # Normalize histogram
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            
            # Extract statistical features from histogram
            features['lbp_uniformity'] = float(np.sum(hist ** 2))
            features['lbp_entropy'] = float(entropy(hist + 1e-10))  # Add small value to avoid log(0)
            features['lbp_mean'] = float(np.mean(lbp_masked))
            features['lbp_std'] = float(np.std(lbp_masked))
            features['lbp_skewness'] = float(self._compute_skewness(lbp_masked))
            features['lbp_kurtosis'] = float(self._compute_kurtosis(lbp_masked))
            
            # Add histogram bins as features (first 10 bins to avoid too many features)
            for i in range(min(10, len(hist))):
                features[f'lbp_hist_bin_{i}'] = float(hist[i])
            
        except Exception as e:
            logger.error(f"LBP feature extraction failed: {e}")
            # Return zero features
            features['lbp_uniformity'] = 0.0
            features['lbp_entropy'] = 0.0
            features['lbp_mean'] = 0.0
            features['lbp_std'] = 0.0
            features['lbp_skewness'] = 0.0
            features['lbp_kurtosis'] = 0.0
            for i in range(10):
                features[f'lbp_hist_bin_{i}'] = 0.0
        
        return features
    
    def extract_gabor_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract Gabor filter response features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of Gabor features
        """
        features = {}
        
        try:
            responses = []
            
            for freq in self.gabor_frequencies:
                for angle in self.gabor_angles:
                    # Apply Gabor filter
                    real, _ = filters.gabor(image, frequency=freq, theta=np.deg2rad(angle))
                    
                    # Apply mask if provided
                    if mask is not None:
                        real_masked = real[mask > 0]
                    else:
                        real_masked = real.flatten()
                    
                    responses.append(real_masked)
                    
                    # Extract statistics from this filter response
                    features[f'gabor_f{freq}_a{angle}_mean'] = float(np.mean(real_masked))
                    features[f'gabor_f{freq}_a{angle}_std'] = float(np.std(real_masked))
                    features[f'gabor_f{freq}_a{angle}_energy'] = float(np.sum(real_masked ** 2))
            
            # Global Gabor statistics
            all_responses = np.concatenate(responses)
            features['gabor_global_mean'] = float(np.mean(all_responses))
            features['gabor_global_std'] = float(np.std(all_responses))
            features['gabor_global_energy'] = float(np.sum(all_responses ** 2))
            
        except Exception as e:
            logger.error(f"Gabor feature extraction failed: {e}")
            # Return zero features
            for freq in self.gabor_frequencies:
                for angle in self.gabor_angles:
                    features[f'gabor_f{freq}_a{angle}_mean'] = 0.0
                    features[f'gabor_f{freq}_a{angle}_std'] = 0.0
                    features[f'gabor_f{freq}_a{angle}_energy'] = 0.0
            features['gabor_global_mean'] = 0.0
            features['gabor_global_std'] = 0.0
            features['gabor_global_energy'] = 0.0
        
        return features
    
    def extract_fractal_dimension(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract fractal dimension using box-counting method.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary containing fractal dimension
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                masked_image = image.copy()
                masked_image[mask == 0] = 0
            else:
                masked_image = image
            
            # Convert to binary for fractal analysis
            threshold = filters.threshold_otsu(masked_image[masked_image > 0])
            binary = masked_image > threshold
            
            # Box-counting method
            scales = np.logspace(0.5, 3, num=10, dtype=int)  # Box sizes from ~3 to 1000
            counts = []
            
            for scale in scales:
                if scale >= min(binary.shape):
                    break
                    
                # Downsample image
                downsampled = measure.block_reduce(binary, (scale, scale), np.max)
                counts.append(np.sum(downsampled))
            
            # Fit line to log-log plot
            if len(counts) > 2:
                scales = scales[:len(counts)]
                log_scales = np.log(scales)
                log_counts = np.log(np.array(counts) + 1)  # +1 to avoid log(0)
                
                # Linear regression
                coeffs = np.polyfit(log_scales, log_counts, 1)
                fractal_dim = -coeffs[0]  # Negative slope gives fractal dimension
                
                features['fractal_dimension'] = float(fractal_dim)
                features['fractal_r_squared'] = float(np.corrcoef(log_scales, log_counts)[0, 1] ** 2)
            else:
                features['fractal_dimension'] = 0.0
                features['fractal_r_squared'] = 0.0
                
        except Exception as e:
            logger.error(f"Fractal dimension extraction failed: {e}")
            features['fractal_dimension'] = 0.0
            features['fractal_r_squared'] = 0.0
        
        return features
    
    def extract_roughness_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract surface roughness metrics.
        
        Args:
            image: Grayscale image (treated as height map)
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of roughness features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                surface = image[mask > 0]
            else:
                surface = image.flatten()
            
            if len(surface) == 0:
                raise ValueError("Empty surface after masking")
            
            # Basic roughness parameters
            mean_height = np.mean(surface)
            features['roughness_ra'] = float(np.mean(np.abs(surface - mean_height)))  # Average roughness
            features['roughness_rq'] = float(np.sqrt(np.mean((surface - mean_height) ** 2)))  # RMS roughness
            features['roughness_rz'] = float(np.max(surface) - np.min(surface))  # Peak-to-valley height
            features['roughness_rsk'] = float(self._compute_skewness(surface - mean_height))  # Skewness
            features['roughness_rku'] = float(self._compute_kurtosis(surface - mean_height))  # Kurtosis
            
            # Additional roughness metrics
            features['roughness_mean'] = float(mean_height)
            features['roughness_std'] = float(np.std(surface))
            features['roughness_range'] = float(np.ptp(surface))  # Peak-to-peak
            
            # Gradient-based roughness (if we have 2D surface)
            if mask is not None:
                try:
                    # Compute gradients
                    grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    
                    # Apply mask
                    gradient_masked = gradient_magnitude[mask > 0]
                    
                    features['roughness_gradient_mean'] = float(np.mean(gradient_masked))
                    features['roughness_gradient_std'] = float(np.std(gradient_masked))
                    features['roughness_gradient_max'] = float(np.max(gradient_masked))
                    
                except Exception as e:
                    logger.warning(f"Gradient roughness computation failed: {e}")
                    features['roughness_gradient_mean'] = 0.0
                    features['roughness_gradient_std'] = 0.0
                    features['roughness_gradient_max'] = 0.0
            else:
                features['roughness_gradient_mean'] = 0.0
                features['roughness_gradient_std'] = 0.0
                features['roughness_gradient_max'] = 0.0
                
        except Exception as e:
            logger.error(f"Roughness feature extraction failed: {e}")
            # Return zero features
            roughness_keys = ['roughness_ra', 'roughness_rq', 'roughness_rz', 'roughness_rsk', 
                            'roughness_rku', 'roughness_mean', 'roughness_std', 'roughness_range',
                            'roughness_gradient_mean', 'roughness_gradient_std', 'roughness_gradient_max']
            for key in roughness_keys:
                features[key] = 0.0
        
        return features
    
    def extract_frequency_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of frequency domain features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                masked_image = image.copy()
                masked_image[mask == 0] = 0
            else:
                masked_image = image
            
            # Compute 2D FFT
            f_transform = fft.fft2(masked_image)
            f_shift = fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Log transform for better visualization
            log_spectrum = np.log(magnitude_spectrum + 1)
            
            # Extract features from frequency domain
            features['fft_dc_component'] = float(magnitude_spectrum[magnitude_spectrum.shape[0]//2, 
                                                                  magnitude_spectrum.shape[1]//2])
            features['fft_mean_magnitude'] = float(np.mean(magnitude_spectrum))
            features['fft_std_magnitude'] = float(np.std(magnitude_spectrum))
            features['fft_max_magnitude'] = float(np.max(magnitude_spectrum))
            features['fft_energy'] = float(np.sum(magnitude_spectrum ** 2))
            
            # Radial frequency analysis
            center = (magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2)
            y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Compute radial profile
            r_max = int(np.min(center))
            radial_profile = []
            for radius in range(1, r_max, max(1, r_max//20)):  # Sample ~20 points
                mask_ring = (r >= radius) & (r < radius + 1)
                if np.any(mask_ring):
                    radial_profile.append(np.mean(magnitude_spectrum[mask_ring]))
            
            if radial_profile:
                features['fft_radial_mean'] = float(np.mean(radial_profile))
                features['fft_radial_std'] = float(np.std(radial_profile))
                features['fft_radial_slope'] = float(np.polyfit(range(len(radial_profile)), radial_profile, 1)[0])
            else:
                features['fft_radial_mean'] = 0.0
                features['fft_radial_std'] = 0.0
                features['fft_radial_slope'] = 0.0
                
        except Exception as e:
            logger.error(f"Frequency feature extraction failed: {e}")
            # Return zero features
            fft_keys = ['fft_dc_component', 'fft_mean_magnitude', 'fft_std_magnitude', 
                       'fft_max_magnitude', 'fft_energy', 'fft_radial_mean', 
                       'fft_radial_std', 'fft_radial_slope']
            for key in fft_keys:
                features[key] = 0.0
        
        return features
    
    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Extract all texture features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary containing all texture features and feature vector
        """
        all_features = {}
        
        # Extract different types of texture features
        glcm_features = self.extract_glcm_features(image, mask)
        lbp_features = self.extract_lbp_features(image, mask)
        gabor_features = self.extract_gabor_features(image, mask)
        fractal_features = self.extract_fractal_dimension(image, mask)
        roughness_features = self.extract_roughness_features(image, mask)
        frequency_features = self.extract_frequency_features(image, mask)
        
        # Combine all features
        all_features.update(glcm_features)
        all_features.update(lbp_features)
        all_features.update(gabor_features)
        all_features.update(fractal_features)
        all_features.update(roughness_features)
        all_features.update(frequency_features)
        
        # Create feature vector
        feature_vector = np.array(list(all_features.values()))
        
        return {
            'individual_features': all_features,
            'feature_vector': feature_vector,
            'feature_names': list(all_features.keys())
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all texture feature names."""
        # This is a comprehensive list of all possible texture features
        feature_names = []
        
        # GLCM features
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in properties:
            feature_names.extend([
                f'glcm_{prop}_mean', f'glcm_{prop}_std', 
                f'glcm_{prop}_max', f'glcm_{prop}_min'
            ])
        
        # LBP features
        feature_names.extend([
            'lbp_uniformity', 'lbp_entropy', 'lbp_mean', 'lbp_std', 
            'lbp_skewness', 'lbp_kurtosis'
        ])
        for i in range(10):
            feature_names.append(f'lbp_hist_bin_{i}')
        
        # Gabor features
        for freq in self.gabor_frequencies:
            for angle in self.gabor_angles:
                feature_names.extend([
                    f'gabor_f{freq}_a{angle}_mean',
                    f'gabor_f{freq}_a{angle}_std',
                    f'gabor_f{freq}_a{angle}_energy'
                ])
        feature_names.extend(['gabor_global_mean', 'gabor_global_std', 'gabor_global_energy'])
        
        # Fractal features
        feature_names.extend(['fractal_dimension', 'fractal_r_squared'])
        
        # Roughness features
        feature_names.extend([
            'roughness_ra', 'roughness_rq', 'roughness_rz', 'roughness_rsk', 
            'roughness_rku', 'roughness_mean', 'roughness_std', 'roughness_range',
            'roughness_gradient_mean', 'roughness_gradient_std', 'roughness_gradient_max'
        ])
        
        # Frequency features
        feature_names.extend([
            'fft_dc_component', 'fft_mean_magnitude', 'fft_std_magnitude', 
            'fft_max_magnitude', 'fft_energy', 'fft_radial_mean', 
            'fft_radial_std', 'fft_radial_slope'
        ])
        
        return feature_names
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis