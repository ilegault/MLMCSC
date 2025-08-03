#!/usr/bin/env python3
"""
Statistical Feature Extraction for Fracture Surface Analysis

This module implements comprehensive statistical analysis including:
- Statistical moments (mean, std, skewness, kurtosis)
- Color histogram statistics
- Intensity distribution analysis
- Local statistical features
- Entropy and information theory metrics
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import entropy, kurtosis, skew
from skimage import filters, exposure
import warnings

logger = logging.getLogger(__name__)


class StatisticalFeatureExtractor:
    """Extracts comprehensive statistical features from fracture surfaces."""
    
    def __init__(self):
        """Initialize statistical feature extractor."""
        self.histogram_bins = 64
        self.local_window_sizes = [5, 9, 15]
        self.percentiles = [5, 10, 25, 50, 75, 90, 95]
        
        logger.debug("StatisticalFeatureExtractor initialized")
    
    def extract_basic_statistics(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract basic statistical moments and descriptive statistics.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of basic statistical features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                pixel_values = image[mask > 0]
            else:
                pixel_values = image.flatten()
            
            if len(pixel_values) == 0:
                # Return zero features if no pixels
                stat_keys = ['mean', 'std', 'var', 'min', 'max', 'range', 'median',
                           'skewness', 'kurtosis', 'cv', 'iqr', 'mad']
                for key in stat_keys:
                    features[f'stat_{key}'] = 0.0
                return features
            
            # Convert to float for calculations
            pixel_values = pixel_values.astype(np.float64)
            
            # Basic moments
            features['stat_mean'] = float(np.mean(pixel_values))
            features['stat_std'] = float(np.std(pixel_values))
            features['stat_var'] = float(np.var(pixel_values))
            features['stat_min'] = float(np.min(pixel_values))
            features['stat_max'] = float(np.max(pixel_values))
            features['stat_range'] = float(np.ptp(pixel_values))  # Peak-to-peak
            features['stat_median'] = float(np.median(pixel_values))
            
            # Higher order moments
            if len(pixel_values) > 1 and np.std(pixel_values) > 0:
                features['stat_skewness'] = float(skew(pixel_values))
                features['stat_kurtosis'] = float(kurtosis(pixel_values))
            else:
                features['stat_skewness'] = 0.0
                features['stat_kurtosis'] = 0.0
            
            # Coefficient of variation
            if features['stat_mean'] != 0:
                features['stat_cv'] = float(features['stat_std'] / abs(features['stat_mean']))
            else:
                features['stat_cv'] = 0.0
            
            # Interquartile range
            q75, q25 = np.percentile(pixel_values, [75, 25])
            features['stat_iqr'] = float(q75 - q25)
            
            # Median absolute deviation
            features['stat_mad'] = float(np.median(np.abs(pixel_values - features['stat_median'])))
            
            # Percentile features
            percentile_values = np.percentile(pixel_values, self.percentiles)
            for i, p in enumerate(self.percentiles):
                features[f'stat_p{p}'] = float(percentile_values[i])
            
        except Exception as e:
            logger.error(f"Basic statistics extraction failed: {e}")
            # Return zero features
            stat_keys = ['mean', 'std', 'var', 'min', 'max', 'range', 'median',
                        'skewness', 'kurtosis', 'cv', 'iqr', 'mad']
            for key in stat_keys:
                features[f'stat_{key}'] = 0.0
            for p in self.percentiles:
                features[f'stat_p{p}'] = 0.0
        
        return features
    
    def extract_histogram_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract histogram-based statistical features.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of histogram features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                pixel_values = image[mask > 0]
            else:
                pixel_values = image.flatten()
            
            if len(pixel_values) == 0:
                # Return zero features
                hist_keys = ['entropy', 'uniformity', 'energy', 'contrast', 'homogeneity']
                for key in hist_keys:
                    features[f'hist_{key}'] = 0.0
                for i in range(min(10, self.histogram_bins)):
                    features[f'hist_bin_{i}'] = 0.0
                return features
            
            # Compute histogram
            hist, bin_edges = np.histogram(pixel_values, bins=self.histogram_bins, 
                                         range=(0, 255), density=True)
            
            # Normalize histogram
            hist = hist / (np.sum(hist) + 1e-10)  # Add small value to avoid division by zero
            
            # Histogram entropy
            features['hist_entropy'] = float(entropy(hist + 1e-10))  # Add small value to avoid log(0)
            
            # Histogram uniformity (energy)
            features['hist_uniformity'] = float(np.sum(hist ** 2))
            features['hist_energy'] = features['hist_uniformity']  # Same as uniformity
            
            # Histogram contrast (measure of spread)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_intensity = np.sum(bin_centers * hist)
            features['hist_contrast'] = float(np.sum(((bin_centers - mean_intensity) ** 2) * hist))
            
            # Histogram homogeneity
            features['hist_homogeneity'] = float(np.sum(hist / (1 + np.abs(np.arange(len(hist)) - mean_intensity/255*len(hist)))))
            
            # First few histogram bins as features (to capture distribution shape)
            for i in range(min(10, len(hist))):
                features[f'hist_bin_{i}'] = float(hist[i])
            
            # Histogram statistics
            features['hist_peak_value'] = float(np.max(hist))
            features['hist_peak_position'] = float(np.argmax(hist))
            features['hist_valley_value'] = float(np.min(hist))
            features['hist_valley_position'] = float(np.argmin(hist))
            
            # Histogram shape analysis
            # Find peaks in histogram
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(hist, height=0.01)  # Find significant peaks
                features['hist_num_peaks'] = float(len(peaks))
                
                if len(peaks) > 0:
                    features['hist_primary_peak'] = float(peaks[np.argmax(hist[peaks])])
                else:
                    features['hist_primary_peak'] = 0.0
                    
            except ImportError:
                # Fallback if scipy.signal is not available
                features['hist_num_peaks'] = 1.0
                features['hist_primary_peak'] = float(np.argmax(hist))
            except Exception as e:
                logger.warning(f"Histogram peak detection failed: {e}")
                features['hist_num_peaks'] = 1.0
                features['hist_primary_peak'] = float(np.argmax(hist))
            
        except Exception as e:
            logger.error(f"Histogram feature extraction failed: {e}")
            # Return zero features
            hist_keys = ['entropy', 'uniformity', 'energy', 'contrast', 'homogeneity',
                        'peak_value', 'peak_position', 'valley_value', 'valley_position',
                        'num_peaks', 'primary_peak']
            for key in hist_keys:
                features[f'hist_{key}'] = 0.0
            for i in range(10):
                features[f'hist_bin_{i}'] = 0.0
        
        return features
    
    def extract_local_statistics(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract local statistical features using sliding windows.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of local statistical features
        """
        features = {}
        
        try:
            for window_size in self.local_window_sizes:
                try:
                    # Compute local mean
                    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
                    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
                    
                    # Compute local variance
                    local_mean_sq = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
                    local_var = local_mean_sq - local_mean**2
                    local_std = np.sqrt(np.maximum(local_var, 0))  # Ensure non-negative
                    
                    # Apply mask if provided
                    if mask is not None:
                        local_mean_masked = local_mean[mask > 0]
                        local_std_masked = local_std[mask > 0]
                        local_var_masked = local_var[mask > 0]
                    else:
                        local_mean_masked = local_mean.flatten()
                        local_std_masked = local_std.flatten()
                        local_var_masked = local_var.flatten()
                    
                    if len(local_mean_masked) == 0:
                        continue
                    
                    # Statistics of local means
                    features[f'local_mean_mean_w{window_size}'] = float(np.mean(local_mean_masked))
                    features[f'local_mean_std_w{window_size}'] = float(np.std(local_mean_masked))
                    features[f'local_mean_range_w{window_size}'] = float(np.ptp(local_mean_masked))
                    
                    # Statistics of local standard deviations
                    features[f'local_std_mean_w{window_size}'] = float(np.mean(local_std_masked))
                    features[f'local_std_std_w{window_size}'] = float(np.std(local_std_masked))
                    features[f'local_std_max_w{window_size}'] = float(np.max(local_std_masked))
                    
                    # Statistics of local variances
                    features[f'local_var_mean_w{window_size}'] = float(np.mean(local_var_masked))
                    features[f'local_var_std_w{window_size}'] = float(np.std(local_var_masked))
                    
                    # Local contrast measures
                    local_contrast = local_std_masked / (local_mean_masked + 1e-10)  # Avoid division by zero
                    features[f'local_contrast_mean_w{window_size}'] = float(np.mean(local_contrast))
                    features[f'local_contrast_std_w{window_size}'] = float(np.std(local_contrast))
                    
                except Exception as e:
                    logger.warning(f"Local statistics computation failed for window size {window_size}: {e}")
                    # Set zero features for this window size
                    local_keys = ['mean_mean', 'mean_std', 'mean_range', 'std_mean', 'std_std', 
                                'std_max', 'var_mean', 'var_std', 'contrast_mean', 'contrast_std']
                    for key in local_keys:
                        features[f'local_{key}_w{window_size}'] = 0.0
                        
        except Exception as e:
            logger.error(f"Local statistics extraction failed: {e}")
            # Return zero features for all window sizes
            for window_size in self.local_window_sizes:
                local_keys = ['mean_mean', 'mean_std', 'mean_range', 'std_mean', 'std_std', 
                            'std_max', 'var_mean', 'var_std', 'contrast_mean', 'contrast_std']
                for key in local_keys:
                    features[f'local_{key}_w{window_size}'] = 0.0
        
        return features
    
    def extract_color_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract color-based statistical features (works with both grayscale and color images).
        
        Args:
            image: Input image (grayscale or color)
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of color statistical features
        """
        features = {}
        
        try:
            # Handle both grayscale and color images
            if len(image.shape) == 3:
                # Color image - analyze each channel
                channels = cv2.split(image)
                channel_names = ['blue', 'green', 'red'] if image.shape[2] == 3 else [f'ch{i}' for i in range(image.shape[2])]
                
                for i, (channel, name) in enumerate(zip(channels, channel_names)):
                    # Apply mask if provided
                    if mask is not None:
                        channel_values = channel[mask > 0]
                    else:
                        channel_values = channel.flatten()
                    
                    if len(channel_values) == 0:
                        continue
                    
                    # Basic statistics for each channel
                    features[f'color_{name}_mean'] = float(np.mean(channel_values))
                    features[f'color_{name}_std'] = float(np.std(channel_values))
                    features[f'color_{name}_range'] = float(np.ptp(channel_values))
                    
                    # Channel histogram
                    hist, _ = np.histogram(channel_values, bins=32, range=(0, 255), density=True)
                    hist = hist / (np.sum(hist) + 1e-10)
                    features[f'color_{name}_entropy'] = float(entropy(hist + 1e-10))
                
                # Inter-channel relationships
                if len(channels) >= 3:
                    try:
                        # Apply mask to all channels
                        if mask is not None:
                            ch0_vals = channels[0][mask > 0].astype(np.float32)
                            ch1_vals = channels[1][mask > 0].astype(np.float32)
                            ch2_vals = channels[2][mask > 0].astype(np.float32)
                        else:
                            ch0_vals = channels[0].flatten().astype(np.float32)
                            ch1_vals = channels[1].flatten().astype(np.float32)
                            ch2_vals = channels[2].flatten().astype(np.float32)
                        
                        if len(ch0_vals) > 1:
                            # Color correlations
                            features['color_corr_01'] = float(np.corrcoef(ch0_vals, ch1_vals)[0, 1])
                            features['color_corr_02'] = float(np.corrcoef(ch0_vals, ch2_vals)[0, 1])
                            features['color_corr_12'] = float(np.corrcoef(ch1_vals, ch2_vals)[0, 1])
                            
                            # Color ratios
                            features['color_ratio_01'] = float(np.mean(ch0_vals) / (np.mean(ch1_vals) + 1e-10))
                            features['color_ratio_02'] = float(np.mean(ch0_vals) / (np.mean(ch2_vals) + 1e-10))
                            features['color_ratio_12'] = float(np.mean(ch1_vals) / (np.mean(ch2_vals) + 1e-10))
                        else:
                            features['color_corr_01'] = 0.0
                            features['color_corr_02'] = 0.0
                            features['color_corr_12'] = 0.0
                            features['color_ratio_01'] = 1.0
                            features['color_ratio_02'] = 1.0
                            features['color_ratio_12'] = 1.0
                            
                    except Exception as e:
                        logger.warning(f"Inter-channel analysis failed: {e}")
                        features['color_corr_01'] = 0.0
                        features['color_corr_02'] = 0.0
                        features['color_corr_12'] = 0.0
                        features['color_ratio_01'] = 1.0
                        features['color_ratio_02'] = 1.0
                        features['color_ratio_12'] = 1.0
                        
            else:
                # Grayscale image - treat as single channel
                if mask is not None:
                    gray_values = image[mask > 0]
                else:
                    gray_values = image.flatten()
                
                if len(gray_values) > 0:
                    features['color_gray_mean'] = float(np.mean(gray_values))
                    features['color_gray_std'] = float(np.std(gray_values))
                    features['color_gray_range'] = float(np.ptp(gray_values))
                    
                    # Grayscale histogram
                    hist, _ = np.histogram(gray_values, bins=32, range=(0, 255), density=True)
                    hist = hist / (np.sum(hist) + 1e-10)
                    features['color_gray_entropy'] = float(entropy(hist + 1e-10))
                else:
                    features['color_gray_mean'] = 0.0
                    features['color_gray_std'] = 0.0
                    features['color_gray_range'] = 0.0
                    features['color_gray_entropy'] = 0.0
                    
        except Exception as e:
            logger.error(f"Color feature extraction failed: {e}")
            # Return minimal features
            if len(image.shape) == 3:
                channel_names = ['blue', 'green', 'red']
                for name in channel_names:
                    features[f'color_{name}_mean'] = 0.0
                    features[f'color_{name}_std'] = 0.0
                    features[f'color_{name}_range'] = 0.0
                    features[f'color_{name}_entropy'] = 0.0
                features['color_corr_01'] = 0.0
                features['color_corr_02'] = 0.0
                features['color_corr_12'] = 0.0
                features['color_ratio_01'] = 1.0
                features['color_ratio_02'] = 1.0
                features['color_ratio_12'] = 1.0
            else:
                features['color_gray_mean'] = 0.0
                features['color_gray_std'] = 0.0
                features['color_gray_range'] = 0.0
                features['color_gray_entropy'] = 0.0
        
        return features
    
    def extract_distribution_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract features related to intensity distribution and information theory.
        
        Args:
            image: Grayscale image
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary of distribution features
        """
        features = {}
        
        try:
            # Apply mask if provided
            if mask is not None:
                pixel_values = image[mask > 0]
            else:
                pixel_values = image.flatten()
            
            if len(pixel_values) == 0:
                # Return zero features
                dist_keys = ['entropy_shannon', 'entropy_renyi', 'mutual_info', 'joint_entropy',
                           'gini_coefficient', 'concentration_ratio', 'diversity_index']
                for key in dist_keys:
                    features[f'dist_{key}'] = 0.0
                return features
            
            # Convert to probability distribution
            hist, _ = np.histogram(pixel_values, bins=self.histogram_bins, 
                                 range=(0, 255), density=True)
            prob_dist = hist / (np.sum(hist) + 1e-10)
            
            # Shannon entropy
            features['dist_entropy_shannon'] = float(entropy(prob_dist + 1e-10))
            
            # Rényi entropy (order 2)
            try:
                renyi_entropy = -np.log(np.sum(prob_dist ** 2))
                features['dist_entropy_renyi'] = float(renyi_entropy)
            except:
                features['dist_entropy_renyi'] = 0.0
            
            # Gini coefficient (measure of inequality)
            try:
                sorted_values = np.sort(pixel_values)
                n = len(sorted_values)
                cumsum = np.cumsum(sorted_values)
                gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                features['dist_gini_coefficient'] = float(gini)
            except:
                features['dist_gini_coefficient'] = 0.0
            
            # Concentration ratio (percentage of pixels in top 10% of intensities)
            try:
                threshold_90 = np.percentile(pixel_values, 90)
                high_intensity_pixels = np.sum(pixel_values >= threshold_90)
                features['dist_concentration_ratio'] = float(high_intensity_pixels / len(pixel_values))
            except:
                features['dist_concentration_ratio'] = 0.0
            
            # Diversity index (Simpson's diversity index)
            try:
                simpson_index = np.sum(prob_dist ** 2)
                diversity_index = 1 - simpson_index
                features['dist_diversity_index'] = float(diversity_index)
            except:
                features['dist_diversity_index'] = 0.0
            
            # Mutual information (simplified version using spatial correlation)
            try:
                if mask is not None and len(image.shape) == 2:
                    # Compute spatial mutual information
                    shifted_image = np.roll(image, 1, axis=0)  # Shift by one pixel
                    
                    # Apply mask to both images
                    original_masked = image[mask > 0]
                    shifted_masked = shifted_image[mask > 0]
                    
                    if len(original_masked) > 1:
                        # Compute joint histogram
                        joint_hist, _, _ = np.histogram2d(
                            original_masked, shifted_masked, 
                            bins=16, range=[[0, 255], [0, 255]], density=True
                        )
                        joint_hist = joint_hist / (np.sum(joint_hist) + 1e-10)
                        
                        # Marginal histograms
                        marginal_x = np.sum(joint_hist, axis=1)
                        marginal_y = np.sum(joint_hist, axis=0)
                        
                        # Mutual information
                        mutual_info = 0.0
                        for i in range(joint_hist.shape[0]):
                            for j in range(joint_hist.shape[1]):
                                if joint_hist[i, j] > 1e-10:
                                    mutual_info += joint_hist[i, j] * np.log(
                                        joint_hist[i, j] / (marginal_x[i] * marginal_y[j] + 1e-10)
                                    )
                        
                        features['dist_mutual_info'] = float(mutual_info)
                        
                        # Joint entropy
                        joint_entropy = entropy(joint_hist.flatten() + 1e-10)
                        features['dist_joint_entropy'] = float(joint_entropy)
                    else:
                        features['dist_mutual_info'] = 0.0
                        features['dist_joint_entropy'] = 0.0
                else:
                    features['dist_mutual_info'] = 0.0
                    features['dist_joint_entropy'] = 0.0
                    
            except Exception as e:
                logger.warning(f"Mutual information computation failed: {e}")
                features['dist_mutual_info'] = 0.0
                features['dist_joint_entropy'] = 0.0
                
        except Exception as e:
            logger.error(f"Distribution feature extraction failed: {e}")
            # Return zero features
            dist_keys = ['entropy_shannon', 'entropy_renyi', 'mutual_info', 'joint_entropy',
                        'gini_coefficient', 'concentration_ratio', 'diversity_index']
            for key in dist_keys:
                features[f'dist_{key}'] = 0.0
        
        return features
    
    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Extract all statistical features.
        
        Args:
            image: Input image (grayscale or color)
            mask: Optional mask to limit analysis region
            
        Returns:
            Dictionary containing all statistical features and feature vector
        """
        all_features = {}
        
        # Convert to grayscale if needed for most statistical analyses
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Extract different types of statistical features
        basic_stats = self.extract_basic_statistics(gray_image, mask)
        hist_features = self.extract_histogram_features(gray_image, mask)
        local_stats = self.extract_local_statistics(gray_image, mask)
        color_features = self.extract_color_features(image, mask)  # Use original image for color
        dist_features = self.extract_distribution_features(gray_image, mask)
        
        # Combine all features
        all_features.update(basic_stats)
        all_features.update(hist_features)
        all_features.update(local_stats)
        all_features.update(color_features)
        all_features.update(dist_features)
        
        # Create feature vector
        feature_vector = np.array(list(all_features.values()))
        
        return {
            'individual_features': all_features,
            'feature_vector': feature_vector,
            'feature_names': list(all_features.keys())
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all statistical feature names."""
        feature_names = []
        
        # Basic statistics
        stat_keys = ['mean', 'std', 'var', 'min', 'max', 'range', 'median',
                    'skewness', 'kurtosis', 'cv', 'iqr', 'mad']
        feature_names.extend([f'stat_{key}' for key in stat_keys])
        
        # Percentiles
        for p in self.percentiles:
            feature_names.append(f'stat_p{p}')
        
        # Histogram features
        hist_keys = ['entropy', 'uniformity', 'energy', 'contrast', 'homogeneity',
                    'peak_value', 'peak_position', 'valley_value', 'valley_position',
                    'num_peaks', 'primary_peak']
        feature_names.extend([f'hist_{key}' for key in hist_keys])
        
        # Histogram bins
        for i in range(10):
            feature_names.append(f'hist_bin_{i}')
        
        # Local statistics
        for window_size in self.local_window_sizes:
            local_keys = ['mean_mean', 'mean_std', 'mean_range', 'std_mean', 'std_std', 
                         'std_max', 'var_mean', 'var_std', 'contrast_mean', 'contrast_std']
            for key in local_keys:
                feature_names.append(f'local_{key}_w{window_size}')
        
        # Color features (assuming both grayscale and color possibilities)
        channel_names = ['blue', 'green', 'red', 'gray']
        for name in channel_names:
            color_keys = ['mean', 'std', 'range', 'entropy']
            for key in color_keys:
                feature_names.append(f'color_{name}_{key}')
        
        # Color correlations and ratios
        feature_names.extend([
            'color_corr_01', 'color_corr_02', 'color_corr_12',
            'color_ratio_01', 'color_ratio_02', 'color_ratio_12'
        ])
        
        # Distribution features
        dist_keys = ['entropy_shannon', 'entropy_renyi', 'mutual_info', 'joint_entropy',
                    'gini_coefficient', 'concentration_ratio', 'diversity_index']
        feature_names.extend([f'dist_{key}' for key in dist_keys])
        
        return feature_names