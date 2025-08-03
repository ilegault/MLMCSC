#!/usr/bin/env python3
"""
Main Feature Extraction Pipeline for Fracture Surface Analysis

This module coordinates feature extraction from YOLO-detected fracture surfaces,
combining multiple feature types into comprehensive feature vectors.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import time

from .texture_features import TextureFeatureExtractor
from .geometric_features import GeometricFeatureExtractor
from .statistical_features import StatisticalFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionResult:
    """Container for feature extraction results."""
    specimen_id: int
    bbox: List[float]  # [x, y, width, height]
    features: Dict[str, Any]
    feature_vector: np.ndarray
    extraction_time: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'specimen_id': self.specimen_id,
            'bbox': self.bbox,
            'features': self.features,
            'feature_vector': self.feature_vector.tolist(),
            'extraction_time': self.extraction_time,
            'timestamp': self.timestamp
        }


class FractureFeatureExtractor:
    """
    Main feature extraction pipeline for fracture surface analysis.
    
    Extracts comprehensive features from YOLO-detected fracture surface regions:
    - Texture features (GLCM, LBP, Gabor)
    - Geometric features (shape, contour analysis)
    - Statistical features (moments, histograms)
    - Surface roughness metrics
    - Crack density estimation
    - Frequency domain features
    """
    
    def __init__(self, 
                 enable_texture: bool = True,
                 enable_geometric: bool = True,
                 enable_statistical: bool = True,
                 preprocessing_config: Optional[Dict] = None):
        """
        Initialize the feature extractor.
        
        Args:
            enable_texture: Enable texture feature extraction
            enable_geometric: Enable geometric feature extraction
            enable_statistical: Enable statistical feature extraction
            preprocessing_config: Configuration for image preprocessing
        """
        self.enable_texture = enable_texture
        self.enable_geometric = enable_geometric
        self.enable_statistical = enable_statistical
        
        # Default preprocessing configuration
        self.preprocessing_config = preprocessing_config or {
            'normalize': True,
            'denoise': True,
            'enhance_contrast': True,
            'gaussian_blur_sigma': 0.5,
            'bilateral_filter': True
        }
        
        # Initialize feature extractors
        self.texture_extractor = TextureFeatureExtractor() if enable_texture else None
        self.geometric_extractor = GeometricFeatureExtractor() if enable_geometric else None
        self.statistical_extractor = StatisticalFeatureExtractor() if enable_statistical else None
        
        # Feature vector configuration
        self.feature_names = []
        self.feature_dimensions = {}
        self._initialize_feature_mapping()
        
        logger.info(f"FractureFeatureExtractor initialized")
        logger.info(f"Enabled extractors: Texture={enable_texture}, "
                   f"Geometric={enable_geometric}, Statistical={enable_statistical}")
    
    def _initialize_feature_mapping(self):
        """Initialize feature names and dimensions mapping."""
        self.feature_names = []
        self.feature_dimensions = {}
        
        if self.texture_extractor:
            texture_names = self.texture_extractor.get_feature_names()
            self.feature_names.extend([f"texture_{name}" for name in texture_names])
            self.feature_dimensions['texture'] = len(texture_names)
        
        if self.geometric_extractor:
            geometric_names = self.geometric_extractor.get_feature_names()
            self.feature_names.extend([f"geometric_{name}" for name in geometric_names])
            self.feature_dimensions['geometric'] = len(geometric_names)
        
        if self.statistical_extractor:
            statistical_names = self.statistical_extractor.get_feature_names()
            self.feature_names.extend([f"statistical_{name}" for name in statistical_names])
            self.feature_dimensions['statistical'] = len(statistical_names)
        
        logger.info(f"Total feature dimensions: {len(self.feature_names)}")
        logger.info(f"Feature breakdown: {self.feature_dimensions}")
    
    def preprocess_region(self, image_region: np.ndarray) -> np.ndarray:
        """
        Preprocess the fracture surface region for feature extraction.
        
        Args:
            image_region: Cropped fracture surface region
            
        Returns:
            Preprocessed image region
        """
        processed = image_region.copy()
        
        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Normalize intensity values
        if self.preprocessing_config.get('normalize', True):
            processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        
        # Denoise
        if self.preprocessing_config.get('denoise', True):
            if self.preprocessing_config.get('bilateral_filter', True):
                processed = cv2.bilateralFilter(processed, 9, 75, 75)
            else:
                processed = cv2.GaussianBlur(
                    processed, 
                    (5, 5), 
                    self.preprocessing_config.get('gaussian_blur_sigma', 0.5)
                )
        
        # Enhance contrast
        if self.preprocessing_config.get('enhance_contrast', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        return processed
    
    def extract_fracture_region(self, 
                              image: np.ndarray, 
                              bbox: List[float],
                              padding: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fracture surface region from image using YOLO bounding box.
        
        Args:
            image: Full input image
            bbox: YOLO bounding box [x, y, width, height]
            padding: Additional padding around bounding box
            
        Returns:
            Tuple of (cropped_region, region_mask)
        """
        x, y, w, h = bbox
        
        # Add padding and ensure bounds
        x1 = max(0, int(x - padding))
        y1 = max(0, int(y - padding))
        x2 = min(image.shape[1], int(x + w + padding))
        y2 = min(image.shape[0], int(y + h + padding))
        
        # Extract region
        region = image[y1:y2, x1:x2]
        
        # Create mask for the actual fracture surface (without padding)
        mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        mask_x1 = int(x - x1)
        mask_y1 = int(y - y1)
        mask_x2 = mask_x1 + int(w)
        mask_y2 = mask_y1 + int(h)
        
        mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
        
        return region, mask
    
    def extract_features(self, 
                        image: np.ndarray,
                        specimen_id: int,
                        bbox: List[float]) -> FeatureExtractionResult:
        """
        Extract comprehensive features from fracture surface region.
        
        Args:
            image: Full input image
            specimen_id: Unique identifier for the specimen
            bbox: YOLO bounding box [x, y, width, height]
            
        Returns:
            FeatureExtractionResult containing all extracted features
        """
        start_time = time.time()
        
        try:
            # Extract fracture surface region
            region, mask = self.extract_fracture_region(image, bbox)
            
            # Preprocess the region
            processed_region = self.preprocess_region(region)
            
            # Initialize feature containers
            all_features = {}
            feature_vectors = []
            
            # Extract texture features
            if self.texture_extractor:
                texture_features = self.texture_extractor.extract_features(
                    processed_region, mask
                )
                all_features['texture'] = texture_features
                feature_vectors.append(texture_features['feature_vector'])
            
            # Extract geometric features
            if self.geometric_extractor:
                geometric_features = self.geometric_extractor.extract_features(
                    processed_region, mask
                )
                all_features['geometric'] = geometric_features
                feature_vectors.append(geometric_features['feature_vector'])
            
            # Extract statistical features
            if self.statistical_extractor:
                statistical_features = self.statistical_extractor.extract_features(
                    processed_region, mask
                )
                all_features['statistical'] = statistical_features
                feature_vectors.append(statistical_features['feature_vector'])
            
            # Combine all feature vectors
            if feature_vectors:
                combined_vector = np.concatenate(feature_vectors)
            else:
                combined_vector = np.array([])
            
            # Calculate extraction time
            extraction_time = time.time() - start_time
            
            # Create result
            result = FeatureExtractionResult(
                specimen_id=specimen_id,
                bbox=bbox,
                features=all_features,
                feature_vector=combined_vector,
                extraction_time=extraction_time,
                timestamp=time.time()
            )
            
            logger.debug(f"Extracted {len(combined_vector)} features for specimen {specimen_id} "
                        f"in {extraction_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Feature extraction failed for specimen {specimen_id}: {e}")
            # Return empty result
            return FeatureExtractionResult(
                specimen_id=specimen_id,
                bbox=bbox,
                features={},
                feature_vector=np.array([]),
                extraction_time=time.time() - start_time,
                timestamp=time.time()
            )
    
    def extract_features_batch(self, 
                             image: np.ndarray,
                             detections: List[Dict]) -> List[FeatureExtractionResult]:
        """
        Extract features from multiple fracture surface detections in batch.
        
        Args:
            image: Full input image
            detections: List of detection dictionaries with 'specimen_id' and 'bbox'
            
        Returns:
            List of FeatureExtractionResult objects
        """
        results = []
        
        for detection in detections:
            result = self.extract_features(
                image=image,
                specimen_id=detection['specimen_id'],
                bbox=detection['bbox']
            )
            results.append(result)
        
        logger.info(f"Extracted features from {len(results)} fracture surfaces")
        return results
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names.copy()
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions breakdown."""
        return self.feature_dimensions.copy()
    
    def save_features(self, 
                     results: List[FeatureExtractionResult], 
                     output_path: Path) -> None:
        """
        Save extracted features to file.
        
        Args:
            results: List of feature extraction results
            output_path: Path to save features
        """
        import json
        
        # Convert results to serializable format
        data = {
            'feature_names': self.feature_names,
            'feature_dimensions': self.feature_dimensions,
            'extraction_results': [result.to_dict() for result in results],
            'metadata': {
                'total_specimens': len(results),
                'feature_vector_length': len(self.feature_names),
                'extraction_timestamp': time.time()
            }
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved features for {len(results)} specimens to {output_path}")
    
    def load_features(self, input_path: Path) -> List[FeatureExtractionResult]:
        """
        Load previously extracted features from file.
        
        Args:
            input_path: Path to load features from
            
        Returns:
            List of FeatureExtractionResult objects
        """
        import json
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        results = []
        for result_data in data['extraction_results']:
            result = FeatureExtractionResult(
                specimen_id=result_data['specimen_id'],
                bbox=result_data['bbox'],
                features=result_data['features'],
                feature_vector=np.array(result_data['feature_vector']),
                extraction_time=result_data['extraction_time'],
                timestamp=result_data['timestamp']
            )
            results.append(result)
        
        logger.info(f"Loaded features for {len(results)} specimens from {input_path}")
        return results