"""
Image preprocessing utilities for Charpy specimen analysis.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image preprocessing for Charpy specimen analysis."""
    
    def __init__(self):
        """Initialize image processor."""
        self.default_size = (512, 512)
        
    def resize_image(self, 
                    image: np.ndarray, 
                    target_size: Tuple[int, int],
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            np.ndarray: Resized image
        """
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas and center image
            if len(image.shape) == 3:
                canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            else:
                canvas = np.zeros((target_h, target_w), dtype=image.dtype)
            
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image values.
        
        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore', 'clahe')
            
        Returns:
            np.ndarray: Normalized image
        """
        if method == 'minmax':
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif method == 'zscore':
            mean = np.mean(image)
            std = np.std(image)
            normalized = (image - mean) / (std + 1e-8)
            return ((normalized + 3) / 6 * 255).clip(0, 255).astype(np.uint8)
        elif method == 'clahe':
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            np.ndarray: Enhanced image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def denoise_image(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median', 'nlm')
            
        Returns:
            np.ndarray: Denoised image
        """
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'nlm':
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen image using unsharp masking.
        
        Args:
            image: Input image
            strength: Sharpening strength (0.5-2.0)
            
        Returns:
            np.ndarray: Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    
    def extract_roi(self, 
                   image: np.ndarray, 
                   bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract region of interest from image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            np.ndarray: ROI image
        """
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    def detect_edges(self, 
                    image: np.ndarray, 
                    method: str = 'canny',
                    **kwargs) -> np.ndarray:
        """
        Detect edges in image.
        
        Args:
            image: Input image
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            **kwargs: Method-specific parameters
            
        Returns:
            np.ndarray: Edge image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'canny':
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            return cv2.Canny(gray, low_threshold, high_threshold)
        elif method == 'sobel':
            ksize = kwargs.get('ksize', 3)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        elif method == 'laplacian':
            ksize = kwargs.get('ksize', 3)
            return cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize).astype(np.uint8)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
    
    def segment_image(self, 
                     image: np.ndarray, 
                     method: str = 'threshold',
                     **kwargs) -> np.ndarray:
        """
        Segment image into regions.
        
        Args:
            image: Input image
            method: Segmentation method ('threshold', 'otsu', 'adaptive', 'watershed')
            **kwargs: Method-specific parameters
            
        Returns:
            np.ndarray: Segmented image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'threshold':
            threshold = kwargs.get('threshold', 127)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            return binary
        elif method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == 'adaptive':
            block_size = kwargs.get('block_size', 11)
            c = kwargs.get('c', 2)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, block_size, c)
        elif method == 'watershed':
            # Simplified watershed implementation
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(image, markers)
            return markers.astype(np.uint8)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def preprocess_for_classification(self, 
                                    image: np.ndarray,
                                    target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Complete preprocessing pipeline for classification.
        
        Args:
            image: Input image
            target_size: Target size for resizing
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if target_size is None:
            target_size = self.default_size
        
        # Resize image
        processed = self.resize_image(image, target_size)
        
        # Denoise
        processed = self.denoise_image(processed, method='bilateral')
        
        # Enhance contrast
        processed = self.normalize_image(processed, method='clahe')
        
        # Slight sharpening
        processed = self.sharpen_image(processed, strength=0.5)
        
        return processed