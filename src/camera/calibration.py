"""
Camera calibration utilities for microscope setup.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import logging

logger = logging.getLogger(__name__)


class CameraCalibration:
    """Camera calibration for microscope imaging."""
    
    def __init__(self):
        """Initialize calibration parameters."""
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coeffs: Optional[np.ndarray] = None
        self.calibration_error: float = 0.0
        self.pixel_to_micron_ratio: float = 1.0
        
    def calibrate_camera(self, 
                        calibration_images: List[np.ndarray],
                        chessboard_size: Tuple[int, int] = (9, 6),
                        square_size: float = 1.0) -> bool:
        """
        Calibrate camera using chessboard pattern.
        
        Args:
            calibration_images: List of calibration images
            chessboard_size: Chessboard pattern size (corners)
            square_size: Size of chessboard squares in real units
            
        Returns:
            bool: True if calibration successful
        """
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        if len(objpoints) < 3:
            logger.error("Not enough valid calibration images found")
            return False
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            self.camera_matrix = mtx
            self.distortion_coeffs = dist
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            self.calibration_error = total_error / len(objpoints)
            logger.info(f"Camera calibration completed. Mean error: {self.calibration_error}")
            return True
        else:
            logger.error("Camera calibration failed")
            return False
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort image using calibration parameters.
        
        Args:
            image: Input distorted image
            
        Returns:
            np.ndarray: Undistorted image
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            logger.warning("Camera not calibrated, returning original image")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
    
    def set_pixel_scale(self, known_distance_pixels: float, known_distance_microns: float) -> None:
        """
        Set pixel to micron conversion ratio.
        
        Args:
            known_distance_pixels: Distance in pixels
            known_distance_microns: Corresponding distance in microns
        """
        self.pixel_to_micron_ratio = known_distance_microns / known_distance_pixels
        logger.info(f"Pixel scale set: {self.pixel_to_micron_ratio} microns/pixel")
    
    def pixels_to_microns(self, pixels: float) -> float:
        """
        Convert pixels to microns.
        
        Args:
            pixels: Distance in pixels
            
        Returns:
            float: Distance in microns
        """
        return pixels * self.pixel_to_micron_ratio
    
    def microns_to_pixels(self, microns: float) -> float:
        """
        Convert microns to pixels.
        
        Args:
            microns: Distance in microns
            
        Returns:
            float: Distance in pixels
        """
        return microns / self.pixel_to_micron_ratio
    
    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration parameters to file.
        
        Args:
            filepath: Path to save calibration file
            
        Returns:
            bool: True if saved successfully
        """
        try:
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                'distortion_coeffs': self.distortion_coeffs.tolist() if self.distortion_coeffs is not None else None,
                'calibration_error': self.calibration_error,
                'pixel_to_micron_ratio': self.pixel_to_micron_ratio
            }
            
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration parameters from file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            if calibration_data['camera_matrix'] is not None:
                self.camera_matrix = np.array(calibration_data['camera_matrix'])
            if calibration_data['distortion_coeffs'] is not None:
                self.distortion_coeffs = np.array(calibration_data['distortion_coeffs'])
            
            self.calibration_error = calibration_data.get('calibration_error', 0.0)
            self.pixel_to_micron_ratio = calibration_data.get('pixel_to_micron_ratio', 1.0)
            
            logger.info(f"Calibration loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False