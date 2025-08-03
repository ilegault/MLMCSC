#!/usr/bin/env python3
"""
Initial Shear Percentage Model Training Script

This script trains the initial online learning model using manually labeled
shear percentage samples. It extracts features from the labeled images and
sets up the baseline model for incremental learning.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.mlmcsc.feature_extraction import FractureFeatureExtractor
from src.mlmcsc.regression import OnlineLearningSystem
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('initial_model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class InitialShearModelTrainer:
    """Trainer for the initial shear percentage prediction model."""
    
    def __init__(self, 
                 yolo_model_path: str,
                 training_data_path: str,
                 output_dir: str = "src/models/shear_prediction"):
        """
        Initialize the trainer.
        
        Args:
            yolo_model_path: Path to trained YOLO model for fracture detection
            training_data_path: Path to manually labeled training data
            output_dir: Directory to save trained models
        """
        self.yolo_model_path = Path(yolo_model_path)
        self.training_data_path = Path(training_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.yolo_model = None
        self.feature_extractor = None
        self.online_learner = None
        
        # Training data
        self.training_samples = []
        self.extracted_features = []
        self.target_values = []
        
        logger.info(f"InitialShearModelTrainer initialized")
        logger.info(f"YOLO model: {self.yolo_model_path}")
        logger.info(f"Training data: {self.training_data_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_yolo_model(self):
        """Load the trained YOLO model for fracture surface detection."""
        try:
            logger.info("Loading YOLO model...")
            self.yolo_model = YOLO(str(self.yolo_model_path))
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def initialize_feature_extractor(self):
        """Initialize the feature extraction pipeline."""
        try:
            logger.info("Initializing feature extractor...")
            self.feature_extractor = FractureFeatureExtractor(
                enable_texture=True,
                enable_geometric=True,
                enable_statistical=True
            )
            logger.info("Feature extractor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {e}")
            raise
    
    def load_training_samples(self):
        """Load manually labeled training samples from directory structure."""
        try:
            logger.info("Loading training samples...")
            
            # Get all shear percentage directories
            shear_dirs = [d for d in self.training_data_path.iterdir() if d.is_dir()]
            shear_dirs.sort()  # Sort to ensure consistent order
            
            for shear_dir in shear_dirs:
                # Extract shear percentage from directory name
                shear_name = shear_dir.name
                try:
                    shear_percentage = float(shear_name.replace('_percent', ''))
                except ValueError:
                    logger.warning(f"Could not parse shear percentage from {shear_name}")
                    continue
                
                # Get all image files in this directory
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(shear_dir.glob(ext))
                
                logger.info(f"Found {len(image_files)} images for {shear_percentage}% shear")
                
                for image_file in image_files:
                    self.training_samples.append({
                        'image_path': image_file,
                        'shear_percentage': shear_percentage,
                        'specimen_id': len(self.training_samples)
                    })
            
            logger.info(f"Loaded {len(self.training_samples)} training samples")
            
            # Print distribution
            shear_counts = {}
            for sample in self.training_samples:
                shear = sample['shear_percentage']
                shear_counts[shear] = shear_counts.get(shear, 0) + 1
            
            logger.info("Shear percentage distribution:")
            for shear in sorted(shear_counts.keys()):
                logger.info(f"  {shear}%: {shear_counts[shear]} samples")
                
        except Exception as e:
            logger.error(f"Failed to load training samples: {e}")
            raise
    
    def detect_fracture_surfaces(self):
        """Use YOLO model to detect fracture surfaces in training images."""
        try:
            logger.info("Detecting fracture surfaces in training images...")
            
            successful_detections = 0
            failed_detections = 0
            
            for i, sample in enumerate(tqdm(self.training_samples, desc="Detecting fracture surfaces")):
                try:
                    # Load image
                    image_path = sample['image_path']
                    image = cv2.imread(str(image_path))
                    
                    if image is None:
                        logger.warning(f"Could not load image: {image_path}")
                        failed_detections += 1
                        continue
                    
                    # Run YOLO detection
                    results = self.yolo_model(image, verbose=False)
                    
                    # Find fracture surface detection (class 2)
                    fracture_detections = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                if int(box.cls) == 2:  # fracture_surface class
                                    # Convert to [x, y, width, height] format
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    bbox = [x1, y1, x2-x1, y2-y1]
                                    confidence = float(box.conf)
                                    fracture_detections.append({
                                        'bbox': bbox,
                                        'confidence': confidence
                                    })
                    
                    if fracture_detections:
                        # Use the detection with highest confidence
                        best_detection = max(fracture_detections, key=lambda x: x['confidence'])
                        sample['fracture_bbox'] = best_detection['bbox']
                        sample['detection_confidence'] = best_detection['confidence']
                        successful_detections += 1
                    else:
                        logger.warning(f"No fracture surface detected in {image_path}")
                        # Use full image as fallback
                        h, w = image.shape[:2]
                        sample['fracture_bbox'] = [0, 0, w, h]
                        sample['detection_confidence'] = 0.0
                        failed_detections += 1
                        
                except Exception as e:
                    logger.error(f"Detection failed for {sample['image_path']}: {e}")
                    failed_detections += 1
                    continue
            
            logger.info(f"Fracture surface detection completed:")
            logger.info(f"  Successful detections: {successful_detections}")
            logger.info(f"  Failed detections: {failed_detections}")
            
        except Exception as e:
            logger.error(f"Fracture surface detection failed: {e}")
            raise
    
    def extract_features(self):
        """Extract features from detected fracture surfaces."""
        try:
            logger.info("Extracting features from fracture surfaces...")
            
            successful_extractions = 0
            failed_extractions = 0
            
            for sample in tqdm(self.training_samples, desc="Extracting features"):
                try:
                    # Load image
                    image = cv2.imread(str(sample['image_path']))
                    if image is None:
                        failed_extractions += 1
                        continue
                    
                    # Extract features using detected bounding box
                    result = self.feature_extractor.extract_features(
                        image=image,
                        specimen_id=sample['specimen_id'],
                        bbox=sample['fracture_bbox']
                    )
                    
                    # Store results
                    self.extracted_features.append(result)
                    self.target_values.append(sample['shear_percentage'])
                    successful_extractions += 1
                    
                except Exception as e:
                    logger.error(f"Feature extraction failed for {sample['image_path']}: {e}")
                    failed_extractions += 1
                    continue
            
            logger.info(f"Feature extraction completed:")
            logger.info(f"  Successful extractions: {successful_extractions}")
            logger.info(f"  Failed extractions: {failed_extractions}")
            logger.info(f"  Feature vector length: {len(self.extracted_features[0].feature_vector) if self.extracted_features else 0}")
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def train_initial_models(self):
        """Train multiple online learning models and select the best one."""
        try:
            logger.info("Training initial online learning models...")
            
            # Prepare feature data
            feature_vectors = [result.feature_vector for result in self.extracted_features]
            feature_names = self.extracted_features[0].feature_names if self.extracted_features else []
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                feature_vectors, self.target_values, 
                test_size=0.2, random_state=42, stratify=None
            )
            
            # Test different online learning models
            model_configs = [
                {'model_type': 'sgd', 'learning_rate': 'constant', 'alpha': 0.0001},
                {'model_type': 'sgd', 'learning_rate': 'adaptive', 'alpha': 0.001},
                {'model_type': 'passive_aggressive', 'alpha': 1.0},
                {'model_type': 'mlp', 'alpha': 0.0001}
            ]
            
            best_model = None
            best_performance = -np.inf
            model_results = []
            
            for config in model_configs:
                try:
                    logger.info(f"Training model: {config}")
                    
                    # Initialize online learner
                    learner = OnlineLearningSystem(
                        target_property='shear_percentage',
                        **config
                    )
                    
                    # Initialize with training data
                    init_performance = learner.initialize_model(
                        feature_data=[{'feature_vector': fv, 'feature_names': feature_names} 
                                    for fv in X_train],
                        target_values=y_train,
                        feature_names=feature_names
                    )
                    
                    # Evaluate on test set
                    test_predictions = learner.predict(
                        [{'feature_vector': fv, 'feature_names': feature_names} 
                         for fv in X_test]
                    )
                    
                    # Calculate test performance
                    test_r2 = r2_score(y_test, test_predictions)
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                    test_mae = mean_absolute_error(y_test, test_predictions)
                    
                    # Cross-validation on full dataset
                    cv_scores = []
                    try:
                        # Simple cross-validation for online models
                        from sklearn.model_selection import KFold
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        
                        for train_idx, val_idx in kf.split(feature_vectors):
                            cv_learner = OnlineLearningSystem(target_property='shear_percentage', **config)
                            cv_X_train = [feature_vectors[i] for i in train_idx]
                            cv_y_train = [self.target_values[i] for i in train_idx]
                            cv_X_val = [feature_vectors[i] for i in val_idx]
                            cv_y_val = [self.target_values[i] for i in val_idx]
                            
                            cv_learner.initialize_model(
                                [{'feature_vector': fv, 'feature_names': feature_names} for fv in cv_X_train],
                                cv_y_train,
                                feature_names
                            )
                            
                            cv_pred = cv_learner.predict(
                                [{'feature_vector': fv, 'feature_names': feature_names} for fv in cv_X_val]
                            )
                            
                            cv_r2 = r2_score(cv_y_val, cv_pred)
                            cv_scores.append(cv_r2)
                            
                    except Exception as e:
                        logger.warning(f"Cross-validation failed for {config}: {e}")
                        cv_scores = [test_r2]  # Fallback to test score
                    
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                    result = {
                        'config': config,
                        'init_performance': init_performance,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'model': learner
                    }
                    
                    model_results.append(result)
                    
                    logger.info(f"Model performance:")
                    logger.info(f"  Test R²: {test_r2:.3f}")
                    logger.info(f"  Test RMSE: {test_rmse:.3f}")
                    logger.info(f"  CV R²: {cv_mean:.3f} ± {cv_std:.3f}")
                    
                    # Select best model based on CV performance
                    if cv_mean > best_performance:
                        best_performance = cv_mean
                        best_model = learner
                        
                except Exception as e:
                    logger.error(f"Training failed for config {config}: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("No models trained successfully")
            
            self.online_learner = best_model
            
            # Save results
            results_file = self.output_dir / 'initial_training_results.json'
            with open(results_file, 'w') as f:
                # Convert results to serializable format
                serializable_results = []
                for result in model_results:
                    serializable_result = result.copy()
                    del serializable_result['model']  # Remove non-serializable model
                    serializable_results.append(serializable_result)
                
                json.dump({
                    'training_timestamp': datetime.now().isoformat(),
                    'total_samples': len(self.training_samples),
                    'successful_features': len(self.extracted_features),
                    'feature_vector_length': len(feature_names),
                    'model_results': serializable_results,
                    'best_model_config': best_model.get_model_info()
                }, f, indent=2)
            
            logger.info(f"Initial model training completed")
            logger.info(f"Best model performance: R² = {best_performance:.3f}")
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Initial model training failed: {e}")
            raise
    
    def save_models_and_data(self):
        """Save trained models and extracted features."""
        try:
            logger.info("Saving models and data...")
            
            # Save online learning model
            model_file = self.output_dir / 'initial_shear_model.joblib'
            self.online_learner.save_model(model_file)
            
            # Save feature extractor configuration
            extractor_config = {
                'feature_names': self.feature_extractor.get_feature_names(),
                'feature_dimensions': self.feature_extractor.get_feature_dimensions(),
                'preprocessing_config': self.feature_extractor.preprocessing_config
            }
            
            config_file = self.output_dir / 'feature_extractor_config.json'
            with open(config_file, 'w') as f:
                json.dump(extractor_config, f, indent=2)
            
            # Save extracted features for future use
            features_file = self.output_dir / 'initial_training_features.json'
            self.feature_extractor.save_features(self.extracted_features, features_file)
            
            # Save training metadata
            metadata = {
                'training_timestamp': datetime.now().isoformat(),
                'yolo_model_path': str(self.yolo_model_path),
                'training_data_path': str(self.training_data_path),
                'total_samples': len(self.training_samples),
                'successful_extractions': len(self.extracted_features),
                'model_info': self.online_learner.get_model_info(),
                'sample_distribution': {}
            }
            
            # Add sample distribution
            for sample in self.training_samples:
                shear = sample['shear_percentage']
                metadata['sample_distribution'][str(shear)] = metadata['sample_distribution'].get(str(shear), 0) + 1
            
            metadata_file = self.output_dir / 'training_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Models and data saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models and data: {e}")
            raise
    
    def create_visualizations(self):
        """Create visualizations of training results."""
        try:
            logger.info("Creating visualizations...")
            
            # Create plots directory
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # 1. Shear percentage distribution
            plt.figure(figsize=(10, 6))
            shear_values = [sample['shear_percentage'] for sample in self.training_samples]
            plt.hist(shear_values, bins=10, alpha=0.7, edgecolor='black')
            plt.xlabel('Shear Percentage (%)')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of Training Samples by Shear Percentage')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'shear_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Prediction vs Actual scatter plot
            if self.online_learner and len(self.extracted_features) > 0:
                feature_data = [{'feature_vector': result.feature_vector, 
                               'feature_names': result.feature_names} 
                              for result in self.extracted_features]
                predictions = self.online_learner.predict(feature_data)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(self.target_values, predictions, alpha=0.6)
                plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
                plt.xlabel('Actual Shear Percentage (%)')
                plt.ylabel('Predicted Shear Percentage (%)')
                plt.title('Predicted vs Actual Shear Percentage')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add R² score to plot
                r2 = r2_score(self.target_values, predictions)
                plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.savefig(plots_dir / 'prediction_vs_actual.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Feature importance (if available)
            try:
                feature_importance = self.online_learner.model.coef_ if hasattr(self.online_learner.model, 'coef_') else None
                if feature_importance is not None:
                    feature_names = self.feature_extractor.get_feature_names()
                    
                    # Get top 20 most important features
                    importance_abs = np.abs(feature_importance)
                    top_indices = np.argsort(importance_abs)[-20:]
                    
                    plt.figure(figsize=(12, 8))
                    plt.barh(range(len(top_indices)), importance_abs[top_indices])
                    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
                    plt.xlabel('Feature Importance (Absolute Coefficient)')
                    plt.title('Top 20 Most Important Features')
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.warning(f"Could not create feature importance plot: {e}")
            
            logger.info(f"Visualizations saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            # Don't raise - visualizations are optional
    
    def run_full_training(self):
        """Run the complete initial model training pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("STARTING INITIAL SHEAR MODEL TRAINING")
            logger.info("=" * 60)
            
            # Step 1: Load YOLO model
            self.load_yolo_model()
            
            # Step 2: Initialize feature extractor
            self.initialize_feature_extractor()
            
            # Step 3: Load training samples
            self.load_training_samples()
            
            # Step 4: Detect fracture surfaces
            self.detect_fracture_surfaces()
            
            # Step 5: Extract features
            self.extract_features()
            
            # Step 6: Train initial models
            self.train_initial_models()
            
            # Step 7: Save models and data
            self.save_models_and_data()
            
            # Step 8: Create visualizations
            self.create_visualizations()
            
            logger.info("=" * 60)
            logger.info("INITIAL SHEAR MODEL TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Initial model training failed: {e}")
            return False


def main():
    """Main function to run initial model training."""
    # Configuration
    YOLO_MODEL_PATH = "src/models/detection/charpy_3class/charpy_3class_20250729_110009/weights/best.pt"
    TRAINING_DATA_PATH = "src/database/samples/shiny_training_data"
    OUTPUT_DIR = "src/models/shear_prediction"
    
    # Create trainer
    trainer = InitialShearModelTrainer(
        yolo_model_path=YOLO_MODEL_PATH,
        training_data_path=TRAINING_DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Run training
    success = trainer.run_full_training()
    
    if success:
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to: {OUTPUT_DIR}")
        logger.info("You can now use the trained model for online learning and predictions.")
    else:
        logger.error("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()