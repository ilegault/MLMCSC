#!/usr/bin/env python3
"""
Feature Extraction Tasks for Celery Workers

This module contains Celery tasks for asynchronous feature extraction
from microscope images.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import cv2
from celery import current_task

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.workers.celery_app import celery_app
from src.mlmcsc.feature_extraction import FractureFeatureExtractor
from src.database import DatabaseManager, load_config
from src.utils.s3_client import S3Client

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='src.workers.tasks.feature_extraction.extract_features')
def extract_features(self, image_id: int, force_recompute: bool = False) -> Dict[str, Any]:
    """
    Extract features from an image asynchronously.
    
    Args:
        image_id: Database ID of the image
        force_recompute: Whether to recompute features if they already exist
        
    Returns:
        Dictionary containing feature extraction results
    """
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'step': 'initializing', 'progress': 0})
        
        # Initialize components
        config = load_config()
        db_manager = DatabaseManager(config.database.get_connection_url())
        feature_extractor = FractureFeatureExtractor()
        s3_client = S3Client() if config.storage.use_s3 else None
        
        # Get image from database
        self.update_state(state='PROGRESS', meta={'step': 'loading_image', 'progress': 10})
        
        with db_manager.get_session() as session:
            from src.database.models import Image
            image_record = session.query(Image).filter(Image.id == image_id).first()
            
            if not image_record:
                raise ValueError(f"Image with ID {image_id} not found")
            
            # Check if features already exist
            if not force_recompute:
                existing_features = db_manager.get_features_by_image_id(image_id)
                if existing_features:
                    logger.info(f"Features already exist for image {image_id}")
                    return {
                        'status': 'skipped',
                        'image_id': image_id,
                        'feature_count': len(existing_features),
                        'message': 'Features already exist'
                    }
        
        # Load image data
        self.update_state(state='PROGRESS', meta={'step': 'loading_image_data', 'progress': 20})
        
        if s3_client and image_record.s3_key:
            # Load from S3
            image_data = s3_client.download_file(image_record.s3_key)
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            # Load from local filesystem
            image_path = Path(image_record.path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image data for image {image_id}")
        
        # Extract features
        self.update_state(state='PROGRESS', meta={'step': 'extracting_features', 'progress': 40})
        
        # Use detection bbox if available, otherwise use full image
        bbox = None
        if hasattr(image_record, 'detection_bbox') and image_record.detection_bbox:
            bbox = image_record.detection_bbox
        
        feature_result = feature_extractor.extract_features(
            image=image,
            specimen_id=image_record.specimen_id,
            bbox=bbox
        )
        
        # Store features in database
        self.update_state(state='PROGRESS', meta={'step': 'storing_features', 'progress': 80})
        
        feature_id = db_manager.store_features(
            image_id=image_id,
            feature_type='comprehensive',
            extractor_version=feature_extractor.get_version(),
            feature_vector=feature_result.feature_vector,
            extraction_time=feature_result.processing_time,
            metadata={
                'feature_names': list(feature_result.features.keys()),
                'feature_count': len(feature_result.feature_vector),
                'extraction_timestamp': datetime.utcnow().isoformat(),
                'worker_id': self.request.id
            }
        )
        
        # Update task completion
        self.update_state(state='PROGRESS', meta={'step': 'completed', 'progress': 100})
        
        result = {
            'status': 'success',
            'image_id': image_id,
            'feature_id': feature_id,
            'feature_count': len(feature_result.feature_vector),
            'processing_time': feature_result.processing_time,
            'extraction_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Features extracted successfully for image {image_id}")
        return result
        
    except Exception as e:
        logger.error(f"Feature extraction failed for image {image_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update task state with error
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'image_id': image_id
            }
        )
        raise

@celery_app.task(bind=True, name='src.workers.tasks.feature_extraction.batch_extract_features')
def batch_extract_features(self, image_ids: List[int], force_recompute: bool = False) -> Dict[str, Any]:
    """
    Extract features from multiple images in batch.
    
    Args:
        image_ids: List of image IDs to process
        force_recompute: Whether to recompute features if they already exist
        
    Returns:
        Dictionary containing batch processing results
    """
    try:
        total_images = len(image_ids)
        processed = 0
        successful = 0
        failed = 0
        skipped = 0
        errors = []
        
        self.update_state(
            state='PROGRESS',
            meta={
                'step': 'batch_processing',
                'progress': 0,
                'total': total_images,
                'processed': 0,
                'successful': 0,
                'failed': 0
            }
        )
        
        for i, image_id in enumerate(image_ids):
            try:
                # Extract features for this image
                result = extract_features.apply(args=[image_id, force_recompute])
                
                if result.successful():
                    result_data = result.get()
                    if result_data['status'] == 'success':
                        successful += 1
                    elif result_data['status'] == 'skipped':
                        skipped += 1
                else:
                    failed += 1
                    errors.append(f"Image {image_id}: {result.traceback}")
                    
            except Exception as e:
                failed += 1
                errors.append(f"Image {image_id}: {str(e)}")
                logger.error(f"Failed to process image {image_id}: {e}")
            
            processed += 1
            progress = int((processed / total_images) * 100)
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={
                    'step': 'batch_processing',
                    'progress': progress,
                    'total': total_images,
                    'processed': processed,
                    'successful': successful,
                    'failed': failed,
                    'skipped': skipped
                }
            )
        
        result = {
            'status': 'completed',
            'total_images': total_images,
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'errors': errors[:10],  # Limit error list
            'completion_time': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Batch feature extraction completed: {successful}/{total_images} successful")
        return result
        
    except Exception as e:
        logger.error(f"Batch feature extraction failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        raise

@celery_app.task(bind=True, name='src.workers.tasks.feature_extraction.extract_features_for_training')
def extract_features_for_training(self, dataset_version: str) -> Dict[str, Any]:
    """
    Extract features for all images in a training dataset.
    
    Args:
        dataset_version: Version identifier for the dataset
        
    Returns:
        Dictionary containing extraction results
    """
    try:
        self.update_state(state='PROGRESS', meta={'step': 'initializing', 'progress': 0})
        
        # Initialize database manager
        config = load_config()
        db_manager = DatabaseManager(config.database.get_connection_url())
        
        # Get all images that need feature extraction
        self.update_state(state='PROGRESS', meta={'step': 'querying_images', 'progress': 10})
        
        with db_manager.get_session() as session:
            from src.database.models import Image, Feature
            
            # Find images without features
            images_without_features = session.query(Image).outerjoin(Feature).filter(
                Feature.id.is_(None)
            ).all()
            
            image_ids = [img.id for img in images_without_features]
        
        if not image_ids:
            return {
                'status': 'completed',
                'message': 'All images already have features extracted',
                'total_images': 0
            }
        
        # Process in batches
        self.update_state(
            state='PROGRESS',
            meta={
                'step': 'batch_processing',
                'progress': 20,
                'total_images': len(image_ids)
            }
        )
        
        # Use batch extraction
        result = batch_extract_features.apply(args=[image_ids, False])
        
        if result.successful():
            batch_result = result.get()
            
            # Record dataset version
            db_manager.record_metric(
                'feature_extraction_batch',
                batch_result['successful'],
                'training',
                'count',
                metadata={
                    'dataset_version': dataset_version,
                    'total_images': batch_result['total_images'],
                    'failed_count': batch_result['failed']
                }
            )
            
            return {
                'status': 'completed',
                'dataset_version': dataset_version,
                **batch_result
            }
        else:
            raise Exception(f"Batch extraction failed: {result.traceback}")
            
    except Exception as e:
        logger.error(f"Training feature extraction failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        raise

@celery_app.task(name='src.workers.tasks.feature_extraction.cleanup_old_features')
def cleanup_old_features(days_old: int = 30) -> Dict[str, Any]:
    """
    Clean up old feature vectors to save storage space.
    
    Args:
        days_old: Remove features older than this many days
        
    Returns:
        Dictionary containing cleanup results
    """
    try:
        config = load_config()
        db_manager = DatabaseManager(config.database.get_connection_url())
        
        # Find old features
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        with db_manager.get_session() as session:
            from src.database.models import Feature
            
            old_features = session.query(Feature).filter(
                Feature.created_at < cutoff_date,
                Feature.feature_type != 'comprehensive'  # Keep comprehensive features
            ).all()
            
            deleted_count = len(old_features)
            
            # Delete old features
            for feature in old_features:
                session.delete(feature)
            
            session.commit()
        
        logger.info(f"Cleaned up {deleted_count} old feature vectors")
        
        return {
            'status': 'completed',
            'deleted_count': deleted_count,
            'cutoff_date': cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Feature cleanup failed: {e}")
        raise