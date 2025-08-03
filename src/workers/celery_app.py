#!/usr/bin/env python3
"""
Celery Application for MLMCSC Async Processing

This module configures Celery for handling asynchronous tasks including:
- Feature extraction
- Model training
- Batch processing
- Data pipeline operations
"""

import os
import logging
from celery import Celery
from celery.schedules import crontab
from kombu import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')

# Create Celery app
celery_app = Celery(
    'mlmcsc',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'src.workers.tasks.feature_extraction',
        'src.workers.tasks.model_training',
        'src.workers.tasks.data_processing',
        'src.workers.tasks.monitoring'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'src.workers.tasks.feature_extraction.*': {'queue': 'feature_extraction'},
        'src.workers.tasks.model_training.*': {'queue': 'model_training'},
        'src.workers.tasks.data_processing.*': {'queue': 'data_processing'},
        'src.workers.tasks.monitoring.*': {'queue': 'monitoring'},
    },
    
    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('feature_extraction', routing_key='feature_extraction'),
        Queue('model_training', routing_key='model_training'),
        Queue('data_processing', routing_key='data_processing'),
        Queue('monitoring', routing_key='monitoring'),
    ),
    
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Task retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'data-quality-check': {
            'task': 'src.workers.tasks.monitoring.check_data_quality',
            'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        },
        'model-performance-check': {
            'task': 'src.workers.tasks.monitoring.check_model_performance',
            'schedule': crontab(minute=30, hour='*/4'),  # Every 4 hours at :30
        },
        'system-health-check': {
            'task': 'src.workers.tasks.monitoring.check_system_health',
            'schedule': crontab(minute='*/15'),  # Every 15 minutes
        },
        'backup-database': {
            'task': 'src.workers.tasks.data_processing.backup_database',
            'schedule': crontab(minute=0, hour=2),  # Daily at 2 AM
        },
        'cleanup-old-files': {
            'task': 'src.workers.tasks.data_processing.cleanup_old_files',
            'schedule': crontab(minute=0, hour=3, day_of_week=0),  # Weekly on Sunday at 3 AM
        },
        'retrain-model': {
            'task': 'src.workers.tasks.model_training.retrain_model',
            'schedule': crontab(minute=0, hour=4, day_of_week=1),  # Weekly on Monday at 4 AM
        },
    },
)

# Task error handling
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup."""
    print(f'Request: {self.request!r}')
    return 'Debug task completed'

# Health check task
@celery_app.task
def health_check():
    """Health check task for monitoring."""
    return {
        'status': 'healthy',
        'timestamp': str(datetime.utcnow()),
        'worker_id': os.getpid()
    }

if __name__ == '__main__':
    celery_app.start()