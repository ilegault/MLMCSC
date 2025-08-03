#!/usr/bin/env python3
"""
Performance Monitoring System for MLMCSC

This module implements comprehensive performance monitoring including:
- Real-time metrics tracking (MAE, RMSE, R²)
- Prediction confidence distribution monitoring
- Label distribution analysis over time
- Technician agreement rate tracking
- Feature importance evolution
- Visualization dashboard components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict, deque
import sqlite3
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics at a point in time."""
    timestamp: str
    mae: float
    rmse: float
    r2: float
    mse: float
    sample_count: int
    prediction_count: int
    avg_confidence: float
    model_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TechnicianMetrics:
    """Container for technician-specific metrics."""
    technician_id: str
    total_labels: int
    avg_label_value: float
    std_label_value: float
    agreement_rate: float
    outlier_rate: float
    last_activity: str
    labels_per_day: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class FeatureImportance:
    """Container for feature importance at a point in time."""
    timestamp: str
    feature_importances: Dict[str, float]
    model_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time metrics collection and storage
    - Historical performance tracking
    - Technician performance analysis
    - Feature importance evolution
    - Automated alerting for performance degradation
    - Rich visualization capabilities
    """
    
    def __init__(self, 
                 storage_path: str = "monitoring",
                 db_path: Optional[str] = None,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 history_window: int = 1000):
        """
        Initialize the performance monitoring system.
        
        Args:
            storage_path: Path to store monitoring data
            db_path: Path to SQLite database (optional)
            alert_thresholds: Thresholds for performance alerts
            history_window: Number of recent metrics to keep in memory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path or str(self.storage_path / "monitoring.db")
        self.history_window = history_window
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'r2_min': 0.7,
            'rmse_max': 10.0,
            'mae_max': 8.0,
            'confidence_min': 0.5
        }
        
        # In-memory storage for recent data
        self.performance_history = deque(maxlen=history_window)
        self.technician_metrics: Dict[str, TechnicianMetrics] = {}
        self.feature_importance_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=history_window)
        self.label_history = deque(maxlen=history_window)
        
        # Initialize database
        self._init_database()
        
        # Load recent data
        self._load_recent_data()
        
        logger.info(f"PerformanceMonitor initialized at {self.storage_path}")
    
    def record_performance_metrics(self,
                                 mae: float,
                                 rmse: float,
                                 r2: float,
                                 mse: float,
                                 sample_count: int,
                                 prediction_count: int,
                                 avg_confidence: float,
                                 model_version: Optional[str] = None) -> None:
        """
        Record performance metrics.
        
        Args:
            mae: Mean Absolute Error
            rmse: Root Mean Square Error
            r2: R-squared score
            mse: Mean Square Error
            sample_count: Number of training samples
            prediction_count: Number of predictions made
            avg_confidence: Average prediction confidence
            model_version: Model version identifier
        """
        try:
            timestamp = datetime.now().isoformat()
            
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                mae=mae,
                rmse=rmse,
                r2=r2,
                mse=mse,
                sample_count=sample_count,
                prediction_count=prediction_count,
                avg_confidence=avg_confidence,
                model_version=model_version
            )
            
            # Add to in-memory storage
            self.performance_history.append(metrics)
            
            # Store in database
            self._store_performance_metrics(metrics)
            
            # Check for alerts
            self._check_performance_alerts(metrics)
            
            logger.debug(f"Recorded performance metrics: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to record performance metrics: {e}")
            raise
    
    def record_prediction(self,
                         prediction: float,
                         confidence: float,
                         actual_label: Optional[float] = None,
                         image_id: Optional[str] = None,
                         technician_id: Optional[str] = None) -> None:
        """
        Record a single prediction.
        
        Args:
            prediction: Model prediction
            confidence: Prediction confidence
            actual_label: Actual label (if available)
            image_id: Image identifier
            technician_id: Technician identifier
        """
        try:
            timestamp = datetime.now().isoformat()
            
            prediction_record = {
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence': confidence,
                'actual_label': actual_label,
                'image_id': image_id,
                'technician_id': technician_id
            }
            
            self.prediction_history.append(prediction_record)
            
            # Store in database
            self._store_prediction(prediction_record)
            
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
    
    def record_label(self,
                    label: float,
                    technician_id: str,
                    image_id: str,
                    confidence: Optional[float] = None) -> None:
        """
        Record a technician label.
        
        Args:
            label: Label value
            technician_id: Technician identifier
            image_id: Image identifier
            confidence: Label confidence (optional)
        """
        try:
            timestamp = datetime.now().isoformat()
            
            label_record = {
                'timestamp': timestamp,
                'label': label,
                'technician_id': technician_id,
                'image_id': image_id,
                'confidence': confidence
            }
            
            self.label_history.append(label_record)
            
            # Store in database
            self._store_label(label_record)
            
            # Update technician metrics
            self._update_technician_metrics(technician_id, label, timestamp)
            
        except Exception as e:
            logger.error(f"Failed to record label: {e}")
    
    def record_feature_importance(self,
                                feature_importances: Dict[str, float],
                                model_version: Optional[str] = None) -> None:
        """
        Record feature importance values.
        
        Args:
            feature_importances: Dictionary of feature names to importance values
            model_version: Model version identifier
        """
        try:
            timestamp = datetime.now().isoformat()
            
            importance_record = FeatureImportance(
                timestamp=timestamp,
                feature_importances=feature_importances,
                model_version=model_version
            )
            
            self.feature_importance_history.append(importance_record)
            
            # Store in database
            self._store_feature_importance(importance_record)
            
        except Exception as e:
            logger.error(f"Failed to record feature importance: {e}")
    
    def get_performance_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get performance summary for the specified time period.
        
        Args:
            days_back: Number of days to include in summary
            
        Returns:
            Performance summary dictionary
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent metrics
            recent_metrics = [
                m for m in self.performance_history
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'No recent performance data available'}
            
            # Calculate summary statistics
            r2_values = [m.r2 for m in recent_metrics]
            rmse_values = [m.rmse for m in recent_metrics]
            mae_values = [m.mae for m in recent_metrics]
            confidence_values = [m.avg_confidence for m in recent_metrics]
            
            summary = {
                'period_days': days_back,
                'total_metrics_recorded': len(recent_metrics),
                'performance_stats': {
                    'r2': {
                        'current': r2_values[-1] if r2_values else 0,
                        'mean': np.mean(r2_values),
                        'std': np.std(r2_values),
                        'min': np.min(r2_values),
                        'max': np.max(r2_values),
                        'trend': self._calculate_trend(r2_values)
                    },
                    'rmse': {
                        'current': rmse_values[-1] if rmse_values else 0,
                        'mean': np.mean(rmse_values),
                        'std': np.std(rmse_values),
                        'min': np.min(rmse_values),
                        'max': np.max(rmse_values),
                        'trend': self._calculate_trend(rmse_values, lower_is_better=True)
                    },
                    'mae': {
                        'current': mae_values[-1] if mae_values else 0,
                        'mean': np.mean(mae_values),
                        'std': np.std(mae_values),
                        'min': np.min(mae_values),
                        'max': np.max(mae_values),
                        'trend': self._calculate_trend(mae_values, lower_is_better=True)
                    },
                    'confidence': {
                        'current': confidence_values[-1] if confidence_values else 0,
                        'mean': np.mean(confidence_values),
                        'std': np.std(confidence_values),
                        'min': np.min(confidence_values),
                        'max': np.max(confidence_values),
                        'trend': self._calculate_trend(confidence_values)
                    }
                },
                'sample_stats': {
                    'total_samples': recent_metrics[-1].sample_count if recent_metrics else 0,
                    'total_predictions': sum(m.prediction_count for m in recent_metrics),
                    'avg_predictions_per_day': sum(m.prediction_count for m in recent_metrics) / days_back
                },
                'alerts': self._get_current_alerts(recent_metrics[-1] if recent_metrics else None)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            raise
    
    def get_technician_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get technician performance summary.
        
        Args:
            days_back: Number of days to include in summary
            
        Returns:
            Technician summary dictionary
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent labels
            recent_labels = [
                l for l in self.label_history
                if datetime.fromisoformat(l['timestamp']) >= cutoff_time
            ]
            
            # Group by technician
            technician_data = defaultdict(list)
            for label_record in recent_labels:
                technician_data[label_record['technician_id']].append(label_record)
            
            # Calculate technician metrics
            technician_summary = {}
            for tech_id, labels in technician_data.items():
                label_values = [l['label'] for l in labels]
                
                technician_summary[tech_id] = {
                    'total_labels': len(labels),
                    'avg_label': np.mean(label_values),
                    'std_label': np.std(label_values),
                    'min_label': np.min(label_values),
                    'max_label': np.max(label_values),
                    'labels_per_day': len(labels) / days_back,
                    'last_activity': max(l['timestamp'] for l in labels),
                    'label_distribution': self._get_label_distribution(label_values)
                }
            
            # Calculate inter-technician agreement
            agreement_stats = self._calculate_technician_agreement(technician_data)
            
            summary = {
                'period_days': days_back,
                'total_technicians': len(technician_summary),
                'total_labels': len(recent_labels),
                'technician_metrics': technician_summary,
                'agreement_stats': agreement_stats,
                'top_performers': self._rank_technicians(technician_summary)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get technician summary: {e}")
            raise
    
    def create_performance_dashboard(self, output_path: str, days_back: int = 30) -> str:
        """
        Create an interactive performance dashboard.
        
        Args:
            output_path: Path to save the dashboard HTML file
            days_back: Number of days of data to include
            
        Returns:
            Path to the created dashboard file
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent data
            recent_metrics = [
                m for m in self.performance_history
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]
            
            recent_predictions = [
                p for p in self.prediction_history
                if datetime.fromisoformat(p['timestamp']) >= cutoff_time
            ]
            
            recent_labels = [
                l for l in self.label_history
                if datetime.fromisoformat(l['timestamp']) >= cutoff_time
            ]
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Performance Metrics Over Time',
                    'Prediction Confidence Distribution',
                    'Label Distribution Over Time',
                    'Technician Activity',
                    'Feature Importance Evolution',
                    'Model Performance Comparison'
                ],
                specs=[
                    [{"secondary_y": True}, {"type": "histogram"}],
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # 1. Performance metrics over time
            if recent_metrics:
                timestamps = [datetime.fromisoformat(m.timestamp) for m in recent_metrics]
                r2_values = [m.r2 for m in recent_metrics]
                rmse_values = [m.rmse for m in recent_metrics]
                mae_values = [m.mae for m in recent_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=r2_values, name='R²', line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=timestamps, y=rmse_values, name='RMSE', line=dict(color='red'), yaxis='y2'),
                    row=1, col=1, secondary_y=True
                )
                fig.add_trace(
                    go.Scatter(x=timestamps, y=mae_values, name='MAE', line=dict(color='orange'), yaxis='y2'),
                    row=1, col=1, secondary_y=True
                )
            
            # 2. Prediction confidence distribution
            if recent_predictions:
                confidences = [p['confidence'] for p in recent_predictions]
                fig.add_trace(
                    go.Histogram(x=confidences, name='Confidence Distribution', nbinsx=20),
                    row=1, col=2
                )
            
            # 3. Label distribution over time
            if recent_labels:
                label_timestamps = [datetime.fromisoformat(l['timestamp']) for l in recent_labels]
                label_values = [l['label'] for l in recent_labels]
                
                fig.add_trace(
                    go.Scatter(
                        x=label_timestamps, 
                        y=label_values, 
                        mode='markers',
                        name='Labels',
                        marker=dict(size=4, opacity=0.6)
                    ),
                    row=2, col=1
                )
            
            # 4. Technician activity
            if recent_labels:
                technician_counts = defaultdict(int)
                for label in recent_labels:
                    technician_counts[label['technician_id']] += 1
                
                tech_ids = list(technician_counts.keys())
                counts = list(technician_counts.values())
                
                fig.add_trace(
                    go.Bar(x=tech_ids, y=counts, name='Labels per Technician'),
                    row=2, col=2
                )
            
            # 5. Feature importance evolution
            if self.feature_importance_history:
                # Get the most recent feature importance
                recent_importance = self.feature_importance_history[-1]
                features = list(recent_importance.feature_importances.keys())
                importances = list(recent_importance.feature_importances.values())
                
                fig.add_trace(
                    go.Bar(x=features, y=importances, name='Feature Importance'),
                    row=3, col=1
                )
            
            # 6. Model performance comparison (if multiple versions)
            version_performance = defaultdict(list)
            for metric in recent_metrics:
                if metric.model_version:
                    version_performance[metric.model_version].append(metric.r2)
            
            if len(version_performance) > 1:
                versions = list(version_performance.keys())
                avg_r2 = [np.mean(scores) for scores in version_performance.values()]
                
                fig.add_trace(
                    go.Bar(x=versions, y=avg_r2, name='Avg R² by Version'),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f'MLMCSC Performance Dashboard - Last {days_back} Days',
                height=1200,
                showlegend=True
            )
            
            # Save dashboard
            fig.write_html(output_path)
            
            logger.info(f"Created performance dashboard at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create performance dashboard: {e}")
            raise
    
    def create_static_plots(self, output_dir: str, days_back: int = 30) -> List[str]:
        """
        Create static matplotlib plots for performance monitoring.
        
        Args:
            output_dir: Directory to save plots
            days_back: Number of days of data to include
            
        Returns:
            List of created plot file paths
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent data
            recent_metrics = [
                m for m in self.performance_history
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]
            
            recent_predictions = [
                p for p in self.prediction_history
                if datetime.fromisoformat(p['timestamp']) >= cutoff_time
            ]
            
            recent_labels = [
                l for l in self.label_history
                if datetime.fromisoformat(l['timestamp']) >= cutoff_time
            ]
            
            created_plots = []
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Performance metrics over time
            if recent_metrics:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                timestamps = [datetime.fromisoformat(m.timestamp) for m in recent_metrics]
                r2_values = [m.r2 for m in recent_metrics]
                rmse_values = [m.rmse for m in recent_metrics]
                mae_values = [m.mae for m in recent_metrics]
                
                # R² plot
                ax1.plot(timestamps, r2_values, 'b-', label='R²', linewidth=2)
                ax1.set_ylabel('R² Score')
                ax1.set_title('Model Performance Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # RMSE and MAE plot
                ax2.plot(timestamps, rmse_values, 'r-', label='RMSE', linewidth=2)
                ax2.plot(timestamps, mae_values, 'orange', label='MAE', linewidth=2)
                ax2.set_ylabel('Error')
                ax2.set_xlabel('Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = output_dir / 'performance_over_time.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                created_plots.append(str(plot_path))
            
            # 2. Prediction confidence distribution
            if recent_predictions:
                plt.figure(figsize=(10, 6))
                confidences = [p['confidence'] for p in recent_predictions]
                
                plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(confidences):.3f}')
                plt.xlabel('Prediction Confidence')
                plt.ylabel('Frequency')
                plt.title('Prediction Confidence Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_path = output_dir / 'confidence_distribution.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                created_plots.append(str(plot_path))
            
            # 3. Label distribution over time
            if recent_labels:
                plt.figure(figsize=(12, 6))
                
                label_timestamps = [datetime.fromisoformat(l['timestamp']) for l in recent_labels]
                label_values = [l['label'] for l in recent_labels]
                
                plt.scatter(label_timestamps, label_values, alpha=0.6, s=20)
                plt.xlabel('Time')
                plt.ylabel('Label Value')
                plt.title('Label Distribution Over Time')
                plt.grid(True, alpha=0.3)
                
                plot_path = output_dir / 'label_distribution.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                created_plots.append(str(plot_path))
            
            # 4. Technician activity
            if recent_labels:
                technician_counts = defaultdict(int)
                for label in recent_labels:
                    technician_counts[label['technician_id']] += 1
                
                plt.figure(figsize=(10, 6))
                tech_ids = list(technician_counts.keys())
                counts = list(technician_counts.values())
                
                plt.bar(tech_ids, counts, color='lightgreen', edgecolor='black')
                plt.xlabel('Technician ID')
                plt.ylabel('Number of Labels')
                plt.title('Technician Activity (Labels Submitted)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                plot_path = output_dir / 'technician_activity.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                created_plots.append(str(plot_path))
            
            # 5. Feature importance (most recent)
            if self.feature_importance_history:
                recent_importance = self.feature_importance_history[-1]
                features = list(recent_importance.feature_importances.keys())
                importances = list(recent_importance.feature_importances.values())
                
                # Sort by importance
                sorted_data = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_data)
                
                plt.figure(figsize=(12, 8))
                plt.barh(features, importances, color='coral', edgecolor='black')
                plt.xlabel('Importance')
                plt.ylabel('Features')
                plt.title('Feature Importance (Most Recent)')
                plt.grid(True, alpha=0.3)
                
                plot_path = output_dir / 'feature_importance.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                created_plots.append(str(plot_path))
            
            logger.info(f"Created {len(created_plots)} static plots in {output_dir}")
            return created_plots
            
        except Exception as e:
            logger.error(f"Failed to create static plots: {e}")
            raise
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current performance alerts."""
        alerts = []
        
        if not self.performance_history:
            return alerts
        
        latest_metrics = self.performance_history[-1]
        
        # R² alert
        if latest_metrics.r2 < self.alert_thresholds['r2_min']:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'metric': 'r2',
                'current_value': latest_metrics.r2,
                'threshold': self.alert_thresholds['r2_min'],
                'message': f"R² score ({latest_metrics.r2:.3f}) below threshold ({self.alert_thresholds['r2_min']:.3f})"
            })
        
        # RMSE alert
        if latest_metrics.rmse > self.alert_thresholds['rmse_max']:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'metric': 'rmse',
                'current_value': latest_metrics.rmse,
                'threshold': self.alert_thresholds['rmse_max'],
                'message': f"RMSE ({latest_metrics.rmse:.3f}) above threshold ({self.alert_thresholds['rmse_max']:.3f})"
            })
        
        # MAE alert
        if latest_metrics.mae > self.alert_thresholds['mae_max']:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'metric': 'mae',
                'current_value': latest_metrics.mae,
                'threshold': self.alert_thresholds['mae_max'],
                'message': f"MAE ({latest_metrics.mae:.3f}) above threshold ({self.alert_thresholds['mae_max']:.3f})"
            })
        
        # Confidence alert
        if latest_metrics.avg_confidence < self.alert_thresholds['confidence_min']:
            alerts.append({
                'type': 'confidence',
                'severity': 'medium',
                'metric': 'confidence',
                'current_value': latest_metrics.avg_confidence,
                'threshold': self.alert_thresholds['confidence_min'],
                'message': f"Average confidence ({latest_metrics.avg_confidence:.3f}) below threshold ({self.alert_thresholds['confidence_min']:.3f})"
            })
        
        return alerts
    
    def export_monitoring_data(self, output_path: str, days_back: int = 30) -> None:
        """Export monitoring data to JSON file."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_days': days_back,
                'performance_metrics': [
                    m.to_dict() for m in self.performance_history
                    if datetime.fromisoformat(m.timestamp) >= cutoff_time
                ],
                'predictions': [
                    p for p in self.prediction_history
                    if datetime.fromisoformat(p['timestamp']) >= cutoff_time
                ],
                'labels': [
                    l for l in self.label_history
                    if datetime.fromisoformat(l['timestamp']) >= cutoff_time
                ],
                'feature_importance': [
                    fi.to_dict() for fi in self.feature_importance_history
                    if datetime.fromisoformat(fi.timestamp) >= cutoff_time
                ],
                'technician_metrics': {
                    tid: tm.to_dict() for tid, tm in self.technician_metrics.items()
                },
                'alert_thresholds': self.alert_thresholds,
                'current_alerts': self.get_alerts()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported monitoring data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export monitoring data: {e}")
            raise
    
    # Private methods
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mae REAL NOT NULL,
                    rmse REAL NOT NULL,
                    r2 REAL NOT NULL,
                    mse REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    prediction_count INTEGER NOT NULL,
                    avg_confidence REAL NOT NULL,
                    model_version TEXT
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    confidence REAL NOT NULL,
                    actual_label REAL,
                    image_id TEXT,
                    technician_id TEXT
                )
            ''')
            
            # Labels table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    label REAL NOT NULL,
                    technician_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    confidence REAL
                )
            ''')
            
            # Feature importance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance REAL NOT NULL,
                    model_version TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, mae, rmse, r2, mse, sample_count, prediction_count, avg_confidence, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.mae, metrics.rmse, metrics.r2, metrics.mse,
                metrics.sample_count, metrics.prediction_count, metrics.avg_confidence,
                metrics.model_version
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
    
    def _store_prediction(self, prediction_record: Dict[str, Any]):
        """Store prediction in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, prediction, confidence, actual_label, image_id, technician_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction_record['timestamp'], prediction_record['prediction'],
                prediction_record['confidence'], prediction_record['actual_label'],
                prediction_record['image_id'], prediction_record['technician_id']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    def _store_label(self, label_record: Dict[str, Any]):
        """Store label in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO labels 
                (timestamp, label, technician_id, image_id, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                label_record['timestamp'], label_record['label'],
                label_record['technician_id'], label_record['image_id'],
                label_record['confidence']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store label: {e}")
    
    def _store_feature_importance(self, importance_record: FeatureImportance):
        """Store feature importance in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for feature_name, importance in importance_record.feature_importances.items():
                cursor.execute('''
                    INSERT INTO feature_importance 
                    (timestamp, feature_name, importance, model_version)
                    VALUES (?, ?, ?, ?)
                ''', (
                    importance_record.timestamp, feature_name, importance,
                    importance_record.model_version
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store feature importance: {e}")
    
    def _load_recent_data(self):
        """Load recent data from database into memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load recent performance metrics
            cursor.execute('''
                SELECT * FROM performance_metrics 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (self.history_window,))
            
            for row in cursor.fetchall():
                metrics = PerformanceMetrics(
                    timestamp=row[1], mae=row[2], rmse=row[3], r2=row[4],
                    mse=row[5], sample_count=row[6], prediction_count=row[7],
                    avg_confidence=row[8], model_version=row[9]
                )
                self.performance_history.appendleft(metrics)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load recent data: {e}")
    
    def _update_technician_metrics(self, technician_id: str, label: float, timestamp: str):
        """Update technician-specific metrics."""
        # This would be implemented with more sophisticated tracking
        # For now, just log the activity
        logger.debug(f"Updated metrics for technician {technician_id}")
    
    def _calculate_trend(self, values: List[float], lower_is_better: bool = False) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        threshold = 0.001  # Minimum slope to consider as trend
        
        if abs(slope) < threshold:
            return 'stable'
        elif slope > 0:
            return 'improving' if not lower_is_better else 'degrading'
        else:
            return 'degrading' if not lower_is_better else 'improving'
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts and log warnings."""
        alerts = self._get_current_alerts(metrics)
        for alert in alerts:
            if alert['severity'] == 'high':
                logger.warning(f"Performance Alert: {alert['message']}")
            else:
                logger.info(f"Performance Notice: {alert['message']}")
    
    def _get_current_alerts(self, metrics: Optional[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """Get current alerts for given metrics."""
        if not metrics:
            return []
        
        alerts = []
        
        if metrics.r2 < self.alert_thresholds['r2_min']:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'metric': 'r2',
                'message': f"R² below threshold: {metrics.r2:.3f} < {self.alert_thresholds['r2_min']:.3f}"
            })
        
        if metrics.rmse > self.alert_thresholds['rmse_max']:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'metric': 'rmse',
                'message': f"RMSE above threshold: {metrics.rmse:.3f} > {self.alert_thresholds['rmse_max']:.3f}"
            })
        
        if metrics.mae > self.alert_thresholds['mae_max']:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'metric': 'mae',
                'message': f"MAE above threshold: {metrics.mae:.3f} > {self.alert_thresholds['mae_max']:.3f}"
            })
        
        if metrics.avg_confidence < self.alert_thresholds['confidence_min']:
            alerts.append({
                'type': 'confidence',
                'severity': 'medium',
                'metric': 'confidence',
                'message': f"Low confidence: {metrics.avg_confidence:.3f} < {self.alert_thresholds['confidence_min']:.3f}"
            })
        
        return alerts
    
    def _get_label_distribution(self, label_values: List[float]) -> Dict[str, Any]:
        """Get label distribution statistics."""
        return {
            'mean': np.mean(label_values),
            'std': np.std(label_values),
            'min': np.min(label_values),
            'max': np.max(label_values),
            'percentiles': {
                '25': np.percentile(label_values, 25),
                '50': np.percentile(label_values, 50),
                '75': np.percentile(label_values, 75)
            }
        }
    
    def _calculate_technician_agreement(self, technician_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate inter-technician agreement statistics."""
        if len(technician_data) < 2:
            return {'agreement_rate': 1.0, 'note': 'Only one technician active'}
        
        # Simple agreement calculation based on label variance
        all_labels = []
        for labels in technician_data.values():
            all_labels.extend([l['label'] for l in labels])
        
        overall_std = np.std(all_labels)
        
        # Calculate per-technician deviation from overall mean
        overall_mean = np.mean(all_labels)
        technician_deviations = {}
        
        for tech_id, labels in technician_data.items():
            tech_labels = [l['label'] for l in labels]
            tech_mean = np.mean(tech_labels)
            deviation = abs(tech_mean - overall_mean)
            technician_deviations[tech_id] = deviation
        
        avg_deviation = np.mean(list(technician_deviations.values()))
        agreement_rate = max(0, 1 - (avg_deviation / overall_std)) if overall_std > 0 else 1.0
        
        return {
            'agreement_rate': agreement_rate,
            'overall_std': overall_std,
            'avg_deviation': avg_deviation,
            'technician_deviations': technician_deviations
        }
    
    def _rank_technicians(self, technician_summary: Dict[str, Dict]) -> Dict[str, Any]:
        """Rank technicians by various metrics."""
        if not technician_summary:
            return {}
        
        # Rank by activity (labels per day)
        activity_ranking = sorted(
            technician_summary.items(),
            key=lambda x: x[1]['labels_per_day'],
            reverse=True
        )
        
        # Rank by consistency (lower std is better)
        consistency_ranking = sorted(
            technician_summary.items(),
            key=lambda x: x[1]['std_label']
        )
        
        return {
            'most_active': activity_ranking[:3],
            'most_consistent': consistency_ranking[:3],
            'total_ranked': len(technician_summary)
        }