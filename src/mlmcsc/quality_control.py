#!/usr/bin/env python3
"""
Quality Control Mechanisms for MLMCSC

This module implements comprehensive quality control mechanisms including:
- Outlier detection for unusual labels
- Inter-rater reliability when multiple technicians label same image
- Automatic flagging of high-variance predictions
- Periodic expert review of contested cases
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass
class QualityFlag:
    """Container for quality control flags."""
    flag_id: str
    flag_type: str  # 'outlier', 'high_variance', 'inter_rater_disagreement', 'expert_review'
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: str
    image_id: str
    technician_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = None
    resolved: bool = False
    resolution_notes: str = ""
    expert_reviewed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        return convert_numpy_types(result)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityFlag':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class InterRaterReliability:
    """Container for inter-rater reliability analysis."""
    image_id: str
    labels: List[float]
    technician_ids: List[str]
    timestamps: List[str]
    mean_label: float
    std_label: float
    coefficient_of_variation: float
    agreement_level: str  # 'excellent', 'good', 'fair', 'poor'
    outlier_technicians: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        return convert_numpy_types(result)


@dataclass
class OutlierDetectionResult:
    """Container for outlier detection results."""
    image_id: str
    label: float
    technician_id: str
    is_outlier: bool
    outlier_score: float
    outlier_method: str
    feature_outliers: Dict[str, bool]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        return convert_numpy_types(result)


class QualityControlSystem:
    """
    Comprehensive quality control system for ensuring data quality.
    
    Features:
    - Statistical outlier detection for labels and features
    - Inter-rater reliability analysis
    - High-variance prediction flagging
    - Expert review workflow
    - Automated quality reporting
    """
    
    def __init__(self, 
                 storage_path: str = "quality_control",
                 outlier_threshold: float = 2.5,
                 variance_threshold: float = 0.3,
                 agreement_threshold: float = 0.15,
                 min_labels_for_reliability: int = 3):
        """
        Initialize the quality control system.
        
        Args:
            storage_path: Path to store quality control data
            outlier_threshold: Z-score threshold for outlier detection
            variance_threshold: Coefficient of variation threshold for high variance
            agreement_threshold: Threshold for inter-rater agreement (CV)
            min_labels_for_reliability: Minimum labels needed for reliability analysis
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.outlier_threshold = outlier_threshold
        self.variance_threshold = variance_threshold
        self.agreement_threshold = agreement_threshold
        self.min_labels_for_reliability = min_labels_for_reliability
        
        # Storage files
        self.flags_file = self.storage_path / "quality_flags.json"
        self.reliability_file = self.storage_path / "inter_rater_reliability.json"
        self.outliers_file = self.storage_path / "outlier_detection.json"
        
        # Data storage
        self.quality_flags: Dict[str, QualityFlag] = {}
        self.reliability_analyses: Dict[str, InterRaterReliability] = {}
        self.outlier_results: Dict[str, OutlierDetectionResult] = {}
        
        # Load existing data
        self._load_existing_data()
        
        # Outlier detection models
        self.label_outlier_detector = None
        self.feature_outlier_detector = None
        
        logger.info(f"QualityControlSystem initialized at {self.storage_path}")
    
    def detect_label_outliers(self, 
                             labels: List[float],
                             technician_ids: List[str],
                             image_ids: List[str],
                             timestamps: List[str],
                             method: str = 'zscore') -> List[OutlierDetectionResult]:
        """
        Detect outliers in technician labels.
        
        Args:
            labels: List of labels
            technician_ids: List of technician IDs
            image_ids: List of image IDs
            timestamps: List of timestamps
            method: Outlier detection method ('zscore', 'iqr', 'isolation_forest')
            
        Returns:
            List of outlier detection results
        """
        try:
            labels_array = np.array(labels)
            results = []
            
            if method == 'zscore':
                # Z-score based outlier detection
                z_scores = np.abs(stats.zscore(labels_array))
                outliers = z_scores > self.outlier_threshold
                
                for i, (label, tech_id, img_id, timestamp) in enumerate(
                    zip(labels, technician_ids, image_ids, timestamps)):
                    
                    result = OutlierDetectionResult(
                        image_id=img_id,
                        label=label,
                        technician_id=tech_id,
                        is_outlier=bool(outliers[i]),
                        outlier_score=float(z_scores[i]),
                        outlier_method='zscore',
                        feature_outliers={},
                        timestamp=timestamp
                    )
                    results.append(result)
                    
                    # Store result
                    result_id = f"{img_id}_{tech_id}_{timestamp}"
                    self.outlier_results[result_id] = result
            
            elif method == 'iqr':
                # Interquartile range based outlier detection
                q1 = np.percentile(labels_array, 25)
                q3 = np.percentile(labels_array, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                for i, (label, tech_id, img_id, timestamp) in enumerate(
                    zip(labels, technician_ids, image_ids, timestamps)):
                    
                    is_outlier = label < lower_bound or label > upper_bound
                    outlier_score = min(abs(label - lower_bound), abs(label - upper_bound)) / iqr
                    
                    result = OutlierDetectionResult(
                        image_id=img_id,
                        label=label,
                        technician_id=tech_id,
                        is_outlier=is_outlier,
                        outlier_score=outlier_score,
                        outlier_method='iqr',
                        feature_outliers={},
                        timestamp=timestamp
                    )
                    results.append(result)
                    
                    # Store result
                    result_id = f"{img_id}_{tech_id}_{timestamp}"
                    self.outlier_results[result_id] = result
            
            elif method == 'isolation_forest':
                # Isolation Forest based outlier detection
                if len(labels) < 10:
                    logger.warning("Not enough samples for Isolation Forest, using Z-score instead")
                    return self.detect_label_outliers(labels, technician_ids, image_ids, timestamps, 'zscore')
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(labels_array.reshape(-1, 1))
                scores = iso_forest.decision_function(labels_array.reshape(-1, 1))
                
                for i, (label, tech_id, img_id, timestamp) in enumerate(
                    zip(labels, technician_ids, image_ids, timestamps)):
                    
                    result = OutlierDetectionResult(
                        image_id=img_id,
                        label=label,
                        technician_id=tech_id,
                        is_outlier=outliers[i] == -1,
                        outlier_score=float(-scores[i]),  # Convert to positive score
                        outlier_method='isolation_forest',
                        feature_outliers={},
                        timestamp=timestamp
                    )
                    results.append(result)
                    
                    # Store result
                    result_id = f"{img_id}_{tech_id}_{timestamp}"
                    self.outlier_results[result_id] = result
            
            # Create quality flags for outliers
            for result in results:
                if result.is_outlier:
                    self._create_quality_flag(
                        flag_type='outlier',
                        severity='medium' if result.outlier_score < self.outlier_threshold * 1.5 else 'high',
                        image_id=result.image_id,
                        technician_id=result.technician_id,
                        description=f"Label outlier detected: {result.label:.2f} "
                                   f"(score: {result.outlier_score:.2f}, method: {method})"
                    )
            
            self._save_data()
            
            outlier_count = sum(1 for r in results if r.is_outlier)
            logger.info(f"Detected {outlier_count} label outliers out of {len(results)} samples using {method}")
            
            return results
            
        except Exception as e:
            logger.error(f"Label outlier detection failed: {e}")
            raise
    
    def detect_feature_outliers(self,
                               features: List[Dict[str, float]],
                               image_ids: List[str],
                               technician_ids: List[str],
                               timestamps: List[str]) -> List[OutlierDetectionResult]:
        """
        Detect outliers in extracted features.
        
        Args:
            features: List of feature dictionaries
            image_ids: List of image IDs
            technician_ids: List of technician IDs
            timestamps: List of timestamps
            
        Returns:
            List of outlier detection results
        """
        try:
            if not features:
                return []
            
            # Convert features to DataFrame
            df = pd.DataFrame(features)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df)
            
            # Use Isolation Forest for multivariate outlier detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(features_scaled)
            scores = iso_forest.decision_function(features_scaled)
            
            results = []
            
            for i, (img_id, tech_id, timestamp) in enumerate(
                zip(image_ids, technician_ids, timestamps)):
                
                # Detect outliers in individual features
                feature_outliers = {}
                for j, feature_name in enumerate(df.columns):
                    z_score = abs(stats.zscore(df[feature_name]))[i]
                    feature_outliers[feature_name] = z_score > self.outlier_threshold
                
                result = OutlierDetectionResult(
                    image_id=img_id,
                    label=0.0,  # Not applicable for feature outliers
                    technician_id=tech_id,
                    is_outlier=outliers[i] == -1,
                    outlier_score=float(-scores[i]),
                    outlier_method='isolation_forest_features',
                    feature_outliers=feature_outliers,
                    timestamp=timestamp
                )
                results.append(result)
                
                # Store result
                result_id = f"{img_id}_{tech_id}_{timestamp}_features"
                self.outlier_results[result_id] = result
                
                # Create quality flag for feature outliers
                if result.is_outlier:
                    outlier_features = [name for name, is_outlier in feature_outliers.items() if is_outlier]
                    self._create_quality_flag(
                        flag_type='outlier',
                        severity='medium',
                        image_id=img_id,
                        technician_id=tech_id,
                        description=f"Feature outliers detected in: {', '.join(outlier_features)}"
                    )
            
            self._save_data()
            
            outlier_count = sum(1 for r in results if r.is_outlier)
            logger.info(f"Detected {outlier_count} feature outliers out of {len(results)} samples")
            
            return results
            
        except Exception as e:
            logger.error(f"Feature outlier detection failed: {e}")
            raise
    
    def analyze_inter_rater_reliability(self, 
                                       labels_by_image: Dict[str, List[Tuple[float, str, str]]]) -> List[InterRaterReliability]:
        """
        Analyze inter-rater reliability for images labeled by multiple technicians.
        
        Args:
            labels_by_image: Dict mapping image_id to list of (label, technician_id, timestamp) tuples
            
        Returns:
            List of inter-rater reliability analyses
        """
        try:
            reliability_results = []
            
            for image_id, label_data in labels_by_image.items():
                if len(label_data) < self.min_labels_for_reliability:
                    continue
                
                labels = [data[0] for data in label_data]
                technician_ids = [data[1] for data in label_data]
                timestamps = [data[2] for data in label_data]
                
                # Calculate statistics
                mean_label = np.mean(labels)
                std_label = np.std(labels)
                cv = std_label / mean_label if mean_label != 0 else float('inf')
                
                # Determine agreement level
                if cv <= 0.05:
                    agreement_level = 'excellent'
                elif cv <= 0.10:
                    agreement_level = 'good'
                elif cv <= self.agreement_threshold:
                    agreement_level = 'fair'
                else:
                    agreement_level = 'poor'
                
                # Identify outlier technicians
                z_scores = np.abs(stats.zscore(labels))
                outlier_technicians = [
                    technician_ids[i] for i, z in enumerate(z_scores) 
                    if z > self.outlier_threshold
                ]
                
                reliability = InterRaterReliability(
                    image_id=image_id,
                    labels=labels,
                    technician_ids=technician_ids,
                    timestamps=timestamps,
                    mean_label=mean_label,
                    std_label=std_label,
                    coefficient_of_variation=cv,
                    agreement_level=agreement_level,
                    outlier_technicians=outlier_technicians
                )
                
                reliability_results.append(reliability)
                self.reliability_analyses[image_id] = reliability
                
                # Create quality flags for poor agreement
                if agreement_level == 'poor':
                    self._create_quality_flag(
                        flag_type='inter_rater_disagreement',
                        severity='high',
                        image_id=image_id,
                        description=f"Poor inter-rater agreement (CV: {cv:.3f}) among {len(labels)} technicians"
                    )
                elif agreement_level == 'fair':
                    self._create_quality_flag(
                        flag_type='inter_rater_disagreement',
                        severity='medium',
                        image_id=image_id,
                        description=f"Fair inter-rater agreement (CV: {cv:.3f}) among {len(labels)} technicians"
                    )
                
                # Create flags for outlier technicians
                for tech_id in outlier_technicians:
                    self._create_quality_flag(
                        flag_type='inter_rater_disagreement',
                        severity='medium',
                        image_id=image_id,
                        technician_id=tech_id,
                        description=f"Technician label significantly differs from consensus"
                    )
            
            self._save_data()
            
            logger.info(f"Analyzed inter-rater reliability for {len(reliability_results)} images")
            return reliability_results
            
        except Exception as e:
            logger.error(f"Inter-rater reliability analysis failed: {e}")
            raise
    
    def flag_high_variance_predictions(self,
                                     predictions: List[float],
                                     confidences: List[float],
                                     image_ids: List[str],
                                     timestamps: List[str]) -> List[str]:
        """
        Flag predictions with high variance/low confidence.
        
        Args:
            predictions: List of model predictions
            confidences: List of prediction confidences
            image_ids: List of image IDs
            timestamps: List of timestamps
            
        Returns:
            List of flagged image IDs
        """
        try:
            flagged_images = []
            
            for pred, conf, img_id, timestamp in zip(predictions, confidences, image_ids, timestamps):
                # Flag based on low confidence
                if conf < 0.5:  # Low confidence threshold
                    severity = 'high' if conf < 0.3 else 'medium'
                    self._create_quality_flag(
                        flag_type='high_variance',
                        severity=severity,
                        image_id=img_id,
                        description=f"Low prediction confidence: {conf:.3f} (prediction: {pred:.2f})",
                        metadata={'prediction': pred, 'confidence': conf, 'timestamp': timestamp}
                    )
                    flagged_images.append(img_id)
            
            # Flag predictions that are statistical outliers
            if len(predictions) > 10:
                z_scores = np.abs(stats.zscore(predictions))
                for i, (z_score, img_id, pred, timestamp) in enumerate(
                    zip(z_scores, image_ids, predictions, timestamps)):
                    
                    if z_score > self.outlier_threshold:
                        self._create_quality_flag(
                            flag_type='high_variance',
                            severity='medium',
                            image_id=img_id,
                            description=f"Prediction is statistical outlier: {pred:.2f} (z-score: {z_score:.2f})",
                            metadata={'prediction': pred, 'z_score': z_score, 'timestamp': timestamp}
                        )
                        if img_id not in flagged_images:
                            flagged_images.append(img_id)
            
            self._save_data()
            
            logger.info(f"Flagged {len(flagged_images)} images with high variance predictions")
            return flagged_images
            
        except Exception as e:
            logger.error(f"High variance prediction flagging failed: {e}")
            raise
    
    def create_expert_review_queue(self, 
                                  max_items: int = 50,
                                  priority_flags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Create a queue of items for expert review.
        
        Args:
            max_items: Maximum number of items in review queue
            priority_flags: List of flag types to prioritize
            
        Returns:
            List of items for expert review
        """
        try:
            priority_flags = priority_flags or ['inter_rater_disagreement', 'outlier', 'high_variance']
            
            # Get unresolved flags
            unresolved_flags = [
                flag for flag in self.quality_flags.values()
                if not flag.resolved and not flag.expert_reviewed
            ]
            
            # Sort by priority and severity
            def get_priority_score(flag):
                type_priority = priority_flags.index(flag.flag_type) if flag.flag_type in priority_flags else len(priority_flags)
                severity_score = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(flag.severity, 0)
                return (type_priority, -severity_score)  # Lower is higher priority
            
            unresolved_flags.sort(key=get_priority_score)
            
            # Create review queue
            review_queue = []
            for flag in unresolved_flags[:max_items]:
                review_item = {
                    'flag_id': flag.flag_id,
                    'flag_type': flag.flag_type,
                    'severity': flag.severity,
                    'image_id': flag.image_id,
                    'technician_id': flag.technician_id,
                    'description': flag.description,
                    'timestamp': flag.timestamp,
                    'metadata': flag.metadata or {}
                }
                
                # Add related information
                if flag.image_id in self.reliability_analyses:
                    reliability = self.reliability_analyses[flag.image_id]
                    review_item['inter_rater_info'] = {
                        'agreement_level': reliability.agreement_level,
                        'coefficient_of_variation': reliability.coefficient_of_variation,
                        'num_technicians': len(reliability.technician_ids),
                        'labels': reliability.labels
                    }
                
                review_queue.append(review_item)
            
            logger.info(f"Created expert review queue with {len(review_queue)} items")
            return review_queue
            
        except Exception as e:
            logger.error(f"Expert review queue creation failed: {e}")
            raise
    
    def resolve_quality_flag(self, 
                            flag_id: str,
                            resolution_notes: str,
                            expert_reviewed: bool = False) -> bool:
        """
        Resolve a quality control flag.
        
        Args:
            flag_id: ID of the flag to resolve
            resolution_notes: Notes about the resolution
            expert_reviewed: Whether this was reviewed by an expert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if flag_id not in self.quality_flags:
                logger.error(f"Quality flag {flag_id} not found")
                return False
            
            flag = self.quality_flags[flag_id]
            flag.resolved = True
            flag.resolution_notes = resolution_notes
            flag.expert_reviewed = expert_reviewed
            
            self._save_data()
            
            logger.info(f"Resolved quality flag {flag_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve quality flag {flag_id}: {e}")
            return False
    
    def get_quality_report(self, 
                          days_back: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive quality control report.
        
        Args:
            days_back: Number of days to include in report
            
        Returns:
            Quality control report
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Filter recent flags
            recent_flags = [
                flag for flag in self.quality_flags.values()
                if datetime.fromisoformat(flag.timestamp) >= cutoff_date
            ]
            
            # Count flags by type and severity
            flag_counts = {}
            severity_counts = {}
            
            for flag in recent_flags:
                flag_counts[flag.flag_type] = flag_counts.get(flag.flag_type, 0) + 1
                severity_counts[flag.severity] = severity_counts.get(flag.severity, 0) + 1
            
            # Resolution statistics
            resolved_count = sum(1 for flag in recent_flags if flag.resolved)
            expert_reviewed_count = sum(1 for flag in recent_flags if flag.expert_reviewed)
            
            # Inter-rater reliability statistics
            recent_reliability = [
                rel for rel in self.reliability_analyses.values()
                if any(datetime.fromisoformat(ts) >= cutoff_date for ts in rel.timestamps)
            ]
            
            agreement_levels = {}
            for rel in recent_reliability:
                level = rel.agreement_level
                agreement_levels[level] = agreement_levels.get(level, 0) + 1
            
            # Outlier statistics
            recent_outliers = [
                result for result in self.outlier_results.values()
                if datetime.fromisoformat(result.timestamp) >= cutoff_date and result.is_outlier
            ]
            
            outlier_methods = {}
            for outlier in recent_outliers:
                method = outlier.outlier_method
                outlier_methods[method] = outlier_methods.get(method, 0) + 1
            
            report = {
                'report_period_days': days_back,
                'report_generated': datetime.now().isoformat(),
                'total_flags': len(recent_flags),
                'flags_by_type': flag_counts,
                'flags_by_severity': severity_counts,
                'resolution_stats': {
                    'resolved_count': resolved_count,
                    'resolution_rate': resolved_count / len(recent_flags) if recent_flags else 0,
                    'expert_reviewed_count': expert_reviewed_count,
                    'expert_review_rate': expert_reviewed_count / len(recent_flags) if recent_flags else 0
                },
                'inter_rater_reliability': {
                    'total_analyses': len(recent_reliability),
                    'agreement_levels': agreement_levels,
                    'avg_coefficient_of_variation': np.mean([rel.coefficient_of_variation for rel in recent_reliability]) if recent_reliability else 0
                },
                'outlier_detection': {
                    'total_outliers': len(recent_outliers),
                    'outliers_by_method': outlier_methods,
                    'outlier_rate': len(recent_outliers) / len(self.outlier_results) if self.outlier_results else 0
                },
                'recommendations': self._generate_recommendations(recent_flags, recent_reliability, recent_outliers)
            }
            
            logger.info(f"Generated quality control report for {days_back} days")
            return report
            
        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            raise
    
    def export_quality_data(self, output_path: str) -> None:
        """Export all quality control data to JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'quality_flags': {fid: flag.to_dict() for fid, flag in self.quality_flags.items()},
                'reliability_analyses': {iid: rel.to_dict() for iid, rel in self.reliability_analyses.items()},
                'outlier_results': {rid: result.to_dict() for rid, result in self.outlier_results.items()},
                'configuration': {
                    'outlier_threshold': self.outlier_threshold,
                    'variance_threshold': self.variance_threshold,
                    'agreement_threshold': self.agreement_threshold,
                    'min_labels_for_reliability': self.min_labels_for_reliability
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported quality control data to {output_path}")
            
        except Exception as e:
            logger.error(f"Quality data export failed: {e}")
            raise
    
    # Private methods
    
    def _create_quality_flag(self,
                           flag_type: str,
                           severity: str,
                           image_id: str,
                           description: str,
                           technician_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new quality control flag."""
        flag_id = f"{flag_type}_{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        flag = QualityFlag(
            flag_id=flag_id,
            flag_type=flag_type,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            image_id=image_id,
            technician_id=technician_id,
            description=description,
            metadata=metadata or {}
        )
        
        self.quality_flags[flag_id] = flag
        return flag_id
    
    def _load_existing_data(self):
        """Load existing quality control data from disk."""
        try:
            # Load quality flags
            if self.flags_file.exists():
                with open(self.flags_file, 'r') as f:
                    flags_data = json.load(f)
                self.quality_flags = {
                    fid: QualityFlag.from_dict(flag_data)
                    for fid, flag_data in flags_data.items()
                }
            
            # Load reliability analyses
            if self.reliability_file.exists():
                with open(self.reliability_file, 'r') as f:
                    reliability_data = json.load(f)
                self.reliability_analyses = {
                    iid: InterRaterReliability(**rel_data)
                    for iid, rel_data in reliability_data.items()
                }
            
            # Load outlier results
            if self.outliers_file.exists():
                with open(self.outliers_file, 'r') as f:
                    outliers_data = json.load(f)
                self.outlier_results = {
                    rid: OutlierDetectionResult.from_dict(result_data)
                    for rid, result_data in outliers_data.items()
                }
            
            logger.info(f"Loaded {len(self.quality_flags)} flags, "
                       f"{len(self.reliability_analyses)} reliability analyses, "
                       f"{len(self.outlier_results)} outlier results")
            
        except Exception as e:
            logger.error(f"Failed to load existing quality control data: {e}")
    
    def _save_data(self):
        """Save quality control data to disk."""
        try:
            # Save quality flags
            with open(self.flags_file, 'w') as f:
                json.dump({fid: flag.to_dict() for fid, flag in self.quality_flags.items()}, f, indent=2)
            
            # Save reliability analyses
            with open(self.reliability_file, 'w') as f:
                json.dump({iid: rel.to_dict() for iid, rel in self.reliability_analyses.items()}, f, indent=2)
            
            # Save outlier results
            with open(self.outliers_file, 'w') as f:
                json.dump({rid: result.to_dict() for rid, result in self.outlier_results.items()}, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save quality control data: {e}")
            raise
    
    def _generate_recommendations(self, 
                                flags: List[QualityFlag],
                                reliability_analyses: List[InterRaterReliability],
                                outliers: List[OutlierDetectionResult]) -> List[str]:
        """Generate recommendations based on quality control analysis."""
        recommendations = []
        
        # Flag-based recommendations
        if len(flags) > 20:
            recommendations.append("High number of quality flags detected. Consider reviewing data collection procedures.")
        
        high_severity_flags = [f for f in flags if f.severity in ['high', 'critical']]
        if len(high_severity_flags) > 5:
            recommendations.append("Multiple high-severity quality issues detected. Immediate expert review recommended.")
        
        # Inter-rater reliability recommendations
        poor_agreement = [r for r in reliability_analyses if r.agreement_level == 'poor']
        if len(poor_agreement) > len(reliability_analyses) * 0.2:
            recommendations.append("Poor inter-rater agreement detected in >20% of multi-labeled images. Consider additional technician training.")
        
        # Outlier recommendations
        if len(outliers) > 10:
            recommendations.append("High number of outliers detected. Review data collection and labeling procedures.")
        
        # Resolution rate recommendations
        unresolved_flags = [f for f in flags if not f.resolved]
        if len(unresolved_flags) > len(flags) * 0.5:
            recommendations.append("Over 50% of quality flags remain unresolved. Increase expert review capacity.")
        
        return recommendations