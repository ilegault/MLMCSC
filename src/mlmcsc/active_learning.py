#!/usr/bin/env python3
"""
Active Learning System for MLMCSC

This module implements advanced active learning techniques including:
- Uncertainty-based sampling (prioritize images where model is uncertain)
- Query-by-committee for diverse model predictions
- Diversity-based sampling to ensure representative data
- Adaptive sampling strategies based on model performance
- Integration with quality control and monitoring systems
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningQuery:
    """Container for active learning query."""
    query_id: str
    image_id: str
    features: Dict[str, float]
    uncertainty_score: float
    diversity_score: float
    combined_score: float
    query_strategy: str
    timestamp: str
    priority: str  # 'high', 'medium', 'low'
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActiveLearningQuery':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class QueryResult:
    """Container for query result after technician labeling."""
    query_id: str
    image_id: str
    technician_id: str
    label: float
    confidence: Optional[float]
    timestamp: str
    feedback_quality: Optional[str] = None  # 'excellent', 'good', 'fair', 'poor'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class QueryStrategy(ABC):
    """Abstract base class for query strategies."""
    
    @abstractmethod
    def select_queries(self, 
                      candidates: List[Dict[str, Any]], 
                      model: Any, 
                      n_queries: int,
                      **kwargs) -> List[ActiveLearningQuery]:
        """
        Select queries based on the strategy.
        
        Args:
            candidates: List of candidate samples
            model: Trained model
            n_queries: Number of queries to select
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of selected queries
        """
        pass


class UncertaintyStrategy(QueryStrategy):
    """Uncertainty-based query strategy."""
    
    def __init__(self, uncertainty_method: str = 'prediction_variance'):
        """
        Initialize uncertainty strategy.
        
        Args:
            uncertainty_method: Method for uncertainty estimation
                - 'prediction_variance': Use prediction variance
                - 'confidence_interval': Use confidence intervals
                - 'ensemble_disagreement': Use ensemble disagreement
        """
        self.uncertainty_method = uncertainty_method
    
    def select_queries(self, 
                      candidates: List[Dict[str, Any]], 
                      model: Any, 
                      n_queries: int,
                      **kwargs) -> List[ActiveLearningQuery]:
        """Select queries based on prediction uncertainty."""
        try:
            if not candidates:
                return []
            
            # Extract features
            features_list = []
            for candidate in candidates:
                if 'features' in candidate:
                    features_list.append(list(candidate['features'].values()))
                else:
                    features_list.append(list(candidate.values()))
            
            X = np.array(features_list)
            
            # Calculate uncertainty scores
            uncertainty_scores = self._calculate_uncertainty(X, model, **kwargs)
            
            # Create queries
            queries = []
            for i, (candidate, uncertainty) in enumerate(zip(candidates, uncertainty_scores)):
                query_id = f"uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                
                # Determine priority based on uncertainty
                if uncertainty > 0.8:
                    priority = 'high'
                elif uncertainty > 0.5:
                    priority = 'medium'
                else:
                    priority = 'low'
                
                query = ActiveLearningQuery(
                    query_id=query_id,
                    image_id=candidate.get('image_id', f'img_{i}'),
                    features=candidate.get('features', candidate),
                    uncertainty_score=uncertainty,
                    diversity_score=0.0,  # Not used in this strategy
                    combined_score=uncertainty,
                    query_strategy='uncertainty',
                    timestamp=datetime.now().isoformat(),
                    priority=priority,
                    metadata={'uncertainty_method': self.uncertainty_method}
                )
                queries.append(query)
            
            # Sort by uncertainty and return top n_queries
            queries.sort(key=lambda q: q.uncertainty_score, reverse=True)
            return queries[:n_queries]
            
        except Exception as e:
            logger.error(f"Uncertainty query selection failed: {e}")
            raise
    
    def _calculate_uncertainty(self, X: np.ndarray, model: Any, **kwargs) -> np.ndarray:
        """Calculate uncertainty scores for samples."""
        try:
            if self.uncertainty_method == 'prediction_variance':
                # Use prediction variance (for ensemble models)
                if hasattr(model, 'estimators_'):
                    # Ensemble model
                    predictions = np.array([estimator.predict(X) for estimator in model.estimators_])
                    uncertainty = np.var(predictions, axis=0)
                else:
                    # Single model - use distance from training data as proxy
                    predictions = model.predict(X)
                    # Normalize to 0-1 range
                    uncertainty = np.abs(predictions - np.mean(predictions)) / (np.std(predictions) + 1e-8)
                
            elif self.uncertainty_method == 'confidence_interval':
                # Use prediction confidence intervals
                predictions = model.predict(X)
                # Simple heuristic: uncertainty based on prediction magnitude
                uncertainty = 1.0 / (1.0 + np.abs(predictions))
                
            elif self.uncertainty_method == 'ensemble_disagreement':
                # Use ensemble disagreement
                ensemble_models = kwargs.get('ensemble_models', [model])
                if len(ensemble_models) > 1:
                    predictions = np.array([m.predict(X) for m in ensemble_models])
                    uncertainty = np.std(predictions, axis=0)
                else:
                    # Fallback to prediction variance
                    predictions = model.predict(X)
                    uncertainty = np.abs(predictions - np.mean(predictions)) / (np.std(predictions) + 1e-8)
            
            else:
                raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
            
            # Normalize to 0-1 range
            if len(uncertainty) > 1:
                uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-8)
            else:
                uncertainty = np.array([0.5])  # Default uncertainty for single sample
            
            return uncertainty
            
        except Exception as e:
            logger.error(f"Uncertainty calculation failed: {e}")
            return np.zeros(len(X))


class DiversityStrategy(QueryStrategy):
    """Diversity-based query strategy."""
    
    def __init__(self, diversity_method: str = 'kmeans_centers'):
        """
        Initialize diversity strategy.
        
        Args:
            diversity_method: Method for diversity sampling
                - 'kmeans_centers': Select cluster centers
                - 'max_distance': Select samples with maximum distance
                - 'representative': Select representative samples
        """
        self.diversity_method = diversity_method
    
    def select_queries(self, 
                      candidates: List[Dict[str, Any]], 
                      model: Any, 
                      n_queries: int,
                      **kwargs) -> List[ActiveLearningQuery]:
        """Select queries based on feature diversity."""
        try:
            if not candidates:
                return []
            
            # Extract features
            features_list = []
            for candidate in candidates:
                if 'features' in candidate:
                    features_list.append(list(candidate['features'].values()))
                else:
                    features_list.append(list(candidate.values()))
            
            X = np.array(features_list)
            
            # Calculate diversity scores
            diversity_scores, selected_indices = self._calculate_diversity(X, n_queries, **kwargs)
            
            # Create queries for selected samples
            queries = []
            for i, idx in enumerate(selected_indices):
                candidate = candidates[idx]
                query_id = f"diversity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                
                # Determine priority based on diversity
                diversity_score = diversity_scores[idx] if idx < len(diversity_scores) else 0.5
                if diversity_score > 0.7:
                    priority = 'high'
                elif diversity_score > 0.4:
                    priority = 'medium'
                else:
                    priority = 'low'
                
                query = ActiveLearningQuery(
                    query_id=query_id,
                    image_id=candidate.get('image_id', f'img_{idx}'),
                    features=candidate.get('features', candidate),
                    uncertainty_score=0.0,  # Not used in this strategy
                    diversity_score=diversity_score,
                    combined_score=diversity_score,
                    query_strategy='diversity',
                    timestamp=datetime.now().isoformat(),
                    priority=priority,
                    metadata={'diversity_method': self.diversity_method}
                )
                queries.append(query)
            
            return queries
            
        except Exception as e:
            logger.error(f"Diversity query selection failed: {e}")
            raise
    
    def _calculate_diversity(self, X: np.ndarray, n_queries: int, **kwargs) -> Tuple[np.ndarray, List[int]]:
        """Calculate diversity scores and select diverse samples."""
        try:
            if self.diversity_method == 'kmeans_centers':
                # Use K-means clustering to find diverse samples
                n_clusters = min(n_queries, len(X))
                if n_clusters < 2:
                    return np.array([1.0] * len(X)), list(range(min(n_queries, len(X))))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                # Select samples closest to cluster centers
                selected_indices = []
                diversity_scores = np.zeros(len(X))
                
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    if not np.any(cluster_mask):
                        continue
                    
                    cluster_samples = X[cluster_mask]
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    
                    # Find sample closest to cluster center
                    distances = np.linalg.norm(cluster_samples - cluster_center, axis=1)
                    closest_idx = np.argmin(distances)
                    
                    # Get original index
                    original_indices = np.where(cluster_mask)[0]
                    selected_idx = original_indices[closest_idx]
                    selected_indices.append(selected_idx)
                    
                    # Assign diversity score (inverse of distance to center)
                    diversity_scores[selected_idx] = 1.0 / (1.0 + distances[closest_idx])
                
            elif self.diversity_method == 'max_distance':
                # Select samples with maximum pairwise distances
                if len(X) < 2:
                    return np.array([1.0] * len(X)), list(range(len(X)))
                
                # Calculate pairwise distances
                distances = squareform(pdist(X))
                
                # Greedy selection of diverse samples
                selected_indices = [0]  # Start with first sample
                
                for _ in range(min(n_queries - 1, len(X) - 1)):
                    # Find sample with maximum minimum distance to selected samples
                    remaining_indices = [i for i in range(len(X)) if i not in selected_indices]
                    if not remaining_indices:
                        break
                    
                    max_min_distance = -1
                    best_idx = remaining_indices[0]
                    
                    for idx in remaining_indices:
                        min_distance = min(distances[idx][selected_idx] for selected_idx in selected_indices)
                        if min_distance > max_min_distance:
                            max_min_distance = min_distance
                            best_idx = idx
                    
                    selected_indices.append(best_idx)
                
                # Calculate diversity scores
                diversity_scores = np.zeros(len(X))
                for idx in selected_indices:
                    if selected_indices:
                        other_selected = [i for i in selected_indices if i != idx]
                        if other_selected:
                            min_dist = min(distances[idx][other_idx] for other_idx in other_selected)
                            diversity_scores[idx] = min_dist / np.max(distances)
                        else:
                            diversity_scores[idx] = 1.0
                
            elif self.diversity_method == 'representative':
                # Select representative samples using density estimation
                # Simple approach: select samples in low-density regions
                
                # Calculate local density for each sample
                distances = squareform(pdist(X))
                bandwidth = np.median(distances[distances > 0]) * 0.5
                
                densities = np.zeros(len(X))
                for i in range(len(X)):
                    # Count neighbors within bandwidth
                    neighbors = np.sum(distances[i] < bandwidth) - 1  # Exclude self
                    densities[i] = neighbors / len(X)
                
                # Select samples with low density (more representative)
                diversity_scores = 1.0 - densities / (np.max(densities) + 1e-8)
                selected_indices = np.argsort(diversity_scores)[-n_queries:].tolist()
            
            else:
                raise ValueError(f"Unknown diversity method: {self.diversity_method}")
            
            return diversity_scores, selected_indices[:n_queries]
            
        except Exception as e:
            logger.error(f"Diversity calculation failed: {e}")
            return np.zeros(len(X)), list(range(min(n_queries, len(X))))


class HybridStrategy(QueryStrategy):
    """Hybrid strategy combining uncertainty and diversity."""
    
    def __init__(self, 
                 uncertainty_weight: float = 0.7,
                 diversity_weight: float = 0.3,
                 uncertainty_method: str = 'prediction_variance',
                 diversity_method: str = 'kmeans_centers'):
        """
        Initialize hybrid strategy.
        
        Args:
            uncertainty_weight: Weight for uncertainty component
            diversity_weight: Weight for diversity component
            uncertainty_method: Method for uncertainty estimation
            diversity_method: Method for diversity sampling
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.uncertainty_strategy = UncertaintyStrategy(uncertainty_method)
        self.diversity_strategy = DiversityStrategy(diversity_method)
    
    def select_queries(self, 
                      candidates: List[Dict[str, Any]], 
                      model: Any, 
                      n_queries: int,
                      **kwargs) -> List[ActiveLearningQuery]:
        """Select queries using hybrid uncertainty-diversity approach."""
        try:
            if not candidates:
                return []
            
            # Get uncertainty scores
            uncertainty_queries = self.uncertainty_strategy.select_queries(
                candidates, model, len(candidates), **kwargs
            )
            
            # Get diversity scores
            diversity_queries = self.diversity_strategy.select_queries(
                candidates, model, len(candidates), **kwargs
            )
            
            # Create mapping from image_id to scores
            uncertainty_map = {q.image_id: q.uncertainty_score for q in uncertainty_queries}
            diversity_map = {q.image_id: q.diversity_score for q in diversity_queries}
            
            # Combine scores
            combined_queries = []
            for i, candidate in enumerate(candidates):
                image_id = candidate.get('image_id', f'img_{i}')
                
                uncertainty_score = uncertainty_map.get(image_id, 0.0)
                diversity_score = diversity_map.get(image_id, 0.0)
                
                combined_score = (self.uncertainty_weight * uncertainty_score + 
                                self.diversity_weight * diversity_score)
                
                query_id = f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                
                # Determine priority based on combined score
                if combined_score > 0.7:
                    priority = 'high'
                elif combined_score > 0.4:
                    priority = 'medium'
                else:
                    priority = 'low'
                
                query = ActiveLearningQuery(
                    query_id=query_id,
                    image_id=image_id,
                    features=candidate.get('features', candidate),
                    uncertainty_score=uncertainty_score,
                    diversity_score=diversity_score,
                    combined_score=combined_score,
                    query_strategy='hybrid',
                    timestamp=datetime.now().isoformat(),
                    priority=priority,
                    metadata={
                        'uncertainty_weight': self.uncertainty_weight,
                        'diversity_weight': self.diversity_weight
                    }
                )
                combined_queries.append(query)
            
            # Sort by combined score and return top n_queries
            combined_queries.sort(key=lambda q: q.combined_score, reverse=True)
            return combined_queries[:n_queries]
            
        except Exception as e:
            logger.error(f"Hybrid query selection failed: {e}")
            raise


class ActiveLearningSystem:
    """
    Comprehensive active learning system for intelligent sample selection.
    
    Features:
    - Multiple query strategies (uncertainty, diversity, hybrid)
    - Adaptive strategy selection based on model performance
    - Integration with quality control and monitoring
    - Query result tracking and feedback analysis
    - Performance-based strategy optimization
    """
    
    def __init__(self, 
                 storage_path: str = "active_learning",
                 default_strategy: str = "hybrid",
                 max_queries_per_batch: int = 10,
                 min_confidence_threshold: float = 0.3):
        """
        Initialize the active learning system.
        
        Args:
            storage_path: Path to store active learning data
            default_strategy: Default query strategy
            max_queries_per_batch: Maximum queries per batch
            min_confidence_threshold: Minimum confidence for automatic labeling
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.default_strategy = default_strategy
        self.max_queries_per_batch = max_queries_per_batch
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize strategies
        self.strategies = {
            'uncertainty': UncertaintyStrategy(),
            'diversity': DiversityStrategy(),
            'hybrid': HybridStrategy()
        }
        
        # Storage
        self.queries_file = self.storage_path / "queries.json"
        self.results_file = self.storage_path / "results.json"
        self.performance_file = self.storage_path / "strategy_performance.json"
        
        # Data
        self.active_queries: Dict[str, ActiveLearningQuery] = {}
        self.query_results: Dict[str, QueryResult] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"ActiveLearningSystem initialized at {self.storage_path}")
        logger.info(f"Default strategy: {default_strategy}, Max queries per batch: {max_queries_per_batch}")
    
    def generate_queries(self, 
                        candidates: List[Dict[str, Any]],
                        model: Any,
                        n_queries: int = None,
                        strategy: str = None,
                        **kwargs) -> List[ActiveLearningQuery]:
        """
        Generate active learning queries for candidate samples.
        
        Args:
            candidates: List of candidate samples with features
            model: Trained model for uncertainty estimation
            n_queries: Number of queries to generate (default: max_queries_per_batch)
            strategy: Query strategy to use (default: default_strategy)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of generated queries
        """
        try:
            n_queries = n_queries or self.max_queries_per_batch
            strategy = strategy or self._select_best_strategy()
            
            if strategy not in self.strategies:
                logger.warning(f"Unknown strategy {strategy}, using {self.default_strategy}")
                strategy = self.default_strategy
            
            # Generate queries using selected strategy
            query_strategy = self.strategies[strategy]
            queries = query_strategy.select_queries(candidates, model, n_queries, **kwargs)
            
            # Store queries
            for query in queries:
                self.active_queries[query.query_id] = query
            
            self._save_data()
            
            logger.info(f"Generated {len(queries)} queries using {strategy} strategy")
            return queries
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            raise
    
    def submit_query_result(self, 
                           query_id: str,
                           technician_id: str,
                           label: float,
                           confidence: Optional[float] = None,
                           feedback_quality: Optional[str] = None) -> bool:
        """
        Submit result for an active learning query.
        
        Args:
            query_id: ID of the query
            technician_id: ID of the technician providing the label
            label: Provided label
            confidence: Confidence in the label (optional)
            feedback_quality: Quality of the feedback (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if query_id not in self.active_queries:
                logger.error(f"Query {query_id} not found")
                return False
            
            query = self.active_queries[query_id]
            
            # Create query result
            result = QueryResult(
                query_id=query_id,
                image_id=query.image_id,
                technician_id=technician_id,
                label=label,
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                feedback_quality=feedback_quality
            )
            
            # Store result
            self.query_results[query_id] = result
            
            # Remove from active queries
            del self.active_queries[query_id]
            
            # Update strategy performance
            self._update_strategy_performance(query, result)
            
            self._save_data()
            
            logger.info(f"Submitted result for query {query_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit query result: {e}")
            return False
    
    def get_priority_queries(self, max_queries: int = None) -> List[ActiveLearningQuery]:
        """
        Get priority queries for technician review.
        
        Args:
            max_queries: Maximum number of queries to return
            
        Returns:
            List of priority queries sorted by importance
        """
        try:
            max_queries = max_queries or self.max_queries_per_batch
            
            # Get all active queries
            queries = list(self.active_queries.values())
            
            # Sort by priority and combined score
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            queries.sort(
                key=lambda q: (priority_order.get(q.priority, 0), q.combined_score),
                reverse=True
            )
            
            return queries[:max_queries]
            
        except Exception as e:
            logger.error(f"Failed to get priority queries: {e}")
            return []
    
    def analyze_query_effectiveness(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze the effectiveness of active learning queries.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Analysis results
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent results
            recent_results = [
                result for result in self.query_results.values()
                if datetime.fromisoformat(result.timestamp) >= cutoff_time
            ]
            
            if not recent_results:
                return {'error': 'No recent query results available'}
            
            # Analyze by strategy
            strategy_analysis = {}
            for result in recent_results:
                query = self._get_query_for_result(result.query_id)
                if query:
                    strategy = query.query_strategy
                    if strategy not in strategy_analysis:
                        strategy_analysis[strategy] = {
                            'total_queries': 0,
                            'avg_uncertainty': 0,
                            'avg_diversity': 0,
                            'avg_label': 0,
                            'label_variance': 0,
                            'response_time': 0
                        }
                    
                    strategy_analysis[strategy]['total_queries'] += 1
                    strategy_analysis[strategy]['avg_uncertainty'] += query.uncertainty_score
                    strategy_analysis[strategy]['avg_diversity'] += query.diversity_score
                    strategy_analysis[strategy]['avg_label'] += result.label
            
            # Calculate averages
            for strategy, data in strategy_analysis.items():
                count = data['total_queries']
                if count > 0:
                    data['avg_uncertainty'] /= count
                    data['avg_diversity'] /= count
                    data['avg_label'] /= count
            
            # Overall statistics
            all_labels = [r.label for r in recent_results]
            all_confidences = [r.confidence for r in recent_results if r.confidence is not None]
            
            analysis = {
                'period_days': days_back,
                'total_queries_completed': len(recent_results),
                'strategy_breakdown': strategy_analysis,
                'overall_stats': {
                    'avg_label': np.mean(all_labels),
                    'label_std': np.std(all_labels),
                    'avg_confidence': np.mean(all_confidences) if all_confidences else None,
                    'unique_technicians': len(set(r.technician_id for r in recent_results))
                },
                'effectiveness_metrics': self._calculate_effectiveness_metrics(recent_results)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query effectiveness analysis failed: {e}")
            raise
    
    def optimize_strategy_parameters(self) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on historical performance.
        
        Returns:
            Optimization results and recommendations
        """
        try:
            if not self.strategy_performance:
                return {'message': 'No performance data available for optimization'}
            
            # Analyze strategy performance
            best_strategy = None
            best_score = -1
            
            optimization_results = {}
            
            for strategy, performance in self.strategy_performance.items():
                avg_score = performance.get('avg_effectiveness', 0)
                total_queries = performance.get('total_queries', 0)
                
                if total_queries >= 10 and avg_score > best_score:  # Minimum queries for reliability
                    best_score = avg_score
                    best_strategy = strategy
                
                optimization_results[strategy] = {
                    'avg_effectiveness': avg_score,
                    'total_queries': total_queries,
                    'recommendation': 'good' if avg_score > 0.7 else 'needs_improvement'
                }
            
            # Generate recommendations
            recommendations = []
            
            if best_strategy:
                recommendations.append(f"Best performing strategy: {best_strategy} (score: {best_score:.3f})")
                
                # Update default strategy if significantly better
                if best_strategy != self.default_strategy and best_score > 0.8:
                    recommendations.append(f"Consider switching default strategy from {self.default_strategy} to {best_strategy}")
            
            # Parameter recommendations
            if 'hybrid' in self.strategy_performance:
                hybrid_perf = self.strategy_performance['hybrid']
                if hybrid_perf.get('avg_effectiveness', 0) < 0.6:
                    recommendations.append("Consider adjusting hybrid strategy weights for better performance")
            
            return {
                'best_strategy': best_strategy,
                'best_score': best_score,
                'strategy_performance': optimization_results,
                'recommendations': recommendations,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            raise
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Get active learning progress and statistics.
        
        Returns:
            Progress statistics
        """
        try:
            total_queries_generated = len(self.active_queries) + len(self.query_results)
            completed_queries = len(self.query_results)
            pending_queries = len(self.active_queries)
            
            # Calculate completion rate
            completion_rate = completed_queries / total_queries_generated if total_queries_generated > 0 else 0
            
            # Analyze query priorities
            priority_distribution = {'high': 0, 'medium': 0, 'low': 0}
            for query in self.active_queries.values():
                priority_distribution[query.priority] = priority_distribution.get(query.priority, 0) + 1
            
            # Recent activity
            recent_results = [
                r for r in self.query_results.values()
                if datetime.fromisoformat(r.timestamp) >= datetime.now() - timedelta(days=7)
            ]
            
            progress = {
                'total_queries_generated': total_queries_generated,
                'completed_queries': completed_queries,
                'pending_queries': pending_queries,
                'completion_rate': completion_rate,
                'priority_distribution': priority_distribution,
                'recent_activity': {
                    'queries_completed_last_7_days': len(recent_results),
                    'avg_queries_per_day': len(recent_results) / 7,
                    'active_technicians': len(set(r.technician_id for r in recent_results))
                },
                'strategy_usage': self._get_strategy_usage_stats()
            }
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to get learning progress: {e}")
            raise
    
    def export_active_learning_data(self, output_path: str) -> None:
        """Export all active learning data to JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'active_queries': {qid: query.to_dict() for qid, query in self.active_queries.items()},
                'query_results': {qid: result.to_dict() for qid, result in self.query_results.items()},
                'strategy_performance': self.strategy_performance,
                'configuration': {
                    'default_strategy': self.default_strategy,
                    'max_queries_per_batch': self.max_queries_per_batch,
                    'min_confidence_threshold': self.min_confidence_threshold
                },
                'statistics': {
                    'total_queries_generated': len(self.active_queries) + len(self.query_results),
                    'completed_queries': len(self.query_results),
                    'pending_queries': len(self.active_queries)
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported active learning data to {output_path}")
            
        except Exception as e:
            logger.error(f"Active learning data export failed: {e}")
            raise
    
    # Private methods
    
    def _select_best_strategy(self) -> str:
        """Select the best performing strategy based on historical data."""
        if not self.strategy_performance:
            return self.default_strategy
        
        best_strategy = self.default_strategy
        best_score = 0
        
        for strategy, performance in self.strategy_performance.items():
            avg_effectiveness = performance.get('avg_effectiveness', 0)
            total_queries = performance.get('total_queries', 0)
            
            # Require minimum queries for reliability
            if total_queries >= 5 and avg_effectiveness > best_score:
                best_score = avg_effectiveness
                best_strategy = strategy
        
        return best_strategy
    
    def _update_strategy_performance(self, query: ActiveLearningQuery, result: QueryResult):
        """Update performance metrics for the query strategy."""
        strategy = query.query_strategy
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'total_queries': 0,
                'avg_effectiveness': 0,
                'total_effectiveness': 0
            }
        
        # Calculate effectiveness score (simple heuristic)
        effectiveness = self._calculate_query_effectiveness(query, result)
        
        perf = self.strategy_performance[strategy]
        perf['total_queries'] += 1
        perf['total_effectiveness'] += effectiveness
        perf['avg_effectiveness'] = perf['total_effectiveness'] / perf['total_queries']
    
    def _calculate_query_effectiveness(self, query: ActiveLearningQuery, result: QueryResult) -> float:
        """Calculate effectiveness score for a query-result pair."""
        # Simple effectiveness calculation based on:
        # 1. Query priority (higher priority = more effective if labeled)
        # 2. Uncertainty score (higher uncertainty = more informative)
        # 3. Result confidence (higher confidence = better quality)
        
        priority_score = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(query.priority, 0.5)
        uncertainty_score = query.uncertainty_score
        confidence_score = result.confidence if result.confidence is not None else 0.5
        
        # Weighted combination
        effectiveness = (0.4 * priority_score + 
                        0.4 * uncertainty_score + 
                        0.2 * confidence_score)
        
        return effectiveness
    
    def _calculate_effectiveness_metrics(self, results: List[QueryResult]) -> Dict[str, float]:
        """Calculate overall effectiveness metrics."""
        if not results:
            return {}
        
        # Information gain proxy: variance in labels
        labels = [r.label for r in results]
        label_variance = np.var(labels)
        
        # Confidence metrics
        confidences = [r.confidence for r in results if r.confidence is not None]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Quality metrics
        quality_scores = []
        quality_mapping = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4}
        for result in results:
            if result.feedback_quality:
                quality_scores.append(quality_mapping.get(result.feedback_quality, 0.5))
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        return {
            'label_variance': label_variance,
            'avg_confidence': avg_confidence,
            'avg_quality': avg_quality,
            'information_gain_proxy': label_variance * avg_confidence
        }
    
    def _get_query_for_result(self, query_id: str) -> Optional[ActiveLearningQuery]:
        """Get the original query for a result (from stored data)."""
        # This would typically load from persistent storage
        # For now, return None as queries are removed after completion
        return None
    
    def _get_strategy_usage_stats(self) -> Dict[str, int]:
        """Get statistics on strategy usage."""
        strategy_counts = {}
        
        # Count active queries by strategy
        for query in self.active_queries.values():
            strategy = query.query_strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Count completed queries by strategy (from performance data)
        for strategy, perf in self.strategy_performance.items():
            total = perf.get('total_queries', 0)
            if strategy in strategy_counts:
                strategy_counts[strategy] += total
            else:
                strategy_counts[strategy] = total
        
        return strategy_counts
    
    def _load_existing_data(self):
        """Load existing active learning data from disk."""
        try:
            # Load active queries
            if self.queries_file.exists():
                with open(self.queries_file, 'r') as f:
                    queries_data = json.load(f)
                self.active_queries = {
                    qid: ActiveLearningQuery.from_dict(query_data)
                    for qid, query_data in queries_data.items()
                }
            
            # Load query results
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    results_data = json.load(f)
                self.query_results = {
                    qid: QueryResult(**result_data)
                    for qid, result_data in results_data.items()
                }
            
            # Load strategy performance
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    self.strategy_performance = json.load(f)
            
            logger.info(f"Loaded {len(self.active_queries)} active queries, "
                       f"{len(self.query_results)} results, "
                       f"{len(self.strategy_performance)} strategy performance records")
            
        except Exception as e:
            logger.error(f"Failed to load existing active learning data: {e}")
    
    def _save_data(self):
        """Save active learning data to disk."""
        try:
            # Save active queries
            with open(self.queries_file, 'w') as f:
                json.dump({qid: query.to_dict() for qid, query in self.active_queries.items()}, f, indent=2)
            
            # Save query results
            with open(self.results_file, 'w') as f:
                json.dump({qid: result.to_dict() for qid, result in self.query_results.items()}, f, indent=2)
            
            # Save strategy performance
            with open(self.performance_file, 'w') as f:
                json.dump(self.strategy_performance, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save active learning data: {e}")
            raise