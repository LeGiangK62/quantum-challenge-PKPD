#!/usr/bin/env python3
"""
Active Learning System for PK/PD Modeling
Select samples based on uncertainty for data efficiency
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from models.uncertainty_quantification import MonteCarloDropout, UncertaintyMetrics
from utils.logging import get_logger

logger = get_logger(__name__)

class ActiveLearningSelector:
    """Active Learning selector based on uncertainty"""
    
    def __init__(self, 
                 uncertainty_estimator: MonteCarloDropout,
                 selection_strategy: str = "uncertainty",
                 diversity_weight: float = 0.1,
                 budget: int = 100):
        """
        Args:
            uncertainty_estimator 
            selection_strategy: selection strategy ("uncertainty", "diversity", "hybrid")
            diversity_weight
            budget: number of samples to select
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.selection_strategy = selection_strategy
        self.diversity_weight = diversity_weight
        self.budget = budget
        self.selected_indices = set()
        
    def select_samples(self, 
                      unlabeled_data: torch.Tensor,
                      labeled_indices: List[int] = None) -> List[int]:
        """
        Select the most informative samples based on uncertainty
        
        Args:
            unlabeled_data: unlabeled data
            labeled_indices: already labeled indices
            
        Returns:
            selected sample indices list
        """
        if labeled_indices is None:
            labeled_indices = []
            
        # Calculate uncertainty
        with torch.no_grad():
            predictions, uncertainties = self.uncertainty_estimator.predict_with_uncertainty(unlabeled_data)
        
        if self.selection_strategy == "uncertainty":
            return self._select_by_uncertainty(uncertainties)
        elif self.selection_strategy == "diversity":
            return self._select_by_diversity(unlabeled_data, labeled_indices)
        elif self.selection_strategy == "hybrid":
            return self._select_hybrid(unlabeled_data, uncertainties, labeled_indices)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
    
    def _select_by_uncertainty(self, uncertainties: torch.Tensor) -> List[int]:
        """Select samples with high uncertainty"""
        # Calculate uncertainty scores (epistemic + aleatoric)
        uncertainty_scores = uncertainties.mean(dim=1)  # [batch_size]
        
        # Select top budget samples
        _, selected_indices = torch.topk(uncertainty_scores, min(self.budget, len(uncertainty_scores)))
        return selected_indices.tolist()
    
    def _select_by_diversity(self, 
                           unlabeled_data: torch.Tensor, 
                           labeled_indices: List[int]) -> List[int]:
        """Sample selection based on diversity (clustering)"""
        from sklearn.cluster import KMeans
        
        # K-means clustering
        n_clusters = min(self.budget, len(unlabeled_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(unlabeled_data.cpu().numpy())
        
        # Select closest sample to center in each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = unlabeled_data[cluster_mask]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Calculate distance to cluster center
            distances = torch.norm(cluster_data - torch.tensor(cluster_center), dim=1)
            closest_idx = torch.argmin(distances)
            
            # Convert to original indices
            original_indices = torch.where(torch.tensor(cluster_mask))[0]
            selected_indices.append(original_indices[closest_idx].item())
        
        return selected_indices
    
    def _select_hybrid(self, 
                      unlabeled_data: torch.Tensor,
                      uncertainties: torch.Tensor,
                      labeled_indices: List[int]) -> List[int]:
        """Hybrid selection based on uncertainty and diversity"""
        # Uncertainty scores
        uncertainty_scores = uncertainties.mean(dim=1)
        
        # Diversity scores (distance from already selected samples)
        diversity_scores = torch.zeros(len(unlabeled_data))
        if labeled_indices:
            labeled_data = unlabeled_data[labeled_indices]
            for i, sample in enumerate(unlabeled_data):
                distances = torch.norm(sample.unsqueeze(0) - labeled_data, dim=1)
                diversity_scores[i] = distances.min()
        else:
            diversity_scores.fill_(1.0)  # All samples have same diversity in first selection
        
        # Normalization
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)
        diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
        
        # Hybrid scores
        hybrid_scores = uncertainty_scores + self.diversity_weight * diversity_scores
        
        # Select top budget samples
        _, selected_indices = torch.topk(hybrid_scores, min(self.budget, len(hybrid_scores)))
        return selected_indices.tolist()

class AdaptiveActiveLearning:
    """Adaptive Active Learning System"""
    
    def __init__(self, 
                 model: nn.Module,
                 uncertainty_estimator: MonteCarloDropout,
                 initial_budget: int = 100,
                 growth_rate: float = 1.2):
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self.initial_budget = initial_budget
        self.growth_rate = growth_rate
        self.iteration = 0
        self.performance_history = []
        
    def adaptive_selection(self, 
                          unlabeled_data: torch.Tensor,
                          current_performance: float) -> Tuple[List[int], Dict]:
        """
        Adjust sample selection strategy adaptively based on performance
        
        Args:
            unlabeled_data: unlabeled data
            current_performance: current model performance (lower is more samples needed)
            
        Returns:
            selected sample indices and metadata
        """
        self.performance_history.append(current_performance)
        self.iteration += 1
        
        # Adjust budget based on performance
        if current_performance < 0.7:  # Select more samples if performance is low
            budget = int(self.initial_budget * self.growth_rate)
            strategy = "uncertainty"  # Prioritize uncertainty
        elif current_performance < 0.85:
            budget = self.initial_budget
            strategy = "hybrid"  # Hybrid approach
        else:
            budget = max(10, int(self.initial_budget * 0.5))  # Select fewer if performance is good
            strategy = "diversity"  # Prioritize diversity
        
        # Create Active Learning selector
        selector = ActiveLearningSelector(
            uncertainty_estimator=self.uncertainty_estimator,
            selection_strategy=strategy,
            budget=budget
        )
        
        # Sample selection
        selected_indices = selector.select_samples(unlabeled_data)
        
        # Metadata
        metadata = {
            "iteration": self.iteration,
            "strategy": strategy,
            "budget": budget,
            "performance": current_performance,
            "uncertainty_threshold": self._calculate_uncertainty_threshold(unlabeled_data)
        }
        
        logger.info(f"Active Learning Iteration {self.iteration}: "
                   f"Strategy={strategy}, Budget={budget}, Performance={current_performance:.3f}")
        
        return selected_indices, metadata
    
    def _calculate_uncertainty_threshold(self, data: torch.Tensor) -> float:
        """Calculate uncertainty threshold"""
        with torch.no_grad():
            _, uncertainties = self.uncertainty_estimator.predict_with_uncertainty(data)
        return uncertainties.mean().item()

class UncertaintyBasedDataAugmentation:
    """Data augmentation based on uncertainty"""
    
    def __init__(self, uncertainty_estimator: MonteCarloDropout):
        self.uncertainty_estimator = uncertainty_estimator
    
    def augment_uncertain_samples(self, 
                                 data: torch.Tensor,
                                 labels: torch.Tensor,
                                 uncertainty_threshold: float = 0.1,
                                 augmentation_factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform data augmentation for samples with high uncertainty
        
        Args:
            data: input data
            labels: labels
            uncertainty_threshold: uncertainty threshold
            augmentation_factor: augmentation factor
            
        Returns:
            augmented data and labels
        """
        with torch.no_grad():
            _, uncertainties = self.uncertainty_estimator.predict_with_uncertainty(data)
        
        # Identify high uncertainty samples
        high_uncertainty_mask = uncertainties.mean(dim=1) > uncertainty_threshold
        uncertain_data = data[high_uncertainty_mask]
        uncertain_labels = labels[high_uncertainty_mask]
        
        if len(uncertain_data) == 0:
            return data, labels
        
        # Data augmentation (simple noise addition)
        augmented_data = []
        augmented_labels = []
        
        for _ in range(augmentation_factor):
            # Add Gaussian noise
            noise = torch.randn_like(uncertain_data) * 0.01
            augmented_data.append(uncertain_data + noise)
            augmented_labels.append(uncertain_labels)
        
        # Combine original and augmented data
        all_data = torch.cat([data] + augmented_data, dim=0)
        all_labels = torch.cat([labels] + augmented_labels, dim=0)
        
        logger.info(f"Data augmentation: {len(uncertain_data)} uncertain samples "
                   f"augmented by factor {augmentation_factor}")
        
        return all_data, all_labels

# Usage example function
def create_active_learning_system(model: nn.Module, 
                                 config) -> AdaptiveActiveLearning:
    """Active Learning System creation"""
    uncertainty_estimator = MonteCarloDropout(
        model=model,
        n_samples=config.uncertainty_samples,
        dropout_rate=config.uncertainty_dropout
    )
    
    return AdaptiveActiveLearning(
        model=model,
        uncertainty_estimator=uncertainty_estimator,
        initial_budget=config.active_learning_budget,
        growth_rate=config.active_learning_growth_rate
    )
    