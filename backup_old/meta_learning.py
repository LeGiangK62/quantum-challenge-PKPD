#!/usr/bin/env python3
"""
Meta-Learning System for PK/PD Modeling
Implementing meta-learning system for PK/PD modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from models.uncertainty_quantification import MonteCarloDropout
from utils.logging import get_logger

logger = get_logger(__name__)

class MetaLearner(nn.Module):
    """Meta-learning based fast adaptation system"""
    
    def __init__(self, 
                 base_model: nn.Module,
                 meta_lr: float = 0.01,
                 adaptation_steps: int = 5,
                 uncertainty_threshold: float = 0.1):
        super().__init__()
        
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.uncertainty_threshold = uncertainty_threshold
        
        # Meta parameters (initial weights)
        self.meta_params = {}
        for name, param in self.base_model.named_parameters():
            self.meta_params[name] = param.clone().detach()
        
        # Uncertainty estimator
        self.uncertainty_estimator = MonteCarloDropout(
            model=self.base_model,
            n_samples=20,
            dropout_rate=0.1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Basic model forward pass"""
        return self.base_model(x)
    
    def adapt_to_task(self, 
                     support_data: torch.Tensor,
                     support_labels: torch.Tensor,
                     query_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast adaptation to new task
        
        Args:
            support_data: Support data (adaptation)
            support_labels: Support labels
            query_data: Query data (prediction)
            
        Returns:
            Adapted prediction and uncertainty
        """
        # Save current model state
        original_params = {}
        for name, param in self.base_model.named_parameters():
            original_params[name] = param.clone()
        
        # Fast adaptation (a few steps)
        adapted_params = self._fast_adaptation(
            support_data, support_labels, self.adaptation_steps
        )
        
        # Predict with adapted parameters
        with torch.no_grad():
            # Temporary parameter replacement
            for name, param in self.base_model.named_parameters():
                param.data = adapted_params[name]
            
            # Prediction
            predictions = self.base_model(query_data)
            
            # Uncertainty estimation
            _, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(query_data)
        
        # Restore original parameters
        for name, param in self.base_model.named_parameters():
            param.data = original_params[name]
        
        return predictions, uncertainty
    
    def _fast_adaptation(self, 
                        support_data: torch.Tensor,
                        support_labels: torch.Tensor,
                        steps: int) -> Dict[str, torch.Tensor]:
        """Fast adaptation"""
        adapted_params = {}
        for name, param in self.base_model.named_parameters():
            adapted_params[name] = param.clone()
        
        # Gradient-based adaptation
        for step in range(steps):
            # Forward pass
            predictions = self.base_model(support_data)
            loss = F.mse_loss(predictions, support_labels)
            
            # Backward pass
            grads = torch.autograd.grad(loss, self.base_model.parameters(), create_graph=True)
            
            # Parameter update
            for i, (name, param) in enumerate(self.base_model.named_parameters()):
                adapted_params[name] = adapted_params[name] - self.meta_lr * grads[i]
                param.data = adapted_params[name]
        
        return adapted_params

class PatientGroupMetaLearner(nn.Module):
    """Patient group meta-learning"""
    
    def __init__(self, 
                 base_model: nn.Module,
                 n_patient_groups: int = 5,
                 meta_lr: float = 0.01,
                 adaptation_steps: int = 3):
        super().__init__()
        
        self.base_model = base_model
        self.n_patient_groups = n_patient_groups
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        
        # Patient group meta-parameters
        self.group_meta_params = nn.ParameterDict()
        for group_id in range(n_patient_groups):
            group_params = {}
            for name, param in self.base_model.named_parameters():
                group_params[f"{name}_group_{group_id}"] = nn.Parameter(param.clone())
            self.group_meta_params.update(group_params)
        
        # Group classifier
        self.group_classifier = nn.Sequential(
            nn.Linear(7, 32),  # Input dimension
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_patient_groups),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Patient group prediction"""
        # Group classification
        group_probs = self.group_classifier(x)
        group_id = torch.argmax(group_probs, dim=1)
        
        # Group prediction
        predictions = []
        for i in range(len(x)):
            # Predict with group parameters
            group_pred = self._predict_with_group_params(x[i:i+1], group_id[i].item())
            predictions.append(group_pred)
        
        predictions = torch.cat(predictions, dim=0)
        return predictions, group_probs
    
    def _predict_with_group_params(self, x: torch.Tensor, group_id: int) -> torch.Tensor:
        """Predict with specific group parameters"""
        # Save current parameters
        original_params = {}
        for name, param in self.base_model.named_parameters():
            original_params[name] = param.clone()
        
        # Replace with group parameters
        for name, param in self.base_model.named_parameters():
            group_param_name = f"{name}_group_{group_id}"
            if group_param_name in self.group_meta_params:
                param.data = self.group_meta_params[group_param_name]
        
        # Prediction
        with torch.no_grad():
            prediction = self.base_model(x)
        
        # Restore original parameters
        for name, param in self.base_model.named_parameters():
            param.data = original_params[name]
        
        return prediction
    
    def adapt_to_new_patient(self, 
                           patient_data: torch.Tensor,
                           patient_labels: torch.Tensor,
                           group_id: int) -> torch.Tensor:
        """Adapt to new patient"""
        # Group-wise adaptation
        adapted_params = {}
        for name, param in self.base_model.named_parameters():
            group_param_name = f"{name}_group_{group_id}"
            if group_param_name in self.group_meta_params:
                adapted_params[group_param_name] = self.group_meta_params[group_param_name].clone()
        
        # Fast adaptation
        for step in range(self.adaptation_steps):
            # Predict with current group parameters
            predictions = self._predict_with_group_params(patient_data, group_id)
            loss = F.mse_loss(predictions, patient_labels)
            
            # Calculate gradient
            grads = torch.autograd.grad(loss, self.group_meta_params.values(), create_graph=True)
            
            # Parameter update
            for i, (name, param) in enumerate(self.group_meta_params.items()):
                if name.endswith(f"_group_{group_id}"):
                    adapted_params[name] = adapted_params[name] - self.meta_lr * grads[i]
        
        return adapted_params

class FewShotLearningSystem(nn.Module):
    """Few-shot Learning system"""
    
    def __init__(self, 
                 base_model: nn.Module,
                 n_way: int = 5,
                 k_shot: int = 3,
                 meta_lr: float = 0.01):
        super().__init__()
        
        self.base_model = base_model
        self.n_way = n_way
        self.k_shot = k_shot
        self.meta_lr = meta_lr
        
        # Meta parameters
        self.meta_params = {}
        for name, param in self.base_model.named_parameters():
            self.meta_params[name] = nn.Parameter(param.clone())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Basic model forward pass"""
        return self.base_model(x)
    
    def meta_train(self, 
                   support_data: torch.Tensor,
                   support_labels: torch.Tensor,
                   query_data: torch.Tensor,
                   query_labels: torch.Tensor) -> torch.Tensor:
        """Meta training"""
        # Inner loop: adapt with support data
        adapted_params = self._inner_loop(support_data, support_labels)
        
        # Outer loop: update meta parameters with query data
        meta_loss = self._outer_loop(query_data, query_labels, adapted_params)
        
        return meta_loss
    
    def _inner_loop(self, 
                   support_data: torch.Tensor,
                   support_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inner loop: fast adaptation"""
        adapted_params = {}
        for name, param in self.meta_params.items():
            adapted_params[name] = param.clone()
        
        # Gradient-based adaptation
        for step in range(5):  # 5-step adaptation
            predictions = self.base_model(support_data)
            loss = F.mse_loss(predictions, support_labels)
            
            grads = torch.autograd.grad(loss, self.base_model.parameters(), create_graph=True)
            
            for i, (name, param) in enumerate(self.base_model.named_parameters()):
                adapted_params[name] = adapted_params[name] - self.meta_lr * grads[i]
                param.data = adapted_params[name]
        
        return adapted_params
    
    def _outer_loop(self, 
                   query_data: torch.Tensor,
                   query_labels: torch.Tensor,
                   adapted_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Outer loop: update meta parameters"""
        # Predict with adapted parameters
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                param.data = adapted_params[name]
            
            predictions = self.base_model(query_data)
            meta_loss = F.mse_loss(predictions, query_labels)
        
        return meta_loss

class AdaptiveMetaLearner(nn.Module):
    """Adaptive meta-learning system"""
    
    def __init__(self, 
                 base_model: nn.Module,
                 uncertainty_estimator: MonteCarloDropout,
                 meta_lr: float = 0.01,
                 max_adaptation_steps: int = 10):
        super().__init__()
        
        self.base_model = base_model
        self.uncertainty_estimator = uncertainty_estimator
        self.meta_lr = meta_lr
        self.max_adaptation_steps = max_adaptation_steps
        
        # Adaptive learning rate
        self.adaptive_lr = nn.Parameter(torch.tensor(meta_lr))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Basic model forward pass"""
        return self.base_model(x)
    
    def adaptive_meta_learning(self, 
                              support_data: torch.Tensor,
                              support_labels: torch.Tensor,
                              query_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Uncertainty-based adaptive meta-learning
        
        Returns:
            Prediction, uncertainty, number of used adaptation steps
        """
        # Save current model state
        original_params = {}
        for name, param in self.base_model.named_parameters():
            original_params[name] = param.clone()
        
        # Adaptive adaptation
        adaptation_steps = 0
        for step in range(self.max_adaptation_steps):
            # Prediction and uncertainty calculation
            predictions = self.base_model(query_data)
            _, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(query_data)
            
            # Early termination if uncertainty is below threshold
            if uncertainty.mean() < 0.05:  # Threshold
                break
            
            # Calculate gradient
            loss = F.mse_loss(predictions, support_labels)
            grads = torch.autograd.grad(loss, self.base_model.parameters(), create_graph=True)
            
            # Update with adaptive learning rate
            for i, (name, param) in enumerate(self.base_model.named_parameters()):
                param.data = param.data - self.adaptive_lr * grads[i]
            
            adaptation_steps += 1
        
        # Final prediction
        final_predictions = self.base_model(query_data)
        _, final_uncertainty = self.uncertainty_estimator.predict_with_uncertainty(query_data)
        
        # Restore original parameters
        for name, param in self.base_model.named_parameters():
            param.data = original_params[name]
        
        logger.info(f"Adaptive meta-learning: {adaptation_steps} steps, "
                   f"final uncertainty: {final_uncertainty.mean().item():.4f}")
        
        return final_predictions, final_uncertainty, adaptation_steps

# Usage example function
def create_meta_learning_system(model_type: str, 
                               base_model: nn.Module,
                               config = None) -> nn.Module:
    """Create meta-learning system"""
    
    if model_type == "basic":
        return MetaLearner(
            base_model=base_model,
            meta_lr=getattr(config, 'meta_lr', 0.01),
            adaptation_steps=getattr(config, 'adaptation_steps', 5)
        )
    elif model_type == "patient_group":
        return PatientGroupMetaLearner(
            base_model=base_model,
            n_patient_groups=getattr(config, 'n_patient_groups', 5),
            meta_lr=getattr(config, 'meta_lr', 0.01)
        )
    elif model_type == "few_shot":
        return FewShotLearningSystem(
            base_model=base_model,
            n_way=getattr(config, 'n_way', 5),
            k_shot=getattr(config, 'k_shot', 3)
        )
    elif model_type == "adaptive":
        uncertainty_estimator = MonteCarloDropout(
            model=base_model,
            n_samples=20,
            dropout_rate=0.1
        )
        return AdaptiveMetaLearner(
            base_model=base_model,
            uncertainty_estimator=uncertainty_estimator,
            meta_lr=getattr(config, 'meta_lr', 0.01)
        )
    else:
        raise ValueError(f"Unknown meta-learning type: {model_type}")
