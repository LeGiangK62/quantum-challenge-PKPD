"""
Uncertainty Quantification for PK/PD Modeling
Module providing various uncertainty estimation methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math

class MonteCarloDropout(nn.Module):
    """
    Uncertainty estimation using Monte Carlo Dropout
    The simplest and most effective method
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 100, dropout_rate: float = 0.1):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with uncertainty
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            mean: Prediction mean [batch_size, output_dim]
            uncertainty: Uncertainty (standard deviation) [batch_size, output_dim]
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.n_samples):
            pred = self.model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, output_dim]
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = torch.sqrt(variance + 1e-8)  # Add small value for numerical stability
        
        return mean, uncertainty
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_uncertainty(x)


class EnsembleUncertainty(nn.Module):
    """
    Uncertainty estimation using ensemble model
    Measure uncertainty by the variance of multiple model predictions
    """
    
    def __init__(self, models: List[nn.Module], n_models: int = 5):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = n_models
        
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uncertainty estimation using ensemble model
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_models, batch_size, output_dim]
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = torch.sqrt(variance + 1e-8)
        
        return mean, uncertainty
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_uncertainty(x)


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer
    Learn the distribution of weights to estimate uncertainty
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Mean and log standard deviation of weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Mean and log standard deviation of bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight_mu, 0, 0.1)
        nn.init.constant_(self.weight_log_sigma, -3.0)  # Small initial uncertainty
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_log_sigma, -3.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample weights and bias
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        # Linear transformation
        output = F.linear(x, weight, bias)
        
        # Calculate uncertainty (approximation)
        uncertainty = torch.sqrt(
            torch.sum((x.unsqueeze(-1) * weight_sigma.T.unsqueeze(0))**2, dim=1) + 
            bias_sigma**2
        )
        
        return output, uncertainty



class UncertaintyAwareLoss(nn.Module):
    """
    Loss function considering uncertainty
    Penalize or reward high uncertainty predictions
    """
    
    def __init__(self, uncertainty_weight: float = 1.0, uncertainty_type: str = "aleatoric"):
        super().__init__()
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_type = uncertainty_type  # "aleatoric", "epistemic", "total"
        
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, 
                uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Loss calculation considering uncertainty
        
        Args:
            prediction: Model prediction [batch_size, output_dim]
            target: Actual value [batch_size, output_dim]
            uncertainty: Uncertainty [batch_size, output_dim]
        """
        # Basic MSE loss
        mse_loss = F.mse_loss(prediction, target, reduction='none')
        
        if self.uncertainty_type == "aleatoric":
            # Aleatoric uncertainty (noise in data itself)
            uncertainty_loss = torch.mean(mse_loss / (2 * uncertainty**2 + 1e-8) + 
                                        torch.log(uncertainty**2 + 1e-8))
        elif self.uncertainty_type == "epistemic":
            # Epistemic uncertainty (model knowledge deficiency)
            uncertainty_loss = torch.mean(mse_loss + self.uncertainty_weight * uncertainty**2)
        else:  # total
            # Total uncertainty
            uncertainty_loss = torch.mean(mse_loss / (uncertainty**2 + 1e-8) + 
                                        torch.log(uncertainty**2 + 1e-8))
        
        return uncertainty_loss


class UncertaintyMetrics:
    """
    Calculation of uncertainty-related metrics
    """
    
    @staticmethod
    def calibration_error(predictions: torch.Tensor, targets: torch.Tensor, 
                         uncertainties: torch.Tensor, n_bins: int = 10) -> float:
        """
        Calculate calibration error of prediction intervals
        """
        # Calculate prediction intervals (95% confidence interval)
        lower_bound = predictions - 1.96 * uncertainties
        upper_bound = predictions + 1.96 * uncertainties
        
        # Check if actual values are within intervals
        in_interval = (targets >= lower_bound) & (targets <= upper_bound)
        coverage = in_interval.float().mean().item()
        
        # Difference from theoretical coverage (95%)
        calibration_error = abs(coverage - 0.95)
        
        return calibration_error
    
    @staticmethod
    def sharpness(uncertainties: torch.Tensor) -> float:
        """
        Sharpness of prediction (average uncertainty)
        Lower is more sharp
        """
        return uncertainties.mean().item()
    
    @staticmethod
    def reliability_diagram(predictions: torch.Tensor, targets: torch.Tensor,
                           uncertainties: torch.Tensor, n_bins: int = 10) -> dict:
        """
        Generate reliability diagram data
        """
        # Divide data into intervals based on uncertainty
        uncertainty_quantiles = torch.quantile(uncertainties, 
                                             torch.linspace(0, 1, n_bins + 1))
        
        bin_data = []
        for i in range(n_bins):
            mask = (uncertainties >= uncertainty_quantiles[i]) & \
                   (uncertainties < uncertainty_quantiles[i + 1])
            
            if mask.sum() > 0:
                bin_predictions = predictions[mask]
                bin_targets = targets[mask]
                bin_uncertainties = uncertainties[mask]
                
                # Average uncertainty within interval
                avg_uncertainty = bin_uncertainties.mean().item()
                
                # Actual error within the interval
                actual_error = F.mse_loss(bin_predictions, bin_targets).item()
                
                bin_data.append({
                    'uncertainty': avg_uncertainty,
                    'actual_error': actual_error,
                    'count': mask.sum().item()
                })
        
        return bin_data


def create_uncertainty_estimator(method: str = "monte_carlo", **kwargs) -> nn.Module:
    """
    Uncertainty estimator factory function
    
    Args:
        method: Uncertainty estimation method ("monte_carlo", "ensemble", "bayesian")
        **kwargs: Additional parameters for each method
    """
    if method == "monte_carlo":
        return MonteCarloDropout(**kwargs)
    elif method == "ensemble":
        return EnsembleUncertainty(**kwargs)
    elif method == "bayesian":
        return BayesianLinear(**kwargs)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
