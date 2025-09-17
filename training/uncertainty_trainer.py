"""
Uncertainty-Aware Training System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from utils.logging import get_logger
from utils.helpers import get_device
from models.uncertainty_quantification import (
    MonteCarloDropout, 
    UncertaintyAwareLoss, 
    UncertaintyMetrics,
    create_uncertainty_estimator
)


class UncertaintyAwareTrainer:
    """
    Uncertainty-aware trainer
    Extend BaseTrainer to add uncertainty estimation functionality
    """
    
    def __init__(self, model, config, loaders, device=None, uncertainty_method="monte_carlo"):
        self.model = model
        self.config = config
        self.loaders = loaders
        self.device = device if device is not None else get_device()
        self.logger = get_logger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup uncertainty estimator
        self.uncertainty_method = uncertainty_method
        self.uncertainty_estimator = self._setup_uncertainty_estimator()
        
        # Setup uncertainty-aware loss function
        self.uncertainty_loss = UncertaintyAwareLoss(
            uncertainty_weight=getattr(config, 'uncertainty_weight', 1.0),
            uncertainty_type=getattr(config, 'uncertainty_type', 'total')
        )
        
        # Setup base loss function (MSE)
        self.base_loss = nn.MSELoss()
        
        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Scheduler setup
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.patience//2, factor=0.5
        )
        
        # Result saving directory
        self.save_dir = Path(config.output_dir) / "models" / config.mode / config.encoder / f"s{config.random_state}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uncertainty metrics
        self.uncertainty_metrics = []
        
        self.logger.info(f"Uncertainty-aware trainer initialized - method: {uncertainty_method}")
        self.logger.info(f"Device: {self.device}")
        
    def _setup_uncertainty_estimator(self):
        """Setup uncertainty estimator"""
        if self.uncertainty_method == "monte_carlo":
            return MonteCarloDropout(
                self.model, 
                n_samples=getattr(self.config, 'uncertainty_samples', 100),
                dropout_rate=getattr(self.config, 'uncertainty_dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
    
    def train_epoch(self, train_loader):
        """Train epoch considering uncertainty"""
        self.model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = self._to_device(batch)
            x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
            target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
            
            self.optimizer.zero_grad()
            
            # Prediction with uncertainty
            if self.uncertainty_method == "monte_carlo":
                prediction, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(x)
            else:
                prediction, uncertainty = self.uncertainty_estimator(x)
            
            # Calculate uncertainty-aware loss
            loss = self.uncertainty_loss(prediction, target, uncertainty)
            
            # Additional regularization loss (to prevent uncertainty from getting too high)
            uncertainty_penalty = getattr(self.config, 'uncertainty_penalty', 0.01)
            if uncertainty_penalty > 0:
                loss += uncertainty_penalty * uncertainty.mean()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_uncertainty += uncertainty.mean().item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_uncertainty = total_uncertainty / num_batches
        
        return avg_loss, avg_uncertainty
    
    def validate_epoch(self, val_loader):
        """Validation with uncertainty"""
        self.model.eval()
        total_loss = 0.0
        total_uncertainty = 0.0
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Prediction with uncertainty
                if self.uncertainty_method == "monte_carlo":
                    prediction, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(x)
                else:
                    prediction, uncertainty = self.uncertainty_estimator(x)
                
                # Calculate base loss
                loss = self.base_loss(prediction, target)
                
                total_loss += loss.item()
                total_uncertainty += uncertainty.mean().item()
                
                all_predictions.append(prediction.cpu())
                all_targets.append(target.cpu())
                all_uncertainties.append(uncertainty.cpu())
        
        # Collect all prediction results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(
            all_predictions, all_targets, all_uncertainties
        )
        
        avg_loss = total_loss / len(val_loader)
        avg_uncertainty = total_uncertainty / len(val_loader)
        
        return avg_loss, avg_uncertainty, uncertainty_metrics
    
    def _compute_uncertainty_metrics(self, predictions, targets, uncertainties):
        """Calculate uncertainty-related metrics"""
        metrics = {}
        
        # Calibration error
        metrics['calibration_error'] = UncertaintyMetrics.calibration_error(
            predictions, targets, uncertainties
        )
        
        # Sharpness
        metrics['sharpness'] = UncertaintyMetrics.sharpness(uncertainties)
        
        # Reliability diagram data
        metrics['reliability_data'] = UncertaintyMetrics.reliability_diagram(
            predictions, targets, uncertainties
        )
        
        # Prediction interval coverage
        lower_bound = predictions - 1.96 * uncertainties
        upper_bound = predictions + 1.96 * uncertainties
        coverage = ((targets >= lower_bound) & (targets <= upper_bound)).float().mean()
        metrics['coverage_95'] = coverage.item()
        
        return metrics
    
    def _to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, (list, tuple)):
            return [item.to(self.device) if torch.is_tensor(item) else item for item in batch]
        elif isinstance(batch, dict):
            return {key: value.to(self.device) if torch.is_tensor(value) else value 
                   for key, value in batch.items()}
        else:
            return batch.to(self.device)
    
    def save_model(self, filename, metrics=None):
        """Save model and uncertainty metrics"""
        model_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'uncertainty_method': self.uncertainty_method,
            'metrics': metrics,
            'uncertainty_metrics': self.uncertainty_metrics
        }, model_path)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename):
        """Load model and uncertainty metrics"""
        model_path = self.save_dir / filename
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'uncertainty_metrics' in checkpoint:
            self.uncertainty_metrics = checkpoint['uncertainty_metrics']
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def predict_with_uncertainty(self, x):
        """Prediction with uncertainty"""
        self.model.eval()
        x = x.to(self.device)
        
        with torch.no_grad():
            if self.uncertainty_method == "monte_carlo":
                prediction, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(x)
            else:
                prediction, uncertainty = self.uncertainty_estimator(x)
        
        return prediction.cpu(), uncertainty.cpu()
    
    def log_uncertainty_metrics(self, epoch, metrics):
        """Log uncertainty metrics"""
        self.logger.info(f"Epoch {epoch} - Uncertainty Metrics:")
        self.logger.info(f"  Calibration Error: {metrics['calibration_error']:.4f}")
        self.logger.info(f"  Sharpness: {metrics['sharpness']:.4f}")
        self.logger.info(f"  95% Coverage: {metrics['coverage_95']:.4f}")
        
        # Log reliability diagram data
        reliability_data = metrics['reliability_data']
        if reliability_data:
            self.logger.info("  Reliability Diagram:")
            for i, bin_data in enumerate(reliability_data[:5]):  # Only top 5 bins
                self.logger.info(f"    Bin {i+1}: Uncertainty={bin_data['uncertainty']:.4f}, "
                               f"Actual Error={bin_data['actual_error']:.4f}, "
                               f"Count={bin_data['count']}")


class UncertaintyAwareSeparateTrainer(UncertaintyAwareTrainer):
    """
    Separate trainer for uncertainty-aware training
    Train PK and PD models with uncertainty
    """
    
    def __init__(self, pk_model, pd_model, config, loaders, device=None, uncertainty_method="monte_carlo"):
        self.pk_model = pk_model
        self.pd_model = pd_model
        self.config = config
        self.loaders = loaders
        self.device = device if device is not None else get_device()
        self.logger = get_logger(__name__)
        
        # Setup uncertainty estimators for PK and PD
        self.pk_uncertainty_estimator = self._setup_uncertainty_estimator(self.pk_model)
        self.pd_uncertainty_estimator = self._setup_uncertainty_estimator(self.pd_model)
        
        # Setup uncertainty-aware loss function
        self.uncertainty_loss = UncertaintyAwareLoss(
            uncertainty_weight=getattr(config, 'uncertainty_weight', 1.0),
            uncertainty_type=getattr(config, 'uncertainty_type', 'total')
        )
        
        # Optimizer setup
        self.pk_optimizer = torch.optim.AdamW(self.pk_model.parameters(), lr=config.learning_rate)
        self.pd_optimizer = torch.optim.AdamW(self.pd_model.parameters(), lr=config.learning_rate)
        
        # Scheduler setup
        self.pk_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pk_optimizer, mode='min', patience=config.patience//2, factor=0.5
        )
        self.pd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pd_optimizer, mode='min', patience=config.patience//2, factor=0.5
        )
        
        # Result saving directory
        self.save_dir = Path(config.output_dir) / "models" / config.mode / config.encoder / f"s{config.random_state}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Uncertainty-aware separate trainer initialized")
    
    def _setup_uncertainty_estimator(self, model):
        """Setup model-specific uncertainty estimator"""
        if self.uncertainty_method == "monte_carlo":
            return MonteCarloDropout(
                model, 
                n_samples=getattr(self.config, 'uncertainty_samples', 100),
                dropout_rate=getattr(self.config, 'uncertainty_dropout', 0.1)
            )
        else:
            raise ValueError(f"Unsupported uncertainty method for separate training: {self.uncertainty_method}")
    
    def train_pk_epoch(self, train_loader):
        """Train PK model epoch"""
        self.pk_model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = self._to_device(batch)
            x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
            target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
            
            self.pk_optimizer.zero_grad()
            
            # PK uncertainty prediction
            prediction, uncertainty = self.pk_uncertainty_estimator.predict_with_uncertainty(x)
            
            # Uncertainty-aware loss
            loss = self.uncertainty_loss(prediction, target, uncertainty)
            
            loss.backward()
            self.pk_optimizer.step()
            
            total_loss += loss.item()
            total_uncertainty += uncertainty.mean().item()
            num_batches += 1
        
        return total_loss / num_batches, total_uncertainty / num_batches
    
    def train_pd_epoch(self, train_loader):
        """Train PD model epoch"""
        self.pd_model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = self._to_device(batch)
            x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
            target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
            
            self.pd_optimizer.zero_grad()
            
            # PD uncertainty prediction
            prediction, uncertainty = self.pd_uncertainty_estimator.predict_with_uncertainty(x)
            
            # Uncertainty-aware loss
            loss = self.uncertainty_loss(prediction, target, uncertainty)
            
            loss.backward()
            self.pd_optimizer.step()
            
            total_loss += loss.item()
            total_uncertainty += uncertainty.mean().item()
            num_batches += 1
        
        return total_loss / num_batches, total_uncertainty / num_batches
