"""
Unified PK/PD Trainer - All training modes unified
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import numpy as np

from utils.logging import get_logger
from utils.helpers import get_device
from models.heads import _reg_metrics


class UnifiedPKPDTrainer:
    """
    Unified PK/PD Trainer - All training modes supported
    """
    
    def __init__(self, model, config, data_loaders, device=None):
        self.model = model
        self.config = config
        self.data_loaders = data_loaders
        self.device = device if device is not None else get_device()
        self.logger = get_logger(__name__)
        self.mode = config.mode
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.patience//2, factor=0.5, verbose=True
        )
        
        # Model save directory
        # Determine encoder name for directory path
        if config.encoder_pk or config.encoder_pd:
            pk_encoder = config.encoder_pk or config.encoder
            pd_encoder = config.encoder_pd or config.encoder
            encoder_name = f"{pk_encoder}-{pd_encoder}"
        else:
            encoder_name = config.encoder
        
        self.model_save_directory = Path(config.output_dir) / "models" / config.mode / encoder_name / f"s{config.random_state}"
        self.model_save_directory.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_pk_rmse = float('inf')
        self.best_pd_rmse = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'pk_train_loss': [],
            'pd_train_loss': [],
            'pk_val_loss': [],
            'pd_val_loss': [],
            # Metrics
            'pk_train_mse': [],
            'pk_train_rmse': [],
            'pk_train_mae': [],
            'pk_train_r2': [],
            'pd_train_mse': [],
            'pd_train_rmse': [],
            'pd_train_mae': [],
            'pd_train_r2': [],
            'pk_val_mse': [],
            'pk_val_rmse': [],
            'pk_val_mae': [],
            'pk_val_r2': [],
            'pd_val_mse': [],
            'pd_val_rmse': [],
            'pd_val_mae': [],
            'pd_val_r2': []
        }
        
        self.logger.info(f"Unified Trainer initialized - Mode: {self.mode}, Device: {self.device}")
        self.logger.info(f"Number of model parameters: {self._count_parameters()}")
    
    def _count_parameters(self) -> int:
        """Calculate number of model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def apply_mixup(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple:
        """Apply Mixup augmentation"""
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample()
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def contrastive_loss(self, features: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """Contrastive loss (NT-Xent)"""
        batch_size = features.size(0)
        
        # Ensure features is 2D for normalization
        if features.dim() == 1:
            features = features.unsqueeze(1)
        
        # Normalization
        features = F.normalize(features, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Diagonal mask (exclude self-similarity)
        mask = torch.eye(batch_size, device=features.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Apply softmax
        logits = F.log_softmax(similarity_matrix, dim=1)
        
        # Ground truth labels (next sample)
        labels = torch.arange(batch_size, device=features.device)
        labels = (labels + 1) % batch_size
        
        # Loss calculation
        loss = F.nll_loss(logits, labels)
        return loss
    
    def _compute_metrics(self, batch: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics (MSE, RMSE, MAE, R²)"""
        metrics = {}
        
        for task in ['pk', 'pd']:
            if task in batch and task in results:
                pred = results[task]['pred']
                target = batch[task]['y'].squeeze(-1)  # [B, 1] -> [B]
                
                # Calculate metrics
                task_metrics = _reg_metrics(pred, target)
                
                # Save metrics
                for metric_name, value in task_metrics.items():
                    metrics[f"{task}_{metric_name}"] = value
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        self.logger.info(f"Training start - Mode: {self.mode}, Epochs: {self.config.epochs}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = self._validate_epoch()
            
            # Record metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['total_loss']):
                self.logger.info(f"Early stopping - Epoch {epoch}")
                break
            
            # Save best model (PD performance priority)
            should_save = False
            if self.mode == "separate":
                # Separate mode: PK and PD each independently select best model
                if val_metrics.get('pk_rmse', float('inf')) < self.best_pk_rmse:
                    self.best_pk_rmse = val_metrics.get('pk_rmse', float('inf'))
                    should_save = True
                    self.logger.info(f"New PK best model - RMSE: {self.best_pk_rmse:.6f}")
                
                if val_metrics.get('pd_rmse', float('inf')) < self.best_pd_rmse:
                    self.best_pd_rmse = val_metrics.get('pd_rmse', float('inf'))
                    should_save = True
                    self.logger.info(f"New PD best model - RMSE: {self.best_pd_rmse:.6f}")
            else:
                # Other modes: PD RMSE selects best model
                if val_metrics.get('pd_rmse', float('inf')) < self.best_pd_rmse:
                    self.best_pd_rmse = val_metrics.get('pd_rmse', float('inf'))
                    should_save = True
                    self.logger.info(f"New PD best model - RMSE: {self.best_pd_rmse:.6f}")
            
            # Maintain existing total_loss criterion (for compatibility)
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                if not should_save:  # If not already saved
                    should_save = True
            
            if should_save:
                self._save_best_model()
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed - Time: {training_time:.2f} seconds")
        
        return self._get_final_results()
    
    def _train_epoch_separate(self) -> Dict[str, float]:
        """Train one epoch for separate mode - PK first, then PD"""
        self.model.train()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0
        }
        
        # Mixed precision settings
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Phase 1: Train PK model
        self.logger.info("=== Phase 1: Training PK Model ===")
        pk_batch_count = 0
        for batch_pk in self.data_loaders['train_pk']:
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch_pk = self._to_device(batch_pk)
            
            # Convert batch to dictionary format
            if isinstance(batch_pk, (list, tuple)):
                x_pk, y_pk = batch_pk[0], batch_pk[1]
                batch_pk_dict = {'x': x_pk, 'y': y_pk}
            else:
                batch_pk_dict = batch_pk
            
            # Forward pass for PK only
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self._compute_separate_loss(batch_pk_dict, None)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self._compute_separate_loss(batch_pk_dict, None)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total'].item()
            pk_loss += loss_dict.get('pk', torch.tensor(0.0)).item()
            num_batches += 1
            
            # Calculate PK metrics
            with torch.no_grad():
                pk_results = self.model({'pk': batch_pk_dict})
                pk_pred = pk_results['pk']['pred']
                pk_target = batch_pk_dict['y'].squeeze(-1)
                
                pk_mse = F.mse_loss(pk_pred, pk_target).item()
                pk_rmse = torch.sqrt(torch.tensor(pk_mse)).item()
                pk_mae = F.l1_loss(pk_pred, pk_target).item()
                
                # R² calculation
                pk_ss_res = torch.sum((pk_target - pk_pred) ** 2)
                pk_ss_tot = torch.sum((pk_target - torch.mean(pk_target)) ** 2)
                pk_r2 = 1 - (pk_ss_res / (pk_ss_tot + 1e-8))
                
                metrics_sum['pk_mse'] += pk_mse
                metrics_sum['pk_rmse'] += pk_rmse
                metrics_sum['pk_mae'] += pk_mae
                metrics_sum['pk_r2'] += pk_r2.item()
            
            pk_batch_count += 1
        
        self.logger.info(f"PK training completed - Processed {pk_batch_count} batches")
        
        # Phase 2: Train PD model
        self.logger.info("=== Phase 2: Training PD Model ===")
        pd_batch_count = 0
        for batch_pd in self.data_loaders['train_pd']:
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch_pd = self._to_device(batch_pd)
            
            # Convert batch to dictionary format
            if isinstance(batch_pd, (list, tuple)):
                x_pd, y_pd = batch_pd[0], batch_pd[1]
                batch_pd_dict = {'x': x_pd, 'y': y_pd}
            else:
                batch_pd_dict = batch_pd
            
            # Forward pass for PD only
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self._compute_separate_loss(None, batch_pd_dict)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self._compute_separate_loss(None, batch_pd_dict)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total'].item()
            pd_loss += loss_dict.get('pd', torch.tensor(0.0)).item()
            num_batches += 1
            
            # Calculate PD metrics
            with torch.no_grad():
                pd_results = self.model({'pd': batch_pd_dict})
                pd_pred = pd_results['pd']['pred']
                pd_target = batch_pd_dict['y'].squeeze(-1)
                
                pd_mse = F.mse_loss(pd_pred, pd_target).item()
                pd_rmse = torch.sqrt(torch.tensor(pd_mse)).item()
                pd_mae = F.l1_loss(pd_pred, pd_target).item()
                
                # R² calculation
                pd_ss_res = torch.sum((pd_target - pd_pred) ** 2)
                pd_ss_tot = torch.sum((pd_target - torch.mean(pd_target)) ** 2)
                pd_r2 = 1 - (pd_ss_res / (pd_ss_tot + 1e-8))
                
                metrics_sum['pd_mse'] += pd_mse
                metrics_sum['pd_rmse'] += pd_rmse
                metrics_sum['pd_mae'] += pd_mae
                metrics_sum['pd_r2'] += pd_r2.item()
            
            pd_batch_count += 1
        
        self.logger.info(f"PD training completed - Processed {pd_batch_count} batches")
        
        # Average metrics
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches,
            'pk_mse': metrics_sum['pk_mse'] / num_batches,
            'pk_rmse': metrics_sum['pk_rmse'] / num_batches,
            'pk_mae': metrics_sum['pk_mae'] / num_batches,
            'pk_r2': metrics_sum['pk_r2'] / num_batches,
            'pd_mse': metrics_sum['pd_mse'] / num_batches,
            'pd_rmse': metrics_sum['pd_rmse'] / num_batches,
            'pd_mae': metrics_sum['pd_mae'] / num_batches,
            'pd_r2': metrics_sum['pd_r2'] / num_batches,
        }
        
        return avg_metrics
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        if self.mode == "separate":
            return self._train_epoch_separate()
        else:
            return self._train_epoch_standard()
    
    def _train_epoch_standard(self) -> Dict[str, float]:
        """Train one epoch for standard modes (joint, shared, etc.)"""
        self.model.train()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0
        }
        
        # Mixed precision settings
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Select mode-specific data loaders
        train_loaders = self._get_train_loaders()
        
        for batch_pk, batch_pd in zip(*train_loaders):
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch_pk = self._to_device(batch_pk)
            batch_pd = self._to_device(batch_pd)
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self._compute_loss(batch_pk, batch_pd)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self._compute_loss(batch_pk, batch_pd)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            # Metrics accumulation
            total_loss += loss_dict['total'].item()
            pk_loss += loss_dict.get('pk', 0.0)
            pd_loss += loss_dict.get('pd', 0.0)
            num_batches += 1
            
            # Metrics calculation (only last batch)
            if num_batches == len(list(zip(*train_loaders))):
                # Convert batch to dictionary
                batch_dict = {}
                if batch_pk is not None:
                    if isinstance(batch_pk, (list, tuple)):
                        x_pk, y_pk = batch_pk
                        batch_dict['pk'] = {'x': x_pk, 'y': y_pk}
                    else:
                        batch_dict['pk'] = batch_pk
                if batch_pd is not None:
                    if isinstance(batch_pd, (list, tuple)):
                        x_pd, y_pd = batch_pd
                        batch_dict['pd'] = {'x': x_pd, 'y': y_pd}
                    else:
                        batch_dict['pd'] = batch_pd
                
                results = self.model(batch_dict)
                batch_metrics = self._compute_metrics(batch_dict, results)
                for key, value in batch_metrics.items():
                    metrics_sum[key] = value
        
        # Calculate average
        result = {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches
        }
        
        # Add metrics
        for key, value in metrics_sum.items():
            result[key] = value
        
        return result
    
    def _validate_epoch_separate(self) -> Dict[str, float]:
        """Validate one epoch for separate mode - PK and PD separately"""
        self.model.eval()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0
        }
        
        with torch.no_grad():
            # Validate PK model
            self.logger.debug("Validating PK model...")
            for batch_pk in self.data_loaders['val_pk']:
                batch_pk = self._to_device(batch_pk)
                
                # Convert batch to dictionary format
                if isinstance(batch_pk, (list, tuple)):
                    x_pk, y_pk = batch_pk[0], batch_pk[1]
                    batch_pk_dict = {'x': x_pk, 'y': y_pk}
                else:
                    batch_pk_dict = batch_pk
                
                loss_dict = self._compute_separate_loss(batch_pk_dict, None)
                
                total_loss += loss_dict['total'].item()
                pk_loss += loss_dict.get('pk', torch.tensor(0.0)).item()
                num_batches += 1
                
                # Calculate PK metrics
                pk_results = self.model({'pk': batch_pk_dict})
                pk_pred = pk_results['pk']['pred']
                pk_target = batch_pk_dict['y'].squeeze(-1)
                
                pk_mse = F.mse_loss(pk_pred, pk_target).item()
                pk_rmse = torch.sqrt(torch.tensor(pk_mse)).item()
                pk_mae = F.l1_loss(pk_pred, pk_target).item()
                
                # R² calculation
                pk_ss_res = torch.sum((pk_target - pk_pred) ** 2)
                pk_ss_tot = torch.sum((pk_target - torch.mean(pk_target)) ** 2)
                pk_r2 = 1 - (pk_ss_res / (pk_ss_tot + 1e-8))
                
                metrics_sum['pk_mse'] += pk_mse
                metrics_sum['pk_rmse'] += pk_rmse
                metrics_sum['pk_mae'] += pk_mae
                metrics_sum['pk_r2'] += pk_r2.item()
            
            # Validate PD model
            self.logger.debug("Validating PD model...")
            for batch_pd in self.data_loaders['val_pd']:
                batch_pd = self._to_device(batch_pd)
                
                # Convert batch to dictionary format
                if isinstance(batch_pd, (list, tuple)):
                    x_pd, y_pd = batch_pd[0], batch_pd[1]
                    batch_pd_dict = {'x': x_pd, 'y': y_pd}
                else:
                    batch_pd_dict = batch_pd
                
                loss_dict = self._compute_separate_loss(None, batch_pd_dict)
                
                total_loss += loss_dict['total'].item()
                pd_loss += loss_dict.get('pd', torch.tensor(0.0)).item()
                num_batches += 1
                
                # Calculate PD metrics
                pd_results = self.model({'pd': batch_pd_dict})
                pd_pred = pd_results['pd']['pred']
                pd_target = batch_pd_dict['y'].squeeze(-1)
                
                pd_mse = F.mse_loss(pd_pred, pd_target).item()
                pd_rmse = torch.sqrt(torch.tensor(pd_mse)).item()
                pd_mae = F.l1_loss(pd_pred, pd_target).item()
                
                # R² calculation
                pd_ss_res = torch.sum((pd_target - pd_pred) ** 2)
                pd_ss_tot = torch.sum((pd_target - torch.mean(pd_target)) ** 2)
                pd_r2 = 1 - (pd_ss_res / (pd_ss_tot + 1e-8))
                
                metrics_sum['pd_mse'] += pd_mse
                metrics_sum['pd_rmse'] += pd_rmse
                metrics_sum['pd_mae'] += pd_mae
                metrics_sum['pd_r2'] += pd_r2.item()
        
        # Average metrics
        result = {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches,
            'pk_mse': metrics_sum['pk_mse'] / num_batches,
            'pk_rmse': metrics_sum['pk_rmse'] / num_batches,
            'pk_mae': metrics_sum['pk_mae'] / num_batches,
            'pk_r2': metrics_sum['pk_r2'] / num_batches,
            'pd_mse': metrics_sum['pd_mse'] / num_batches,
            'pd_rmse': metrics_sum['pd_rmse'] / num_batches,
            'pd_mae': metrics_sum['pd_mae'] / num_batches,
            'pd_r2': metrics_sum['pd_r2'] / num_batches,
        }
        
        return result
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate one epoch"""
        if self.mode == "separate":
            return self._validate_epoch_separate()
        else:
            return self._validate_epoch_standard()
    
    def _validate_epoch_standard(self) -> Dict[str, float]:
        """Validate one epoch for standard modes"""
        self.model.eval()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0
        }
        
        val_loaders = self._get_val_loaders()
        
        with torch.no_grad():
            for batch_pk, batch_pd in zip(*val_loaders):
                batch_pk = self._to_device(batch_pk)
                batch_pd = self._to_device(batch_pd)
                
                loss_dict = self._compute_loss(batch_pk, batch_pd)
                
                total_loss += loss_dict['total'].item()
                pk_loss += loss_dict.get('pk', 0.0)
                pd_loss += loss_dict.get('pd', 0.0)
                num_batches += 1
                
                # Metrics calculation (only last batch)
                if num_batches == len(list(zip(*val_loaders))):
                    # Convert batch to dictionary
                    batch_dict = {}
                    if batch_pk is not None:
                        if isinstance(batch_pk, (list, tuple)):
                            x_pk, y_pk = batch_pk
                            batch_dict['pk'] = {'x': x_pk, 'y': y_pk}
                        else:
                            batch_dict['pk'] = batch_pk
                    if batch_pd is not None:
                        if isinstance(batch_pd, (list, tuple)):
                            x_pd, y_pd = batch_pd
                            batch_dict['pd'] = {'x': x_pd, 'y': y_pd}
                        else:
                            batch_dict['pd'] = batch_pd
                    
                    results = self.model(batch_dict)
                    batch_metrics = self._compute_metrics(batch_dict, results)
                    for key, value in batch_metrics.items():
                        metrics_sum[key] = value
        
        # Calculate average
        result = {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches
        }
        
        # Add metrics
        for key, value in metrics_sum.items():
            result[key] = value
        
        return result
    
    def _compute_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Calculate loss - different logic for each mode"""
        if self.mode == "separate":
            return self._compute_separate_loss(batch_pk, batch_pd)
        elif self.mode in ["joint", "dual_stage", "integrated"]:
            return self._compute_dual_branch_loss(batch_pk, batch_pd)
        elif self.mode == "shared":
            return self._compute_shared_loss(batch_pk, batch_pd)
        elif self.mode == "two_stage_shared":
            return self._compute_two_stage_shared_loss(batch_pk, batch_pd)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _compute_separate_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Calculate separate mode loss with mixup and contrastive learning"""
        losses = {}
        
        # PK loss
        if batch_pk is not None:
            # Convert batch to dictionary
            if isinstance(batch_pk, (list, tuple)):
                x_pk, y_pk = batch_pk
                batch_pk_dict = {'x': x_pk, 'y': y_pk}
            else:
                batch_pk_dict = batch_pk
            
            pk_results = self.model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pk_features = pk_results['pk'].get('z', None)
            
            # Original PK loss
            pk_loss = F.mse_loss(pk_pred, pk_target)
            
            # Add contrastive loss for PK
            if self.config.lambda_contrast > 0 and pk_features is not None:
                contrast_loss_pk = self.contrastive_loss(pk_features, self.config.temperature)
                pk_loss = pk_loss + self.config.lambda_contrast * contrast_loss_pk
            
            # Apply mixup for PK if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pk, y_a_pk, y_b_pk, lam_pk = self.apply_mixup(
                    batch_pk_dict['x'], pk_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_pk = {'x': mixed_x_pk, 'y': pk_target}
                pk_results_mix = self.model({'pk': mixed_batch_pk})
                pk_pred_mix = pk_results_mix['pk']['pred']
                
                # Mixup loss
                pk_loss_mix = lam_pk * F.mse_loss(pk_pred_mix, y_a_pk) + (1 - lam_pk) * F.mse_loss(pk_pred_mix, y_b_pk)
                
                # Combine original and mixup losses
                pk_loss = 0.7 * pk_loss + 0.3 * pk_loss_mix
            
            losses['pk'] = pk_loss
        
        # PD loss
        if batch_pd is not None:
            # Convert batch to dictionary
            if isinstance(batch_pd, (list, tuple)):
                x_pd, y_pd = batch_pd
                batch_pd_dict = {'x': x_pd, 'y': y_pd}
            else:
                batch_pd_dict = batch_pd
            
            pd_results = self.model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pd_features = pd_results['pd'].get('z', None)
            
            # Original PD loss
            pd_loss = F.mse_loss(pd_pred, pd_target)
            
            # Add contrastive loss for PD
            if self.config.lambda_contrast > 0 and pd_features is not None:
                contrast_loss_pd = self.contrastive_loss(pd_features, self.config.temperature)
                pd_loss = pd_loss + self.config.lambda_contrast * contrast_loss_pd
            
            # Apply mixup for PD if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pd, y_a_pd, y_b_pd, lam_pd = self.apply_mixup(
                    batch_pd_dict['x'], pd_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_pd = {'x': mixed_x_pd, 'y': pd_target}
                pd_results_mix = self.model({'pd': mixed_batch_pd})
                pd_pred_mix = pd_results_mix['pd']['pred']
                
                # Mixup loss
                pd_loss_mix = lam_pd * F.mse_loss(pd_pred_mix, y_a_pd) + (1 - lam_pd) * F.mse_loss(pd_pred_mix, y_b_pd)
                
                # Combine original and mixup losses
                pd_loss = 0.7 * pd_loss + 0.3 * pd_loss_mix
            
            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_dual_branch_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Calculate dual branch mode loss with mixup and contrastive learning"""
        losses = {}
        
        # Construct batch dictionary
        batch_dict = {}
        if batch_pk is not None:
            # Convert batch to dictionary
            if isinstance(batch_pk, (list, tuple)):
                x_pk, y_pk = batch_pk
                batch_pk_dict = {'x': x_pk, 'y': y_pk}
            else:
                batch_pk_dict = batch_pk
            batch_dict['pk'] = batch_pk_dict
        
        if batch_pd is not None:
            # Convert batch to dictionary
            if isinstance(batch_pd, (list, tuple)):
                x_pd, y_pd = batch_pd
                batch_pd_dict = {'x': x_pd, 'y': y_pd}
            else:
                batch_pd_dict = batch_pd
            batch_dict['pd'] = batch_pd_dict
        
        # Forward pass
        results = self.model(batch_dict)
        
        # PK loss
        if 'pk' in results:
            pk_pred = results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pk_features = results['pk'].get('z', None)
            
            # Original PK loss
            pk_loss = F.mse_loss(pk_pred, pk_target)
            
            # Add contrastive loss for PK
            if self.config.lambda_contrast > 0 and pk_features is not None:
                contrast_loss_pk = self.contrastive_loss(pk_features, self.config.temperature)
                pk_loss = pk_loss + self.config.lambda_contrast * contrast_loss_pk
            
            # Apply mixup for PK if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pk, y_a_pk, y_b_pk, lam_pk = self.apply_mixup(
                    batch_pk_dict['x'], pk_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_dict = batch_dict.copy()
                mixed_batch_dict['pk'] = {'x': mixed_x_pk, 'y': pk_target}
                pk_results_mix = self.model(mixed_batch_dict)
                pk_pred_mix = pk_results_mix['pk']['pred']
                
                # Mixup loss
                pk_loss_mix = lam_pk * F.mse_loss(pk_pred_mix, y_a_pk) + (1 - lam_pk) * F.mse_loss(pk_pred_mix, y_b_pk)
                
                # Combine original and mixup losses
                pk_loss = 0.7 * pk_loss + 0.3 * pk_loss_mix
            
            losses['pk'] = pk_loss
        
        # PD loss
        if 'pd' in results:
            pd_pred = results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pd_features = results['pd'].get('z', None)
            
            # Original PD loss
            pd_loss = F.mse_loss(pd_pred, pd_target)
            
            # Add contrastive loss for PD
            if self.config.lambda_contrast > 0 and pd_features is not None:
                contrast_loss_pd = self.contrastive_loss(pd_features, self.config.temperature)
                pd_loss = pd_loss + self.config.lambda_contrast * contrast_loss_pd
            
            # Apply mixup for PD if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pd, y_a_pd, y_b_pd, lam_pd = self.apply_mixup(
                    batch_pd_dict['x'], pd_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_dict = batch_dict.copy()
                mixed_batch_dict['pd'] = {'x': mixed_x_pd, 'y': pd_target}
                pd_results_mix = self.model(mixed_batch_dict)
                pd_pred_mix = pd_results_mix['pd']['pred']
                
                # Mixup loss
                pd_loss_mix = lam_pd * F.mse_loss(pd_pred_mix, y_a_pd) + (1 - lam_pd) * F.mse_loss(pd_pred_mix, y_b_pd)
                
                # Combine original and mixup losses
                pd_loss = 0.7 * pd_loss + 0.3 * pd_loss_mix
            
            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_shared_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Calculate shared mode loss with mixup and contrastive learning"""
        losses = {}
        
        # PK loss
        if batch_pk is not None:
            # Convert batch to dictionary
            if isinstance(batch_pk, (list, tuple)):
                x_pk, y_pk = batch_pk
                batch_pk_dict = {'x': x_pk, 'y': y_pk}
            else:
                batch_pk_dict = batch_pk
            
            pk_results = self.model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pk_features = pk_results['pk'].get('z', None)
            
            # Original PK loss
            pk_loss = F.mse_loss(pk_pred, pk_target)
            
            # Add contrastive loss for PK
            if self.config.lambda_contrast > 0 and pk_features is not None:
                contrast_loss_pk = self.contrastive_loss(pk_features, self.config.temperature)
                pk_loss = pk_loss + self.config.lambda_contrast * contrast_loss_pk
            
            # Apply mixup for PK if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pk, y_a_pk, y_b_pk, lam_pk = self.apply_mixup(
                    batch_pk_dict['x'], pk_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_pk = {'x': mixed_x_pk, 'y': pk_target}
                pk_results_mix = self.model({'pk': mixed_batch_pk})
                pk_pred_mix = pk_results_mix['pk']['pred']
                
                # Mixup loss
                pk_loss_mix = lam_pk * F.mse_loss(pk_pred_mix, y_a_pk) + (1 - lam_pk) * F.mse_loss(pk_pred_mix, y_b_pk)
                
                # Combine original and mixup losses
                pk_loss = 0.7 * pk_loss + 0.3 * pk_loss_mix
            
            losses['pk'] = pk_loss
        
        # PD loss
        if batch_pd is not None:
            # Convert batch to dictionary
            if isinstance(batch_pd, (list, tuple)):
                x_pd, y_pd = batch_pd
                batch_pd_dict = {'x': x_pd, 'y': y_pd}
            else:
                batch_pd_dict = batch_pd
            
            pd_results = self.model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pd_features = pd_results['pd'].get('z', None)
            
            # Original PD loss
            pd_loss = F.mse_loss(pd_pred, pd_target)
            
            # Add contrastive loss for PD
            if self.config.lambda_contrast > 0 and pd_features is not None:
                contrast_loss_pd = self.contrastive_loss(pd_features, self.config.temperature)
                pd_loss = pd_loss + self.config.lambda_contrast * contrast_loss_pd
            
            # Apply mixup for PD if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pd, y_a_pd, y_b_pd, lam_pd = self.apply_mixup(
                    batch_pd_dict['x'], pd_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_pd = {'x': mixed_x_pd, 'y': pd_target}
                pd_results_mix = self.model({'pd': mixed_batch_pd})
                pd_pred_mix = pd_results_mix['pd']['pred']
                
                # Mixup loss
                pd_loss_mix = lam_pd * F.mse_loss(pd_pred_mix, y_a_pd) + (1 - lam_pd) * F.mse_loss(pd_pred_mix, y_b_pd)
                
                # Combine original and mixup losses
                pd_loss = 0.7 * pd_loss + 0.3 * pd_loss_mix
            
            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_two_stage_shared_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Calculate two-stage shared mode loss with mixup and contrastive learning"""
        losses = {}
        
        # Stage 1: PK prediction
        if batch_pk is not None:
            # Convert batch to dictionary
            if isinstance(batch_pk, (list, tuple)):
                x_pk, y_pk = batch_pk
                batch_pk_dict = {'x': x_pk, 'y': y_pk}
            else:
                batch_pk_dict = batch_pk
            
            pk_results = self.model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pk_features = pk_results['pk'].get('z', None)
            
            # Original PK loss
            pk_loss = F.mse_loss(pk_pred, pk_target)
            
            # Add contrastive loss for PK
            if self.config.lambda_contrast > 0 and pk_features is not None:
                contrast_loss_pk = self.contrastive_loss(pk_features, self.config.temperature)
                pk_loss = pk_loss + self.config.lambda_contrast * contrast_loss_pk
            
            # Apply mixup for PK if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pk, y_a_pk, y_b_pk, lam_pk = self.apply_mixup(
                    batch_pk_dict['x'], pk_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_pk = {'x': mixed_x_pk, 'y': pk_target}
                pk_results_mix = self.model({'pk': mixed_batch_pk})
                pk_pred_mix = pk_results_mix['pk']['pred']
                
                # Mixup loss
                pk_loss_mix = lam_pk * F.mse_loss(pk_pred_mix, y_a_pk) + (1 - lam_pk) * F.mse_loss(pk_pred_mix, y_b_pk)
                
                # Combine original and mixup losses
                pk_loss = 0.7 * pk_loss + 0.3 * pk_loss_mix
            
            losses['pk'] = pk_loss
        
        # Stage 2: PD prediction (PK information included)
        if batch_pd is not None:
            # Convert batch to dictionary
            if isinstance(batch_pd, (list, tuple)):
                x_pd, y_pd = batch_pd
                batch_pd_dict = {'x': x_pd, 'y': y_pd}
            else:
                batch_pd_dict = batch_pd
            
            pd_results = self.model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)  # [B, 1] -> [B]
            pd_features = pd_results['pd'].get('z', None)
            
            # Original PD loss
            pd_loss = F.mse_loss(pd_pred, pd_target)
            
            # Add contrastive loss for PD
            if self.config.lambda_contrast > 0 and pd_features is not None:
                contrast_loss_pd = self.contrastive_loss(pd_features, self.config.temperature)
                pd_loss = pd_loss + self.config.lambda_contrast * contrast_loss_pd
            
            # Apply mixup for PD if enabled
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x_pd, y_a_pd, y_b_pd, lam_pd = self.apply_mixup(
                    batch_pd_dict['x'], pd_target, self.config.mixup_alpha
                )
                
                # Create mixed batch
                mixed_batch_pd = {'x': mixed_x_pd, 'y': pd_target}
                pd_results_mix = self.model({'pd': mixed_batch_pd})
                pd_pred_mix = pd_results_mix['pd']['pred']
                
                # Mixup loss
                pd_loss_mix = lam_pd * F.mse_loss(pd_pred_mix, y_a_pd) + (1 - lam_pd) * F.mse_loss(pd_pred_mix, y_b_pd)
                
                # Combine original and mixup losses
                pd_loss = 0.7 * pd_loss + 0.3 * pd_loss_mix
            
            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _get_train_loaders(self) -> List[Any]:
        """Return training data loaders"""
        if self.mode == "separate":
            return [self.data_loaders['train_pk'], self.data_loaders['train_pd']]
        else:
            return [self.data_loaders['train_pk'], self.data_loaders['train_pd']]
    
    def _get_val_loaders(self) -> List[Any]:
        """Return validation data loaders"""
        if self.mode == "separate":
            return [self.data_loaders['val_pk'], self.data_loaders['val_pd']]
        else:
            return [self.data_loaders['val_pk'], self.data_loaders['val_pd']]
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
        else:
            return batch.to(self.device)
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics"""
        # Loss history recording
        self.training_history['train_loss'].append(train_metrics['total_loss'])
        self.training_history['val_loss'].append(val_metrics['total_loss'])
        self.training_history['pk_train_loss'].append(train_metrics.get('pk_loss', 0.0))
        self.training_history['pd_train_loss'].append(train_metrics.get('pd_loss', 0.0))
        self.training_history['pk_val_loss'].append(val_metrics.get('pk_loss', 0.0))
        self.training_history['pd_val_loss'].append(val_metrics.get('pd_loss', 0.0))
        
        # Metrics history recording
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            self.training_history[f'pk_train_{metric}'].append(train_metrics.get(f'pk_{metric}', 0.0))
            self.training_history[f'pd_train_{metric}'].append(train_metrics.get(f'pd_{metric}', 0.0))
            self.training_history[f'pk_val_{metric}'].append(val_metrics.get(f'pk_{metric}', 0.0))
            self.training_history[f'pd_val_{metric}'].append(val_metrics.get(f'pd_{metric}', 0.0))
        
        # Log output
        self.logger.info(
            f"Epoch {epoch:4d} | Train |"
            f"Loss: {train_metrics['total_loss']:.6f} | "
            f"PK RMSE: {train_metrics.get('pk_rmse', 0.0):.6f} | "
            f"PD RMSE: {train_metrics.get('pd_rmse', 0.0):.6f} | "
            f"PK_R²: {train_metrics.get('pk_r2', 0.0):.4f} | "
            f"PD_R²: {train_metrics.get('pd_r2', 0.0):.4f}"
        )
        self.logger.info(
            f"Epoch {epoch:4d} | Valid |"
            f"Loss: {val_metrics['total_loss']:.6f} | "
            f"PK RMSE: {val_metrics.get('pk_rmse', 0.0):.6f} | "
            f"PD RMSE: {val_metrics.get('pd_rmse', 0.0):.6f} | "
            f"PK_R²: {val_metrics.get('pk_r2', 0.0):.4f} | "
            f"PD_R²: {val_metrics.get('pd_r2', 0.0):.4f}"
        )
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping"""
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def _save_best_model(self):
        """Save best model"""
        model_path = self.model_save_directory / "best_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'val_loss': self.best_val_loss,
            'config': self.config
        }, model_path)
    
    def save_model(self, filename: str):
        """Save model"""
        model_path = self.model_save_directory / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")
    
    def _get_final_results(self) -> Dict[str, Any]:
        """Return final results"""
        return {
            'best_val_loss': self.best_val_loss,
            'best_pk_rmse': self.best_pk_rmse,
            'best_pd_rmse': self.best_pd_rmse,
            'final_epoch': self.epoch,
            'training_history': self.training_history,
            'model_info': self.model.get_model_info()
        }
