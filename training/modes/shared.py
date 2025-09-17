"""
Shared training mode
"""

import torch
from ..trainer import BaseTrainer
from ..unified_logger import UnifiedMetricsLogger
from utils.logging import get_logger
from utils.helpers import ReIter, roundrobin_loaders, rr_val, get_device


class SharedTrainer(BaseTrainer):
    """Shared training: shared encoder with separate heads"""
    
    def __init__(self, model, config, loaders, device=None):
        # Don't call super().__init__ if model is None
        if model is not None:
            super().__init__(model, config, loaders, device)
        else:
            # Initialize without model
            self.config = config
            self.loaders = loaders
            self.device = device if device is not None else get_device()
            self.logger = get_logger(__name__)
            
            # Result saving directory
            from pathlib import Path
            self.save_dir = Path(config.output_dir) / "models" / config.mode / config.encoder / f"s{config.random_state}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize scheduler (will be set up properly when we have model)
            self.scheduler = None
        
        self.metrics_logger = UnifiedMetricsLogger("shared", config)
    
    def predict(self, x):
        """Make prediction with shared model"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.eval()
            x = x.to(self.device)
            
            with torch.no_grad():
                # Use shared encoder
                shared_encoded = self.model['encoder'](x)
                
                # Get PD prediction (use PD head)
                pd_pred = self.model['head_pd']['mean'](shared_encoded)
                
                return pd_pred.cpu()
        else:
            raise ValueError("Shared model not loaded")
    
    def train(self):
        """Execute shared training"""
        self.logger.info("Shared training started")
        
        # Create combined loader
        train_combo = ReIter(roundrobin_loaders, self.loaders["train_pk"], self.loaders["train_pd"])
        val_combo = ReIter(rr_val, self.loaders["val_pk"], self.loaders["val_pd"])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_combo)
            val_loss = self.validate_epoch(val_combo)
            
            self.scheduler.step(val_loss)
            
            # Compute validation metrics
            val_metrics = self.metrics_logger.compute_pk_pd_metrics(self.model, self.loaders, self.device)
            
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True
                self.save_model("best_shared_model.pth", val_metrics['pk'])
                self.metrics_logger.log_best_model_saved(epoch, val_metrics)
            else:
                patience_counter += 1
            
            # Log epoch results with PK/PD separation
            self.metrics_logger.log_epoch_results(epoch, train_loss, val_loss, val_metrics, is_best)
            
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Compute final test metrics
        test_metrics = self.metrics_logger.compute_test_metrics(self.model, self.loaders, self.device)
        
        # Log final results with PK/PD separation
        self.metrics_logger.log_final_results(val_metrics, test_metrics)
        
        results = {
            "best_val_loss": best_val_loss,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "epochs": epoch+1,
            "mode": "shared"
        }
        
        return results
    
    def _compute_validation_metrics(self):
        """Calculate validation metrics with PK/PD separation"""
        self.model.eval()
        
        # PK metrics
        pk_preds = []
        pk_targets = []
        with torch.no_grad():
            for batch in self.loaders["val_pk"]:
                batch = self._to_device(batch)
                pred, _, _ = self.model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                pk_preds.append(pred.cpu())
                pk_targets.append(target.cpu())
        
        # PD metrics
        pd_preds = []
        pd_targets = []
        with torch.no_grad():
            for batch in self.loaders["val_pd"]:
                batch = self._to_device(batch)
                pred, _, _ = self.model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                pd_preds.append(pred.cpu())
                pd_targets.append(target.cpu())
        
        # Calculate separate metrics
        pk_preds = torch.cat(pk_preds, dim=0)
        pk_targets = torch.cat(pk_targets, dim=0)
        pd_preds = torch.cat(pd_preds, dim=0)
        pd_targets = torch.cat(pd_targets, dim=0)
        
        pk_metrics = self.compute_metrics(pk_preds, pk_targets)
        pd_metrics = self.compute_metrics(pd_preds, pd_targets)
        
        # Calculate combined metrics
        all_preds = torch.cat([pk_preds, pd_preds], dim=0)
        all_targets = torch.cat([pk_targets, pd_targets], dim=0)
        combined_metrics = self.compute_metrics(all_preds, all_targets)
        
        return {
            'pk': pk_metrics,
            'pd': pd_metrics,
            'combined': combined_metrics
        }
    
    def _compute_test_metrics(self):
        """Calculate test metrics with PK/PD separation"""
        self.model.eval()
        
        # PK test metrics
        pk_preds = []
        pk_targets = []
        with torch.no_grad():
            for batch in self.loaders["test_pk"]:
                batch = self._to_device(batch)
                pred, _, _ = self.model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                pk_preds.append(pred.cpu())
                pk_targets.append(target.cpu())
        
        # PD test metrics
        pd_preds = []
        pd_targets = []
        with torch.no_grad():
            for batch in self.loaders["test_pd"]:
                batch = self._to_device(batch)
                pred, _, _ = self.model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                pd_preds.append(pred.cpu())
                pd_targets.append(target.cpu())
        
        # Calculate separate metrics
        pk_preds = torch.cat(pk_preds, dim=0)
        pk_targets = torch.cat(pk_targets, dim=0)
        pd_preds = torch.cat(pd_preds, dim=0)
        pd_targets = torch.cat(pd_targets, dim=0)
        
        pk_metrics = self.compute_metrics(pk_preds, pk_targets)
        pd_metrics = self.compute_metrics(pd_preds, pd_targets)
        
        # Calculate combined metrics
        all_preds = torch.cat([pk_preds, pd_preds], dim=0)
        all_targets = torch.cat([pk_targets, pd_targets], dim=0)
        combined_metrics = self.compute_metrics(all_preds, all_targets)
        
        return {
            'pk': pk_metrics,
            'pd': pd_metrics,
            'combined': combined_metrics
        }
    
    def log_epoch_results_separated(self, epoch, train_loss, val_loss, val_metrics, is_best=False):
        """Log epoch results with PK/PD separation"""
        if self.config.verbose or epoch % 100 == 0 or is_best:
            pk_metrics = val_metrics['pk']
            pd_metrics = val_metrics['pd']
            combined_metrics = val_metrics['combined']
            
            self.logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.logger.info(f"  PK  - MSE: {pk_metrics['mse']:.4f}, RMSE: {pk_metrics['rmse']:.4f}, MAE: {pk_metrics['mae']:.4f}, R²: {pk_metrics['r2']:.4f}")
            self.logger.info(f"  PD  - MSE: {pd_metrics['mse']:.4f}, RMSE: {pd_metrics['rmse']:.4f}, MAE: {pd_metrics['mae']:.4f}, R²: {pd_metrics['r2']:.4f}")
            
            if is_best:
                self.logger.info("   Best model saved!")
    
    def log_final_results_separated(self, val_metrics, test_metrics):
        """Log final results with PK/PD separation"""
        self.logger.info("=" * 80)
        self.logger.info("FINAL RESULTS - SHARED TRAINING (PK/PD SEPARATED)")
        self.logger.info("=" * 80)
        
        # Validation results
        self.logger.info("VALIDATION RESULTS:")
        pk_val = val_metrics['pk']
        pd_val = val_metrics['pd']
        all_val = val_metrics['combined']
        
        self.logger.info(f"  PK  - MSE: {pk_val['mse']:.4f}, RMSE: {pk_val['rmse']:.4f}, MAE: {pk_val['mae']:.4f}, R²: {pk_val['r2']:.4f}")
        self.logger.info(f"  PD  - MSE: {pd_val['mse']:.4f}, RMSE: {pd_val['rmse']:.4f}, MAE: {pd_val['mae']:.4f}, R²: {pd_val['r2']:.4f}")
        
        # Test results
        self.logger.info("TEST RESULTS:")
        pk_test = test_metrics['pk']
        pd_test = test_metrics['pd']
        all_test = test_metrics['combined']
        
        self.logger.info(f"  PK  - MSE: {pk_test['mse']:.4f}, RMSE: {pk_test['rmse']:.4f}, MAE: {pk_test['mae']:.4f}, R²: {pk_test['r2']:.4f}")
        self.logger.info(f"  PD  - MSE: {pd_test['mse']:.4f}, RMSE: {pd_test['rmse']:.4f}, MAE: {pd_test['mae']:.4f}, R²: {pd_test['r2']:.4f}")
        
        self.logger.info("=" * 80)
    
    def compute_loss(self, batch):
        """Shared training loss computation with improved mixup (always learn original + optional mixup)"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        
        # Determine if this is PK or PD data based on input dimensions
        # In shared mode, we always process data as PD data (which includes both PK and PD processing)
        is_pk_data = False  # Don't use PK-only processing
        is_pd_data = True   # Always use PD processing (which includes PK)
        
        # Always learn from original data first
        if hasattr(self.model, 'forward_pk') and hasattr(self.model, 'forward_pd'):
            if is_pk_data:
                # This is PK data - only compute PK loss
                pk_pred_orig, z_pk_orig, _ = self.model.forward_pk(batch)
                
                # Adjust tensor dimensions for PK
                if pk_pred_orig.dim() == 1 and target.dim() == 2:
                    target_pk = target.squeeze(-1)
                elif pk_pred_orig.dim() == 2 and target.dim() == 1:
                    pk_pred_orig = pk_pred_orig.squeeze(-1)
                elif pk_pred_orig.dim() == 0:
                    pk_pred_orig = pk_pred_orig.unsqueeze(0)
                elif target.dim() == 0:
                    target_pk = target.unsqueeze(0)
                else:
                    target_pk = target
                
                # Compute PK loss only with PK target
                pk_loss_orig = torch.nn.functional.mse_loss(pk_pred_orig, target_pk)
                loss_orig = pk_loss_orig
                
                # Add contrastive loss for PK
                if self.config.lambda_contrast > 0 and z_pk_orig is not None:
                    contrast_loss_pk_orig = self.contrastive_loss(z_pk_orig, self.config.temperature)
                    loss_orig = loss_orig + self.config.lambda_contrast * contrast_loss_pk_orig
                
                return loss_orig
                
            elif is_pd_data:
                # This is PD data - compute both PK and PD losses
                # First get PK prediction (using PK input dimensions)
                pk_input = x[:, :11]  # Use first 11 features for PK
                pk_batch = (pk_input, target)  # Use same target for PK
                
                pk_pred_orig, z_pk_orig, _ = self.model.forward_pk(pk_batch)
                pd_pred_orig, z_pd_orig, _ = self.model.forward_pd(batch)
                
                # Adjust tensor dimensions for PK
                if pk_pred_orig.dim() == 1 and target.dim() == 2:
                    target_pk = target.squeeze(-1)
                elif pk_pred_orig.dim() == 2 and target.dim() == 1:
                    pk_pred_orig = pk_pred_orig.squeeze(-1)
                elif pk_pred_orig.dim() == 0:
                    pk_pred_orig = pk_pred_orig.unsqueeze(0)
                elif target.dim() == 0:
                    target_pk = target.unsqueeze(0)
                else:
                    target_pk = target
                
                # Adjust tensor dimensions for PD
                if pd_pred_orig.dim() == 1 and target.dim() == 2:
                    target_pd = target.squeeze(-1)
                elif pd_pred_orig.dim() == 2 and target.dim() == 1:
                    pd_pred_orig = pd_pred_orig.squeeze(-1)
                elif pd_pred_orig.dim() == 0:
                    pd_pred_orig = pd_pred_orig.unsqueeze(0)
                elif target.dim() == 0:
                    target_pd = target.unsqueeze(0)
                else:
                    target_pd = target
                
                # Original loss (always computed) - PK with PK target, PD with PD target
                pk_loss_orig = torch.nn.functional.mse_loss(pk_pred_orig, target_pk)
                pd_loss_orig = torch.nn.functional.mse_loss(pd_pred_orig, target_pd)
                # Give higher weight to PD (PK: 0.3, PD: 0.7)
                loss_orig = 0.3 * pk_loss_orig + 0.7 * pd_loss_orig
            
            # Add contrastive loss for original
            if self.config.lambda_contrast > 0:
                if z_pk_orig is not None:
                    contrast_loss_pk_orig = self.contrastive_loss(z_pk_orig, self.config.temperature)
                    loss_orig = loss_orig + self.config.lambda_contrast * contrast_loss_pk_orig
                if z_pd_orig is not None:
                    contrast_loss_pd_orig = self.contrastive_loss(z_pd_orig, self.config.temperature)
                    loss_orig = loss_orig + self.config.lambda_contrast * contrast_loss_pd_orig
            
            # Apply mixup as additional regularization (if enabled)
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                # For mixup, we need to handle PK and PD targets separately
                if isinstance(batch, dict):
                    target_pk_mix = batch.get("y_pk", target)  # PK target
                    target_pd_mix = batch.get("y_pd", target)  # PD target
                else:
                    # For tuple/list format, assume target contains both PK and PD targets
                    if target.dim() == 2 and target.shape[1] >= 2:
                        target_pk_mix = target[:, 0]  # First column is PK target
                        target_pd_mix = target[:, 1]  # Second column is PD target
                    else:
                        # Fallback: use same target for both (not ideal but prevents errors)
                        target_pk_mix = target
                        target_pd_mix = target
                
                # Apply mixup to PK target
                mixed_x_pk, y_a_pk, y_b_pk, lam_pk = self.apply_mixup(x[:, :11], target_pk_mix, self.config.mixup_alpha)
                # Apply mixup to PD target  
                mixed_x_pd, y_a_pd, y_b_pd, lam_pd = self.apply_mixup(x, target_pd_mix, self.config.mixup_alpha)
                
                # Create mixed batches
                pk_mixed_batch = (mixed_x_pk, target_pk_mix) if isinstance(batch, (list, tuple)) else {"x": mixed_x_pk, "y": target_pk_mix}
                pd_mixed_batch = (mixed_x_pd, target_pd_mix) if isinstance(batch, (list, tuple)) else {"x": mixed_x_pd, "y": target_pd_mix}
                
                # Forward pass with mixed data
                pk_pred_mix, z_pk_mix, _ = self.model.forward_pk(pk_mixed_batch)
                pd_pred_mix, z_pd_mix, _ = self.model.forward_pd(pd_mixed_batch)
                
                # Adjust tensor dimensions for mixup PK
                if pk_pred_mix.dim() == 1 and y_a_pk.dim() == 2:
                    y_a_pk = y_a_pk.squeeze(-1)
                    y_b_pk = y_b_pk.squeeze(-1)
                elif pk_pred_mix.dim() == 2 and y_a_pk.dim() == 1:
                    pk_pred_mix = pk_pred_mix.squeeze(-1)
                
                # Adjust tensor dimensions for mixup PD
                if pd_pred_mix.dim() == 1 and y_a_pd.dim() == 2:
                    y_a_pd = y_a_pd.squeeze(-1)
                    y_b_pd = y_b_pd.squeeze(-1)
                elif pd_pred_mix.dim() == 2 and y_a_pd.dim() == 1:
                    pd_pred_mix = pd_pred_mix.squeeze(-1)
                
                # Mixup loss calculation with separate targets
                y_a_pk_flat = y_a_pk.squeeze(-1) if y_a_pk.dim() > 1 else y_a_pk
                y_b_pk_flat = y_b_pk.squeeze(-1) if y_b_pk.dim() > 1 else y_b_pk
                y_a_pd_flat = y_a_pd.squeeze(-1) if y_a_pd.dim() > 1 else y_a_pd
                y_b_pd_flat = y_b_pd.squeeze(-1) if y_b_pd.dim() > 1 else y_b_pd
                
                pk_loss_mix = lam_pk * torch.nn.functional.mse_loss(pk_pred_mix, y_a_pk_flat) + (1 - lam_pk) * torch.nn.functional.mse_loss(pk_pred_mix, y_b_pk_flat)
                pd_loss_mix = lam_pd * torch.nn.functional.mse_loss(pd_pred_mix, y_a_pd_flat) + (1 - lam_pd) * torch.nn.functional.mse_loss(pd_pred_mix, y_b_pd_flat)
                loss_mix = pk_loss_mix + pd_loss_mix
                
                # Add contrastive loss for mixup
                if self.config.lambda_contrast > 0:
                    if z_pk_mix is not None:
                        contrast_loss_pk_mix = self.contrastive_loss(z_pk_mix, self.config.temperature)
                        loss_mix = loss_mix + self.config.lambda_contrast * contrast_loss_pk_mix
                    if z_pd_mix is not None:
                        contrast_loss_pd_mix = self.contrastive_loss(z_pd_mix, self.config.temperature)
                        loss_mix = loss_mix + self.config.lambda_contrast * contrast_loss_pd_mix
                
                # Combine original and mixup losses (weighted)
                # Original gets higher weight to ensure it's always learned
                total_loss = 0.7 * loss_orig + 0.3 * loss_mix
            else:
                # Only original loss
                total_loss = loss_orig
            
            return total_loss
        else:
            # For general model - always learn from original data first
            # DualBranchPKPD returns 4 values: pk_outs, pd_outs, z_pk, z_pd
            outputs = self.model(batch)
            if len(outputs) == 4:
                pk_outs, pd_outs, z_pk, z_pd = outputs
                # Use PD output as the main prediction
                if isinstance(pd_outs, dict):
                    pred_orig = pd_outs['pred']
                else:
                    pred_orig = pd_outs
                z_orig = z_pd
            else:
                # Fallback for other models that return 3 values
                pred_orig, z_orig, _ = outputs
            
            # Adjust tensor dimensions for original
            if pred_orig.dim() == 1 and target.dim() == 2:
                target_orig = target.squeeze(-1)
            elif pred_orig.dim() == 2 and target.dim() == 1:
                pred_orig = pred_orig.squeeze(-1)
            elif pred_orig.dim() == 0:
                pred_orig = pred_orig.unsqueeze(0)
            elif target.dim() == 0:
                target_orig = target.unsqueeze(0)
            else:
                target_orig = target
            
            # Original loss (always computed)
            loss_orig = torch.nn.functional.mse_loss(pred_orig, target_orig)
            
            # Add contrastive loss for original
            if self.config.lambda_contrast > 0 and z_orig is not None:
                contrast_loss_orig = self.contrastive_loss(z_orig, self.config.temperature)
                loss_orig = loss_orig + self.config.lambda_contrast * contrast_loss_orig
            
            # Apply mixup as additional regularization (if enabled)
            if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
                mixed_x, y_a, y_b, lam = self.apply_mixup(x, target, self.config.mixup_alpha)
                
                # Create mixed batch
                mixed_batch = (mixed_x, target) if isinstance(batch, (list, tuple)) else {"x": mixed_x, "y": target}
                
                # Forward pass with mixed data
                outputs_mix = self.model(mixed_batch)
                if len(outputs_mix) == 4:
                    pk_outs_mix, pd_outs_mix, z_pk_mix, z_pd_mix = outputs_mix
                    # Use PD output as the main prediction
                    if isinstance(pd_outs_mix, dict):
                        pred_mix = pd_outs_mix['pred']
                    else:
                        pred_mix = pd_outs_mix
                    z_mix = z_pd_mix
                else:
                    # Fallback for other models that return 3 values
                    pred_mix, z_mix, _ = outputs_mix
                
                # Adjust tensor dimensions for mixup
                if pred_mix.dim() == 1 and y_a.dim() == 2:
                    y_a = y_a.squeeze(-1)
                    y_b = y_b.squeeze(-1)
                elif pred_mix.dim() == 2 and y_a.dim() == 1:
                    pred_mix = pred_mix.squeeze(-1)
                
                # Mixup loss calculation
                y_a_flat = y_a.squeeze(-1) if y_a.dim() > 1 else y_a
                y_b_flat = y_b.squeeze(-1) if y_b.dim() > 1 else y_b
                loss_mix = lam * torch.nn.functional.mse_loss(pred_mix, y_a_flat) + (1 - lam) * torch.nn.functional.mse_loss(pred_mix, y_b_flat)
                
                # Add contrastive loss for mixup
                if self.config.lambda_contrast > 0 and z_mix is not None:
                    contrast_loss_mix = self.contrastive_loss(z_mix, self.config.temperature)
                    loss_mix = loss_mix + self.config.lambda_contrast * contrast_loss_mix
                
                # Combine original and mixup losses (weighted)
                # Original gets higher weight to ensure it's always learned
                total_loss = 0.7 * loss_orig + 0.3 * loss_mix
            else:
                # Only original loss
                total_loss = loss_orig
            
            return total_loss
