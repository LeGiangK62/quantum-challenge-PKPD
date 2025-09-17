"""
Dual stage training mode
"""

import torch
from ..trainer import BaseTrainer
from ..unified_logger import UnifiedMetricsLogger
from utils.logging import get_logger
from utils.helpers import ReIter, roundrobin_loaders, rr_val


class DualStageTrainer(BaseTrainer):
    """Dual stage training: x -> front_encoder -> z -> back_encoder -> z2"""
    
    def __init__(self, model, config, loaders, device=None):
        super().__init__(model, config, loaders, device)
        self.logger = get_logger(__name__)
        self.metrics_logger = UnifiedMetricsLogger("dual_stage", config)
    
    def train(self):
        """Execute dual stage training"""
        self.logger.info("Dual stage training started")
        
        # Create combined loader
        train_combo = ReIter(roundrobin_loaders, self.loaders["train_pk"], self.loaders["train_pd"])
        val_combo = ReIter(rr_val, self.loaders["val_pk"], self.loaders["val_pd"])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_combo)
            val_loss = self.validate_epoch(val_combo)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("best_dual_stage_model.pth")
                # Compute and log detailed metrics when best model is saved
                val_metrics = self.metrics_logger.compute_pk_pd_metrics(self.model, self.loaders, self.device)
                self.metrics_logger.log_best_model_saved(epoch, val_metrics)
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                self.logger.info(f"Dual stage training early stopping: {epoch+1} epochs")
                break
            
            # Log epoch results with unified logger
            if epoch % 100 == 0 or epoch < 10:
                val_metrics = self.metrics_logger.compute_pk_pd_metrics(self.model, self.loaders, self.device)
                self.metrics_logger.log_epoch_results(epoch, train_loss, val_loss, val_metrics, False)
        
        # Log final results
        val_metrics = self.metrics_logger.compute_pk_pd_metrics(self.model, self.loaders, self.device)
        test_metrics = self.metrics_logger.compute_test_metrics(self.model, self.loaders, self.device)
        self.metrics_logger.log_final_results(val_metrics, test_metrics)
        
        results = {
            "best_val_loss": best_val_loss,
            "epochs": epoch+1,
            "mode": "dual_stage"
        }
        
        self.logger.info("Dual stage training completed")
        return results
    
    def _compute_validation_metrics(self):
        """Calculate validation metrics"""
        val_combo = ReIter(rr_val, self.loaders["val_pk"], self.loaders["val_pd"])
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_combo:
                batch = self._to_device(batch)
                pred, _, _ = self.model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Adjust tensor dimensions
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Calculate overall metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return self.compute_metrics(all_preds, all_targets)
    
    def compute_loss(self, batch):
        """Dual stage loss computation with improved mixup (always learn original + optional mixup)"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        
        # Determine if this is PK or PD data based on input dimensions
        # PK data has 11 features, PD data has 12 features
        is_pk_data = x.shape[1] == 11
        is_pd_data = x.shape[1] == 12
        
        # Always learn from original data first
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
            # Extract separate PK and PD targets from batch
            if isinstance(batch, dict):
                target_pk = batch.get("y_pk", target)  # PK target
                target_pd = batch.get("y_pd", target)  # PD target
            else:
                # For tuple/list format, assume target contains both PK and PD targets
                # If target is 2D, first column is PK, second is PD
                if target.dim() == 2 and target.shape[1] >= 2:
                    target_pk = target[:, 0]  # First column is PK target
                    target_pd = target[:, 1]  # Second column is PD target
                else:
                    # Fallback: use same target for both (not ideal but prevents errors)
                    target_pk = target
                    target_pd = target
            
            # First get PK prediction (using PK input dimensions)
            pk_input = x[:, :11]  # Use first 11 features for PK
            pk_batch = (pk_input, target_pk)  # Use PK target
            
            pk_pred_orig, z_pk_orig, _ = self.model.forward_pk(pk_batch)
            
            # Adjust tensor dimensions for PK
            if pk_pred_orig.dim() == 1 and target_pk.dim() == 2:
                target_pk = target_pk.squeeze(-1)
            elif pk_pred_orig.dim() == 2 and target_pk.dim() == 1:
                pk_pred_orig = pk_pred_orig.squeeze(-1)
            elif pk_pred_orig.dim() == 0:
                pk_pred_orig = pk_pred_orig.unsqueeze(0)
            elif target_pk.dim() == 0:
                target_pk = target_pk.unsqueeze(0)
            
            # PD loss - original (using z_pk_orig)
            pd_pred_orig, z_pd_orig, _ = self.model.forward_pd(z_pk_orig, batch)
            
            # Adjust tensor dimensions for PD
            if pd_pred_orig.dim() == 1 and target_pd.dim() == 2:
                target_pd = target_pd.squeeze(-1)
            elif pd_pred_orig.dim() == 2 and target_pd.dim() == 1:
                pd_pred_orig = pd_pred_orig.squeeze(-1)
            elif pd_pred_orig.dim() == 0:
                pd_pred_orig = pd_pred_orig.unsqueeze(0)
            elif target_pd.dim() == 0:
                target_pd = target_pd.unsqueeze(0)
            
            # Original loss (always computed) - PK with PK target, PD with PD target
            pk_loss_orig = torch.nn.functional.mse_loss(pk_pred_orig, target_pk)
            pd_loss_orig = torch.nn.functional.mse_loss(pd_pred_orig, target_pd)
            loss_orig = pk_loss_orig + pd_loss_orig
            
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
            if is_pd_data:
                # For PD data, apply mixup to both PK and PD targets
                if isinstance(batch, dict):
                    target_pk_mix = batch.get("y_pk", target)
                    target_pd_mix = batch.get("y_pd", target)
                else:
                    if target.dim() == 2 and target.shape[1] >= 2:
                        target_pk_mix = target[:, 0]
                        target_pd_mix = target[:, 1]
                    else:
                        target_pk_mix = target
                        target_pd_mix = target
                
                # Apply mixup to PK target
                mixed_x_pk, y_a_pk, y_b_pk, lam_pk = self.apply_mixup(x[:, :11], target_pk_mix, self.config.mixup_alpha)
                # Apply mixup to PD target  
                mixed_x_pd, y_a_pd, y_b_pd, lam_pd = self.apply_mixup(x, target_pd_mix, self.config.mixup_alpha)
                
                # Create mixed batches
                pk_mixed_batch = (mixed_x_pk, target_pk_mix) if isinstance(batch, (list, tuple)) else {"x": mixed_x_pk, "y": target_pk_mix}
                pd_mixed_batch = (mixed_x_pd, target_pd_mix) if isinstance(batch, (list, tuple)) else {"x": mixed_x_pd, "y": target_pd_mix}
                
                # PK loss - mixup
                pk_pred_mix, z_pk_mix, _ = self.model.forward_pk(pk_mixed_batch)
                
                # Adjust tensor dimensions for mixup PK
                if pk_pred_mix.dim() == 1 and y_a_pk.dim() == 2:
                    y_a_pk = y_a_pk.squeeze(-1)
                    y_b_pk = y_b_pk.squeeze(-1)
                elif pk_pred_mix.dim() == 2 and y_a_pk.dim() == 1:
                    pk_pred_mix = pk_pred_mix.squeeze(-1)
                
                # PD loss - mixup (using z_pk_mix)
                pd_pred_mix, z_pd_mix, _ = self.model.forward_pd(z_pk_mix, pd_mixed_batch)
                
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
            else:
                # For PK data, use standard mixup
                mixed_x, y_a, y_b, lam = self.apply_mixup(x, target, self.config.mixup_alpha)
                
                # Create mixed batch
                mixed_batch = (mixed_x, target) if isinstance(batch, (list, tuple)) else {"x": mixed_x, "y": target}
                
                # PK loss - mixup
                pk_pred_mix, z_pk_mix, _ = self.model.forward_pk(mixed_batch)
                
                # Adjust tensor dimensions for mixup PK
                if pk_pred_mix.dim() == 1 and y_a.dim() == 2:
                    y_a = y_a.squeeze(-1)
                    y_b = y_b.squeeze(-1)
                elif pk_pred_mix.dim() == 2 and y_a.dim() == 1:
                    pk_pred_mix = pk_pred_mix.squeeze(-1)
                
                # Mixup loss calculation
                y_a_flat = y_a.squeeze(-1) if y_a.dim() > 1 else y_a
                y_b_flat = y_b.squeeze(-1) if y_b.dim() > 1 else y_b
                loss_mix = lam * torch.nn.functional.mse_loss(pk_pred_mix, y_a_flat) + (1 - lam) * torch.nn.functional.mse_loss(pk_pred_mix, y_b_flat)
            
            # Add contrastive loss for mixup
            if self.config.lambda_contrast > 0:
                if is_pd_data:
                    # For PD data, add contrastive loss for both PK and PD
                    if z_pk_mix is not None:
                        contrast_loss_pk_mix = self.contrastive_loss(z_pk_mix, self.config.temperature)
                        loss_mix = loss_mix + self.config.lambda_contrast * contrast_loss_pk_mix
                    if z_pd_mix is not None:
                        contrast_loss_pd_mix = self.contrastive_loss(z_pd_mix, self.config.temperature)
                        loss_mix = loss_mix + self.config.lambda_contrast * contrast_loss_pd_mix
                else:
                    # For PK data, only add PK contrastive loss
                    if z_pk_mix is not None:
                        contrast_loss_pk_mix = self.contrastive_loss(z_pk_mix, self.config.temperature)
                        loss_mix = loss_mix + self.config.lambda_contrast * contrast_loss_pk_mix
            
            # Combine original and mixup losses (weighted)
            # Original gets higher weight to ensure it's always learned
            total_loss = 0.7 * loss_orig + 0.3 * loss_mix
        else:
            # Only original loss
            total_loss = loss_orig
        
        return total_loss
