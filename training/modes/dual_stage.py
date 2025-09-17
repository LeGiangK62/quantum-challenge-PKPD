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
        
        # Always learn from original data first
        # PK loss - original
        pk_pred_orig, z_pk_orig, _ = self.model.forward_pk(batch)
        
        # Adjust tensor dimensions for original PK
        if pk_pred_orig.dim() == 1 and target.dim() == 2:
            target_orig = target.squeeze(-1)
        elif pk_pred_orig.dim() == 2 and target.dim() == 1:
            pk_pred_orig = pk_pred_orig.squeeze(-1)
        elif pk_pred_orig.dim() == 0:
            pk_pred_orig = pk_pred_orig.unsqueeze(0)
        elif target.dim() == 0:
            target_orig = target.unsqueeze(0)
        else:
            target_orig = target
        
        # PD loss - original (using z_pk_orig)
        pd_pred_orig, z_pd_orig, _ = self.model.forward_pd(z_pk_orig, batch)
        
        # Adjust tensor dimensions for original PD
        if pd_pred_orig.dim() == 1 and target_orig.dim() == 2:
            target_orig = target_orig.squeeze(-1)
        elif pd_pred_orig.dim() == 2 and target_orig.dim() == 1:
            pd_pred_orig = pd_pred_orig.squeeze(-1)
        elif pd_pred_orig.dim() == 0:
            pd_pred_orig = pd_pred_orig.unsqueeze(0)
        elif target_orig.dim() == 0:
            target_orig = target_orig.unsqueeze(0)
        
        # Original loss (always computed)
        pk_loss_orig = torch.nn.functional.mse_loss(pk_pred_orig, target_orig)
        pd_loss_orig = torch.nn.functional.mse_loss(pd_pred_orig, target_orig)
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
            
            # PD loss - mixup (using z_pk_mix)
            pd_pred_mix, z_pd_mix, _ = self.model.forward_pd(z_pk_mix, mixed_batch)
            
            # Adjust tensor dimensions for mixup PD
            if pd_pred_mix.dim() == 1 and y_a.dim() == 2:
                y_a = y_a.squeeze(-1)
                y_b = y_b.squeeze(-1)
            elif pd_pred_mix.dim() == 2 and y_a.dim() == 1:
                pd_pred_mix = pd_pred_mix.squeeze(-1)
            
            # Mixup loss calculation
            y_a_flat = y_a.squeeze(-1) if y_a.dim() > 1 else y_a
            y_b_flat = y_b.squeeze(-1) if y_b.dim() > 1 else y_b
            pk_loss_mix = lam * torch.nn.functional.mse_loss(pk_pred_mix, y_a_flat) + (1 - lam) * torch.nn.functional.mse_loss(pk_pred_mix, y_b_flat)
            pd_loss_mix = lam * torch.nn.functional.mse_loss(pd_pred_mix, y_a_flat) + (1 - lam) * torch.nn.functional.mse_loss(pd_pred_mix, y_b_flat)
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
