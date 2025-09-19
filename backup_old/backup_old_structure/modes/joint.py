"""
Joint training mode
"""

import torch
from ..trainer import BaseTrainer
from ..unified_logger import UnifiedMetricsLogger
from utils.logging import get_logger
from utils.helpers import ReIter, roundrobin_loaders, rr_val, get_device


class JointTrainer(BaseTrainer):
    """Joint training: train PK and PD simultaneously"""
    
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
        
        self.metrics_logger = UnifiedMetricsLogger("joint", config)
    
    def predict(self, x):
        """Make prediction with hybrid joint model"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.eval()
            x = x.to(self.device)
            
            with torch.no_grad():
                batch = (x, None)  # Create batch format
                
                # Determine if this is PK or PD data
                is_pd_data = x.shape[1] > 10  # PD data has more features (including PD_BASELINE)
                
                if is_pd_data:
                    # PD data: predict PD directly using PD_BASELINE
                    pd_outs, z_pd, _ = self.model.forward_pd(batch)
                    pd_pred = pd_outs['pred'] if isinstance(pd_outs, dict) else pd_outs
                    return pd_pred.cpu()
                else:
                    # PK data: predict PK first, then use PK prediction for PD
                    pk_outs, z_pk, _ = self.model.forward_pk(batch)
                    pk_pred = pk_outs['pred'] if isinstance(pk_outs, dict) else pk_outs
                    
                    # Create PD input by concatenating PK prediction to original features
                    pd_input = torch.cat([x, pk_pred.unsqueeze(-1)], dim=-1)
                    pd_batch = (pd_input, None)
                    
                    # Get PD prediction using the concatenated input
                    pd_outs, z_pd, _ = self.model.forward_pd(pd_batch, pk_pred, z_pk)
                    pd_pred = pd_outs['pred'] if isinstance(pd_outs, dict) else pd_outs
                    
                    # Return PD prediction as final biomarker
                    return pd_pred.cpu()
        else:
            raise ValueError("Joint model not loaded")
    
    def train(self):
        """Execute joint training with hybrid PK/PD data usage"""
        self.logger.info("Joint training started with hybrid PK/PD data usage")
        
        train_pk = self.loaders["train_pk"]
        train_pd = self.loaders["train_pd"]
        val_pk = self.loaders["val_pk"]
        val_pd = self.loaders["val_pd"]
        
        best_val_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Hybrid training: Use both PK and PD data in each epoch
            train_loss_pk = self.train_epoch(train_pk)
            train_loss_pd = self.train_epoch(train_pd)
            
            # Combined loss (weighted average)
            train_loss = 0.4 * train_loss_pk + 0.6 * train_loss_pd
            
            if epoch % 50 == 0 or epoch < 10:
                self.logger.info(f"Epoch {epoch+1}: PK loss: {train_loss_pk:.6f}, PD loss: {train_loss_pd:.6f}, Combined: {train_loss:.6f}")
            
            # Use PD data for validation (since PD is the final target)
            val_loss = self.validate_epoch(val_pd)
            
            self.scheduler.step(val_loss)
            
            # Compute validation metrics for RMSE-based selection
            val_metrics = self.metrics_logger.compute_pk_pd_metrics(self.model, self.loaders, self.device)
            # Use PD RMSE for best model selection (final target is PD prediction)
            pd_rmse = val_metrics['pd']['rmse']
            
            if pd_rmse < best_val_rmse:
                best_val_rmse = pd_rmse
                patience_counter = 0
                self.save_model("best_joint_model.pth")
                # Log detailed metrics when best model is saved
                self.metrics_logger.log_best_model_saved(epoch, val_metrics)
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                self.logger.info(f"Joint training early stopping: {epoch+1} epochs")
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
            "best_val_rmse": best_val_rmse,
            "epochs": epoch+1,
            "mode": "joint"
        }
        
        self.logger.info("Joint training completed")
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
                pk_outs, pd_outs, _, _ = self.model(batch)
                # Extract prediction from PK outputs
                if isinstance(pk_outs, dict):
                    pred = pk_outs['pred']
                else:
                    pred = pk_outs
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
                pk_outs, pd_outs, _, _ = self.model(batch)
                # Extract prediction from PD outputs
                if isinstance(pd_outs, dict):
                    pred = pd_outs['pred']
                else:
                    pred = pd_outs
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
        
        return {
            'pk': pk_metrics,
            'pd': pd_metrics
        }
    
    def compute_loss(self, batch):
        """joint training loss computation: Handle both PK and PD data appropriately"""
