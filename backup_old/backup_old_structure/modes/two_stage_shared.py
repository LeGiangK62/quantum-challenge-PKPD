"""
Two-Stage Shared Training Mode
Stage 1: Train encoder with PK data
Stage 2: Fine-tune with PD data
"""

import torch
from ..trainer import BaseTrainer
from ..unified_logger import UnifiedMetricsLogger
from utils.logging import get_logger
from utils.helpers import ReIter, roundrobin_loaders, rr_val, get_device


class TwoStageSharedTrainer(BaseTrainer):
    """Two-stage shared training: PK first, then PD fine-tuning"""
    
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
        
        self.metrics_logger = UnifiedMetricsLogger("two_stage_shared", config)
    
    def predict(self, x):
        """Make prediction with two-stage shared model"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.eval()
            x = x.to(self.device)
            
            with torch.no_grad():
                # Create batch for forward_pd
                batch = (x, torch.zeros(x.size(0), 1).to(self.device))  # Dummy target
                
                # Get PD prediction (use PD head)
                pd_pred, _, _ = self.model.forward_pd(batch)
                
                return pd_pred.cpu()
        else:
            raise ValueError("Two-stage shared model not loaded")
    
    def train(self):
        """Execute two-stage shared training"""
        self.logger.info("Two-stage shared training started")
        
        # Stage 1: Train encoder with PK data
        self.logger.info("=== Stage 1: PK Training ===")
        pk_results = self._train_pk_stage()
        
        # Stage 2: Fine-tune with PD data
        self.logger.info("=== Stage 2: PD Fine-tuning ===")
        pd_results = self._train_pd_stage()
        
        # Compute final validation and test metrics
        val_metrics = self.metrics_logger.compute_pk_pd_metrics(self.model, self.loaders, self.device)
        test_metrics = self.metrics_logger.compute_test_metrics(self.model, self.loaders, self.device)
        
        # Log final results
        self.metrics_logger.log_final_results(val_metrics, test_metrics)
        
        results = {
            "pk_stage": pk_results,
            "pd_stage": pd_results,
            "test_metrics": test_metrics,
            "mode": "two_stage_shared"
        }
        
        return results
    
    def _train_pk_stage(self):
        """Stage 1: Train encoder with PK data"""
        self.logger.info("Training encoder with PK data...")
        
        # Use only PK data
        pk_loader = self.loaders["train_pk"]
        pk_val_loader = self.loaders["val_pk"]
        
        best_val_rmse = float('inf')
        patience_counter = 0
        
        # Set optimizer for PK training
        pk_optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.head_pk.parameters(), 'lr': self.config.learning_rate}
        ])
        
        pk_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            pk_optimizer, mode='min', patience=10, factor=0.5
        )
        
        for epoch in range(self.config.epochs // 2):  # PK uses half of total epochs
            # PK training
            self.model.train()
            train_loss = 0.0
            for batch in pk_loader:
                batch = self._to_device(batch)
                
                pk_optimizer.zero_grad()
                loss = self._compute_pk_loss(batch)
                loss.backward()
                pk_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(pk_loader)
            
            # PK validation
            val_loss, val_metrics = self._validate_pk(pk_val_loader)
            
            pk_scheduler.step(val_loss)
            
            if val_loss < best_val_rmse:
                best_val_rmse = val_loss
                patience_counter = 0
                self.logger.info(f"PK Stage - Best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"PK Stage Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if patience_counter >= self.config.patience // 2:
                self.logger.info(f"PK Stage early stopping at epoch {epoch+1}")
                break
        
        self.logger.info(f"PK Stage completed - Best Val Loss: {best_val_rmse:.4f}")
        return {"best_val_rmse": best_val_rmse, "epochs": epoch+1}
    
    def _train_pd_stage(self):
        """Stage 2: Fine-tune with PD data"""
        self.logger.info("Fine-tuning with PD data...")
        
        # Use PD data
        pd_loader = self.loaders["train_pd"]
        pd_val_loader = self.loaders["val_pd"]
        
        best_val_rmse = float('inf')
        patience_counter = 0
        
        # Set optimizer for PD fine-tuning (encoder with low LR)
        pd_optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': self.config.learning_rate * 0.1},  # Low LR
            {'params': self.model.head_pd.parameters(), 'lr': self.config.learning_rate}  # High LR
        ])
        
        pd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            pd_optimizer, mode='min', patience=10, factor=0.5
        )
        
        for epoch in range(self.config.epochs // 2):  # PD uses half of total epochs
            # PD training
            self.model.train()
            train_loss = 0.0
            for batch in pd_loader:
                batch = self._to_device(batch)
                
                pd_optimizer.zero_grad()
                loss = self._compute_pd_loss(batch)
                loss.backward()
                pd_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(pd_loader)
            
            # PD validation
            val_loss, val_metrics = self._validate_pd(pd_val_loader)
            
            pd_scheduler.step(val_loss)
            
            if val_loss < best_val_rmse:
                best_val_rmse = val_loss
                patience_counter = 0
                self.save_model("best_two_stage_shared_model.pth", val_metrics)
                self.logger.info(f"PD Stage - Best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"PD Stage Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if patience_counter >= self.config.patience // 2:
                self.logger.info(f"PD Stage early stopping at epoch {epoch+1}")
                break
        
        self.logger.info(f"PD Stage completed - Best Val Loss: {best_val_rmse:.4f}")
        return {"best_val_rmse": best_val_rmse, "epochs": epoch+1}
    
    def _compute_pk_loss(self, batch):
        """Calculate PK loss"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        
        # PK prediction
        pk_pred, _, _ = self.model.forward_pk(batch)
        
        # Calculate loss
        if pk_pred.dim() == 1 and target.dim() == 2:
            target = target.squeeze(-1)
        elif pk_pred.dim() == 2 and target.dim() == 1:
            pk_pred = pk_pred.squeeze(-1)
        
        return torch.nn.functional.mse_loss(pk_pred, target)
    
    def _compute_pd_loss(self, batch):
        """Calculate PD loss"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        
        # PD prediction
        pd_pred, _, _ = self.model.forward_pd(batch)
        
        # Calculate loss
        if pd_pred.dim() == 1 and target.dim() == 2:
            target = target.squeeze(-1)
        elif pd_pred.dim() == 2 and target.dim() == 1:
            pd_pred = pd_pred.squeeze(-1)
        
        return torch.nn.functional.mse_loss(pd_pred, target)
    
    def _validate_pk(self, val_loader):
        """PK validation"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                pred, _, _ = self.model.forward_pk(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                loss = torch.nn.functional.mse_loss(pred, target)
                total_loss += loss.item()
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(val_loader), metrics
    
    def _validate_pd(self, val_loader):
        """PD validation"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                pred, _, _ = self.model.forward_pd(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                loss = torch.nn.functional.mse_loss(pred, target)
                total_loss += loss.item()
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(val_loader), metrics
