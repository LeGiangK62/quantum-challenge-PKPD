"""
Separate training mode
"""

import torch
from ..trainer import BaseTrainer
from utils.logging import get_logger
from utils.helpers import ReIter, roundrobin_loaders, rr_val, get_device


class PKTrainer(BaseTrainer):
    """PK-specific trainer"""
    
    def __init__(self, model, config, loaders, device=None):
        super().__init__(model, config, loaders, device)

    def compute_loss(self, batch):
        """PK loss computation with improved mixup (always learn original + optional mixup)"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        
        # Always learn from original data
        pred_orig, z_orig, _ = self.model(x)
        
        # Ensure tensor dimensions match for original
        target_orig = target  # Initialize target_orig
        if pred_orig.dim() == 1 and target.dim() == 2:
            target_orig = target.squeeze(-1)
        elif pred_orig.dim() == 2 and target.dim() == 1:
            pred_orig = pred_orig.squeeze(-1)
        elif pred_orig.dim() == 0:  # Handle scalar prediction
            pred_orig = pred_orig.unsqueeze(0)
        elif target.dim() == 0:  # Handle scalar target
            target_orig = target.unsqueeze(0)
        
        # Ensure both tensors have the same shape for original
        if pred_orig.shape != target_orig.shape:
            if pred_orig.numel() == target_orig.numel():
                pred_orig = pred_orig.view_as(target_orig)
            else:
                # If shapes are completely different, use the smaller one
                min_size = min(pred_orig.numel(), target_orig.numel())
                pred_orig = pred_orig.view(-1)[:min_size]
                target_orig = target_orig.view(-1)[:min_size]
        
        # Original loss (always computed)
        loss_orig = torch.nn.functional.mse_loss(pred_orig, target_orig)
        
        # Add contrastive loss for original
        if self.config.lambda_contrast > 0 and z_orig is not None:
            contrast_loss_orig = self.contrastive_loss(z_orig, self.config.temperature)
            loss_orig = loss_orig + self.config.lambda_contrast * contrast_loss_orig
        
        # Apply mixup as additional regularization (if enabled)
        if self.config.use_mixup and torch.rand(()) < self.config.mixup_prob:
            mixed_x, y_a, y_b, lam = self.apply_mixup(x, target, self.config.mixup_alpha)
            pred_mix, z_mix, _ = self.model(mixed_x)
            
            # Ensure tensor dimensions match for mixup
            if pred_mix.dim() == 1 and y_a.dim() == 2:
                y_a = y_a.squeeze(-1)
                y_b = y_b.squeeze(-1)
            elif pred_mix.dim() == 2 and y_a.dim() == 1:
                pred_mix = pred_mix.squeeze(-1)
            
            loss_a = torch.nn.functional.mse_loss(pred_mix, y_a)
            loss_b = torch.nn.functional.mse_loss(pred_mix, y_b)
            loss_mix = lam * loss_a + (1 - lam) * loss_b
            
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


class PDTrainer(BaseTrainer):
    """PD-specific trainer"""
    
    def __init__(self, model, config, loaders, device=None):
        super().__init__(model, config, loaders, device)
    
    def compute_loss(self, batch):
        """PD loss computation with improved mixup (always learn original + optional mixup)"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        
        # Always learn from original data
        pred_orig, z_orig, _ = self.model(x)
        
        # Ensure tensor dimensions match for original
        target_orig = target  # Initialize target_orig
        if pred_orig.dim() == 1 and target.dim() == 2:
            target_orig = target.squeeze(-1)
        elif pred_orig.dim() == 2 and target.dim() == 1:
            pred_orig = pred_orig.squeeze(-1)
        elif pred_orig.dim() == 0:  # Handle scalar prediction
            pred_orig = pred_orig.unsqueeze(0)
        elif target.dim() == 0:  # Handle scalar target
            target_orig = target.unsqueeze(0)
        
        # Ensure both tensors have the same shape for original
        if pred_orig.shape != target_orig.shape:
            if pred_orig.numel() == target_orig.numel():
                pred_orig = pred_orig.view_as(target_orig)
            else:
                # If shapes are completely different, use the smaller one
                min_size = min(pred_orig.numel(), target_orig.numel())
                pred_orig = pred_orig.view(-1)[:min_size]
                target_orig = target_orig.view(-1)[:min_size]
        
        # Original loss (always computed)
        loss_orig = torch.nn.functional.mse_loss(pred_orig, target_orig)
        
        # Add contrastive loss for original
        if self.config.lambda_contrast > 0 and z_orig is not None:
            contrast_loss_orig = self.contrastive_loss(z_orig, self.config.temperature)
            loss_orig = loss_orig + self.config.lambda_contrast * contrast_loss_orig
        
        # Apply mixup as additional regularization (if enabled)
        if self.config.use_mixup and torch.rand(()) < self.config.mixup_prob:
            mixed_x, y_a, y_b, lam = self.apply_mixup(x, target, self.config.mixup_alpha)
            pred_mix, z_mix, _ = self.model(mixed_x)
            
            # Ensure tensor dimensions match for mixup
            if pred_mix.dim() == 1 and y_a.dim() == 2:
                y_a = y_a.squeeze(-1)
                y_b = y_b.squeeze(-1)
            elif pred_mix.dim() == 2 and y_a.dim() == 1:
                pred_mix = pred_mix.squeeze(-1)
            
            loss_a = torch.nn.functional.mse_loss(pred_mix, y_a)
            loss_b = torch.nn.functional.mse_loss(pred_mix, y_b)
            loss_mix = lam * loss_a + (1 - lam) * loss_b
            
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


class SeparateTrainer(BaseTrainer):
    """Separate training: PK first, then PD (with PK predictions)"""
    
    def __init__(self, model, config, loaders, device=None):
        # Don't call super().__init__ since we don't have a model yet
        self.config = config
        self.loaders = loaders
        self.logger = get_logger(__name__)
        self.device = device if device is not None else get_device()
        
        # Result saving directory
        from pathlib import Path
        run_name = config.run_name if config.run_name else "default_run"
        # Create hierarchical model save directory: models/mode/encoder/s{seed}/
        self.save_dir = Path(config.output_dir) / "models" / config.mode / config.encoder / f"s{config.random_state}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scheduler (will be set up properly when we have models)
        self.scheduler = None
        
        self.logger.info(f"SeparateTrainer initialization completed - device: {self.device}")
    
    def _to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, (list, tuple)):
            return [item.to(self.device) if torch.is_tensor(item) else item for item in batch]
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            return batch.to(self.device)
    
    def train(self):
        """Execute separate training"""
        self.logger.info("Separate training: PK -> PD")
        
        # 1st step: PK model training
        self.logger.info("PK training...")
        pk_results = self._train_pk()
        
        # 2nd step: PD model training (with PK predictions)
        self.logger.info("PD training...")
        pd_results = self._train_pd()
        
        # Compute final test metrics for both PK and PD using trained models
        # Note: We would need to store the best models from training to use them here
        # For now, we'll use the temporary model approach
        pk_test_metrics = self._compute_test_metrics(self.loaders["test_pk"])
        pd_test_metrics = self._compute_test_metrics(self.loaders["test_pd"])
        
        # Log final results
        self.logger.info("FINAL RESULTS - SEPARATE TRAINING")
        self.logger.info("=" * 80)
        self.logger.info("VALIDATION RESULTS:")
        self.logger.info(f"  PK  - MSE: {pk_results.get('val_metrics', {}).get('mse', 0):.4f}, RMSE: {pk_results.get('val_metrics', {}).get('rmse', 0):.4f}, MAE: {pk_results.get('val_metrics', {}).get('mae', 0):.4f}, R²: {pk_results.get('val_metrics', {}).get('r2', 0):.4f}")
        self.logger.info(f"  PD  - MSE: {pd_results.get('val_metrics', {}).get('mse', 0):.4f}, RMSE: {pd_results.get('val_metrics', {}).get('rmse', 0):.4f}, MAE: {pd_results.get('val_metrics', {}).get('mae', 0):.4f}, R²: {pd_results.get('val_metrics', {}).get('r2', 0):.4f}")
        self.logger.info("TEST RESULTS:")
        self.logger.info(f"  PK  - MSE: {pk_test_metrics['mse']:.4f}, RMSE: {pk_test_metrics['rmse']:.4f}, MAE: {pk_test_metrics['mae']:.4f}, R²: {pk_test_metrics['r2']:.4f}")
        self.logger.info(f"  PD  - MSE: {pd_test_metrics['mse']:.4f}, RMSE: {pd_test_metrics['rmse']:.4f}, MAE: {pd_test_metrics['mae']:.4f}, R²: {pd_test_metrics['r2']:.4f}")
        self.logger.info("=" * 80)
        
        # Integrate results
        results = {
            "pk": pk_results,
            "pd": pd_results,
            "pk_test_metrics": pk_test_metrics,
            "pd_test_metrics": pd_test_metrics,
            "mode": "separate"
        }
        
        return results
    
    def _train_pk(self):
        """Train PK model"""
        self.logger.info("Starting PK model creation...")
        
        # Validate data loaders
        if "train_pk" not in self.loaders or "val_pk" not in self.loaders:
            raise ValueError("PK data loaders not found in loaders dictionary")
        
        self.logger.info("PK data loaders validated")
        
        # Create PK-specific model (from existing structure)
        from utils.helpers import build_encoder, build_head
        
        pk_input_dim = self.loaders["train_pk"].dataset.tensors[0].shape[1]
        if pk_input_dim <= 0:
            raise ValueError(f"Invalid PK input dimension: {pk_input_dim}")
        
        self.logger.info(f"PK input dimension: {pk_input_dim}")
        
        pk_encoder = build_encoder(self.config.encoder, pk_input_dim, self.config)
        pk_head = build_head("mse", self.config.hidden, "pk", self.config)
        
        self.logger.info("PK encoder and head created")
        
        # Create PK model
        from models.architectures import EncHeadModel
        pk_model = EncHeadModel(pk_encoder, pk_head)
        
        # Store PK model as attribute for saving
        self.pk_model = pk_model
        
        self.logger.info("PK model created")
        
        # Create PK trainer
        pk_trainer = PKTrainer(pk_model, self.config, self.loaders, device=self.device)
        
        self.logger.info("PK trainer created")
        
        # Train PK with limited epochs for debugging
        max_epochs = self.config.epochs
        self.logger.info(f"Starting PK training for {max_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            self.logger.info(f"PK Epoch {epoch+1}/{max_epochs}")
            
            train_loss = pk_trainer.train_epoch(self.loaders["train_pk"])
            train_metrics = self._compute_train_metrics(pk_model, self.loaders["train_pk"])
            
            val_loss, val_metrics = self._validate_with_metrics(pk_model, self.loaders["val_pk"])
            
            # Log detailed metrics every epoch
            self.logger.info(f"PK Epoch {epoch+1} - Train: Loss={train_loss:.4f}, MSE={train_metrics['mse']:.4f}, RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, R²={train_metrics['r2']:.4f}")
            self.logger.info(f"PK Epoch {epoch+1} - Val:  Loss={val_loss:.4f}, MSE={val_metrics['mse']:.4f}, RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}, R²={val_metrics['r2']:.4f}")
            
            if hasattr(pk_trainer, 'scheduler') and pk_trainer.scheduler is not None:
                pk_trainer.scheduler.step(val_loss)
            
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True
                pk_trainer.save_model("best_pk_model.pth", val_metrics)
            else:
                patience_counter += 1
            
            # Log epoch results
            pk_trainer.log_epoch_results(epoch, train_loss, val_loss, val_metrics, is_best)
            
            if patience_counter >= self.config.patience:
                self.logger.info(f"PK training early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("PK training completed")
        return {"best_val_loss": best_val_loss, "val_metrics": val_metrics, "epochs": epoch+1}
    
    def _compute_train_metrics(self, model, train_loader):
        """Compute training metrics"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in train_loader:
                batch = self._to_device(batch)
                pred, _, _ = model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Calculate total metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return metrics
    
    def _validate_with_metrics(self, model, val_loader):
        """Validate and calculate metrics"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                pred, _, _ = model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                loss = torch.nn.functional.mse_loss(pred, target)
                total_loss += loss.item()
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Calculate total metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(val_loader), metrics
    
    def _train_pd(self):
        """Train PD model (including PK predictions)"""
        self.logger.info("Starting PD model creation...")
        
        # Validate data loaders
        if "train_pd" not in self.loaders or "val_pd" not in self.loaders:
            raise ValueError("PD data loaders not found in loaders dictionary")
        
        self.logger.info("PD data loaders validated")
        
        # Create PD-specific model
        from utils.helpers import build_encoder, build_head
        
        pd_input_dim = self.loaders["train_pd"].dataset.tensors[0].shape[1]
        if pd_input_dim <= 0:
            raise ValueError(f"Invalid PD input dimension: {pd_input_dim}")
        
        self.logger.info(f"PD input dimension: {pd_input_dim}")
        
        pd_encoder = build_encoder(self.config.encoder, pd_input_dim, self.config)
        pd_head = build_head("mse", self.config.hidden, "pd", self.config)
        
        self.logger.info("PD encoder and head created")
        
        # Create PD model
        from models.architectures import EncHeadModel
        pd_model = EncHeadModel(pd_encoder, pd_head)
        
        # Store PD model as attribute for saving
        self.pd_model = pd_model
        
        self.logger.info("PD model created")
        
        # Create PD trainer
        pd_trainer = PDTrainer(pd_model, self.config, self.loaders, device=self.device)
        
        self.logger.info("PD trainer created")
        
        # Train PD with limited epochs for debugging
        max_epochs = self.config.epochs
        self.logger.info(f"Starting PD training for {max_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            self.logger.info(f"PD Epoch {epoch+1}/{max_epochs}")
            
            train_loss = pd_trainer.train_epoch(self.loaders["train_pd"])
            train_metrics = self._compute_train_metrics(pd_model, self.loaders["train_pd"])
            
            val_loss, val_metrics = self._validate_with_metrics(pd_model, self.loaders["val_pd"])
            
            # Log detailed metrics every epoch
            self.logger.info(f"PD Epoch {epoch+1} - Train: Loss={train_loss:.4f}, MSE={train_metrics['mse']:.4f}, RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, R²={train_metrics['r2']:.4f}")
            self.logger.info(f"PD Epoch {epoch+1} - Val:  Loss={val_loss:.4f}, MSE={val_metrics['mse']:.4f}, RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}, R²={val_metrics['r2']:.4f}")
            
            if hasattr(pd_trainer, 'scheduler') and pd_trainer.scheduler is not None:
                pd_trainer.scheduler.step(val_loss)
            
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True
                pd_trainer.save_model("best_pd_model.pth", val_metrics)
            else:
                patience_counter += 1
            
            # Log epoch results
            pd_trainer.log_epoch_results(epoch, train_loss, val_loss, val_metrics, is_best)
            
            if patience_counter >= self.config.patience:
                self.logger.info(f"PD training early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("PD training completed")
        return {"best_val_loss": best_val_loss, "val_metrics": val_metrics, "epochs": epoch+1}
    
    def _compute_test_metrics(self, test_loader, model=None):
        """Calculate test metrics with optional model parameter to avoid creating new models"""
        if model is None:
            # Only create a temporary model if none provided
            from utils.helpers import build_encoder, build_head
            from models.architectures import EncHeadModel
            
            input_dim = test_loader.dataset.tensors[0].shape[1]
            encoder = build_encoder(self.config.encoder, input_dim, self.config)
            head = build_head("mse", self.config.hidden, "test", self.config)
            model = EncHeadModel(encoder, head)
            model.to(self.device)
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = self._to_device(batch)
                pred, _, _ = model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match
                if pred.dim() == 1 and target.dim() == 2:
                    target = target.squeeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    pred = pred.squeeze(-1)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Calculate total metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return self.compute_metrics(all_preds, all_targets)
    
    def compute_loss(self, batch):
        """Calculate loss"""
        # Basic MSE loss
        pred, _, _ = self.model(batch)
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        return torch.nn.functional.mse_loss(pred, target)
    
    def save_model(self, filename="final_model.pth", metrics=None):
        """Save both PK and PD models"""
        # Save PK model
        pk_save_path = self.save_dir / f"pk_{filename}"
        if hasattr(self, 'pk_model') and self.pk_model is not None:
            torch.save({
                'model_state_dict': self.pk_model.state_dict(),
                'model_type': 'pk',
                'config': self.config,
                'metrics': metrics
            }, pk_save_path)
            self.logger.info(f"PK model saved: {pk_save_path}")
        
        # Save PD model
        pd_save_path = self.save_dir / f"pd_{filename}"
        if hasattr(self, 'pd_model') and self.pd_model is not None:
            torch.save({
                'model_state_dict': self.pd_model.state_dict(),
                'model_type': 'pd',
                'config': self.config,
                'metrics': metrics
            }, pd_save_path)
            self.logger.info(f"PD model saved: {pd_save_path}")
        
        # Also save a combined checkpoint for convenience
        combined_save_path = self.save_dir / filename
        combined_data = {
            'config': self.config,
            'metrics': metrics,
            'model_type': 'separate_pk_pd'
        }
        
        if hasattr(self, 'pk_model') and self.pk_model is not None:
            combined_data['pk_model_state_dict'] = self.pk_model.state_dict()
        
        if hasattr(self, 'pd_model') and self.pd_model is not None:
            combined_data['pd_model_state_dict'] = self.pd_model.state_dict()
        
        torch.save(combined_data, combined_save_path)
        self.logger.info(f"Combined model saved: {combined_save_path}")
        
        # Simple save message with metrics
        if metrics:
            self.logger.info(f"Saved! MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        else:
            self.logger.info("Saved!")
    
    def predict(self, x):
        """Make prediction with separate PK/PD models"""
        # For separate mode, we need both PK and PD predictions
        if hasattr(self, 'pk_model') and hasattr(self, 'pd_model'):
            self.pk_model.eval()
            self.pd_model.eval()
            x = x.to(self.device)
            
            with torch.no_grad():
                # Get PK prediction - use encoder and head separately
                pk_encoded = self.pk_model['encoder'](x)
                pk_pred = self.pk_model['head']['mean'](pk_encoded)
                
                # For PD prediction, we need to prepare the right input
                # PD model expects 12 dimensions, but we have 11 from PK input
                # We need to add one more feature (PK prediction or another feature)
                if x.shape[1] == 11:  # PK model input size
                    # Add PK prediction as additional feature for PD model
                    pd_input = torch.cat([x, pk_pred], dim=-1)
                else:
                    # Use original features if already correct size
                    pd_input = x
                
                pd_encoded = self.pd_model['encoder'](pd_input)
                pd_pred = self.pd_model['head']['mean'](pd_encoded)
                
                return pd_pred.cpu()  # Return PD prediction as final biomarker
        else:
            raise ValueError("PK and PD models not loaded")