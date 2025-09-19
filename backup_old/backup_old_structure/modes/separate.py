"""
Separate training mode
"""

import torch
from ..trainer import BaseTrainer
from utils.logging import get_logger
from utils.helpers import ReIter, roundrobin_loaders, rr_val, get_device
from utils.helpers import build_encoder, build_head
from models.architectures import EncHeadModel

class BaseSingleTrainer(BaseTrainer):
    """Base trainer for single model (PK or PD) with common loss computation"""
    
    def __init__(self, model, config, loaders, device=None):
        super().__init__(model, config, loaders, device)

    def compute_loss(self, batch):
        """Common loss computation with improved mixup (always learn original + optional mixup)"""
        
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        
        # Always learn from original data
        pred_orig, z_orig, _ = self.model(x)
        
        # Ensure tensor dimensions match using helper function
        pred_orig, target_orig = self._ensure_tensor_compatibility(pred_orig, target)
        
        # Original loss (always computed)
        loss_orig = torch.nn.functional.mse_loss(pred_orig, target_orig)
        
        # Add contrastive loss for original
        if self.config.lambda_contrast > 0 and z_orig is not None:
            contrast_loss_orig = self.contrastive_loss(z_orig, self.config.temperature)
            loss_orig = loss_orig + self.config.lambda_contrast * contrast_loss_orig
        
        # Apply mixup as additional regularization (if enabled)
        if self.config.use_mixup and torch.rand(1).item() < self.config.mixup_prob:
            mixed_x, y_a, y_b, lam = self.apply_mixup(x, target, self.config.mixup_alpha)
            pred_mix, z_mix, _ = self.model(mixed_x)
            
            # Ensure tensor dimensions match using helper function
            pred_mix, y_a = self._ensure_tensor_compatibility(pred_mix, y_a)
            pred_mix, y_b = self._ensure_tensor_compatibility(pred_mix, y_b)
            
            loss_mix = lam * torch.nn.functional.mse_loss(pred_mix, y_a) + (1 - lam) * torch.nn.functional.mse_loss(pred_mix, y_b)
            
            # Add contrastive loss for mixup
            if self.config.lambda_contrast > 0 and z_mix is not None:
                contrast_loss_mix = self.contrastive_loss(z_mix, self.config.temperature)
                loss_mix = loss_mix + self.config.lambda_contrast * contrast_loss_mix
            
            # Combine original and mixup losses (weighted)
            original_weight = getattr(self.config, 'original_weight', 0.7)
            mixup_weight = getattr(self.config, 'mixup_weight', 0.3)
            total_loss = original_weight * loss_orig + mixup_weight * loss_mix
        else:
            # Only original loss
            total_loss = loss_orig
        
        return total_loss


class PKTrainer(BaseSingleTrainer):
    """PK-specific trainer"""
    pass


class PDTrainer(BaseSingleTrainer):
    """PD-specific trainer"""
    pass


class SeparateTrainer(BaseTrainer):
    """Separate training: PK first, then PD (with PK predictions)"""
    
    def __init__(self, model, config, data_loaders, device=None):
        # Follow consistent pattern with other trainers
        if model is not None and model != "separate_placeholder":
            super().__init__(model, config, data_loaders, device)
        else:
            # Initialize without model (separate mode creates models internally)
            self.config = config
            self.data_loaders = data_loaders
            self.logger = get_logger(__name__)
            self.device = device if device is not None else get_device()
        
        # Result saving directory
        from pathlib import Path
        run_name = config.run_name if config.run_name else "default_run"
        # Create hierarchical model save directory: models/mode/encoder/s{seed}/
        self.model_save_directory = Path(config.output_dir) / "models" / config.mode / config.encoder_pk + "_" + config.encoder_pd / f"s{config.random_state}"
        self.model_save_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize scheduler (will be set up properly when we have models)
        self.learning_rate_scheduler = None
        
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
    
    def _train_single_model(self, model_type, loaders_key):
        """Common single model training logic"""
        self.logger.info(f"Starting {model_type} model creation...")
        
        # Validate data loaders
        training_loader_key = f"train_{model_type.lower()}"
        validation_loader_key = f"val_{model_type.lower()}"
        
        if training_loader_key not in self.data_loaders or validation_loader_key not in self.data_loaders:
            raise ValueError(f"{model_type} data loaders not found in data_loaders dictionary")
        self.logger.info(f"{model_type} data loaders validated")
        
        # Create model
        input_dimension = self.data_loaders[training_loader_key].dataset.tensors[0].shape[1]
        print(f"input_dimension: {input_dimension}")
        if input_dimension <= 0:
            raise ValueError(f"Invalid {model_type} input dimension: {input_dimension}")
        
        print(f"input_dimension: {input_dimension}")
        self.logger.info(f"{model_type} input dimension: {input_dimension}")
        
        # Use specific encoder for PK/PD if provided
        if model_type == "PK" and self.config.encoder_pk is not None:
            encoder_type = self.config.encoder_pk
        elif model_type == "PD" and self.config.encoder_pd is not None:
            encoder_type = self.config.encoder_pd
        else:
            encoder_type = self.config.encoder
        
        # Validate encoder type
        valid_encoders = ["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"]
        if encoder_type not in valid_encoders:
            raise ValueError(f"Invalid {model_type} encoder: {encoder_type}. Valid options: {valid_encoders}")
        
        encoder = build_encoder(encoder_type, input_dimension, self.config)
        head = build_head("mse", self.config.hidden)
        model = EncHeadModel(encoder, head)
        
        self.logger.info(f"{model_type} model: {encoder_type} encoder (input_dim={input_dimension}, output_dim={encoder.out_dim})")
        
        # Store model as attribute for saving
        setattr(self, f"{model_type.lower()}_model", model)
        
        self.logger.info(f"{model_type} model created")
        
        # Create trainer
        trainer_class = PKTrainer if model_type == "PK" else PDTrainer
        trainer = trainer_class(model, self.config, self.data_loaders, device=self.device)
        
        self.logger.info(f"{model_type} trainer created")
        
        # Train model
        self.logger.info(f"Starting {model_type} training for {self.config.epochs} epochs...")
        
        best_val_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            training_loss = trainer.train_epoch(self.data_loaders[training_loader_key])
            validation_loss, validation_metrics = self._validate_with_metrics(model, self.data_loaders[validation_loader_key])
            
            if hasattr(trainer, 'learning_rate_scheduler') and trainer.learning_rate_scheduler is not None:
                trainer.learning_rate_scheduler.step(validation_loss)
            
            is_best_model = False
            # Use RMSE for best model selection (more intuitive than MSE)
            if validation_metrics['rmse'] < best_val_rmse:
                best_val_rmse = validation_metrics['rmse']
                patience_counter = 0
                is_best_model = True
                best_epoch = epoch+1
                trainer.save_model(f"best_{model_type.lower()}_model.pth", validation_metrics)
                # Log detailed metrics when best model is saved
                self.logger.info(f"[Epoch {epoch+1:3d}/{self.config.epochs}] {model_type} - NEW BEST MODEL! [BEST]")
                self.logger.info(f"  Train Loss: {training_loss:.4f} | Valid Loss: {validation_loss:.4f}")
                self.logger.info(f"  RMSE: {validation_metrics['rmse']:.4f} | R²: {validation_metrics['r2']:.4f}")
                self.logger.info(f"  Full Metrics - MSE: {validation_metrics['mse']:.4f}, MAE: {validation_metrics['mae']:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"[Epoch {epoch+1:3d}/{self.config.epochs}] {model_type} - Progress Update")
                self.logger.info(f"  Train Loss: {training_loss:.4f} | Valid Loss: {validation_loss:.4f}")
                self.logger.info(f"  RMSE: {validation_metrics['rmse']:.4f} | R²: {validation_metrics['r2']:.4f}")
                self.logger.info(f"  Best RMSE: {best_val_rmse:.4f} | Patience: {patience_counter}/{self.config.patience}")

            if patience_counter >= self.config.patience:
                self.logger.info(f"{model_type} training early stopping at epoch {epoch+1}")
                break
        
        self.logger.info(f"{model_type} training completed")
        return {"best_val_rmse": best_val_rmse, "val_metrics": validation_metrics, "epochs": epoch+1}
    
    def _train_pk(self):
        """Train PK model"""
        return self._train_single_model("PK", "train_pk")
    
    def _compute_metrics_for_loader(self, model, loader):
        """Compute metrics for any loader (train/val/test)"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                pred, _, _ = model(batch)
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Ensure tensor dimensions match using helper function
                pred, target = self._ensure_tensor_compatibility(pred, target)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return self.compute_metrics(all_preds, all_targets)
    
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
                
                # Ensure tensor dimensions match using helper function
                pred, target = self._ensure_tensor_compatibility(pred, target)
                
                loss = torch.nn.functional.mse_loss(pred, target)
                total_loss += loss.item()
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(val_loader), metrics
    
    def _train_pd(self):
        """Train PD model"""
        return self._train_single_model("PD", "train_pd")
    
    def _compute_test_metrics(self, test_loader, model=None):
        """Calculate test metrics with optional model parameter"""
        if model is None:
            # Create temporary model if none provided
            input_dim = test_loader.dataset.tensors[0].shape[1]
            encoder = build_encoder(self.config.encoder, input_dim, self.config)
            head = build_head("mse", self.config.hidden)
            model = EncHeadModel(encoder, head)
            model.to(self.device)
        
        return self._compute_metrics_for_loader(model, test_loader)
    
    def compute_loss(self, batch):
        """Calculate loss"""
        # basic MSE loss
        pred, _, _ = self.model(batch)
        target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
        return torch.nn.functional.mse_loss(pred, target)
    
    def save_model(self, filename="final_model.pth", metrics=None):
        """save both PK and PD models"""
        # save PK model
        pk_save_path = self.save_dir / f"pk_{filename}"
        if hasattr(self, 'pk_model') and self.pk_model is not None:
            torch.save({
                'model_state_dict': self.pk_model.state_dict(),
                'model_type': 'pk',
                'config': self.config,
                'metrics': metrics
            }, pk_save_path)
            self.logger.info(f"PK model saved: {pk_save_path}")
        
        # save PD model
        pd_save_path = self.save_dir / f"pd_{filename}"
        if hasattr(self, 'pd_model') and self.pd_model is not None:
            torch.save({
                'model_state_dict': self.pd_model.state_dict(),
                'model_type': 'pd',
                'config': self.config,
                'metrics': metrics
            }, pd_save_path)
            self.logger.info(f"PD model saved: {pd_save_path}")
        
        # also save a combined checkpoint for convenience
        from models.architectures import EncHeadModel
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
            self.logger.info(f"Model saved: R²={metrics['r2']:.4f}")
        else:
            self.logger.info("Model saved")
    
    def predict(self, x):
        """Make prediction with separate PK/PD models"""
        # for separate mode, we need both PK and PD predictions
        if hasattr(self, 'pk_model') and hasattr(self, 'pd_model'):
            self.pk_model.eval()
            self.pd_model.eval()
            x = x.to(self.device)
            
            with torch.no_grad():
                pk_encoded = self.pk_model['encoder'](x)
                pk_pred = self.pk_model['head']['mean'](pk_encoded)
                
                if x.shape[1] == 11:  # PK model input size
                    pd_input = torch.cat([x, pk_pred], dim=-1)
                else:
                    # use original features if already correct size
                    pd_input = x
                
                pd_encoded = self.pd_model['encoder'](pd_input)
                pd_pred = self.pd_model['head']['mean'](pd_encoded)
                
                return pd_pred.cpu()  # return PD prediction as final biomarker
        else:
            raise ValueError("PK and PD models not loaded")