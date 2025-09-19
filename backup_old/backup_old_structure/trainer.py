"""
Base trainer class
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from utils.logging import get_logger
from utils.helpers import get_device


class BaseTrainer:
    """Base trainer class for model training and evaluation"""
    
    def __init__(self, model, config, data_loaders, device=None):
        self.model = model
        self.config = config
        self.data_loaders = data_loaders
        self.logger = get_logger(__name__)
        self.device = device if device is not None else get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.patience//2, factor=0.5
        )

        self.model_save_directory = Path(config.output_dir) / "models" / config.mode / config.encoder / f"s{config.random_state}"
        self.model_save_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Trainer initialization completed - device: {self.device}")
        self.logger.info(f"Model parameter count: {self._count_model_parameters()}")
    
    def _count_model_parameters(self):
        """Calculate total number of trainable model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, training_data_loader):
        """Train model for one epoch"""
        self.model.train()
        total_epoch_loss = 0.0
        number_of_batches = 0
        
        # Use mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        for batch in training_data_loader:  # Training loop
            self.optimizer.zero_grad()
            
            batch = self._to_device(batch)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = self.compute_loss(batch)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            number_of_batches += 1
        
        return total_epoch_loss / number_of_batches
    
    def validate_epoch(self, validation_data_loader):
        """Validate model for one epoch"""
        self.model.eval()
        total_validation_loss = 0.0
        number_of_validation_batches = 0
        
        with torch.no_grad():
            for batch in validation_data_loader:
                batch = self._to_device(batch)
                batch_loss = self.compute_loss(batch)
                total_validation_loss += batch_loss.item()
                number_of_validation_batches += 1
        
        return total_validation_loss / number_of_validation_batches
    
    def compute_loss(self, batch):
        """Loss computation - to be implemented by subclasses"""
        raise NotImplementedError("Must be implemented by subclasses")
    
    def apply_mixup(self, x, y, alpha=0.2):
        """Apply Mixup augmentation with proper alpha handling"""
        try:
            if alpha <= 0:
                return x, y, y, 1.0
            
            if x.size(0) < 2:
                return x, y, y, 1.0
            
            lam = torch.distributions.Beta(alpha, alpha).sample()
            
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
            
        except Exception as e:
            self.logger.warning(f"Mixup failed: {e}, using original data")
            return x, y, y, 1.0
    
    def apply_dual_mixup(self, x, y_pk, y_pd, alpha=0.2):
        """Apply Mixup augmentation for dual targets (PK and PD) with consistent lambda"""
        try:
            if alpha <= 0:
                return x, y_pk, y_pk, y_pd, y_pd, 1.0
            
            if x.size(0) < 2:
                return x, y_pk, y_pk, y_pd, y_pd, 1.0
            
            lam = torch.distributions.Beta(alpha, alpha).sample()
            
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a_pk, y_b_pk = y_pk, y_pk[index]
            y_a_pd, y_b_pd = y_pd, y_pd[index]
            return mixed_x, y_a_pk, y_b_pk, y_a_pd, y_b_pd, lam
            
        except Exception as e:
            self.logger.warning(f"Dual mixup failed: {e}, using original data")
            return x, y_pk, y_pk, y_pd, y_pd, 1.0
    
    def _ensure_tensor_compatibility(self, pred, target):
        """Ensure prediction and target tensors have compatible dimensions"""
        original_pred_shape = pred.shape
        original_target_shape = target.shape
        
        if pred.shape == target.shape:
            return pred, target
        
        # Handle scalar predictions/targets
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        # Handle different dimensions
        if pred.dim() == 1 and target.dim() == 2:
            target = target.squeeze(-1)
        elif pred.dim() == 2 and target.dim() == 1:
            pred = pred.squeeze(-1)
        
        # If shapes still don't match, try to reshape
        if pred.shape != target.shape:
            if pred.numel() == target.numel():
                # Same number of elements, reshape to match target
                pred = pred.view_as(target)
            else:
                # Different number of elements, use the smaller size
                min_size = min(pred.numel(), target.numel())
                pred = pred.view(-1)[:min_size]
                target = target.view(-1)[:min_size]
        
        # Log dimension changes for debugging
        if original_pred_shape != pred.shape or original_target_shape != target.shape:
            self.logger.debug(f"Tensor dimension adjusted: pred {original_pred_shape} -> {pred.shape}, target {original_target_shape} -> {target.shape}")
        
        return pred, target
    
    def contrastive_loss(self, features, temperature=0.2):
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
    
    def compute_metrics(self, pred, target):
        """Calculate metrics (MSE, RMSE, MAE, R2)"""
        # Adjust tensor dimensions
        if pred.dim() == 1 and target.dim() == 2:
            target = target.squeeze(-1)
        elif pred.dim() == 2 and target.dim() == 1:
            pred = pred.squeeze(-1)
        
        # MSE
        mse = F.mse_loss(pred, target).item()
        
        # RMSE
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # MAE
        mae = F.l1_loss(pred, target).item()
        
        # R2
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def log_epoch_results(self, epoch, train_loss, val_loss, val_metrics=None, is_best=False):
        """Log epoch results with clean formatting"""
        # Only log when best model is saved or every 50 epochs
        if is_best:
            if val_metrics:
                self.logger.info(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, R²={val_metrics['r2']:.4f} [BEST]")
            else:
                self.logger.info(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f} [BEST]")
        elif epoch % 50 == 0:
            if val_metrics:
                self.logger.info(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, R²={val_metrics['r2']:.4f}")
            else:
                self.logger.info(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    def log_final_results(self, val_metrics, test_metrics):
        """Log final validation and test results"""
        self.logger.info("=" * 60)
        self.logger.info("FINAL RESULTS")
        self.logger.info("=" * 60)
        
        if val_metrics:
            self.logger.info(f"Validation - MSE: {val_metrics['mse']:.4f}, RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        if test_metrics:
            self.logger.info(f"Test       - MSE: {test_metrics['mse']:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        self.logger.info("=" * 60)
    
    def _to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, (list, tuple)):
            return [item.to(self.device) if torch.is_tensor(item) else item for item in batch]
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            return batch.to(self.device)
    
    def save_model(self, model_filename="best_model.pth", validation_metrics=None):
        """Save trained model with state and metrics"""
        model_save_path = self.model_save_directory / model_filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': validation_metrics
        }, model_save_path)
        
        # Log save message with metrics if available
        if validation_metrics:
            self.logger.info(f"Model saved: R²={validation_metrics['r2']:.4f}")
        else:
            self.logger.info("Model saved")
    
    def load_model(self, model_filename="best_model.pth"):
        """Load trained model from file"""
        model_load_path = self.model_save_directory / model_filename
        if model_load_path.exists():
            model_checkpoint = torch.load(model_load_path, map_location=self.device)
            self.model.load_state_dict(model_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
            self.logger.info(f"Model loaded: {model_load_path}")
        else:
            self.logger.warning(f"Model file does not exist: {model_load_path}")
    
    def predict(self, x):
        """Make prediction with the model"""
        self.model.eval()
        x = x.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, '__call__'):
                # Handle different model architectures
                if hasattr(self.model, 'forward'):
                    pred = self.model(x)
                else:
                    pred = self.model(x)
                
                # Extract prediction from tuple if needed
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]  # Take the first element (prediction)
                
                return pred.cpu()
            else:
                raise ValueError("Model is not callable")