"""
Base trainer class
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from utils.logging import get_logger
from utils.helpers import get_device


class BaseTrainer:
    """Base trainer class"""
    
    def __init__(self, model, config, loaders, device=None):
        self.model = model
        self.config = config
        self.loaders = loaders
        self.logger = get_logger(__name__)
        self.device = device if device is not None else get_device()
        
        # Move model to device
        self.model.to(self.device)
        
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
        # Create hierarchical model save directory: models/mode/encoder/s{seed}/
        self.save_dir = Path(config.output_dir) / "models" / config.mode / config.encoder / f"s{config.random_state}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Trainer initialization completed - device: {self.device}")
        self.logger.info(f"Model parameter count: {self._count_parameters()}")
    
    def _count_parameters(self):
        """Calculate model parameter count"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Use mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = self._to_device(batch)
            
            # Forward pass with mixed precision
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
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def compute_loss(self, batch):
        """Loss computation - to be implemented by subclasses"""
        raise NotImplementedError("Must be implemented by subclasses")
    
    def apply_mixup(self, x, y, alpha=0.2):
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
        if self.config.verbose:
            if val_metrics:
                self.logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Metrics - MSE: {val_metrics['mse']:.4f}, RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
            else:
                self.logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}")
        else:
            # Non-verbose: only log every 100 epochs or best models
            if epoch % 100 == 0 or is_best:
                if val_metrics:
                    self.logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Metrics - MSE: {val_metrics['mse']:.4f}, RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
                else:
                    self.logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}")
    
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
    
    def save_model(self, filename="best_model.pth", metrics=None):
        """Save model"""
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics
        }, save_path)
        
        # Simple save message with metrics
        if metrics:
            self.logger.info(f"Saved! MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        else:
            self.logger.info("Saved!")
    
    def load_model(self, filename="best_model.pth"):
        """Load model"""
        load_path = self.save_dir / filename
        if load_path.exists():
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"Model loaded: {load_path}")
        else:
            self.logger.warning(f"Model file does not exist: {load_path}")
    
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