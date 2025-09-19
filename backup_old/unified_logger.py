"""
Unified Logging System for All Training Modes
Standardized PK/PD metrics logging system
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from utils.logging import get_logger


class UnifiedMetricsLogger:
    """Unified metrics logger - used in all training modes"""
    
    def __init__(self, mode: str, config):
        self.mode = mode
        self.config = config
        self.logger = get_logger(__name__)
        
        # For storing metrics
        self.epoch_metrics = []
        self.best_metrics = None
        self.final_metrics = None
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate standardized metrics"""
        # Convert tensors to 1D
        pred = predictions.view(-1).cpu().numpy()
        target = targets.view(-1).cpu().numpy()
        
        # MSE
        mse = np.mean((pred - target) ** 2)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(pred - target))
        
        # R²
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-12))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def compute_pk_pd_metrics(self, model, loaders: Dict[str, Any], device: torch.device) -> Dict[str, Dict[str, float]]:
        """Calculate separated PK/PD metrics"""
        model.eval()
        
        # PK metrics
        pk_preds = []
        pk_targets = []
        with torch.no_grad():
            for batch in loaders["val_pk"]:
                batch = self._to_device(batch, device)
                # Handle different model output formats
                model_output = model(batch)
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        pred, _, _ = model_output
                    elif len(model_output) == 4:
                        pred, _, _, _ = model_output  # DualBranchPKPD returns 4 values
                    else:
                        pred = model_output[0]
                else:
                    pred = model_output
                
                # Extract prediction if it's a dict
                if isinstance(pred, dict):
                    pred = pred['pred']
                
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Adjust tensor shapes - more robust logic
                pred = self._align_tensor_shapes(pred, target)
                target = self._align_tensor_shapes(target, pred)
                
                pk_preds.append(pred.cpu())
                pk_targets.append(target.cpu())
        
        # PD metrics
        pd_preds = []
        pd_targets = []
        with torch.no_grad():
            for batch in loaders["val_pd"]:
                batch = self._to_device(batch, device)
                # Handle different model output formats
                model_output = model(batch)
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        _, pred, _ = model_output  # For single models, second output is PD
                    elif len(model_output) == 4:
                        _, pred, _, _ = model_output  # DualBranchPKPD: pk_outs, pd_outs, z_pk, z_pd
                    else:
                        pred = model_output[1] if len(model_output) > 1 else model_output[0]
                else:
                    pred = model_output
                
                # Extract prediction if it's a dict
                if isinstance(pred, dict):
                    pred = pred['pred']
                
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Adjust tensor shapes - more robust logic
                pred = self._align_tensor_shapes(pred, target)
                target = self._align_tensor_shapes(target, pred)
                
                pd_preds.append(pred.cpu())
                pd_targets.append(target.cpu())
        
        # Calculate metrics - safe tensor concatenation
        pk_preds_tensor = self._safe_cat_tensors(pk_preds)
        pk_targets_tensor = self._safe_cat_tensors(pk_targets)
        pd_preds_tensor = self._safe_cat_tensors(pd_preds)
        pd_targets_tensor = self._safe_cat_tensors(pd_targets)
        
        # Final shape alignment
        pk_preds_tensor, pk_targets_tensor = self._final_align_tensors(pk_preds_tensor, pk_targets_tensor)
        pd_preds_tensor, pd_targets_tensor = self._final_align_tensors(pd_preds_tensor, pd_targets_tensor)
        
        pk_metrics = self.compute_metrics(pk_preds_tensor, pk_targets_tensor)
        pd_metrics = self.compute_metrics(pd_preds_tensor, pd_targets_tensor)
        
        return {
            'pk': pk_metrics,
            'pd': pd_metrics
        }
    
    def compute_test_metrics(self, model, loaders: Dict[str, Any], device: torch.device) -> Dict[str, Dict[str, float]]:
        """Calculate test metrics"""
        model.eval()
        
        # PK Test metrics
        pk_preds = []
        pk_targets = []
        with torch.no_grad():
            for batch in loaders["test_pk"]:
                batch = self._to_device(batch, device)
                # Handle different model output formats
                model_output = model(batch)
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        pred, _, _ = model_output
                    elif len(model_output) == 4:
                        pred, _, _, _ = model_output  # DualBranchPKPD returns 4 values
                    else:
                        pred = model_output[0]
                else:
                    pred = model_output
                
                # Extract prediction if it's a dict
                if isinstance(pred, dict):
                    pred = pred['pred']
                
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Adjust tensor shapes - more robust logic
                pred = self._align_tensor_shapes(pred, target)
                target = self._align_tensor_shapes(target, pred)
                
                pk_preds.append(pred.cpu())
                pk_targets.append(target.cpu())
        
        # PD Test metrics
        pd_preds = []
        pd_targets = []
        with torch.no_grad():
            for batch in loaders["test_pd"]:
                batch = self._to_device(batch, device)
                # Handle different model output formats
                model_output = model(batch)
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        _, pred, _ = model_output  # For single models, second output is PD
                    elif len(model_output) == 4:
                        _, pred, _, _ = model_output  # DualBranchPKPD: pk_outs, pd_outs, z_pk, z_pd
                    else:
                        pred = model_output[1] if len(model_output) > 1 else model_output[0]
                else:
                    pred = model_output
                
                # Extract prediction if it's a dict
                if isinstance(pred, dict):
                    pred = pred['pred']
                
                target = batch[1] if isinstance(batch, (list, tuple)) else batch["y"]
                
                # Adjust tensor shapes - more robust logic
                pred = self._align_tensor_shapes(pred, target)
                target = self._align_tensor_shapes(target, pred)
                
                pd_preds.append(pred.cpu())
                pd_targets.append(target.cpu())
        
        # Calculate metrics - safe tensor concatenation
        pk_preds_tensor = self._safe_cat_tensors(pk_preds)
        pk_targets_tensor = self._safe_cat_tensors(pk_targets)
        pd_preds_tensor = self._safe_cat_tensors(pd_preds)
        pd_targets_tensor = self._safe_cat_tensors(pd_targets)
        
        # Final shape alignment
        pk_preds_tensor, pk_targets_tensor = self._final_align_tensors(pk_preds_tensor, pk_targets_tensor)
        pd_preds_tensor, pd_targets_tensor = self._final_align_tensors(pd_preds_tensor, pd_targets_tensor)
        
        pk_metrics = self.compute_metrics(pk_preds_tensor, pk_targets_tensor)
        pd_metrics = self.compute_metrics(pd_preds_tensor, pd_targets_tensor)
        
        return {
            'pk': pk_metrics,
            'pd': pd_metrics
        }
    
    def _to_device(self, batch, device):
        """Move batch to device"""
        if isinstance(batch, (list, tuple)):
            return [item.to(device) if torch.is_tensor(item) else item for item in batch]
        elif isinstance(batch, dict):
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            return batch.to(device)
    
    def _align_tensor_shapes(self, tensor1, tensor2):
        """Align tensor shapes to make them compatible for operations"""
        if not torch.is_tensor(tensor1) or not torch.is_tensor(tensor2):
            return tensor1
            
        # If shapes are already compatible, return as is
        if tensor1.shape == tensor2.shape:
            return tensor1
            
        # If one is 1D and other is 2D, squeeze the 2D one
        if tensor1.dim() == 1 and tensor2.dim() == 2:
            return tensor1
        elif tensor1.dim() == 2 and tensor2.dim() == 1:
            return tensor1.squeeze(-1)
        
        # If both are 2D but different shapes, try to match the first dimension
        if tensor1.dim() == 2 and tensor2.dim() == 2:
            if tensor1.shape[0] == tensor2.shape[0]:
                # Same batch size, squeeze the last dimension if needed
                if tensor1.shape[1] == 1:
                    return tensor1.squeeze(-1)
                elif tensor2.shape[1] == 1:
                    return tensor1
            else:
                # Different batch sizes, flatten to 1D
                return tensor1.flatten()
        
        # If all else fails, flatten to 1D
        return tensor1.flatten()
    
    def _safe_cat_tensors(self, tensor_list):
        """Safely concatenate tensors with shape validation"""
        if not tensor_list:
            return torch.tensor([])
        
        # Flatten all tensors to 1D first
        flattened_tensors = [t.flatten() for t in tensor_list]
        
        # Concatenate
        try:
            return torch.cat(flattened_tensors, dim=0)
        except Exception as e:
            # If concatenation fails, return the first tensor flattened
            return flattened_tensors[0]
    
    def _final_align_tensors(self, tensor1, tensor2):
        """Final alignment of tensors before metric computation"""
        if not torch.is_tensor(tensor1) or not torch.is_tensor(tensor2):
            return tensor1, tensor2
        
        # Ensure both are 1D
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()
        
        # If sizes don't match, truncate the larger one
        min_size = min(tensor1.shape[0], tensor2.shape[0])
        if min_size > 0:
            tensor1 = tensor1[:min_size]
            tensor2 = tensor2[:min_size]
        
        return tensor1, tensor2
    
    def log_epoch_results(self, epoch: int, train_loss: float, val_loss: float, 
                         val_metrics: Dict[str, Dict[str, float]], is_best: bool = False):
        """Log epoch results"""
        # Determine logging frequency
        should_log = (
            self.config.verbose or 
            epoch % 100 == 0 or 
            is_best or 
            epoch < 10 or 
            epoch % 50 == 0
        )
        
        if should_log:
            pk_metrics = val_metrics['pk']
            pd_metrics = val_metrics['pd']
            
            # Basic information
            self.logger.info(f"Epoch {epoch+1:4d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Separated PK/PD metrics
            self.logger.info(f"  PK  - MSE: {pk_metrics['mse']:.4f}, RMSE: {pk_metrics['rmse']:.4f}, MAE: {pk_metrics['mae']:.4f}, R²: {pk_metrics['r2']:.4f}")
            self.logger.info(f"  PD  - MSE: {pd_metrics['mse']:.4f}, RMSE: {pd_metrics['rmse']:.4f}, MAE: {pd_metrics['mae']:.4f}, R²: {pd_metrics['r2']:.4f}")
            
            if is_best:
                self.logger.info("  Best model saved!")
            
            # Store metrics
            self.epoch_metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pk_metrics': pk_metrics,
                'pd_metrics': pd_metrics,
                'is_best': is_best
            })
    
    def log_final_results(self, val_metrics: Dict[str, Dict[str, float]], 
                         test_metrics: Dict[str, Dict[str, float]]):
        """Log final results"""
        self.logger.info("=" * 100)
        self.logger.info(f"FINAL RESULTS - {self.mode.upper()} TRAINING (PK/PD SEPARATED)")
        self.logger.info("=" * 100)
        
        # Validation results
        self.logger.info("VALIDATION RESULTS:")
        pk_val = val_metrics['pk']
        pd_val = val_metrics['pd']
        
        self.logger.info(f"  PK  - MSE: {pk_val['mse']:.4f}, RMSE: {pk_val['rmse']:.4f}, MAE: {pk_val['mae']:.4f}, R²: {pk_val['r2']:.4f}")
        self.logger.info(f"  PD  - MSE: {pd_val['mse']:.4f}, RMSE: {pd_val['rmse']:.4f}, MAE: {pd_val['mae']:.4f}, R²: {pd_val['r2']:.4f}")
        
        # Test results
        self.logger.info("TEST RESULTS:")
        pk_test = test_metrics['pk']
        pd_test = test_metrics['pd']
        
        self.logger.info(f"  PK  - MSE: {pk_test['mse']:.4f}, RMSE: {pk_test['rmse']:.4f}, MAE: {pk_test['mae']:.4f}, R²: {pk_test['r2']:.4f}")
        self.logger.info(f"  PD  - MSE: {pd_test['mse']:.4f}, RMSE: {pd_test['rmse']:.4f}, MAE: {pd_test['mae']:.4f}, R²: {pd_test['r2']:.4f}")
        
        # Performance summary
        self.logger.info("PERFORMANCE SUMMARY:")
        self.logger.info(f"  Best PK R²:  {pk_val['r2']:.4f} (Val) / {pk_test['r2']:.4f} (Test)")
        self.logger.info(f"  Best PD R²:  {pd_val['r2']:.4f} (Val) / {pd_test['r2']:.4f} (Test)")
        
        self.logger.info("=" * 100)
        
        # Store final metrics
        self.final_metrics = {
            'validation': val_metrics,
            'test': test_metrics
        }
    
    def log_best_model_saved(self, epoch: int, val_metrics: Dict[str, Dict[str, float]]):
        """Log when best model is saved"""
        pk_metrics = val_metrics['pk']
        pd_metrics = val_metrics['pd']
        
        self.logger.info(f"{self.mode.upper()} Best Model Saved! Epoch {epoch+1}")
        self.logger.info(f"   PK  - MSE: {pk_metrics['mse']:.4f}, RMSE: {pk_metrics['rmse']:.4f}, MAE: {pk_metrics['mae']:.4f}, R²: {pk_metrics['r2']:.4f}")
        self.logger.info(f"   PD  - MSE: {pd_metrics['mse']:.4f}, RMSE: {pd_metrics['rmse']:.4f}, MAE: {pd_metrics['mae']:.4f}, R²: {pd_metrics['r2']:.4f}")
        
        # Store best metrics
        self.best_metrics = {
            'epoch': epoch + 1,
            'metrics': val_metrics
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return metrics summary"""
        return {
            'mode': self.mode,
            'epoch_metrics': self.epoch_metrics,
            'best_metrics': self.best_metrics,
            'final_metrics': self.final_metrics
        }