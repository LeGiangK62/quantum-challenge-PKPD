"""
Helper functions for PK/PD modeling
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# Additional utility functions
import os
from pathlib import Path

def build_encoder(encoder_type, input_dim, config):
    """Build encoder"""
    if encoder_type == "mlp":
        from models.encoders import MLPEncoder
        return MLPEncoder(input_dim, config.hidden, config.depth, config.dropout)
    
    elif encoder_type == "resmlp":
        from models.encoders import ResMLPEncoder
        return ResMLPEncoder(input_dim, config.hidden, config.depth, config.dropout)
    
    elif encoder_type == "moe":
        from models.encoders import MoEEncoder
        return MoEEncoder(
            input_dim=input_dim,
            hidden_dim=config.hidden,
            num_layers=config.depth,
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            dropout=config.dropout
        )
    
    elif encoder_type == "resmlp_moe":
        from models.resmlp_moe_encoder import create_resmlp_moe_encoder
        return create_resmlp_moe_encoder(
            in_dim=input_dim,
            hidden_dim=config.hidden,
            num_layers=config.depth,
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            variant="standard"
        )
    
    elif encoder_type == "adaptive_resmlp_moe":
        from models.resmlp_moe_encoder import create_resmlp_moe_encoder
        return create_resmlp_moe_encoder(
            in_dim=input_dim,
            hidden_dim=config.hidden,
            num_layers=config.depth,
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            variant="adaptive"
        )
    
 - fallback to MLP
    
    else:
        # Fallback to MLP for unknown encoders
        from models.encoders import MLPEncoder
        return MLPEncoder(input_dim, config.hidden, config.depth, config.dropout)

def build_head(head_type, hidden_dim, for_branch, config):
    """Build head"""
    from models.heads import MSEHead
    return MSEHead(hidden_dim)

def scaling_and_prepare_loader(data, features, batch_size, lambda_ctr, target_col, num_workers, pin_memory, drop_last_train):
    """Prepare data loader"""
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    
    # Extract features and target
    if isinstance(data, dict):
        # Handle train/val/test splits
        train_data = data['train']
        val_data = data['val'] 
        test_data = data['test']
    else:
        # Single dataset - split it
        train_data = data
        val_data = data
        test_data = data
    
    # Prepare training data
    X_train = train_data[features].values.astype(np.float32)
    y_train = train_data[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Fit scaler on training data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled), 
        torch.tensor(y_train_scaled)
    )
    
    # Prepare validation data
    X_val = val_data[features].values.astype(np.float32)
    y_val = val_data[target_col].values.astype(np.float32).reshape(-1, 1)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled),
        torch.tensor(y_val_scaled)
    )
    
    # Prepare test data
    X_test = test_data[features].values.astype(np.float32)
    y_test = test_data[target_col].values.astype(np.float32).reshape(-1, 1)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled),
        torch.tensor(y_test_scaled)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=drop_last_train
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return scaler_X, train_loader, val_loader, test_loader

class ReIter:
    """Re-iterable wrapper for data loaders"""
    def __init__(self, func, *loaders):
        self.func = func
        self.loaders = loaders
    
    def __iter__(self):
        return self.func(*self.loaders)

def roundrobin_loaders(*loaders):
    """Round-robin through multiple loaders"""
    from itertools import cycle
    # Create iterators for each loader
    iterators = [iter(loader) for loader in loaders]
    # Cycle through the iterators
    for iterator in cycle(iterators):
        try:
            yield next(iterator)
        except StopIteration:
            # If one loader is exhausted, remove it from the cycle
            iterators = [it for it in iterators if it is not iterator]
            if not iterators:
                break

def rr_val(*loaders):
    """Round-robin validation"""
    return roundrobin_loaders(*loaders)

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_device(device_id=0):
    """Return available device with specified device ID"""
    import torch
    if torch.cuda.is_available():
        if device_id >= torch.cuda.device_count():
            print(f"Warning: Device {device_id} not available. Using device 0 instead.")
            device_id = 0
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """Calculate number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)