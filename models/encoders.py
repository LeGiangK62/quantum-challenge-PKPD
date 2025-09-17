"""
Encoder models - Unified structure
"""

import math
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# Import ResMLP + MoE encoder
from .resmlp_moe_encoder import (
    ResMLPMoEEncoder,
    AdaptiveResMLPMoEEncoder,
    create_resmlp_moe_encoder
)

# =========================
# Pooling Layer
# =========================

class Pooling(nn.Module):
    def __init__(self, in_dim: int, hidden: int, mode: str = "attn"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in ["mean", "max", "min", "attn"]:
            raise ValueError(f"Unknown pooling mode: {self.mode}. Choose from 'mean', 'max', 'min', 'attn'.")

        if self.mode == "attn":
            self.proj = nn.Linear(in_dim, hidden)
            self.score = nn.Linear(hidden, 1)
            self.norm = nn.LayerNorm(hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return x

        if self.mode == "mean":
            return x.mean(dim=1)
        
        elif self.mode == "max":
            return x.max(dim=1).values
        
        elif self.mode == "min":
            return x.min(dim=1).values

        elif self.mode == "attn":
            # [B, T, F] -> [B, T, H]
            h = self.proj(x)
            
            # [B, T, H] -> [B, T, 1] -> [B, T]
            a = self.score(torch.tanh(h)).squeeze(-1)
            
            # [B, T] -> [B, T, 1]
            w = torch.softmax(a, dim=1).unsqueeze(-1)
            
            # [B, T, H] * [B, T, 1] -> [B, T, H] -> [B, H]
            z = (h * w).sum(dim=1)
            
            # [B, H]
            return self.norm(z)

# =========================
# Base Encoder
# =========================

class BaseEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# =========================
# MLP Encoder
# =========================

class MLPEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        time_pool: str = None,  # "mean"/"max"/"attn"/None
        use_input_ln: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.time_pool = time_pool

        if time_pool in ["mean", "max", "min", "attn"]:
            self.pooling = Pooling(in_dim, hidden, mode=time_pool)
            d = hidden
        else:
            self.pooling = None
            d = in_dim

        layers = []
        if use_input_ln:
            layers.append(nn.LayerNorm(d))
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = self.pooling(x) # [B, N, F] -> [B, F]
        return self.net(x) # [B, F] -> [B, H]

# =========================
# ResNet Block
# =========================

class ResBlock(nn.Module):
    def __init__(
            self, d: int, 
            dropout: float = 0.0
        ):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.ln(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + self.drop(h)

# =========================
# ResMLP Encoder
# =========================

class ResMLPEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        time_pool: str = "mean",
    ):
        super().__init__()
        self.pooling = Pooling(in_dim, hidden, mode=time_pool) if time_pool in ["mean", "max", "min", "attn"] else None

        stem_in = hidden if self.pooling is not None else in_dim
        
        self.stem = nn.Sequential(nn.Linear(stem_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(
            *[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)]
        )
        self.final_ln = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and self.pooling is not None:
            x = self.pooling(x)
        elif x.dim() == 2 and self.pooling is not None:
            # For 2D input with pooling, skip pooling and use direct projection
            # Create a projection layer to match hidden dimension
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(x.size(-1), self.hidden_dim).to(x.device)
            x = self.input_proj(x)
            # Skip stem since we already projected to hidden dimension
            x = self.blocks(x)
            return self.final_ln(x)

        x = self.stem(x)
        x = self.blocks(x)
        return self.final_ln(x)

# =========================
# MoE (Mixture of Experts)
# =========================

class MoEBlock(nn.Module):
    """Mixture of Experts block with residual connection."""
    
    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(in_dim, num_experts)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(in_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [B, num_experts]
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)  # Renormalize top-k scores
        
        # Apply experts
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))
        
        # Combine expert outputs
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            # Handle both 1D and 2D cases
            if top_k_indices.dim() == 1:
                expert_idx = top_k_indices[i:i+1]  # Keep as 1D tensor
                expert_score = top_k_scores[i:i+1]  # Keep as 1D tensor
            else:
                expert_idx = top_k_indices[:, i]  # [B]
                expert_score = top_k_scores[:, i:i+1]  # [B, 1]
            
            # Gather expert outputs
            if expert_idx.dim() == 1 and expert_idx.size(0) == 1:  # single sample case
                expert_output = expert_outputs[expert_idx[0].item()]
            else:
                expert_output = torch.stack([expert_outputs[j][b] for b, j in enumerate(expert_idx)])
            output += expert_score * expert_output
        
        # Residual connection
        return self.layer_norm(x + output)


class MoEEncoder(BaseEncoder):
    """Mixture of Experts encoder with residual connections."""
    
    def __init__(self, in_dim: int, hidden_dims: List[int], num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = hidden_dims[-1]
        
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(MoEBlock(prev_dim, hidden_dim, num_experts, top_k, dropout, activation))
            # MoEBlock maintains the same dimension, so prev_dim stays the same
            # Only change dimension if we want to project
            if hidden_dim != prev_dim:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =========================
# Dual Stage Encoder
# =========================

class SimpleMLPEncoder(BaseEncoder):
    """Simple MLP encoder for dual-stage model."""
    
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = hidden_dims[-1]
        
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualStageEncoder(BaseEncoder):
    """Dual-stage encoder: front encoder -> z -> back encoder -> z2."""
    
    def __init__(self, in_dim: int, front_dims: List[int], back_dims: List[int], 
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.in_dim = in_dim
        self.front_dims = front_dims
        self.back_dims = back_dims
        self.out_dim = back_dims[-1]
        
        # Front encoder (for PK)
        self.front_encoder = SimpleMLPEncoder(in_dim, front_dims, dropout)
        
        # Back encoder (for PD)
        self.back_encoder = SimpleMLPEncoder(front_dims[-1], back_dims, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Front encoder
        z = self.front_encoder(x)  # [B, front_dims[-1]]
        
        # Back encoder
        z2 = self.back_encoder(z)  # [B, back_dims[-1]]
        
        return z2
    
    def forward_front(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through front encoder only (for PK)."""
        return self.front_encoder(x)
    
    def forward_back(self, z: torch.Tensor) -> torch.Tensor:
        """Forward through back encoder only (for PD)."""
        return self.back_encoder(z)
