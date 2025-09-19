"""
Encoder models - Unified structure
All encoders in one place for better organization
"""

import math
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# ResMLP Encoder (Basic)
# =========================

class ResMLPEncoder(BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        time_pool: str = None,  # Changed default to None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden
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
# MoE (Mixture of Experts) - Basic
# =========================

class MoEBlock(nn.Module):
    """Basic Mixture of Experts block with residual connection."""
    
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
    """Basic Mixture of Experts encoder with residual connections."""
    
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
# Advanced ResMLP + MoE Components
# =========================

class ResidualMLPBlock(nn.Module):
    """Advanced Residual MLP Block with LayerNorm and Dropout"""
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP layers
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout for residual connection
        self.dropout_layer = nn.Dropout(dropout)
    
    def _get_activation(self, activation: str):
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection
        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.dropout_layer(x)
        x = x + residual
        
        # Second residual connection
        residual = x
        x = self.norm2(x)
        return x + residual


class AdvancedMoEBlock(nn.Module):
    """Advanced MoE Block with Top-K routing and load balancing"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_capacity_factor: float = 1.25,
        dropout: float = 0.1,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.jitter_noise = jitter_noise
        
        # Router (Gating network)
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
        # Auxiliary loss for load balancing
        self.aux_loss_weight = 0.01
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            output: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            aux_loss: auxiliary loss for load balancing
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1] if x.dim() == 3 else 1
        
        # Flatten for processing
        if x.dim() == 3:
            x_flat = x.view(-1, self.hidden_dim)
        else:
            x_flat = x
        
        # Router logits
        router_logits = self.router(x_flat)  # [batch_size*seq_len, num_experts]
        
        # Add jitter noise for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
        
        # Top-K selection
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Expert capacity
        expert_capacity = int(self.expert_capacity_factor * batch_size * seq_len / self.num_experts)
        
        # Process through experts
        output = torch.zeros_like(x_flat)
        aux_loss = 0.0
        
        for i in range(self.top_k):
            # Handle both 1D and 2D cases
            if top_k_indices.dim() == 1:
                expert_idx = top_k_indices[i:i+1]  # Keep as 1D tensor
                expert_probs = top_k_probs[i:i+1]
            else:
                expert_idx = top_k_indices[:, i]
                expert_probs = top_k_probs[:, i]
            
            # Get expert outputs
            expert_outputs = []
            for j in range(self.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    # Ensure mask and x_flat have compatible shapes
                    if mask.dim() == 0:  # scalar mask
                        mask = mask.unsqueeze(0)
                    if expert_idx.dim() == 1 and expert_idx.size(0) == 1:  # single sample case
                        expert_input = x_flat
                    else:
                        expert_input = x_flat[mask]
                    expert_output = self.experts[j](expert_input)
                    expert_outputs.append((mask, expert_output * expert_probs[mask].unsqueeze(-1)))
            
            # Combine expert outputs
            for mask, expert_output in expert_outputs:
                # Ensure mask and output have compatible shapes
                if mask.dim() == 0:  # scalar mask
                    mask = mask.unsqueeze(0)
                if expert_idx.dim() == 1 and expert_idx.size(0) == 1:  # single sample case
                    # Ensure shapes match for broadcasting
                    if expert_output.dim() > output.dim():
                        expert_output = expert_output.squeeze(0)
                    output += expert_output
                else:
                    output[mask] += expert_output
        
        # Reshape back to original shape
        if x.dim() == 3:
            output = output.view(batch_size, seq_len, self.hidden_dim)
        
        # Calculate auxiliary loss for load balancing
        if self.training:
            # Fraction of tokens routed to each expert
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.top_k):
                # Handle both 1D and 2D cases
                if top_k_indices.dim() == 1:
                    expert_idx = top_k_indices[i:i+1]  # Keep as 1D tensor
                else:
                    expert_idx = top_k_indices[:, i]
                for j in range(self.num_experts):
                    expert_counts[j] += (expert_idx == j).float().sum()
            
            # Load balancing loss
            expert_fraction = expert_counts / (batch_size * seq_len * self.top_k)
            aux_loss = self.aux_loss_weight * (expert_fraction * torch.log(expert_fraction + 1e-8)).sum()
        
        return output, aux_loss


class ResMLPMoEBlock(nn.Module):
    """Combined ResMLP + Advanced MoE Block"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.resmlp = ResidualMLPBlock(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation
        )
        
        self.moe = AdvancedMoEBlock(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ResMLP processing
        x = self.resmlp(x)
        
        # MoE processing
        x, aux_loss = self.moe(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x, aux_loss


class ResMLPMoEEncoder(BaseEncoder):
    """
    Advanced ResMLP + MoE Hybrid Encoder
    Transformer-like stacking: [ResMLP->MoE]->[ResMLP->MoE]->[ResMLP->MoE]
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        time_pool: Optional[str] = None,  # "mean"/"max"/"attn"/None
        use_input_projection: bool = True,
        use_output_projection: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.num_layers = num_layers
        self.time_pool = time_pool
        
        # Input projection
        if use_input_projection:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.input_proj = nn.Identity()
        
        # ResMLP + MoE blocks
        self.blocks = nn.ModuleList([
            ResMLPMoEBlock(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # Time pooling (if needed)
        if time_pool in ["mean", "max", "min", "attn"]:
            if time_pool == "attn":
                self.pooling = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                    nn.Softmax(dim=1)
                )
            else:
                self.pooling = time_pool
        else:
            self.pooling = None
        
        # Output projection
        if use_output_projection:
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.output_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, in_dim] or [batch_size, in_dim]
        Returns:
            output: [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            total_aux_loss: total auxiliary loss from all MoE blocks
        """
        # Input projection
        x = self.input_proj(x)
        
        # Process through ResMLP + MoE blocks
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        
        # Time pooling (if needed)
        if self.pooling is not None and x.dim() == 3:
            if self.pooling == "mean":
                x = x.mean(dim=1)
            elif self.pooling == "max":
                x = x.max(dim=1).values
            elif self.pooling == "min":
                x = x.min(dim=1).values
            elif isinstance(self.pooling, nn.Module):
                # Attention pooling
                weights = self.pooling(x)  # [batch_size, seq_len, 1]
                x = (x * weights).sum(dim=1)  # [batch_size, hidden_dim]
        
        # Output projection
        x = self.output_proj(x)
        
        return x, total_aux_loss


class AdaptiveResMLPMoEEncoder(ResMLPMoEEncoder):
    """
    Adaptive ResMLP + MoE Encoder with dynamic expert selection
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        time_pool: Optional[str] = None,
        adaptive_experts: bool = True,
        expert_growth_rate: float = 1.2
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
            time_pool=time_pool
        )
        
        self.adaptive_experts = adaptive_experts
        self.expert_growth_rate = expert_growth_rate
        
        if adaptive_experts:
            # Expert importance tracking
            self.expert_importance = nn.Parameter(torch.ones(num_experts))
            self.importance_decay = 0.99
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Update expert importance
        if self.adaptive_experts and self.training:
            with torch.no_grad():
                self.expert_importance.data *= self.importance_decay
        
        # Standard forward pass
        output, total_aux_loss = super().forward(x)
        
        # Adaptive expert selection (future enhancement)
        if self.adaptive_experts and self.training:
            # Could implement dynamic expert selection based on importance
            pass
        
        return output, total_aux_loss

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

# =========================
# Factory Functions
# =========================

def create_resmlp_moe_encoder(
    in_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_experts: int = 8,
    top_k: int = 2,
    variant: str = "standard"
) -> ResMLPMoEEncoder:
    """
    Factory function to create ResMLP + MoE encoders
    
    Args:
        in_dim: Input dimension
        hidden_dim: Hidden dimension
        num_layers: Number of ResMLP+MoE blocks
        num_experts: Number of experts in MoE
        top_k: Number of experts to use per token
        variant: "standard" or "adaptive"
    """
    if variant == "adaptive":
        return AdaptiveResMLPMoEEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k
        )
    else:
        return ResMLPMoEEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k
        )