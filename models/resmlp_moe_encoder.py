"""
ResMLP + MoE Hybrid Encoder
Transformer-like stacking: [ResMLP->MoE]->[ResMLP->MoE]->[ResMLP->MoE]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import math

class ResidualMLPBlock(nn.Module):
    """Residual MLP Block with LayerNorm and Dropout"""
    
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


class MoEBlock(nn.Module):
    """Mixture of Experts Block with Top-K routing"""
    
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
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
            
            # Get expert outputs
            expert_outputs = []
            for j in range(self.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[j](expert_input)
                    expert_outputs.append((mask, expert_output * expert_probs[mask].unsqueeze(-1)))
            
            # Combine expert outputs
            for mask, expert_output in expert_outputs:
                output[mask] += expert_output
        
        # Reshape back to original shape
        if x.dim() == 3:
            output = output.view(batch_size, seq_len, self.hidden_dim)
        
        # Calculate auxiliary loss for load balancing
        if self.training:
            # Fraction of tokens routed to each expert
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                for j in range(self.num_experts):
                    expert_counts[j] += (expert_idx == j).float().sum()
            
            # Load balancing loss
            expert_fraction = expert_counts / (batch_size * seq_len * self.top_k)
            aux_loss = self.aux_loss_weight * (expert_fraction * torch.log(expert_fraction + 1e-8)).sum()
        
        return output, aux_loss


class ResMLPMoEBlock(nn.Module):
    """Combined ResMLP + MoE Block"""
    
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
        
        self.moe = MoEBlock(
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


class ResMLPMoEEncoder(nn.Module):
    """
    ResMLP + MoE Hybrid Encoder
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
        self.out_dim = hidden_dim  # Add out_dim attribute
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


# Factory function
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
