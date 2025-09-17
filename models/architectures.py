"""
Complete architecture models - Unified structure
"""

import math
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import BaseEncoder
from .heads import BaseHead

# =========================
# Utils
# =========================

def _maybe_time_pool(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    x: [B,F] or [B,T,F]
    mode: "mean" | "max"
    """
    if x.dim() == 3:
        if mode == "mean":
            return x.mean(dim=1)
        elif mode == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
    return x

# =========================
# Encoder + Head (single branch)
# =========================

class EncHeadModel(nn.Module):
    """
    Generic wrapper for an encoder + head.
    forward(batch) -> (pred, z, outs)
    - batch can be dict with 'x' (and optional extra keys) or a tuple (x, y, ...).
    - extra_keys (if provided) are pooled/reshaped and concatenated to x.
    """
    def __init__(
        self,
        encoder: BaseEncoder,
        head: BaseHead,
        extra_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.extra_keys = tuple(extra_keys or [])

    def _get_x(self, batch: Union[Dict[str, Any], List, Tuple]) -> torch.Tensor:
        # Normalize to a feature tensor; pool time axis if needed.
        if isinstance(batch, dict):
            x = batch["x"]
            # concatenate selected extra features, if any
            if self.extra_keys:
                extras = []
                for k in self.extra_keys:
                    v = batch.get(k, None)
                    if torch.is_tensor(v):
                        v = _maybe_time_pool(v) if v.dim() == 3 else v
                        if v.dim() == 1:
                            v = v.unsqueeze(-1)  # [B] -> [B,1]
                        extras.append(v.float())
                if extras:
                    x = _maybe_time_pool(x)
                    x = torch.cat([x.float()] + extras, dim=-1)
            # encode expects [B,F]; if [B,T,F], pool to [B,F]
            x = _maybe_time_pool(x)
            return x.float()
        # tuple/list: assume (x, y, ...)
        x = batch[0]
        x = _maybe_time_pool(x)
        return x.float()

    def forward(
        self, batch: Union[Dict[str, Any], Tuple, List]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = self._get_x(batch)
        z = self.encoder(x)                              # [B, D]
        # Handle tuple output from ResMLP+MoE encoder
        if isinstance(z, tuple):
            z, aux_loss = z
            # Store aux_loss for later use if needed
            if not hasattr(self, '_aux_losses'):
                self._aux_losses = []
            self._aux_losses.append(aux_loss)
        outs = self.head(z, batch if isinstance(batch, dict) else {})
        pred = outs["pred"]
        return pred, z, outs

# =========================
# Dual Branch PK/PD Model
# =========================

class DualBranchPKPD(nn.Module):
    """
    Two encoders + two heads for PK and PD branches.
    PD forward ALWAYS uses PK prediction as a feature.
    """
    def __init__(self, enc_pk, enc_pd, head_pk, head_pd,
                 proj_pk=None, proj_pd=None,
                 *,                    # --- NEW: Force connection option ---
                 concat_mode: str = "input",  # "input" | "latent"
                 pk_input_key: str = "x_pk_for_pd",
                 pk_detach: bool = True):
        super().__init__()
        self.enc_pk, self.enc_pd = enc_pk, enc_pd
        self.head_pk, self.head_pd = head_pk, head_pd
        self.proj_pk, self.proj_pd = proj_pk, proj_pd

        assert concat_mode in ("input", "latent")
        self.concat_mode = concat_mode
        self.pk_input_key = pk_input_key
        self.pk_detach = bool(pk_detach)

    def forward_pk(self, batch: Dict[str, Any]):
        # Handle both dict and tuple/list batch formats
        if isinstance(batch, dict):
            x = _maybe_time_pool(batch["x"]).float()
        else:
            # Assume batch is tuple/list format: (X, y, ...)
            x = _maybe_time_pool(batch[0]).float()
        
        # If input has more than 11 features, use only the first 11 for PK
        if x.shape[1] > 11:
            x = x[:, :11]
        
        z1 = self.enc_pk(x)                 # [B, D1]
        # Handle tuple output from ResMLP+MoE encoder
        if isinstance(z1, tuple):
            z1, aux_loss = z1
            # Store aux_loss for later use if needed
            if not hasattr(self, '_aux_losses'):
                self._aux_losses = []
            self._aux_losses.append(aux_loss)
        outs = self.head_pk(z1, batch)      # must include 'pred'
        return outs, z1, outs  # Return 3 values for compatibility

    # --- NEW: small util ---
    @staticmethod
    def _concat_feature(X: torch.Tensor, pkv) -> torch.Tensor:
        """Concat along feature dim; supports [B,F] or [B,T,F]."""
        # Handle case where pkv might be a dict
        if isinstance(pkv, dict):
            if 'pred' in pkv:
                pkv = pkv['pred']
            else:
                raise ValueError("PK prediction dict must contain 'pred' key")
        
        if pkv.dim() == 1:
            pkv = pkv.unsqueeze(-1)  # [B,1]
        if X.dim() == 3:  # [B,T,F]
            if pkv.dim() == 2:       # [B,F2] -> broadcast over T
                pkv = pkv.unsqueeze(1).expand(-1, X.size(1), -1)
            return torch.cat([X, pkv], dim=-1)
        else:  # [B,F]
            return torch.cat([X, pkv], dim=-1)

    def forward_pd(self, batch: Dict[str, Any], pk_pred: torch.Tensor = None, z_pk: torch.Tensor = None):
        # Handle case where pk_pred is a dict (from forward method)
        if isinstance(pk_pred, dict):
            if 'pred' in pk_pred:
                pk_pred = pk_pred['pred']
            else:
                # If no 'pred' key, create dummy prediction
                x_pd = _maybe_time_pool(batch["x"] if isinstance(batch, dict) else batch[0]).float()
                pk_pred = torch.zeros(x_pd.size(0), 1, device=x_pd.device)
        
        if pk_pred is None:
            # Get PK prediction from batch
            if isinstance(batch, dict):
                pk_pred = batch.get(self.pk_input_key)
                if pk_pred is None:
                    # If no PK prediction in batch, create a dummy one
                    # This is a fallback for cases where PK prediction is not available
                    x_pd = _maybe_time_pool(batch["x"]).float()
                    pk_pred = torch.zeros(x_pd.size(0), 1, device=x_pd.device)
            else:
                # For tuple/list batch format, create dummy PK prediction
                x_pd = _maybe_time_pool(batch[0]).float()
                pk_pred = torch.zeros(x_pd.size(0), 1, device=x_pd.device)
        
        # Ensure pk_pred has the right shape for concatenation
        if isinstance(pk_pred, torch.Tensor):
            if pk_pred.dim() == 1:
                pk_pred = pk_pred.unsqueeze(-1)  # [B] -> [B, 1]
        
        # Debug logging removed for clean output
        
        if self.pk_detach and hasattr(pk_pred, 'detach'):
            pk_pred = pk_pred.detach()
        
        # Handle both dict and tuple/list batch formats
        if isinstance(batch, dict):
            x_pd = _maybe_time_pool(batch["x"]).float()
        else:
            # Assume batch is tuple/list format: (X, y, ...)
            x_pd = _maybe_time_pool(batch[0]).float()
        
        if self.concat_mode == "input":
            # For input concatenation, we need to ensure pk_pred has the right shape
            if isinstance(pk_pred, dict):
                pk_pred_tensor = pk_pred['pred']
            else:
                pk_pred_tensor = pk_pred
            
            # Ensure pk_pred has the same batch size and time dimension as x_pd
            if pk_pred_tensor.dim() == 1:
                pk_pred_tensor = pk_pred_tensor.unsqueeze(-1)  # [B, 1]
            elif pk_pred_tensor.dim() == 2 and x_pd.dim() == 3:
                pk_pred_tensor = pk_pred_tensor.unsqueeze(1).expand(-1, x_pd.size(1), -1)  # [B, T, 1]
            
            x_concat = torch.cat([x_pd, pk_pred_tensor], dim=-1)
            # Debug: Print shapes (remove after fixing)
            # print(f"DEBUG forward_pd: x_pd.shape = {x_pd.shape}")
            # print(f"DEBUG forward_pd: pk_pred_tensor.shape = {pk_pred_tensor.shape}")
            # print(f"DEBUG forward_pd: x_concat.shape = {x_concat.shape}")
            # print(f"DEBUG forward_pd: self.enc_pd.in_dim = {self.enc_pd.in_dim}")
            z2 = self.enc_pd(x_concat)
            # Handle tuple output from ResMLP+MoE encoder
            if isinstance(z2, tuple):
                z2, aux_loss = z2
                # Store aux_loss for later use if needed
                if not hasattr(self, '_aux_losses'):
                    self._aux_losses = []
                self._aux_losses.append(aux_loss)
        else:  # "latent"
            z_pd = self.enc_pd(x_pd)
            # Handle tuple output from ResMLP+MoE encoder
            if isinstance(z_pd, tuple):
                z_pd, aux_loss = z_pd
                # Store aux_loss for later use if needed
                if not hasattr(self, '_aux_losses'):
                    self._aux_losses = []
                self._aux_losses.append(aux_loss)
            
            if z_pk is not None and self.pk_detach:
                z_pk = z_pk.detach()
            z_concat = torch.cat([z_pd, z_pk], dim=-1)
            z2 = z_concat
        
        outs = self.head_pd(z2, batch)
        return outs, z2, outs  # Return 3 values for compatibility

    def forward(self, batch: Dict[str, Any]):
        """Full forward pass: PK -> PD"""
        # Determine if this is PK or PD data based on input dimensions
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        is_pk_data = x.shape[1] == 11
        is_pd_data = x.shape[1] == 12
        
        if is_pk_data:
            # This is PK data - only return PK output
            pk_outs, z_pk, _ = self.forward_pk(batch)
            return pk_outs, None, z_pk, None
        elif is_pd_data:
            # This is PD data - return both PK and PD outputs
            pk_outs, z_pk, _ = self.forward_pk(batch)
            pd_outs, z_pd, _ = self.forward_pd(batch, pk_outs, z_pk)
            return pk_outs, pd_outs, z_pk, z_pd
        else:
            # Fallback - assume it's PD data
            pk_outs, z_pk, _ = self.forward_pk(batch)
            pd_outs, z_pd, _ = self.forward_pd(batch, pk_outs, z_pk)
            return pk_outs, pd_outs, z_pk, z_pd

# =========================
# Dual Stage PK/PD Model
# =========================

class DualStagePKPDModel(nn.Module):
    """
    Dual-stage model: x -> front_encoder -> z -> back_encoder -> z2 -> head -> y_hat
    z is used for PK, z2 is used for PD
    """
    
    def __init__(self, front_encoder, back_encoder, head_pk, head_pd, max_input_dim=None, pk_input_dim=None):
        super().__init__()
        self.front_encoder = front_encoder  # For PD: x_pd -> z
        self.back_encoder = back_encoder    # For PD: z -> z2
        self.head_pk = head_pk
        self.head_pd = head_pd
        self.max_input_dim = max_input_dim
        self.pk_input_dim = pk_input_dim
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to max_input_dim if needed."""
        if self.max_input_dim is not None and x.shape[-1] < self.max_input_dim:
            padding = torch.zeros(x.shape[0], self.max_input_dim - x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=-1)
        return x
    
    def forward_pk(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward for PK: x_pd -> front_encoder -> z -> head_pk -> y_hat_pk"""
        # Handle both dict and tuple/list batch formats
        if isinstance(batch, dict):
            x = _maybe_time_pool(batch["x"]).float()
        else:
            # Assume batch is tuple/list format: (X, y, ...)
            x = _maybe_time_pool(batch[0]).float()
        # Ensure x is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        x = self._pad_input(x)  # Pad if needed
        z = self.front_encoder(x)  # [B, front_dim]
        outs = self.head_pk(z, batch)
        return outs["pred"], z, outs
    
    def forward_pd(self, z_pk: torch.Tensor, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward for PD: z_pk -> back_encoder -> z2 -> head_pd -> y_hat"""
        z2 = self.back_encoder(z_pk)      # [B, back_dim]
        outs = self.head_pd(z2, batch)
        return outs["pred"], z2, outs
    
    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generic forward for convenience:
        - If batch contains 'branch' ('pk' or 'pd'), route accordingly.
        - Else defaults to PD (common in PD-first evaluation).
        """
        branch = str(batch.get("branch", "pd")).lower() if isinstance(batch, dict) else "pd"
        if branch == "pk":
            return self.forward_pk(batch)
        else:
            # For PD, we need PK representation first
            _, z_pk, _ = self.forward_pk(batch)
            return self.forward_pd(z_pk, batch)


# =========================
# Shared Encoder Model
# =========================

class SharedEncModel(nn.Module):
    """
    Shared encoder model for PK/PD with separate heads.
    """
    def __init__(
        self,
        encoder: BaseEncoder,
        head_pk: BaseHead,
        head_pd: BaseHead,
        pk_input_dim: int = None,
        pd_input_dim: int = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.head_pk = head_pk
        self.head_pd = head_pd
        self.pk_input_dim = pk_input_dim
        self.pd_input_dim = pd_input_dim
    
    def _pad_input(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pad input tensor to target dimension"""
        if x.shape[1] < target_dim:
            padding = torch.zeros(x.shape[0], target_dim - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        return x
    
    def forward_pk(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward for PK branch"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        
        # Pad input if necessary
        if self.pk_input_dim and x.shape[1] < self.encoder.in_dim:
            x = self._pad_input(x, self.encoder.in_dim)
        
        # Create new batch with padded input
        if isinstance(batch, (list, tuple)):
            new_batch = (x, batch[1])
        else:
            new_batch = {**batch, "x": x}
        
        z = self.encoder(x)
        # Handle tuple output from ResMLP+MoE encoder
        if isinstance(z, tuple):
            z, aux_loss = z
            # Store aux_loss for later use if needed
            if not hasattr(self, '_aux_losses'):
                self._aux_losses = []
            self._aux_losses.append(aux_loss)
        outs = self.head_pk(z, new_batch)
        return outs["pred"], z, outs
    
    def forward_pd(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward for PD branch"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        
        # Pad input if necessary
        if self.pd_input_dim and x.shape[1] < self.encoder.in_dim:
            x = self._pad_input(x, self.encoder.in_dim)
        
        # Create new batch with padded input
        if isinstance(batch, (list, tuple)):
            new_batch = (x, batch[1])
        else:
            new_batch = {**batch, "x": x}
        
        z = self.encoder(x)
        # Handle tuple output from ResMLP+MoE encoder
        if isinstance(z, tuple):
            z, aux_loss = z
            # Store aux_loss for later use if needed
            if not hasattr(self, '_aux_losses'):
                self._aux_losses = []
            self._aux_losses.append(aux_loss)
        outs = self.head_pd(z, new_batch)
        return outs["pred"], z, outs
    
    def forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generic forward - defaults to PK branch
        """
        branch = str(batch.get("branch", "pk")).lower() if isinstance(batch, dict) else "pk"
        if branch == "pd":
            return self.forward_pd(batch)
        else:
            return self.forward_pk(batch)
