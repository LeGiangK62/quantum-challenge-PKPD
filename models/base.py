"""
Model factory for PK/PD modeling.
"""

from utils.logging import get_logger
from .architectures import EncHeadModel, DualBranchPKPD, SharedEncModel, DualStagePKPDModel
from utils.helpers import build_encoder, build_head
from .encoders import SimpleMLPEncoder

def get_model(model_type, args, loaders, branch=None):
    """Get a model instance based on the specified type."""
    logger = get_logger(__name__)
    
    logger.info(f"get_model called with model_type: {model_type}, type: {type(model_type)}")
    
    if model_type == 'enc_head':
        return _get_enc_head_model(args, loaders, branch)
    elif model_type == 'dual_branch':
        return _get_dual_branch_model(args, loaders)
    elif model_type == 'shared':
        return _get_shared_model(args, loaders)
    elif model_type == 'dual_stage':
        return _get_dual_stage_model(args, loaders)
    elif model_type == 'two_stage_shared':
        return _get_shared_model(args, loaders)  # Same as shared model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _get_enc_head_model(args, loaders, branch):
    """Get encoder-head model."""

    def infer_input_dim(loader, branch):
        return loader.dataset.tensors[0].shape[1]
    
    # Get input dimension
    loader_key = f"train_{branch}"
    F_in = infer_input_dim(loaders[loader_key])
    
    # Build encoder and head
    encoder = build_encoder(args.encoder, F_in, args)
    head = build_head(getattr(args, f'head_{branch}'), encoder.out_dim, branch, args)
    
    return EncHeadModel(encoder=encoder, head=head)


def _get_dual_branch_model(args, loaders):
    """Get dual branch model."""
    def infer_input_dim(loader, branch=None):
        return loader.dataset.tensors[0].shape[1]
    
    # Get actual feature dimensions based on feature engineering
    if hasattr(args, 'use_feature_engineering') and args.use_feature_engineering:
        # With feature engineering: PK uses 11 features, PD uses 12 features
        F_in_pk = 11
        F_in_pd = 12
    else:
        # Without feature engineering: PK uses 7 features, PD uses 7 features
        F_in_pk = 7
        F_in_pd = 7
    
    # Build encoders and heads
    enc_pk = build_encoder(args.encoder, F_in_pk, args)
    # For PD encoder, we need to account for PK prediction concatenation
    # When concat_mode="input", PD encoder receives F_in_pd + 1 features (PD features + PK prediction)
    enc_pd = build_encoder(args.encoder, F_in_pd + 1, args)
    head_pk = build_head(args.head_pk, enc_pk.out_dim, "pk", args)
    head_pd = build_head(args.head_pd, enc_pd.out_dim, "pd", args)
    
    return DualBranchPKPD(enc_pk, enc_pd, head_pk, head_pd)


def _get_shared_model(args, loaders):
    """Get shared encoder model."""
    def infer_input_dim(loader, branch=None):
        return loader.dataset.tensors[0].shape[1]
    
    # Get input dimensions
    F_in_pk = infer_input_dim(loaders["train_pk"])
    F_in_pd = infer_input_dim(loaders["train_pd"])
    
    # For shared mode, use the maximum dimension and pad if necessary
    F_in_max = max(F_in_pk, F_in_pd)
    
    # Build shared encoder with maximum input dimension
    encoder = build_encoder(args.encoder, F_in_max, args)
    head_pk = build_head(args.head_pk, encoder.out_dim, "pk", args)
    head_pd = build_head(args.head_pd, encoder.out_dim, "pd", args)
    
    return SharedEncModel(encoder=encoder, head_pk=head_pk, head_pd=head_pd, 
                         pk_input_dim=F_in_pk, pd_input_dim=F_in_pd)


def _get_dual_stage_model(args, loaders):
    """Get dual stage model."""
    def infer_input_dim(loader, branch=None):
        return loader.dataset.tensors[0].shape[1]
    
    # Get input dimensions
    F_in_pk = infer_input_dim(loaders["train_pk"])
    F_in_pd = infer_input_dim(loaders["train_pd"])
    
    # Build front and back encoders
    if F_in_pk != F_in_pd:
        max_dim = max(F_in_pk, F_in_pd)
        front_encoder = SimpleMLPEncoder(in_dim=max_dim, hidden_dims=args.front_dims, dropout=args.dropout)
    else:
        front_encoder = SimpleMLPEncoder(in_dim=F_in_pd, hidden_dims=args.front_dims, dropout=args.dropout)
    
    back_encoder = SimpleMLPEncoder(in_dim=front_encoder.out_dim, hidden_dims=args.back_dims, dropout=args.dropout)
    
    # Build heads
    head_pk = build_head(args.head_pk, front_encoder.out_dim, "pk", args)
    head_pd = build_head(args.head_pd, back_encoder.out_dim, "pd", args)
    
    return DualStagePKPDModel(
        front_encoder=front_encoder,
        back_encoder=back_encoder,
        head_pk=head_pk,
        head_pd=head_pd,
        max_input_dim=max(F_in_pk, F_in_pd) if F_in_pk != F_in_pd else None,
        pk_input_dim=F_in_pk
    )
