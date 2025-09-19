"""
Model factory for PK/PD modeling.
"""

from utils.logging import get_logger
from .architectures import EncHeadModel, DualBranchPKPD, SharedEncModel, DualStagePKPDModel
from utils.helpers import build_encoder, build_head
from .encoders import SimpleMLPEncoder

def create_model(model_type, config, loaders, pk_features=None, pd_features=None, branch=None):
    """Create a model instance based on the specified type."""
    logger = get_logger(__name__)
    
    logger.info(f"Creating {model_type} model")
    
    if model_type == 'one_model':
        return _create_one_model(config, loaders, branch, pk_features, pd_features)
    elif model_type == 'two_encoder_model':
        return _create_two_encoder_model(config, loaders, pk_features, pd_features)
    elif model_type == 'shared_model':
        return _create_shared_model(config, loaders, pk_features, pd_features)
    elif model_type == 'dual_stage':
        return _create_dual_stage_model(config, loaders, pk_features, pd_features)
    elif model_type == 'two_stage_shared_model':
        return _create_two_stage_shared_model(config, loaders, pk_features, pd_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _create_one_model(config, loaders, branch, pk_features=None, pd_features=None):
    """Create single encoder-head model."""
    def get_input_dim(loader, branch):
        return loader.dataset.tensors[0].shape[1]
    
    train_key = f"train_{branch}"
    input_dim = get_input_dim(loaders[train_key], branch)
    
    encoder = build_encoder(config.encoder, input_dim, config)
    head = build_head(getattr(config, f'head_{branch}'), encoder.out_dim, branch, config)
    
    return EncHeadModel(encoder=encoder, head=head)


def _create_two_encoder_model(config, data_loaders, pk_feature_names=None, pd_feature_names=None):
    """Create dual branch model with potentially different encoders for PK and PD."""
    logger = get_logger(__name__)
    
    # Get input dimensions from feature names
    pk_input_dimension = len(pk_feature_names)
    pd_input_dimension = len(pd_feature_names)
    
    # Determine encoder types for PK and PD
    pk_encoder_type = config.encoder_pk if config.encoder_pk is not None else config.encoder
    pd_encoder_type = config.encoder_pd if config.encoder_pd is not None else config.encoder
    
    # Validate encoder types
    valid_encoder_types = ["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"]
    if pk_encoder_type not in valid_encoder_types:
        raise ValueError(f"Invalid PK encoder: {pk_encoder_type}. Valid options: {valid_encoder_types}")
    if pd_encoder_type not in valid_encoder_types:
        raise ValueError(f"Invalid PD encoder: {pd_encoder_type}. Valid options: {valid_encoder_types}")
    
    logger.info(f"Building dual branch model:")
    logger.info(f"  PK: {pk_encoder_type} encoder (input_dim={pk_input_dimension})")
    logger.info(f"  PD: {pd_encoder_type} encoder (input_dim={pd_input_dimension})")
    
    # Create encoders
    pk_encoder = build_encoder(pk_encoder_type, pk_input_dimension, config)
    pd_encoder = build_encoder(pd_encoder_type, pd_input_dimension, config)

    # Create heads
    pk_head = build_head(config.head_pk, pk_encoder.out_dim)
    pd_head = build_head(config.head_pd, pd_encoder.out_dim)
    
    logger.info(f"  PK encoder output_dim: {pk_encoder.out_dim}")
    logger.info(f"  PD encoder output_dim: {pd_encoder.out_dim}")
    
    return DualBranchPKPD(pk_encoder, pd_encoder, pk_head, pd_head, concat_mode="latent")


def _create_shared_model(config, data_loaders, pk_feature_names=None, pd_feature_names=None):
    """Create shared encoder model with separate heads for PK and PD."""
    def get_input_dimension(data_loader, branch_name=None):
        return data_loader.dataset.tensors[0].shape[1]
    
    pk_input_dimension = get_input_dimension(data_loaders["train_pk"])
    pd_input_dimension = get_input_dimension(data_loaders["train_pd"])
    
    max_input_dimension = max(pk_input_dimension, pd_input_dimension)
    
    shared_encoder = build_encoder(config.encoder, max_input_dimension, config)
    pk_head = build_head(config.head_pk, shared_encoder.out_dim)
    pd_head = build_head(config.head_pd, shared_encoder.out_dim)
    
    return SharedEncModel(encoder=shared_encoder, head_pk=pk_head, head_pd=pd_head, 
                         pk_input_dim=pk_input_dimension, pd_input_dim=pd_input_dimension)


def _create_two_stage_shared_model(config, data_loaders, pk_feature_names=None, pd_feature_names=None):
    """Create dual stage model with potentially different encoders."""
    logger = get_logger(__name__)
    
    def get_input_dimension(data_loader, branch_name=None):
        return data_loader.dataset.tensors[0].shape[1]
    
    # Get input dimensions
    pk_input_dimension = get_input_dimension(data_loaders["train_pk"])
    pd_input_dimension = get_input_dimension(data_loaders["train_pd"])
    
    # Determine encoder types for front and back stages
    front_encoder_type = config.encoder_pk if config.encoder_pk is not None else config.encoder
    back_encoder_type = config.encoder_pd if config.encoder_pd is not None else config.encoder
    
    # Validate encoder types
    valid_encoder_types = ["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"]
    if front_encoder_type not in valid_encoder_types:
        raise ValueError(f"Invalid front encoder: {front_encoder_type}. Valid options: {valid_encoder_types}")
    if back_encoder_type not in valid_encoder_types:
        raise ValueError(f"Invalid back encoder: {back_encoder_type}. Valid options: {valid_encoder_types}")
    
    logger.info(f"Building dual stage model:")
    logger.info(f"  Front: {front_encoder_type} encoder (input_dim={max(pk_input_dimension, pd_input_dimension)})")
    logger.info(f"  Back: {back_encoder_type} encoder (input_dim=front.out_dim)")
    
    # Create front and back encoders with potentially different types
    if pk_input_dimension != pd_input_dimension:
        max_input_dimension = max(pk_input_dimension, pd_input_dimension)
        front_encoder = build_encoder(front_encoder_type, max_input_dimension, config)
    else:
        front_encoder = build_encoder(front_encoder_type, pd_input_dimension, config)
    
    # Back encoder uses the output dimension of front encoder
    back_encoder = build_encoder(back_encoder_type, front_encoder.out_dim, config)
    
    # Create heads
    pk_head = build_head(config.head_pk, front_encoder.out_dim)
    pd_head = build_head(config.head_pd, back_encoder.out_dim)
    
    logger.info(f"  Front encoder output_dim: {front_encoder.out_dim}")
    logger.info(f"  Back encoder output_dim: {back_encoder.out_dim}")
    
    return DualStagePKPDModel(
        front_encoder=front_encoder,
        back_encoder=back_encoder,
        head_pk=pk_head,
        head_pd=pd_head,
        max_input_dim=max(pk_input_dimension, pd_input_dimension) if pk_input_dimension != pd_input_dimension else None,
        pk_input_dim=pk_input_dimension
    )


def _create_dual_stage_model(config, data_loaders, pk_feature_names=None, pd_feature_names=None):
    """Create dual stage model (alias for two_stage_shared_model)."""
    return _create_two_stage_shared_model(config, data_loaders, pk_feature_names, pd_feature_names)
