"""
 Factory functions
"""

from utils.logging import get_logger
from models.unified_model import UnifiedPKPDModel
from training.unified_trainer import UnifiedPKPDTrainer


def create_model(config, loaders, pk_features, pd_features):
    """Create unified model"""
    logger = get_logger(__name__)
    
    # Calculate input dimensions
    pk_input_dim = len(pk_features)
    pd_input_dim = len(pd_features)
    
    logger.info(f"Model creation - Mode: {config.mode}")
    logger.info(f"PK input dimension: {pk_input_dim}, PD input dimension: {pd_input_dim}")
    
    # Encoder information logging
    pk_encoder = config.encoder_pk or config.encoder
    pd_encoder = config.encoder_pd or config.encoder
    if pk_encoder != pd_encoder:
        logger.info(f"Encoder settings - PK: {pk_encoder}, PD: {pd_encoder}")
    else:
        logger.info(f"Encoder settings - Common: {pk_encoder}")
    
    # Create unified model
    model = UnifiedPKPDModel(
        config=config,
        pk_features=pk_features,
        pd_features=pd_features,
        pk_input_dim=pk_input_dim,
        pd_input_dim=pd_input_dim
    )
    
    logger.info(f"Model creation completed: {type(model).__name__}")
    return model


def create_trainer(config, model, loaders, device):
    """Create unified trainer"""
    logger = get_logger(__name__)
    
    logger.info(f"Trainer creation - Mode: {config.mode}")
    
    # Create unified trainer
    trainer = UnifiedPKPDTrainer(
        model=model,
        config=config,
        data_loaders=loaders,
        device=device
    )
    
    logger.info(f"Trainer creation completed: {type(trainer).__name__}")
    return trainer
