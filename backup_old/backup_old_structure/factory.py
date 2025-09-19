"""
Factory functions for creating models and trainers
"""

from utils.logging import get_logger
from models.base import create_model


def create_model_factory(config, loaders, pk_features=None, pd_features=None):
    """Model creation using the proper model factory"""
    logger = get_logger(__name__)
    
    # Map training modes to model types
    mode_to_model_type = {
        "separate": "one_model",
        "joint": "two_encoder_model",
        "dual_stage": "two_encoder_model",
        "integrated": "two_encoder_model",
        "shared": "shared_model", 
        "two_stage_shared": "two_stage_shared_model"
    }
    model_type = mode_to_model_type.get(config.mode, config.mode)
    
    # Log encoder configuration
    encoder_info = f"Encoder: {config.encoder}"
    if config.encoder_pk is not None or config.encoder_pd is not None:
        encoder_info += f" | PK: {config.encoder_pk or config.encoder} | PD: {config.encoder_pd or config.encoder}"
    
    logger.info(f"Creating {config.mode} model (type: {model_type})")
    logger.info(f"Encoder config: {encoder_info}")
    
    if config.mode == "separate":
        logger.info("Separate mode: creating placeholder model for consistent interface")
        # Return a placeholder model for consistent interface
        # The trainer will create actual PK/PD models internally using encoder_pk/encoder_pd
        return "separate_placeholder"
    model = create_model(model_type, config, loaders, pk_features, pd_features)
    logger.info(f"Model creation completed: {type(model).__name__}")
    return model


def create_trainer_factory(config, model, loaders, device):
    """Trainer creation"""
    logger = get_logger(__name__)
    
    if config.mode == "separate":
        from training.modes import SeparateTrainer
        trainer = SeparateTrainer(None, config, loaders, device)
    elif config.mode == "joint":
        from training.modes import JointTrainer
        trainer = JointTrainer(model, config, loaders, device)
    elif config.mode == "shared":
        from training.modes import SharedTrainer
        trainer = SharedTrainer(model, config, loaders, device)
    elif config.mode == "dual_stage":
        from training.modes import DualStageTrainer
        trainer = DualStageTrainer(model, config, loaders, device)
    elif config.mode == "integrated":
        from training.modes import IntegratedTrainer
        trainer = IntegratedTrainer(model, config, loaders, device)
    elif config.mode == "two_stage_shared":
        from training.modes.two_stage_shared import TwoStageSharedTrainer
        trainer = TwoStageSharedTrainer(model, config, loaders, device)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")
    
    logger.info(f"Trainer creation completed: {type(trainer).__name__}")
    return trainer
