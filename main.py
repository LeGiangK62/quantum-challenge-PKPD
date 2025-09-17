#!/usr/bin/env python3
"""
PK/PD Modeling System
Clean and extensible structure with competition task solving capabilities
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import parse_args
from utils.logging import setup_logging, get_logger
from data import DataLoader, Preprocessor, DataSplitter
from models.architectures import *
from training.modes import *
from utils.helpers import scaling_and_prepare_loader, get_device


def main():
    """Main function with competition task solving capabilities"""
    # Parse configuration
    config = parse_args()
    
    
    # Generate run name with improved format
    if not config.run_name:
        # Create human-readable timestamp (YYMMDD_HHMM)
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        
        # Create postfix based on configuration (only add if features are enabled)
        postfix_parts = []
        if config.use_feature_engineering:
            postfix_parts.append("fe")
        if hasattr(config, 'use_mixup') and config.use_mixup:
            postfix_parts.append("mixup")
        if hasattr(config, 'use_contrast') and config.use_contrast:
            postfix_parts.append("contrast")
        if hasattr(config, 'use_uncertainty') and config.use_uncertainty:
            postfix_parts.append("uncertainty")
        if hasattr(config, 'use_active_learning') and config.use_active_learning:
            postfix_parts.append("active")
        if hasattr(config, 'use_meta_learning') and config.use_meta_learning:
            postfix_parts.append("meta")
        
        # Only add postfix if there are enabled features
        postfix = "_".join(postfix_parts) if postfix_parts else ""
        
        # Create clean run name
        if postfix:
            config.run_name = f"{config.mode}_{config.encoder}_s{config.random_state}_{timestamp}_{postfix}"
        else:
            config.run_name = f"{config.mode}_{config.encoder}_s{config.random_state}_{timestamp}"
    else:
        # User provided run_name - use it as is
        logger = get_logger(__name__)
        logger.info(f"Using user-provided run name: {config.run_name}")
    
    # Setup logging with hierarchical directory structure
    log_dir = f"{config.output_dir}/logs/{config.mode}/{config.encoder}/s{config.random_state}"
    log_file = setup_logging(log_dir, config.verbose, config.run_name)
    logger = get_logger(__name__)
    
    # Force flush to ensure output is visible
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    logger.info("PK/PD Modeling System")
    logger.info(f"Run Name: {config.run_name}")
    device = get_device(config.device_id)
    logger.info(f"Mode: {config.mode} | Encoder: {config.encoder} | Epochs: {config.epochs} | Batch: {config.batch_size} | Device: {device}")
    
    # 1. Data loading
    data_loader = DataLoader(config)
    data = data_loader.load_data()
    
    # 2. Data preprocessing
    preprocessor = Preprocessor(config)
    df_final, pk_features, pd_features = preprocessor.process(data["df_obs"], data["df_dose"])
    
    # 3. Data splitting
    splitter = DataSplitter(config)
    splits = splitter.split(df_final, data["df_dose"], pk_features, pd_features)
    
    # 4. Data loader creation
    pin_mem = get_device().type == 'cuda' # pin memory for faster data transfer
    
    # Optimize num_workers based on CPU cores
    import os
    num_workers = min(4, os.cpu_count() or 1) # limit number of workers to 4
    
    pk_scaler, train_loader_pk, valid_loader_pk, test_loader_pk = scaling_and_prepare_loader(
        splits['pk'], pk_features, 
        batch_size=config.batch_size, lambda_ctr=0.0,
        target_col="DV", num_workers=num_workers, pin_memory=pin_mem, drop_last_train=True
    )
    
    pd_scaler, train_loader_pd, valid_loader_pd, test_loader_pd = scaling_and_prepare_loader(
        splits['pd'], pd_features, 
        batch_size=config.batch_size, lambda_ctr=0.0,
        target_col="DV", num_workers=num_workers, pin_memory=pin_mem, drop_last_train=True
    )
    
    loaders = {
        "train_pk": train_loader_pk, "val_pk": valid_loader_pk, "test_pk": test_loader_pk,
        "train_pd": train_loader_pd, "val_pd": valid_loader_pd, "test_pd": test_loader_pd,
    }
    
    # 5. Model creation
    model = create_model(config, loaders)
    
    # 6. Trainer creation and training
    logger.info("Training started...")
    trainer = create_trainer(config, model, loaders, device)
    results = trainer.train()
    
    # 7. Results output (already handled by trainer's log_final_results)
    logger.info("Training completed!")
    
    # 8. Model saving with distinguishable name
    model_filename = f"{config.run_name}_final_model.pth"
    trainer.save_model(model_filename)
    logger.info(f"Model saved as: {model_filename}")
    
    logger.info("Execution completed!")
    


def create_model(config, loaders):
    """Model creation using the proper model factory"""
    logger = get_logger(__name__)
    
    # Map training modes to model types
    mode_to_model_type = {
        "separate": "enc_head",  # Separate trainer creates its own models
        "joint": "dual_branch",
        "shared": "shared", 
        "dual_stage": "dual_stage",
        "integrated": "dual_branch",  # Integrated uses dual branch
        "two_stage_shared": "two_stage_shared"  # Two-stage shared uses shared model
    }
    
    model_type = mode_to_model_type.get(config.mode, config.mode)
    
    # For separate mode, return None as the trainer creates its own models
    if config.mode == "separate":
        logger.info("Separate mode: trainer will create models internally")
        return None
    
    # Use the proper model factory from models/base.py
    from models.base import get_model
    
    logger.info(f"Creating {config.mode} model (type: {model_type})")
    model = get_model(model_type, config, loaders)
    
    logger.info(f"Model creation completed: {type(model).__name__}")
    return model


def create_trainer(config, model, loaders, device):
    """Trainer creation"""
    logger = get_logger(__name__)
    
    if config.mode == "separate":
        from training.modes import SeparateTrainer
        # Separate trainer doesn't need a pre-created model
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


if __name__ == "__main__":
    exit(main())
