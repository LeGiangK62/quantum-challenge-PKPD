#!/usr/bin/env python3
"""
PK/PD Modeling System
Clean and extensible structure with competition task solving capabilities
"""

import sys
import time
import argparse
from pathlib import Path
import json
import pickle 
# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
import os

from config import parse_args
from utils.logging import setup_logging, get_logger
from data.loaders import load_estdata, use_feature_engineering
from data.splits import prepare_for_split
from models.architectures import *
from training.modes import *
from utils.helpers import scaling_and_prepare_loader, get_device, generate_run_name
from utils.factory import create_model_factory, create_trainer_factory   

def main():
    """Main function with competition task solving capabilities"""
    # Parse configuration
    config = parse_args()
    
    # Generate run name with improved format
    if not config.run_name:
        config.run_name = generate_run_name(config)
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
    # Log encoder configuration
    encoder_info = f"Encoder: {config.encoder}"
    if config.encoder_pk is not None or config.encoder_pd is not None:
        encoder_info += f" | PK: {config.encoder_pk or config.encoder} | PD: {config.encoder_pd or config.encoder}"
    
    logger.info(f"Mode: {config.mode} | {encoder_info} | Epochs: {config.epochs} | Batch: {config.batch_size} | Device: {device} \n")
    
    # 1. Data loading
    logger.info(f"\n Data loading started: {config.csv_path}")
    df_all, df_obs, df_dose = load_estdata(config.csv_path)
    logger.info(f"Successfully loaded CSV data: {df_all.shape}")
    logger.info(f"Observations: {df_obs.shape}, Dosing: {df_dose.shape}")
    
    # 2. Data preprocessing
    logger.info("\nData preprocessing started")
    if config.use_feature_engineering:
        logger.info("- Feature engineering applied")
        df_final, pk_features, pd_features = use_feature_engineering(
            df_obs=df_obs, df_dose=df_dose,
            use_perkg=config.perkg,
            target="dv",
            use_pd_baseline_for_dv=True,
            allow_future_dose=config.allow_future_dose,
            time_windows=config.time_windows
        )
        logger.info(f"- Feature engineering completed. PK features: {len(pk_features)}, PD features: {len(pd_features)}")
    else:
        logger.info("- Basic preprocessing applied")
        df_final = df_obs.copy()
        feature_cols = [col for col in df_final.columns 
                      if col not in ['ID', 'TIME', 'DV', 'DVID']]
        pk_features = feature_cols
        pd_features = feature_cols
    
    logger.info(f"- Preprocessing completed - Final data shape: {df_final.shape}")
    logger.info(f"- PK features: {len(pk_features)}, PD features: {len(pd_features)}")
    
    # 3. Data splitting
    logger.info("\nData splitting started")
    # Separate PK/PD data
    pk_df = df_final[df_final["DVID"] == 1].copy()
    pd_df = df_final[df_final["DVID"] == 2].copy()
    logger.info(f"- PK data: {pk_df.shape}, PD data: {pd_df.shape}")
    
    pk_splits, pd_splits, global_splits, _ = prepare_for_split(
        df_final=df_final, df_dose=df_dose,
        pk_df=pk_df, pd_df=pd_df,
        split_strategy=config.split_strategy,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
        dose_bins=4,
        id_universe="union",
        verbose=True,
    )
    logger.info(f" - {config.split_strategy} splitting completed successfully")
            
    splits = {
        'pk': pk_splits,
        'pd': pd_splits,
        'global': global_splits,
        'pk_features': pk_features,
        'pd_features': pd_features
    }
    
    logger.info(" - Data splitting completed")
    
    # 4. Data loader creation
    pin_mem = get_device().type == 'cuda' # pin memory for faster data transfer
    
    # Optimize num_workers based on CPU cores
    num_workers = min(4, os.cpu_count() or 1) # limit number of workers to 4
    
    logger.info("\nData loader creation started")
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
    logger.info(" - Data loader creation completed")
    logger.info(f" - train_loader_pk: {len(train_loader_pk.dataset)}")
    logger.info(f" - valid_loader_pk: {len(valid_loader_pk.dataset)}")
    logger.info(f" - test_loader_pk: {len(test_loader_pk.dataset)}")
    logger.info(f" - train_loader_pd: {len(train_loader_pd.dataset)}")
    logger.info(f" - valid_loader_pd: {len(valid_loader_pd.dataset)}")
    logger.info(f" - test_loader_pd: {len(test_loader_pd.dataset)}")
    
    loaders = {
        "train_pk": train_loader_pk, "val_pk": valid_loader_pk, "test_pk": test_loader_pk,
        "train_pd": train_loader_pd, "val_pd": valid_loader_pd, "test_pd": test_loader_pd,
    }
    
    # 5. Model creation
    logger.info("\nModel creation started")
    model = create_model_factory(config, loaders, pk_features, pd_features)
    
    # 6. Trainer creation and training
    logger.info("\nTrainer creation and training started")
    trainer = create_trainer_factory(config, model, loaders, device)
    results = trainer.train()
    
    # 7. Results output (already handled by trainer's log_final_results)
    logger.info(" - Training completed!")
    
    # 8. Model saving with distinguishable name
    model_filename = f"{config.run_name}_final_model.pth"
    trainer.save_model(model_filename)
    logger.info(f" - Model saved as: {model_filename}")

    # save config
    configs_dir = f"{config.output_dir}/configs"
    os.makedirs(configs_dir, exist_ok=True)
    with open(f"{configs_dir}/{config.run_name}.json", "w") as f:
        json.dump(config.__dict__, f)
    logger.info(f" - Config saved as: {configs_dir}/{config.run_name}.json")

    # save scalers
    scalers_dir = f"{config.output_dir}/scalers"
    os.makedirs(scalers_dir, exist_ok=True)
    with open(f"{scalers_dir}/{config.run_name}.pkl", "wb") as f:
        pickle.dump(pk_scaler, f)
        pickle.dump(pd_scaler, f)
    logger.info(f" - Scalers saved as: {scalers_dir}/{config.run_name}.pkl")

    logger.info(" - Execution completed!")
    



if __name__ == "__main__":
    exit(main())
