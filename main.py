#!/usr/bin/env python3
"""
PK/PD Modeling System
"""

import sys
import time
import argparse
from pathlib import Path
import json
import pickle
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import parse_args
from utils.logging import setup_logging, get_logger
from data.loaders import load_estdata, use_feature_engineering
from data.splits import prepare_for_split
from utils.helpers import scaling_and_prepare_loader, get_device, generate_run_name
from utils.factory import create_model, create_trainer


def main():
    """Main function"""
    # Parse configuration
    config = parse_args()
    
    # Generate run name
    if not config.run_name:
        config.run_name = generate_run_name(config)
    
    # Setup logging with hierarchical directory structure
    if config.encoder_pk or config.encoder_pd:
        pk_encoder = config.encoder_pk or config.encoder
        pd_encoder = config.encoder_pd or config.encoder
        encoder_name = f"{pk_encoder}-{pd_encoder}"
    else:
        encoder_name = config.encoder
    
    # Create hierarchical log directory: logs/{run_name}/{mode}/{encoder}/s{seed}/
    if config.run_name:
        log_dir = f"{config.output_dir}/logs/{config.run_name}/{config.mode}/{encoder_name}/s{config.random_state}"
    else:
        # Fallback to the old structure if no run_name
        log_dir = f"{config.output_dir}/logs/{config.mode}/{encoder_name}/s{config.random_state}"
    
    log_file = setup_logging(log_dir, config.verbose, config.run_name)
    logger = get_logger(__name__)
    
    # Flush output
    sys.stdout.flush()
    sys.stderr.flush()
    
    logger.info("=== PK/PD Modeling ===")
    logger.info(f"Run name: {config.run_name}")
    
    # Encoder information
    if config.encoder_pk or config.encoder_pd:
        pk_encoder = config.encoder_pk or config.encoder
        pd_encoder = config.encoder_pd or config.encoder
        logger.info(    f"Mode: {config.mode} | PK Encoder: {pk_encoder} | PD Encoder: {pd_encoder} | Epochs: {config.epochs}")
    else:
        logger.info(f"Mode: {config.mode} | Encoder: {config.encoder} | Epochs: {config.epochs}")
    
    logger.info(f"Batch size: {config.batch_size} | Learning rate: {config.learning_rate}")
    
    device = get_device(config.device_id)
    logger.info(f"Device: {device}")
    
    # === 1. Data loading ===
    logger.info(f"\n=== 1. Data loading ===")
    logger.info(f"CSV file: {config.csv_path}")
    
    df_all, df_obs, df_dose = load_estdata(config.csv_path)
    logger.info(f"Data loading completed - Total: {df_all.shape}, Observed: {df_obs.shape}, Dose: {df_dose.shape}")
    
    # === 2. Data preprocessing ===
    logger.info(f"\n=== 2. Data preprocessing ===")
    
    if config.use_feature_engineering:
        logger.info("Feature engineering applied")
        df_final, pk_features, pd_features = use_feature_engineering(
            df_obs=df_obs, df_dose=df_dose,
            use_perkg=config.perkg,
            target="dv",
            use_pd_baseline_for_dv=True,
            allow_future_dose=config.allow_future_dose,
            time_windows=config.time_windows
        )
        logger.info(f"Feature engineering completed - PK features: {len(pk_features)}, PD features: {len(pd_features)}")
    else:
        logger.info("Basic preprocessing applied")
        df_final = df_obs.copy()
        feature_cols = [col for col in df_final.columns 
                      if col not in ['ID', 'TIME', 'DV', 'DVID']]
        pk_features = feature_cols
        pd_features = feature_cols
    
    logger.info(f"Preprocessing completed - Final data shape: {df_final.shape}")
    
    # === 3. Data splitting ===
    logger.info(f"\n=== 3. Data splitting ===")
    
    # PK/PD data split
    pk_df = df_final[df_final["DVID"] == 1].copy()
    pd_df = df_final[df_final["DVID"] == 2].copy()
    logger.info(f"PK data: {pk_df.shape}, PD data: {pd_df.shape}")
    
    # Data splitting
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
    logger.info(f"Data splitting completed - Strategy: {config.split_strategy}")
    
    splits = {
        'pk': pk_splits,
        'pd': pd_splits,
        'global': global_splits,
        'pk_features': pk_features,
        'pd_features': pd_features
    }
    
    # === 4. Data loader creation ===
    logger.info(f"\n=== 4. Data loader creation ===")
    
    pin_mem = get_device().type == 'cuda'
    num_workers = min(4, os.cpu_count() or 1)
    
    # PK data loader
    pk_scaler, train_loader_pk, valid_loader_pk, test_loader_pk = scaling_and_prepare_loader(
        splits['pk'], pk_features, 
        batch_size=config.batch_size, lambda_ctr=0.0,
        target_col="DV", num_workers=num_workers, pin_memory=pin_mem, drop_last_train=True
    )
    
    # PD data loader
    pd_scaler, train_loader_pd, valid_loader_pd, test_loader_pd = scaling_and_prepare_loader(
        splits['pd'], pd_features, 
        batch_size=config.batch_size, lambda_ctr=0.0,
        target_col="DV", num_workers=num_workers, pin_memory=pin_mem, drop_last_train=True
    )
    
    loaders = {
        "train_pk": train_loader_pk, "val_pk": valid_loader_pk, "test_pk": test_loader_pk,
        "train_pd": train_loader_pd, "val_pd": valid_loader_pd, "test_pd": test_loader_pd,
    }
    
    logger.info("Data loader creation completed")
    logger.info(f"PK - Train: {len(train_loader_pk.dataset)}, Val: {len(valid_loader_pk.dataset)}, Test: {len(test_loader_pk.dataset)}")
    logger.info(f"PD - Train: {len(train_loader_pd.dataset)}, Val: {len(valid_loader_pd.dataset)}, Test: {len(test_loader_pd.dataset)}")
    
    # === 5. Model creation ===
    logger.info(f"\n=== 5. Model creation ===")
    
    model = create_model(config, loaders, pk_features, pd_features)
    logger.info(f"Model creation completed - Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # === 6. Trainer creation and training ===
    logger.info(f"\n=== 6. Training start ===")
    
    trainer = create_trainer(config, model, loaders, device)
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"Training completed - Time: {training_time:.2f} seconds")
    logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
    logger.info(f"Best PK RMSE: {results['best_pk_rmse']:.6f}")
    logger.info(f"Best PD RMSE: {results['best_pd_rmse']:.6f}")
    
    # === 7. Results saving ===
    logger.info(f"\n=== 7. Results saving ===")
    
    # Create hierarchical directory structure for all outputs
    # Structure: runs/{run_name}/{mode}/{encoder}/s{seed}/
    if config.encoder_pk or config.encoder_pd:
        pk_encoder = config.encoder_pk or config.encoder
        pd_encoder = config.encoder_pd or config.encoder
        encoder_name = f"{pk_encoder}-{pd_encoder}"
    else:
        encoder_name = config.encoder
    
    run_dir = f"{config.output_dir}/runs/{config.run_name}/{config.mode}/{encoder_name}/s{config.random_state}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Model saving
    model_path = f"{run_dir}/model.pth"
    # Temporarily override the trainer's save directory
    from pathlib import Path
    original_save_dir = trainer.model_save_directory
    trainer.model_save_directory = Path(run_dir)
    trainer.save_model("model.pth")
    trainer.model_save_directory = original_save_dir  # Restore original
    logger.info(f"Model saved: {model_path}")
    
    # Configuration saving
    config_path = f"{run_dir}/config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"Configuration saved: {config_path}")
    
    # Scaler saving
    scaler_path = f"{run_dir}/scalers.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(pk_scaler, f)
        pickle.dump(pd_scaler, f)
    logger.info(f"Scaler saved: {scaler_path}")
    
    # Results saving
    results_path = f"{run_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved: {results_path}")
    
    # Create symlink for backward compatibility
    try:
        # Create symlinks in the old structure for easy access
        old_model_path = f"{config.output_dir}/models/{config.mode}/{encoder_name}/s{config.random_state}/{config.run_name}.pth"
        os.makedirs(os.path.dirname(old_model_path), exist_ok=True)
        if not os.path.exists(old_model_path):
            os.symlink(os.path.abspath(model_path), old_model_path)
            logger.info(f"Symlink created: {old_model_path} -> {model_path}")
    except Exception as e:
        logger.warning(f"Could not create symlink: {e}")
    
    logger.info(f"All outputs saved to: {run_dir}")
    
    logger.info("=== Execution completed ===")
    return 0


if __name__ == "__main__":
    exit(main())
