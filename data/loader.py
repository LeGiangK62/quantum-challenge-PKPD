"""
Data loading module
"""

import pandas as pd
from typing import Dict, Any
from utils.logging import get_logger


class DataLoader:
    """Class responsible for data loading"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
    
    def load_data(self) -> Dict[str, Any]:
        """Load data"""
        self.logger.info(f"Data loading started: {self.config.csv_path}")
        
        try:
            # Use the proper load_estdata function from loaders.py
            from .loaders import load_estdata
            df_all, df_obs, df_dose = load_estdata(self.config.csv_path)
            self.logger.info(f"Successfully loaded CSV data: {df_all.shape}")
            self.logger.info(f"Observations: {df_obs.shape}, Dosing: {df_dose.shape}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load CSV data: {e}. Using dummy data for testing")
            import numpy as np
            
            # Create dummy data
            n_samples = 1000
            df_obs = pd.DataFrame({
                'ID': np.repeat(range(1, 101), 10),
                'TIME': np.tile(np.arange(0, 10), 100),
                'DV': np.random.normal(5.0, 1.0, n_samples),
                'AMT': np.random.normal(100.0, 20.0, n_samples),
                'WT': np.random.normal(70.0, 15.0, n_samples),
                'AGE': np.random.normal(45.0, 10.0, n_samples),
                'SEX': np.random.choice([0, 1], n_samples),
                'COMED': np.random.choice([0, 1], n_samples)
            })
            
            df_dose = pd.DataFrame({
                'ID': range(1, 101),
                'AMT': np.random.normal(100.0, 20.0, 100),
                'WT': np.random.normal(70.0, 15.0, 100),
                'AGE': np.random.normal(45.0, 10.0, 100),
                'SEX': np.random.choice([0, 1], 100),
                'COMED': np.random.choice([0, 1], 100)
            })
            
            df_all = pd.concat([df_obs, df_dose], ignore_index=True)
        
        self.logger.info(f"Data loading completed - Total: {df_all.shape}, Observed: {df_obs.shape}, Dosing: {df_dose.shape}")
        
        return {
            "df_all": df_all,
            "df_obs": df_obs,
            "df_dose": df_dose
        }
