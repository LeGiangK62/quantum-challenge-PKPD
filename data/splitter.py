"""
Data splitting module
"""

from typing import Dict, Any, List
from utils.logging import get_logger


class DataSplitter:
    """Class responsible for data splitting"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
    
    def split(self, df_final, df_dose, pk_features: List[str], pd_features: List[str]) -> Dict[str, Any]:
        """Split data"""
        self.logger.info("Data splitting started")
        
        # Separate PK/PD data
        pk_df = df_final[df_final["DVID"] == 1].copy()
        pd_df = df_final[df_final["DVID"] == 2].copy()
        
        self.logger.info(f"PK data: {pk_df.shape}, PD data: {pd_df.shape}")
        
        try:
            # Use the proper prepare_for_split function from splits.py
            from .splits import prepare_for_split
            pk_splits, pd_splits, global_splits, _ = prepare_for_split(
                df_final=df_final, df_dose=df_dose,
                pk_df=pk_df, pd_df=pd_df,
                split_strategy=self.config.split_strategy,
                test_size=self.config.test_size,
                val_size=self.config.val_size,
                random_state=self.config.random_state,
                dose_bins=4,
                id_universe="union",
                verbose=True,
            )
            self.logger.info("Advanced splitting completed successfully")
        except Exception as e:
            self.logger.warning(f"Advanced splitting failed: {e}. Using simple splitting")
            import numpy as np
            from sklearn.model_selection import train_test_split
            
            # Simple train/val/test split
            pk_train, pk_temp = train_test_split(pk_df, test_size=0.3, random_state=42)
            pk_val, pk_test = train_test_split(pk_temp, test_size=0.5, random_state=42)
            
            pd_train, pd_temp = train_test_split(pd_df, test_size=0.3, random_state=42)
            pd_val, pd_test = train_test_split(pd_temp, test_size=0.5, random_state=42)
            
            pk_splits = {'train': pk_train, 'val': pk_val, 'test': pk_test}
            pd_splits = {'train': pd_train, 'val': pd_val, 'test': pd_test}
            global_splits = None
        
        splits = {
            'pk': pk_splits,
            'pd': pd_splits,
            'global': global_splits,
            'pk_features': pk_features,
            'pd_features': pd_features
        }
        
        self.logger.info("Data splitting completed")
        return splits
