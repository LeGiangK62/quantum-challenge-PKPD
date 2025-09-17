"""
Data preprocessing module
"""

from typing import Tuple, List
from utils.logging import get_logger


class Preprocessor:
    """Class responsible for data preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        self.pk_features = None
        self.pd_features = None
    
    def process(self, df_obs, df_dose) -> Tuple[any, List[str], List[str]]:
        """Data preprocessing"""
        self.logger.info("Data preprocessing started")
        
        if self.config.use_feature_engineering:
            self.logger.info("Feature engineering applied")
            try:
                from .loaders import use_feature_engineering
                df_final, pk_features, pd_features = use_feature_engineering(
                    df_obs=df_obs, df_dose=df_dose,
                    use_perkg=self.config.perkg,
                    target="dv",
                    use_pd_baseline_for_dv=True,
                    allow_future_dose=self.config.allow_future_dose
                )
                self.logger.info(f"Feature engineering completed. PK features: {len(pk_features)}, PD features: {len(pd_features)}")
            except Exception as e:
                self.logger.warning(f"Feature engineering failed: {e}. Using basic preprocessing")
                df_final = df_obs.copy()
                feature_cols = [col for col in df_final.columns 
                              if col not in ['ID', 'TIME', 'DV', 'DVID']]
                pk_features = feature_cols
                pd_features = feature_cols
        else:
            self.logger.info("Basic preprocessing applied")
            df_final = df_obs.copy()
            # Exclude non-feature columns
            feature_cols = [col for col in df_final.columns 
                          if col not in ['ID', 'TIME', 'DV', 'DVID']]
            pk_features = feature_cols
            pd_features = feature_cols
        
        self.pk_features = pk_features
        self.pd_features = pd_features
        
        self.logger.info(f"Preprocessing completed - Final data shape: {df_final.shape}")
        self.logger.info(f"PK features: {len(pk_features)}, PD features: {len(pd_features)}")
        
        return df_final, pk_features, pd_features
    
    def get_features(self) -> Tuple[List[str], List[str]]:
        """Return feature lists"""
        if self.pk_features is None or self.pd_features is None:
            raise RuntimeError("Preprocessing has not been executed yet. Call process() first.")
        return self.pk_features, self.pd_features
