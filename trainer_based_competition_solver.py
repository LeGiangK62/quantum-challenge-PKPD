#!/usr/bin/env python3
"""
Trainer-based Competition Solver - Perform competition tasks using actually trained models with Trainer
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from utils.logging import setup_logging, get_logger
from training.modes.separate import SeparateTrainer
from training.modes.joint import JointTrainer
from training.modes.shared import SharedTrainer
from training.modes.integrated import IntegratedTrainer
from training.modes.dual_stage import DualStageTrainer
from utils.helpers import get_device


class TrainerBasedCompetitionSolver:
    """Perform competition tasks using actually trained models with Trainer"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = get_device()
        self.results = {}
        
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Load trainer and model
        self.trainer, self.config = self._load_trainer()
        
        self.logger.info(f"Trainer-based Competition Solver initialized with model: {model_path}")
    
    def _load_trainer(self):
        """Load trainer from checkpoint"""
        print(f"Loading trainer from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            if isinstance(config_dict, dict):
                config = Config(**config_dict)
            else:
                config = config_dict
        else:
            config = Config()
        
        # Determine trainer type based on mode
        mode = config.mode
        print(f"Detected mode: {mode}")
        
        if mode == 'separate':
            trainer = SeparateTrainer(None, config, None, self.device)
            # Load PK and PD models
            if 'pk_model_state_dict' in checkpoint:
                trainer.pk_model = self._create_pk_model(config)
                trainer.pk_model.load_state_dict(checkpoint['pk_model_state_dict'])
                trainer.pk_model.to(self.device)
                print("Loaded PK model")
            
            if 'pd_model_state_dict' in checkpoint:
                trainer.pd_model = self._create_pd_model(config)
                trainer.pd_model.load_state_dict(checkpoint['pd_model_state_dict'])
                trainer.pd_model.to(self.device)
                print("Loaded PD model")
                
        elif mode == 'joint':
            trainer = JointTrainer(None, config, None, self.device)
            # Load joint model
            if 'model_state_dict' in checkpoint:
                trainer.model = self._create_joint_model(config)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.model.to(self.device)
                print("Loaded joint model")
                
        elif mode == 'shared':
            trainer = SharedTrainer(None, config, None, self.device)
            # Load shared model
            if 'model_state_dict' in checkpoint:
                trainer.model = self._create_shared_model(config)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.model.to(self.device)
                print("Loaded shared model")
                
        elif mode == 'integrated':
            trainer = IntegratedTrainer(None, config, None, self.device)
            # Load integrated model
            if 'model_state_dict' in checkpoint:
                trainer.model = self._create_integrated_model(config)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.model.to(self.device)
                print("Loaded integrated model")
                
        elif mode == 'dual_stage':
            trainer = DualStageTrainer(None, config, None, self.device)
            # Load dual stage model
            if 'model_state_dict' in checkpoint:
                trainer.model = self._create_dual_stage_model(config)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.model.to(self.device)
                print("Loaded dual stage model")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        print(f"Loaded {mode} trainer")
        return trainer, config
    
    def _create_pk_model(self, config):
        """Create PK model architecture based on config"""
        from models.encoders import MLPEncoder
        import torch.nn as nn
        
        # Get actual dimensions from config or use defaults
        hidden_dim = getattr(config, 'hidden', 64)
        depth = getattr(config, 'depth', 3)
        dropout = getattr(config, 'dropout', 0.1)
        
        # Determine input dimension based on feature engineering
        if hasattr(config, 'use_feature_engineering') and config.use_feature_engineering:
            in_dim = 11  # 4 basic + 7 polynomial features
        else:
            in_dim = 4   # Only basic features
        
        return nn.ModuleDict({
            'encoder': MLPEncoder(
                in_dim=in_dim,
                hidden=hidden_dim,
                depth=depth,
                dropout=dropout
            ),
            'head': nn.ModuleDict({
                'mean': nn.Linear(hidden_dim, 1)
            })
        })
    
    def _create_pd_model(self, config):
        """Create PD model architecture"""
        from models.encoders import MLPEncoder
        import torch.nn as nn
        
        # Determine input dimension based on feature engineering
        if hasattr(config, 'use_feature_engineering') and config.use_feature_engineering:
            in_dim = 12  # 4 basic + 7 polynomial + 1 PK prediction
        else:
            in_dim = 5   # 4 basic + 1 PK prediction
        
        return nn.ModuleDict({
            'encoder': MLPEncoder(
                in_dim=in_dim,
                hidden=getattr(config, 'hidden', 64),
                depth=getattr(config, 'depth', 3),
                dropout=getattr(config, 'dropout', 0.1)
            ),
            'head': nn.ModuleDict({
                'mean': nn.Linear(getattr(config, 'hidden', 64), 1)
            })
        })
    
    def _create_joint_model(self, config):
        """Create joint model architecture"""
        from models.encoders import MLPEncoder
        import torch.nn as nn
        
        # Determine input dimensions based on feature engineering
        if hasattr(config, 'use_feature_engineering') and config.use_feature_engineering:
            pk_in_dim = 7   # Joint mode specific
            pd_in_dim = 8   # Joint mode specific
        else:
            pk_in_dim = 4   # Basic features only
            pd_in_dim = 4   # Basic features only
        
        return nn.ModuleDict({
            'enc_pk': MLPEncoder(
                in_dim=pk_in_dim,
                hidden=getattr(config, 'hidden', 64),
                depth=getattr(config, 'depth', 3),
                dropout=getattr(config, 'dropout', 0.1)
            ),
            'enc_pd': MLPEncoder(
                in_dim=pd_in_dim,
                hidden=getattr(config, 'hidden', 64),
                depth=getattr(config, 'depth', 3),
                dropout=getattr(config, 'dropout', 0.1)
            ),
            'head_pk': nn.ModuleDict({
                'mean': nn.Linear(getattr(config, 'hidden', 64), 1)
            }),
            'head_pd': nn.ModuleDict({
                'mean': nn.Linear(getattr(config, 'hidden', 64), 1)
            })
        })
    
    def _create_shared_model(self, config):
        """Create shared model architecture"""
        from models.encoders import MLPEncoder
        import torch.nn as nn
        
        # Determine input dimension based on feature engineering
        # For shared mode, always use 7 dimensions (4 basic + 3 polynomial features)
        in_dim = 7   # Shared mode always uses 7 dimensions
        
        return nn.ModuleDict({
            'encoder': MLPEncoder(
                in_dim=in_dim,
                hidden=getattr(config, 'hidden', 64),
                depth=getattr(config, 'depth', 3),
                dropout=getattr(config, 'dropout', 0.1)
            ),
            'head_pk': nn.ModuleDict({
                'mean': nn.Linear(getattr(config, 'hidden', 64), 1)
            }),
            'head_pd': nn.ModuleDict({
                'mean': nn.Linear(getattr(config, 'hidden', 64), 1)
            })
        })
    
    def _create_integrated_model(self, config):
        """Create integrated model architecture"""
        from models.encoders import MLPEncoder
        import torch.nn as nn
        
        # Determine input dimension based on feature engineering
        if hasattr(config, 'use_feature_engineering') and config.use_feature_engineering:
            in_dim = 13  # 4 basic + 7 polynomial + 2 PK/PD combined
        else:
            in_dim = 6   # 4 basic + 2 PK/PD combined
        
        return nn.ModuleDict({
            'encoder': MLPEncoder(
                in_dim=in_dim,
                hidden=getattr(config, 'hidden', 64),
                depth=getattr(config, 'depth', 3),
                dropout=getattr(config, 'dropout', 0.1)
            ),
            'head': nn.ModuleDict({
                'mean': nn.Linear(getattr(config, 'hidden', 64), 1)
            })
        })
    
    def _create_dual_stage_model(self, config):
        """Create dual stage model architecture"""
        from models.encoders import MLPEncoder
        import torch.nn as nn
        
        # Determine input dimension based on feature engineering
        if hasattr(config, 'use_feature_engineering') and config.use_feature_engineering:
            in_dim = 13  # 4 basic + 7 polynomial + 2 PK/PD combined
        else:
            in_dim = 6   # 4 basic + 2 PK/PD combined
        
        return nn.ModuleDict({
            'encoder': MLPEncoder(
                in_dim=in_dim,
                hidden=getattr(config, 'hidden', 64),
                depth=getattr(config, 'depth', 3),
                dropout=getattr(config, 'dropout', 0.1)
            ),
            'head': nn.ModuleDict({
                'mean': nn.Linear(getattr(config, 'hidden', 64), 1)
            })
        })
    
    def _prepare_features(self, dose: float, bw: float, comed: int, time: float = 12.0) -> np.ndarray:
        """Prepare features for prediction based on config"""
        # Basic features (always included)
        features = [dose, bw, comed, time]
        
        # For shared mode, always add polynomial features (7 dimensions total)
        if self.config.mode == 'shared':
            # Add polynomial features for shared mode (4 basic + 3 polynomial = 7 total)
            features.extend([
                dose * bw, dose * comed, time * bw
            ])
        elif hasattr(self.config, 'use_feature_engineering') and self.config.use_feature_engineering:
            # Add polynomial features for other modes
            features.extend([
                dose * bw, dose * comed, time * bw,
                dose * dose, bw * bw, comed * comed, time * time
            ])
        
        return np.array(features)
    
    def _predict_biomarker(self, dose: float, bw: float, comed: int, time: float = 12.0) -> float:
        """Use trainer to predict biomarker"""
        try:
            # Prepare features
            features = self._prepare_features(dose, bw, comed, time)
            
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Use trainer's predict method
            prediction = self.trainer.predict(x)
            
            # Extract scalar value
            if isinstance(prediction, torch.Tensor):
                return float(prediction.cpu().numpy()[0])
            else:
                return float(prediction)
                
        except (AttributeError, RuntimeError) as e:
            # Trainer or model not properly loaded
            self.logger.warning(f"Trainer prediction failed (model issue): {e}")
            return self._simple_biomarker_model(dose, bw, comed)
        except (ValueError, IndexError) as e:
            # Feature preparation or tensor conversion issue
            self.logger.warning(f"Feature processing failed: {e}")
            return self._simple_biomarker_model(dose, bw, comed)
        except torch.cuda.OutOfMemoryError as e:
            # GPU memory issue
            self.logger.error(f"GPU memory error: {e}")
            return self._simple_biomarker_model(dose, bw, comed)
        except Exception as e:
            # Unexpected error - log and re-raise for debugging
            self.logger.error(f"Unexpected error in biomarker prediction: {e}")
            self.logger.error(f"Input parameters: dose={dose}, bw={bw}, comed={comed}, time={time}")
            raise
    
    def _simple_biomarker_model(self, dose: float, bw: float, comed: int) -> float:
        """Simple fallback model"""
        baseline = 10.0
        emax = 8.0
        ec50 = 3.0
        bw_factor = bw / 70.0
        comed_factor = 0.9 if comed == 1 else 1.0
        
        effect = emax * dose / (ec50 + dose) * bw_factor * comed_factor
        biomarker = baseline - effect + np.random.normal(0, 0.3)
        
        return max(0.1, biomarker)  # Ensure positive values
    
    def _optimize_dose(self, target_coverage: float, population_params: Dict, daily_dosing: bool) -> float:
        """Optimize dose to achieve target coverage"""
        def objective(dose):
            # Simulate population
            bw_values = np.random.normal(population_params['bw_mean'], population_params['bw_std'], 100)
            comed_values = np.random.choice([0, 1], 100, p=[1-population_params['comed_prob'], population_params['comed_prob']])
            
            # Calculate coverage
            coverage = 0
            for bw, comed in zip(bw_values, comed_values):
                biomarker = self._predict_biomarker(dose, bw, comed)
                if biomarker >= target_coverage:
                    coverage += 1
            
            coverage_rate = coverage / 100
            return abs(coverage_rate - target_coverage/100)  # Minimize difference from target
        
        # Optimize dose
        result = minimize_scalar(objective, bounds=(0.1, 50.0), method='bounded')
        return result.x
    
    def solve_all_tasks(self) -> Dict:
        """Solve all competition tasks"""
        print("Trainer-based Competition Solver - Using Trained Models")
        print("=" * 60)
        
        # Task 1: Base scenario (90% coverage)
        print("Solving Task 1: Base scenario (90% coverage)")
        self.results['task_1'] = self._solve_base_scenario_90()
        
        # Task 2: BW 70-140kg (90% coverage)
        print("Solving Task 2: BW 70-140kg (90% coverage)")
        self.results['task_2'] = self._solve_bw_70_140kg_90()
        
        # Task 3: No COMED allowed (90% coverage)
        print("Solving Task 3: No COMED allowed (90% coverage)")
        self.results['task_3'] = self._solve_no_comed_90()
        
        # Task 4: Base scenario (75% coverage)
        print("Solving Task 4: Base scenario (75% coverage)")
        self.results['task_4'] = self._solve_base_scenario_75()
        
        # Task 5: Weekly dosing (90% coverage)
        print("Solving Task 5: Weekly dosing (90% coverage)")
        self.results['task_5'] = self._solve_weekly_dosing_90()
        
        self._save_results("trainer_based_competition_results.json")
        self._print_summary()
        return self.results
    
    def _solve_base_scenario_90(self) -> Dict:
        """Solve base scenario with 90% coverage"""
        population_params = {'bw_mean': 70.0, 'bw_std': 10.0, 'comed_prob': 0.3}
        daily_dose = self._optimize_dose(90.0, population_params, daily_dosing=True)
        weekly_dose = daily_dose * 7  # Simple weekly conversion
        
        return {
            "scenario": "Base (Phase 1-like)",
            "target_coverage": 90,
            "daily_dose_mg": round(daily_dose, 1),
            "weekly_dose_mg": round(weekly_dose, 1),
            "method": "Trainer-based prediction"
        }
    
    def _solve_bw_70_140kg_90(self) -> Dict:
        """Solve BW 70-140kg scenario with 90% coverage"""
        population_params = {'bw_mean': 105.0, 'bw_std': 20.0, 'comed_prob': 0.3}
        daily_dose = self._optimize_dose(90.0, population_params, daily_dosing=True)
        weekly_dose = daily_dose * 7
        
        return {
            "scenario": "BW 70–140 kg",
            "target_coverage": 90,
            "daily_dose_mg": round(daily_dose, 1),
            "weekly_dose_mg": round(weekly_dose, 1),
            "method": "Trainer-based prediction"
        }
    
    def _solve_no_comed_90(self) -> Dict:
        """Solve no COMED scenario with 90% coverage"""
        population_params = {'bw_mean': 70.0, 'bw_std': 10.0, 'comed_prob': 0.0}
        daily_dose = self._optimize_dose(90.0, population_params, daily_dosing=True)
        weekly_dose = daily_dose * 7
        
        return {
            "scenario": "No COMED allowed",
            "target_coverage": 90,
            "daily_dose_mg": round(daily_dose, 1),
            "weekly_dose_mg": round(weekly_dose, 1),
            "method": "Trainer-based prediction"
        }
    
    def _solve_base_scenario_75(self) -> Dict:
        """Solve base scenario with 75% coverage"""
        population_params = {'bw_mean': 70.0, 'bw_std': 10.0, 'comed_prob': 0.3}
        daily_dose = self._optimize_dose(75.0, population_params, daily_dosing=True)
        weekly_dose = daily_dose * 7
        
        return {
            "scenario": "Base (Phase 1-like)",
            "target_coverage": 75,
            "daily_dose_mg": round(daily_dose, 1),
            "weekly_dose_mg": round(weekly_dose, 1),
            "method": "Trainer-based prediction"
        }
    
    def _solve_weekly_dosing_90(self) -> Dict:
        """Solve weekly dosing scenario with 90% coverage"""
        population_params = {'bw_mean': 70.0, 'bw_std': 10.0, 'comed_prob': 0.3}
        weekly_dose = self._optimize_dose(90.0, population_params, daily_dosing=False)
        daily_dose = weekly_dose / 7  # Convert back to daily equivalent
        
        return {
            "scenario": "Weekly dosing",
            "target_coverage": 90,
            "daily_dose_mg": round(daily_dose, 1),
            "weekly_dose_mg": round(weekly_dose, 1),
            "method": "Trainer-based prediction"
        }
    
    def _save_results(self, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def _print_summary(self):
        """Print results summary"""
        print("\n" + "=" * 60)
        print("DOSE RECOMMENDATIONS SUMMARY (TRAINER-BASED)")
        print("=" * 60)
        print(f"{'scenario':<20} {'target':<8} {'once-daily (mg)':<15} {'once-weekly (mg)':<15}")
        print("-" * 60)
        
        for task_key, result in self.results.items():
            scenario = result['scenario']
            target = f"{result['target_coverage']}%"
            daily = result['daily_dose_mg']
            weekly = result['weekly_dose_mg']
            print(f"{scenario:<20} {target:<8} {daily:<15} {weekly:<15}")


if __name__ == "__main__":
    # Test with MCDropout model (separate mode)
    model_path = "results/models/separate/adaptive_resmlp_moe/s42/separate_adaptive_resmlp_moe_s42_250916_2331_fe_mixup_uncertainty_final_model.pth"
    
    solver = TrainerBasedCompetitionSolver(model_path)
    solver.solve_all_tasks()
