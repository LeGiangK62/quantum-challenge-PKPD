"""
Configuration management module
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional

def parse_time_windows(time_windows_str: str) -> List[int]:
    """Parse comma-separated time windows string into list of integers"""
    if not time_windows_str:
        return None
    try:
        return [int(x.strip()) for x in time_windows_str.split(',') if x.strip()]
    except ValueError:
        raise ValueError(f"Invalid time windows format: {time_windows_str}. Use comma-separated integers (e.g., '24,48,72,96,120,144,168')")


@dataclass
class Config:
    """Class to manage all configurations"""
    
    # Basic settings
    mode: str = "separate" # "separate", "joint", "shared", "dual_stage", "integrated", "two_stage_shared"
    csv_path: str = "data/EstData.csv"
    
    # Training settings
    epochs: int = 3000
    batch_size: int = 32 
    learning_rate: float = 1e-3
    patience: int = 300
    
    # Model settings
    encoder: str = "mlp"
    encoder_pk: str = None  # PK 전용 encoder (None이면 encoder 사용)
    encoder_pd: str = None  # PD 전용 encoder (None이면 encoder 사용)
    head_pk: str = "mse"
    head_pd: str = "mse"
    
    # Model hyperparameters
    hidden: int = 64
    depth: int = 3
    dropout: float = 0.1
    time_pool: bool = False
    res_blocks: int = 3
    gauss_tied: bool = False
    emax_conditional: bool = False
    emax_use_projC: bool = False
    
    # MoE settings
    num_experts: int = 4
    top_k: int = 2
    
    # Dual stage settings
    front_dims: List[int] = None
    back_dims: List[int] = None
        
    # Data preprocessing
    use_feature_engineering: bool = False
    perkg: bool = False
    allow_future_dose: bool = False
    
    # Time window settings for feature engineering
    time_windows: List[int] = None  # Custom time windows (default: [24, 48, 72, 96, 120, 144, 168] for EstData.csv)
    auto_time_windows: bool = True  # Auto-determine optimal windows
    
    # Data splitting
    split_strategy: str = "stratify_dose_even" # "stratify_dose_even", "stratify_dose_even_no_placebo_test", "leave_one_dose_out", "random_subject", "only_bw_range"
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    
    # Augmentation and normalization
    use_mixup: bool = False
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.5
    
    # Contrastive learning
    lambda_contrast: float = 0.0
    temperature: float = 0.2
    
    # Uncertainty quantification settings
    use_uncertainty: bool = False
    uncertainty_method: str = "monte_carlo"  # "monte_carlo", "ensemble", "bayesian",
    uncertainty_samples: int = 100
    uncertainty_dropout: float = 0.1
    uncertainty_weight: float = 1.0
    uncertainty_type: str = "total"  # "aleatoric", "epistemic", "total"
    uncertainty_penalty: float = 0.01
    
    # Active Learning settings
    use_active_learning: bool = False
    active_learning_strategy: str = "uncertainty"  # "uncertainty", "diversity", "hybrid"
    active_learning_budget: int = 100
    active_learning_growth_rate: float = 1.2
    active_learning_diversity_weight: float = 0.1
    
    # Meta-Learning settings
    use_meta_learning: bool = False
    meta_learning_type: str = "basic"  # "basic", "patient_group", "few_shot", "adaptive"
    meta_lr: float = 0.01
    adaptation_steps: int = 5
    n_patient_groups: int = 5
    n_way: int = 5
    k_shot: int = 3
    
    # Output settings
    output_dir: str = "results"
    run_name: Optional[str] = None
    verbose: bool = False
    
    # Device settings
    device_id: int = 0
    
    # Competition settings
    competition_mode: bool = False
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.front_dims is None:
            self.front_dims = [128, 64]
        if self.back_dims is None:
            self.back_dims = [64, 32]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PK/PD Modeling System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Basic settings
    parser.add_argument("--mode", choices=["separate", "joint", "shared", "dual_stage", "integrated", "two_stage_shared"], 
                       default="separate", help="Training mode")
    parser.add_argument("--csv", default="data/EstData.csv", help="Data CSV file path")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    
    # Model settings
    parser.add_argument("--encoder", 
                    choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"], 
                    default="mlp", help="Default encoder type")
    parser.add_argument("--encoder_pk", 
                    choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"], 
                    default=None, help="PK-specific encoder type (overrides --encoder). Use different encoders for PK/PD to optimize for their different characteristics.")
    parser.add_argument("--encoder_pd", 
                    choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"], 
                    default=None, help="PD-specific encoder type (overrides --encoder). Use different encoders for PK/PD to optimize for their different characteristics.")
    parser.add_argument("--head_pk", choices=["mse", "gauss", "poisson"], 
                       default="mse", help="PK head type")
    parser.add_argument("--head_pd", choices=["mse", "gauss", "poisson", "emax"], 
                       default="mse", help="PD head type")
    
    # Model hyperparameters
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=3, help="Network depth")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")
    
    # MoE settings
    parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top experts to use")
    
    # Dual stage settings
    parser.add_argument("--front_dims", type=int, nargs="+", default=[128, 64], 
                       help="Front encoder hidden dimensions")
    parser.add_argument("--back_dims", type=int, nargs="+", default=[64, 32], 
                       help="Back encoder hidden dimensions")
    
    # Data preprocessing
    parser.add_argument("--use_fe", action="store_true", help="Feature engineering")
    parser.add_argument("--perkg", action="store_true", help="Per kg dose")
    parser.add_argument("--allow_future_dose", action="store_true", help="Allow future dose information")
    
    # Time window settings for feature engineering
    parser.add_argument("--time_windows", type=str, default=None,
                       help="Custom time windows for dose history features (comma-separated, e.g., '24,48,72,96,120,144,168')")
    parser.add_argument("--auto_time_windows", action="store_true", default=True,
                       help="Automatically determine optimal time windows based on dose patterns")
    
    # Data splitting
    parser.add_argument("--split_strategy", default="stratify_dose_even", 
                       help="Data splitting strategy: stratify_dose_even, stratify_dose_even_no_placebo_test, leave_one_dose_out, random_subject, only_bw_range")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # Augmentation and normalization
    parser.add_argument("--use_mixup", action="store_true", help="Use mixup augmentation")
    parser.add_argument("--mixup_prob", type=float, default=0.0, help="Mixup probability")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha parameter")
    
    # Contrastive learning
    parser.add_argument("--lambda_contrast", type=float, default=0.0, help="Contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=0.2, help="Contrastive learning temperature")
    
    # Uncertainty quantification settings
    parser.add_argument("--use_uncertainty", action="store_true", help="Enable uncertainty quantification")
    parser.add_argument("--uncertainty_method", type=str, default="monte_carlo", 
                       choices=["monte_carlo", "ensemble", "bayesian"],
                       help="Uncertainty estimation method")
    parser.add_argument("--uncertainty_samples", type=int, default=100, 
                       help="Number of Monte Carlo samples for uncertainty estimation")
    parser.add_argument("--uncertainty_dropout", type=float, default=0.1, 
                       help="Dropout rate for uncertainty estimation")
    parser.add_argument("--uncertainty_weight", type=float, default=1.0, 
                       help="Weight for uncertainty loss")
    parser.add_argument("--uncertainty_type", type=str, default="total", 
                       choices=["aleatoric", "epistemic", "total"],
                       help="Type of uncertainty to estimate")
    parser.add_argument("--uncertainty_penalty", type=float, default=0.01, 
                       help="Penalty weight for high uncertainty")
    
    # Active Learning settings
    parser.add_argument("--use_active_learning", action="store_true", help="Enable active learning")
    parser.add_argument("--active_learning_strategy", type=str, default="uncertainty",
                       choices=["uncertainty", "diversity", "hybrid"],
                       help="Active learning selection strategy")
    parser.add_argument("--active_learning_budget", type=int, default=100,
                       help="Number of samples to select in each active learning iteration")
    parser.add_argument("--active_learning_growth_rate", type=float, default=1.2,
                       help="Growth rate for adaptive budget")
    parser.add_argument("--active_learning_diversity_weight", type=float, default=0.1,
                       help="Weight for diversity in hybrid selection")
    
    
    # Meta-Learning settings
    parser.add_argument("--use_meta_learning", action="store_true", help="Enable meta-learning")
    parser.add_argument("--meta_learning_type", type=str, default="basic",
                       choices=["basic", "patient_group", "few_shot", "adaptive"],
                       help="Type of meta-learning system")
    parser.add_argument("--meta_lr", type=float, default=0.01, help="Meta-learning rate")
    parser.add_argument("--adaptation_steps", type=int, default=5, help="Number of adaptation steps")
    parser.add_argument("--n_patient_groups", type=int, default=5, help="Number of patient groups")
    parser.add_argument("--n_way", type=int, default=5, help="Number of ways in few-shot learning")
    parser.add_argument("--k_shot", type=int, default=3, help="Number of shots in few-shot learning")
    
    # Output settings
    parser.add_argument("--out_dir", default="results", help="Result output directory")
    parser.add_argument("--run_name", help="Run name (automatically generated)")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    
    # Device settings
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID (0, 1, 2, ...)")
    
    # Competition settings
    parser.add_argument("--competition", action="store_true", help="Run competition task analysis")
    parser.add_argument("--model_path", help="Path to trained model for competition analysis")
    parser.add_argument("--config_path", help="Path to model configuration file")
    
    return parser


def parse_args() -> Config:
    """Parse command line arguments and create Config object"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create Config object
    config = Config(
        # For basic settings
        mode=args.mode, # Training mode
        csv_path=args.csv, # Data CSV file path

        # For training settings
        epochs=args.epochs, # Number of training epochs
        batch_size=args.batch_size, # Batch size
        learning_rate=args.lr, # Learning rate
        patience=args.patience, # Early stopping patience

        # For model settings
        encoder=args.encoder, # Default encoder type
        encoder_pk=args.encoder_pk, # PK-specific encoder type
        encoder_pd=args.encoder_pd, # PD-specific encoder type
        head_pk=args.head_pk, # PK head type
        head_pd=args.head_pd, # PD head type
        hidden=args.hidden, # Hidden dimension
        depth=args.depth, # Network depth
        dropout=args.dropout, # Dropout ratio

        # For MoE mode
        num_experts=args.num_experts, # Number of MoE experts
        top_k=args.top_k, # Number of top experts to use

        # For dual stage mode
        front_dims=args.front_dims, # Front encoder hidden dimensions
        back_dims=args.back_dims, # Back encoder hidden dimensions

        # For data preprocessing
        use_feature_engineering=args.use_fe, # Feature engineering
        perkg=args.perkg, # Per kg dose
        allow_future_dose=args.allow_future_dose, # Allow future dose information
        
        # Time window settings (default optimized for EstData.csv: 24h dosing)
        time_windows=parse_time_windows(args.time_windows) if args.time_windows else [24, 48, 72, 96, 120, 144, 168],
        auto_time_windows=args.auto_time_windows,
        split_strategy=args.split_strategy,
        test_size=args.test_size, # Test set size
        val_size=args.val_size, # Validation set size
        random_state=args.random_state, # Random seed

        # For augmentation and normalization
        use_mixup=args.use_mixup, # Use mixup augmentation
        mixup_prob=args.mixup_prob, # Mixup probability
        mixup_alpha=args.mixup_alpha, # Mixup alpha parameter

        # For contrastive learning
        lambda_contrast=args.lambda_contrast, # Contrastive loss weight
        temperature=args.temperature, # Contrastive learning temperature
        
        # For uncertainty quantification
        use_uncertainty=args.use_uncertainty, # Enable uncertainty quantification
        uncertainty_method=args.uncertainty_method, # Uncertainty estimation method
        uncertainty_samples=args.uncertainty_samples, # Number of Monte Carlo samples
        uncertainty_dropout=args.uncertainty_dropout, # Dropout rate for uncertainty
        uncertainty_weight=args.uncertainty_weight, # Uncertainty loss weight
        uncertainty_type=args.uncertainty_type, # Type of uncertainty
        uncertainty_penalty=args.uncertainty_penalty, # Uncertainty penalty weight
        
        output_dir=args.out_dir, # Result output directory  

        # For output settings
        run_name=args.run_name, # Run name (automatically generated)    
        verbose=args.verbose, # Detailed output
        
        # For device settings
        device_id=args.device_id, # CUDA device ID

        # For competition settings
        competition_mode=args.competition, # Run competition task analysis
        model_path=args.model_path, # Path to trained model for competition analysis
        config_path=args.config_path # Path to model configuration file
    )
    
    return config
