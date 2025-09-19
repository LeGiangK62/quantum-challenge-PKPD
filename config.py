"""
 Configuration
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Configuration"""
    
    # === Basic settings ===
    mode: str = "separate"  # "separate", "joint", "shared", "dual_stage", "integrated", "two_stage_shared"
    csv_path: str = "data/EstData.csv"
    
    # === Training settings ===
    epochs: int = 3000
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 300
    
    # === Model settings ===
    encoder: str = "mlp"  # "mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"
    encoder_pk: Optional[str] = None  # PK-specific encoder
    encoder_pd: Optional[str] = None  # PD-specific encoder
    head_pk: str = "mse"  # "mse", "gauss", "poisson"
    head_pd: str = "mse"  # "mse", "gauss", "poisson", "emax"
    
    # === Model hyperparameters ===
    hidden: int = 64
    depth: int = 3
    dropout: float = 0.1
    
    # === MoE settings ===
    num_experts: int = 4
    top_k: int = 2
    
    # === Data preprocessing ===
    use_feature_engineering: bool = False
    perkg: bool = False
    allow_future_dose: bool = False
    time_windows: List[int] = None
    
    # === Augmentation and regularization ===
    use_mixup: bool = False
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.5
    
    # === Contrastive learning ===
    lambda_contrast: float = 0.0
    temperature: float = 0.2
    
    # === Data splitting ===
    split_strategy: str = "stratify_dose_even"
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    
    # === Output settings ===
    output_dir: str = "results"
    run_name: Optional[str] = None
    verbose: bool = False
    device_id: int = 0
    
    def __post_init__(self):
        """Initialization after processing"""
        if self.time_windows is None:
            self.time_windows = [24, 48, 72, 96, 120, 144, 168]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="PK/PD Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # === Basic settings ===
    parser.add_argument("--mode", 
                       choices=["separate", "joint", "shared", "dual_stage", "integrated", "two_stage_shared"], 
                       default="separate", help="Training mode")
    parser.add_argument("--csv", default="data/EstData.csv", help="Data CSV file path")
    
    # === Training settings ===
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=300, help="Early stopping patience")
    
    # === Model settings ===
    parser.add_argument("--encoder", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"], 
                       default="mlp", help="Default encoder type")
    parser.add_argument("--encoder_pk", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"], 
                       default=None, help="PK-specific encoder type")
    parser.add_argument("--encoder_pd", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe"], 
                       default=None, help="PD-specific encoder type")
    parser.add_argument("--head_pk", choices=["mse", "gauss", "poisson"], 
                       default="mse", help="PK head type")
    parser.add_argument("--head_pd", choices=["mse", "gauss", "poisson", "emax"], 
                       default="mse", help="PD head type")
    
    # === Model hyperparameters ===
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=3, help="Network depth")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")
    
    # === MoE settings ===
    parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top experts to use")
    
    # === Data preprocessing ===
    parser.add_argument("--use_fe", action="store_true", help="Feature engineering")
    parser.add_argument("--perkg", action="store_true", help="Per kg dose")
    parser.add_argument("--allow_future_dose", action="store_true", help="Allow future dose information")
    parser.add_argument("--time_windows", type=str, default=None,
                       help="Time windows (comma-separated, e.g., '24,48,72,96,120,144,168')")
    
    # === Augmentation and regularization ===
    parser.add_argument("--use_mixup", action="store_true", help="Use mixup augmentation")
    parser.add_argument("--mixup_prob", type=float, default=0.0, help="Mixup probability")
    parser.add_argument("--mixup_alpha", type=float, default=0.5, help="Mixup alpha parameter")
    
    # === Contrastive learning ===
    parser.add_argument("--lambda_contrast", type=float, default=0.0, help="Contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=0.2, help="Contrastive learning temperature")
    
    # === Data splitting ===
    parser.add_argument("--split_strategy", default="stratify_dose_even", 
                       help="Data splitting strategy")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # === Output settings ===
    parser.add_argument("--out_dir", default="results", help="Result output directory")
    parser.add_argument("--run_name", help="Run name (auto-generated)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    
    return parser


def parse_time_windows(time_windows_str: str) -> List[int]:
    """Parse time windows string"""
    if not time_windows_str:
        return None
    try:
        return [int(x.strip()) for x in time_windows_str.split(',') if x.strip()]
    except ValueError:
        raise ValueError(f"Invalid time windows format: {time_windows_str}. Use comma-separated integers (e.g., '24,48,72,96,120,144,168')")


def parse_args() -> Config:
    """Parse command line arguments and create Config object"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create Config object
    config = Config(
        # Basic settings
        mode=args.mode,
        csv_path=args.csv,
        
        # Training settings
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        
        # Model settings
        encoder=args.encoder,
        encoder_pk=args.encoder_pk,
        encoder_pd=args.encoder_pd,
        head_pk=args.head_pk,
        head_pd=args.head_pd,
        
        # Model hyperparameters
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        
        # MoE settings
        num_experts=args.num_experts,
        top_k=args.top_k,
        
        # Data preprocessing
        use_feature_engineering=args.use_fe,
        perkg=args.perkg,
        allow_future_dose=args.allow_future_dose,
        time_windows=parse_time_windows(args.time_windows) if args.time_windows else None,
        
        # Augmentation and regularization
        use_mixup=args.use_mixup,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        
        # Contrastive learning
        lambda_contrast=args.lambda_contrast,
        temperature=args.temperature,
        
        # Data splitting
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        
        # Output settings
        output_dir=args.out_dir,
        run_name=args.run_name,
        verbose=args.verbose,
        device_id=args.device_id
    )
    
    return config
