#!/bin/bash

# =============================================================================
# Seed-based Experiment Runner
# =============================================================================
# This script runs experiments with multiple seeds (1,2,3,4,5) for statistical significance
# Usage: ./run_seed_experiments.sh [options]
# 
# Options:
#   --mode MODE                    Training mode (default: separate)
#   --encoder ENCODER              Default encoder (default: mlp)
#   --encoder_pk ENCODER_PK        PK-specific encoder
#   --encoder_pd ENCODER_PD        PD-specific encoder
#   --use_fe                       Enable feature engineering
#   --use_mixup                    Enable mixup augmentation
#   --lambda_contrast LAMBDA       Contrastive learning weight
#   --epochs EPOCHS                Number of epochs (default: 50)
#   --batch_size BATCH_SIZE        Batch size (default: 32)
#   --lr LEARNING_RATE             Learning rate (default: 0.001)
#   --seeds SEEDS                  Comma-separated seeds (default: 1,2,3,4,5)
#   --run_name RUN_NAME            Base run name (default: seed_exp)
#   --help                         Show this help message
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_MODE="separate"
DEFAULT_ENCODER="mlp"
DEFAULT_EPOCHS=3000
DEFAULT_BATCH_SIZE=32
DEFAULT_LR=0.001
DEFAULT_PATIENCE=300
DEFAULT_SEEDS="1,2,3,4,5"
DEFAULT_RUN_NAME="seed_exp"

# Initialize variables
MODE="$DEFAULT_MODE"
ENCODER="$DEFAULT_ENCODER"
ENCODER_PK=""
ENCODER_PD=""
USE_FE=false
USE_MIXUP=false
LAMBDA_CONTRAST=""
EPOCHS="$DEFAULT_EPOCHS"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
LR="$DEFAULT_LR"
SEEDS="$DEFAULT_SEEDS"
RUN_NAME="$DEFAULT_RUN_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_seed() {
    echo -e "${CYAN}[SEED $1]${NC} $2"
}

show_help() {
    cat << EOF
Seed-based Experiment Runner

Usage: $0 [options]

Options:
  --mode MODE                    Training mode (default: $DEFAULT_MODE)
                                Available: separate, joint, shared, dual_stage, integrated, two_stage_shared
  --encoder ENCODER              Default encoder (default: $DEFAULT_ENCODER)
                                Available: mlp, resmlp, moe, resmlp_moe, adaptive_resmlp_moe, cnn
  --encoder_pk ENCODER_PK        PK-specific encoder
  --encoder_pd ENCODER_PD        PD-specific encoder
  --use_fe                       Enable feature engineering
  --use_mixup                    Enable mixup augmentation
  --lambda_contrast LAMBDA       Contrastive learning weight (e.g., 0.1)
  --epochs EPOCHS                Number of epochs (default: $DEFAULT_EPOCHS)
  --batch_size BATCH_SIZE        Batch size (default: $DEFAULT_BATCH_SIZE)
  --lr LEARNING_RATE             Learning rate (default: $DEFAULT_LR)
  --seeds SEEDS                  Comma-separated seeds (default: $DEFAULT_SEEDS)
  --run_name RUN_NAME            Base run name (default: $DEFAULT_RUN_NAME)
  --help                         Show this help message

Examples:
  # Basic experiment with default settings (creates: seed_exp_separate_mlp_TIMESTAMP)
  $0

  # Experiment with specific mode and encoder (creates: my_exp_joint_resmlp_moe_TIMESTAMP)
  $0 --mode joint --encoder resmlp_moe --run_name my_exp

  # Experiment with different PK/PD encoders (creates: test_joint_mlp-cnn_TIMESTAMP)
  $0 --mode joint --encoder_pk mlp --encoder_pd cnn --run_name test

  # Experiment with feature engineering and mixup (creates: exp_separate_mlp_fe_mixup_TIMESTAMP)
  $0 --mode separate --encoder mlp --use_fe --use_mixup --run_name exp

  # Experiment with contrastive learning (creates: test_shared_resmlp_fe_contrast0.1_TIMESTAMP)
  $0 --mode shared --encoder resmlp --use_fe --lambda_contrast 0.1 --run_name test

  # Custom seeds and run name (creates: my_exp_separate_mlp_TIMESTAMP)
  $0 --seeds 1,2,3 --run_name my_experiment --mode separate --encoder mlp

  # Full custom experiment (creates: full_joint_resmlp-cnn_fe_mixup_contrast0.1_ep100_bs64_lr0.0005_TIMESTAMP)
  $0 --mode joint --encoder_pk resmlp --encoder_pd cnn --use_fe --use_mixup --lambda_contrast 0.1 --epochs 100 --batch_size 64 --lr 0.0005 --seeds 1,2,3,4,5,6,7,8,9,10 --run_name full

Note: Run names are automatically generated to include all configuration options,
      ensuring no experiments overwrite each other's results.
EOF
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --encoder)
                ENCODER="$2"
                shift 2
                ;;
            --encoder_pk)
                ENCODER_PK="$2"
                shift 2
                ;;
            --encoder_pd)
                ENCODER_PD="$2"
                shift 2
                ;;
            --use_fe)
                USE_FE=true
                shift
                ;;
            --use_mixup)
                USE_MIXUP=true
                shift
                ;;
            --lambda_contrast)
                LAMBDA_CONTRAST="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --batch_size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --lr)
                LR="$2"
                shift 2
                ;;
            --seeds)
                SEEDS="$2"
                shift 2
                ;;
            --run_name)
                RUN_NAME="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                log_info "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Experiment Functions
# =============================================================================

# Function to run a single experiment with specific seed
run_single_experiment() {
    local seed="$1"
    local run_name="$2"
    
    log_seed "$seed" "Starting experiment: $run_name"
    
    # Build command
    local cmd="python main.py"
    cmd="$cmd --run_name $run_name"
    cmd="$cmd --mode $MODE"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --lr $LR"
    cmd="$cmd --patience $DEFAULT_PATIENCE"
    cmd="$cmd --random_state $seed"
    
    # Add encoder arguments
    if [[ -n "$ENCODER_PK" && -n "$ENCODER_PD" ]]; then
        cmd="$cmd --encoder_pk $ENCODER_PK --encoder_pd $ENCODER_PD"
    else
        cmd="$cmd --encoder $ENCODER"
    fi
    
    # Add feature engineering
    if [[ "$USE_FE" == true ]]; then
        cmd="$cmd --use_fe"
    fi
    
    # Add mixup
    if [[ "$USE_MIXUP" == true ]]; then
        cmd="$cmd --use_mixup"
    fi
    
    # Add contrastive learning
    if [[ -n "$LAMBDA_CONTRAST" ]]; then
        cmd="$cmd --lambda_contrast $LAMBDA_CONTRAST"
    fi
    
    # Run experiment
    local start_time=$(date +%s)
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_seed "$seed" "Completed successfully in ${duration}s"
        return 0
    else
        log_seed "$seed" "Failed"
        return 1
    fi
}

# Function to generate descriptive run name
generate_run_name() {
    local base_name="$1"
    local timestamp="$2"
    
    # Build descriptive name with options
    local desc_parts=()
    
    # Add mode
    desc_parts+=("$MODE")
    
    # Add encoder info
    if [[ -n "$ENCODER_PK" && -n "$ENCODER_PD" ]]; then
        desc_parts+=("${ENCODER_PK}-${ENCODER_PD}")
    else
        desc_parts+=("$ENCODER")
    fi
    
    # Add feature flags
    if [[ "$USE_FE" == true ]]; then
        desc_parts+=("fe")
    fi
    
    if [[ "$USE_MIXUP" == true ]]; then
        desc_parts+=("mixup")
    fi
    
    if [[ -n "$LAMBDA_CONTRAST" ]]; then
        desc_parts+=("contrast${LAMBDA_CONTRAST}")
    fi
    
    # Add hyperparameters if different from defaults
    if [[ "$EPOCHS" != "$DEFAULT_EPOCHS" ]]; then
        desc_parts+=("ep${EPOCHS}")
    fi
    
    if [[ "$BATCH_SIZE" != "$DEFAULT_BATCH_SIZE" ]]; then
        desc_parts+=("bs${BATCH_SIZE}")
    fi
    
    if [[ "$LR" != "$DEFAULT_LR" ]]; then
        desc_parts+=("lr${LR}")
    fi
    
    # Join parts with underscore
    local desc_name=$(IFS="_"; echo "${desc_parts[*]}")
    
    # Create final run name
    echo "${base_name}_${desc_name}_${timestamp}"
}

# Function to run all seed experiments
run_seed_experiments() {
    local timestamp=$(date +"%y%m%d_%H%M%S")
    local base_run_name=$(generate_run_name "$RUN_NAME" "$timestamp")
    
    # Convert seeds string to array
    IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
    
    log_info "Starting seed-based experiments"
    log_info "Base run name: $base_run_name"
    log_info "Seeds: ${SEED_ARRAY[*]}"
    log_info "Configuration:"
    log_info "  Mode: $MODE"
    if [[ -n "$ENCODER_PK" && -n "$ENCODER_PD" ]]; then
        log_info "  Encoder PK: $ENCODER_PK"
        log_info "  Encoder PD: $ENCODER_PD"
    else
        log_info "  Encoder: $ENCODER"
    fi
    log_info "  Epochs: $EPOCHS"
    log_info "  Batch size: $BATCH_SIZE"
    log_info "  Learning rate: $LR"
    log_info "  Feature engineering: $USE_FE"
    log_info "  Mixup: $USE_MIXUP"
    if [[ -n "$LAMBDA_CONTRAST" ]]; then
        log_info "  Lambda contrast: $LAMBDA_CONTRAST"
    fi
    
    # Initialize counters
    local total_experiments=${#SEED_ARRAY[@]}
    local successful_experiments=0
    local failed_experiments=0
    local failed_seeds=()
    
    # Create results directory
    local results_dir="results/seed_experiments"
    mkdir -p "$results_dir"
    
    # Create log file
    local log_file="$results_dir/${base_run_name}_log.txt"
    log_info "Logging to: $log_file"
    
    # Run experiments for each seed
    for seed in "${SEED_ARRAY[@]}"; do
        local run_name="${base_run_name}_s${seed}"
        
        echo "==========================================" >> "$log_file"
        echo "Seed: $seed, Run name: $run_name" >> "$log_file"
        echo "Started at: $(date)" >> "$log_file"
        echo "==========================================" >> "$log_file"
        
        if run_single_experiment "$seed" "$run_name" >> "$log_file" 2>&1; then
            successful_experiments=$((successful_experiments + 1))
            echo "Result: SUCCESS" >> "$log_file"
        else
            failed_experiments=$((failed_experiments + 1))
            failed_seeds+=("$seed")
            echo "Result: FAILED" >> "$log_file"
        fi
        
        echo "Completed at: $(date)" >> "$log_file"
        echo "" >> "$log_file"
    done
    
    # Create summary
    local summary_file="$results_dir/${base_run_name}_summary.txt"
    cat > "$summary_file" << EOF
# Seed Experiment Summary
# Base run name: $base_run_name
# Timestamp: $timestamp
# Total experiments: $total_experiments
# Successful: $successful_experiments
# Failed: $failed_experiments

## Configuration
- Mode: $MODE
- Encoder: $ENCODER
- Encoder PK: $ENCODER_PK
- Encoder PD: $ENCODER_PD
- Epochs: $EPOCHS
- Batch size: $BATCH_SIZE
- Learning rate: $LR
- Feature engineering: $USE_FE
- Mixup: $USE_MIXUP
- Lambda contrast: $LAMBDA_CONTRAST
- Seeds: ${SEED_ARRAY[*]}

## Results
- Successful seeds: $successful_experiments
- Failed seeds: $failed_experiments
EOF

    if [[ ${#failed_seeds[@]} -gt 0 ]]; then
        echo "- Failed seed list: ${failed_seeds[*]}" >> "$summary_file"
    fi

    echo "" >> "$summary_file"
    echo "## Individual Results" >> "$summary_file"
    echo "Check results/runs/ for individual experiment results" >> "$summary_file"
    echo "Log file: $log_file" >> "$summary_file"
    
    # Final report
    echo "=========================================="
    log_info "Seed experiments completed!"
    log_info "Total experiments: $total_experiments"
    log_success "Successful: $successful_experiments"
    if [[ $failed_experiments -gt 0 ]]; then
        log_warning "Failed: $failed_experiments"
        log_warning "Failed seeds: ${failed_seeds[*]}"
    fi
    log_info "Results saved to: $results_dir"
    log_info "Summary: $summary_file"
    log_info "Log: $log_file"
    
    # Show results directory structure
    log_info "Results directory structure:"
    echo "results/"
    echo "├── seed_experiments/"
    echo "│   ├── ${base_run_name}_summary.txt"
    echo "│   └── ${base_run_name}_log.txt"
    echo "└── runs/"
    for seed in "${SEED_ARRAY[@]}"; do
        echo "    └── ${base_run_name}_s${seed}/"
        echo "        └── $MODE/"
        if [[ -n "$ENCODER_PK" && -n "$ENCODER_PD" ]]; then
            echo "            └── ${ENCODER_PK}-${ENCODER_PD}/"
        else
            echo "            └── $ENCODER/"
        fi
        echo "                └── s${seed}/"
        echo "                    ├── model.pth"
        echo "                    ├── config.json"
        echo "                    ├── scalers.pkl"
        echo "                    └── results.json"
    done
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Run seed experiments
    run_seed_experiments
}

# =============================================================================
# Script Execution
# =============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
