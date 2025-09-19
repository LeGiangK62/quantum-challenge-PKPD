#!/bin/bash

# =============================================================================
# Seed Experiment Examples
# =============================================================================
# This script provides examples of how to use run_seed_experiments.sh
# Usage: ./seed_experiment_examples.sh [example_name]
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

show_examples() {
    cat << EOF
Available examples:

1. basic          - Basic experiment with default settings (5 seeds)
2. mode_test      - Test different modes with same encoder
3. encoder_test   - Test different encoders with same mode
4. mixed_encoders - Test different PK/PD encoder combinations
5. feature_test   - Test feature engineering effects
6. mixup_test     - Test mixup augmentation effects
7. contrast_test  - Test contrastive learning effects
8. full_test      - Full experiment with all features
9. custom_seeds   - Custom seed range example
10. quick_test    - Quick test with 2 seeds and 5 epochs

Usage: ./seed_experiment_examples.sh [example_name]
EOF
}

run_example() {
    local example_name="$1"
    local cmd=""
    
    case "$example_name" in
        "basic")
            log_info "Running basic experiment with default settings..."
            cmd="./run_seed_experiments.sh"
            ;;
        "mode_test")
            log_info "Testing different modes with MLP encoder..."
            cmd="./run_seed_experiments.sh --mode joint --encoder mlp --run_name mode_test"
            ;;
        "encoder_test")
            log_info "Testing different encoders with separate mode..."
            cmd="./run_seed_experiments.sh --mode separate --encoder resmlp_moe --run_name encoder_test"
            ;;
        "mixed_encoders")
            log_info "Testing different PK/PD encoder combinations..."
            cmd="./run_seed_experiments.sh --mode joint --encoder_pk mlp --encoder_pd cnn --run_name mixed_encoders"
            ;;
        "feature_test")
            log_info "Testing feature engineering effects..."
            cmd="./run_seed_experiments.sh --mode separate --encoder resmlp_moe --use_fe --run_name feature_test"
            ;;
        "mixup_test")
            log_info "Testing mixup augmentation effects..."
            cmd="./run_seed_experiments.sh --mode separate --encoder resmlp_moe --use_fe --use_mixup --run_name mixup_test"
            ;;
        "contrast_test")
            log_info "Testing contrastive learning effects..."
            cmd="./run_seed_experiments.sh --mode shared --encoder resmlp --use_fe --lambda_contrast 0.1 --run_name contrast_test"
            ;;
        "full_test")
            log_info "Running full experiment with all features..."
            cmd="./run_seed_experiments.sh --mode joint --encoder_pk resmlp --encoder_pd cnn --use_fe --use_mixup --lambda_contrast 0.1 --epochs 100 --run_name full_test"
            ;;
        "custom_seeds")
            log_info "Testing with custom seed range..."
            cmd="./run_seed_experiments.sh --seeds 1,2,3,4,5,6,7,8,9,10 --run_name custom_seeds --mode separate --encoder mlp"
            ;;
        "quick_test")
            log_info "Running quick test with 2 seeds and 5 epochs..."
            cmd="./run_seed_experiments.sh --seeds 1,2 --epochs 5 --run_name quick_test --mode separate --encoder mlp"
            ;;
        *)
            log_warning "Unknown example: $example_name"
            show_examples
            exit 1
            ;;
    esac
    
    log_info "Command: $cmd"
    echo "=========================================="
    
    if eval "$cmd"; then
        log_success "Example '$example_name' completed successfully!"
    else
        log_warning "Example '$example_name' failed!"
        exit 1
    fi
}

main() {
    if [[ $# -eq 0 ]]; then
        show_examples
        exit 0
    fi
    
    local example_name="$1"
    run_example "$example_name"
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
