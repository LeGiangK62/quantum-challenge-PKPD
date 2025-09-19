#!/bin/bash

# =============================================================================
# PK/PD Modeling - Comprehensive Experiment Runner
# =============================================================================
# This script runs experiments across different modes and encoders
# Usage: ./run_experiments.sh [experiment_type]
# 
# Experiment types:
# - basic: Basic mode-encoder combinations
# - advanced: Advanced combinations with different PK/PD encoders
# - ablation: Ablation studies
# - all: Run all experiments
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Base configuration
BASE_EPOCHS=50
BASE_BATCH_SIZE=32
BASE_LR=0.001
BASE_PATIENCE=20

# Experiment directories
EXPERIMENT_DIR="results/experiments"
LOG_DIR="results/experiment_logs"
TIMESTAMP=$(date +"%y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to run a single experiment
run_experiment() {
    local run_name="$1"
    local mode="$2"
    local encoder="$3"
    local encoder_pk="$4"
    local encoder_pd="$5"
    local epochs="$6"
    local batch_size="$7"
    local lr="$8"
    local extra_args="$9"
    
    log_info "Running experiment: $run_name"
    log_info "Mode: $mode, Encoder: $encoder, PK: $encoder_pk, PD: $encoder_pd"
    
    # Build command
    local cmd="python main.py"
    cmd="$cmd --run_name $run_name"
    cmd="$cmd --mode $mode"
    cmd="$cmd --epochs $epochs"
    cmd="$cmd --batch_size $batch_size"
    cmd="$cmd --lr $lr"
    cmd="$cmd --patience $BASE_PATIENCE"
    
    # Add encoder arguments
    if [[ -n "$encoder_pk" && -n "$encoder_pd" ]]; then
        cmd="$cmd --encoder_pk $encoder_pk --encoder_pd $encoder_pd"
    else
        cmd="$cmd --encoder $encoder"
    fi
    
    # Add extra arguments
    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi
    
    # Run experiment
    local start_time=$(date +%s)
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Experiment $run_name completed in ${duration}s"
        return 0
    else
        log_error "Experiment $run_name failed"
        return 1
    fi
}

# Function to create experiment summary
create_summary() {
    local experiment_type="$1"
    local summary_file="$EXPERIMENT_DIR/${experiment_type}_summary_${TIMESTAMP}.txt"
    
    log_info "Creating experiment summary: $summary_file"
    
    cat > "$summary_file" << EOF
# Experiment Summary: $experiment_type
# Timestamp: $TIMESTAMP
# Total experiments: $total_experiments
# Successful: $successful_experiments
# Failed: $failed_experiments

## Configuration
- Base epochs: $BASE_EPOCHS
- Base batch size: $BASE_BATCH_SIZE
- Base learning rate: $BASE_LR
- Base patience: $BASE_PATIENCE

## Results Directory
$EXPERIMENT_DIR

## Log Directory
$LOG_DIR

EOF
}

# =============================================================================
# Experiment Definitions
# =============================================================================

# Basic experiments: each mode with each encoder
run_basic_experiments() {
    log_info "Starting basic experiments..."
    
    local modes=("separate" "joint" "shared" "dual_stage" "integrated" "two_stage_shared")
    local encoders=("mlp" "resmlp" "moe" "resmlp_moe" "adaptive_resmlp_moe" "cnn")
    
    for mode in "${modes[@]}"; do
        for encoder in "${encoders[@]}"; do
            local run_name="basic_${mode}_${encoder}_${TIMESTAMP}"
            run_experiment "$run_name" "$mode" "$encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" ""
        done
    done
}

# Advanced experiments: different PK/PD encoder combinations
run_advanced_experiments() {
    log_info "Starting advanced experiments..."
    
    local modes=("joint" "dual_stage" "integrated")
    local pk_encoders=("mlp" "resmlp" "cnn")
    local pd_encoders=("mlp" "resmlp" "cnn")
    
    for mode in "${modes[@]}"; do
        for pk_encoder in "${pk_encoders[@]}"; do
            for pd_encoder in "${pd_encoders[@]}"; do
                if [[ "$pk_encoder" != "$pd_encoder" ]]; then
                    local run_name="advanced_${mode}_${pk_encoder}_${pd_encoder}_${TIMESTAMP}"
                    run_experiment "$run_name" "$mode" "" "$pk_encoder" "$pd_encoder" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" ""
                fi
            done
        done
    done
}

# Ablation experiments: feature engineering and augmentation
run_ablation_experiments() {
    log_info "Starting ablation experiments..."
    
    local base_mode="separate"
    local base_encoder="resmlp_moe"
    
    # Feature engineering ablation
    local run_name="ablation_no_fe_${TIMESTAMP}"
    run_experiment "$run_name" "$base_mode" "$base_encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" ""
    
    local run_name="ablation_with_fe_${TIMESTAMP}"
    run_experiment "$run_name" "$base_mode" "$base_encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--use_fe"
    
    # Mixup ablation
    local run_name="ablation_no_mixup_${TIMESTAMP}"
    run_experiment "$run_name" "$base_mode" "$base_encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--use_fe"
    
    local run_name="ablation_with_mixup_${TIMESTAMP}"
    run_experiment "$run_name" "$base_mode" "$base_encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--use_fe --use_mixup"
    
    # Contrastive learning ablation
    local run_name="ablation_no_contrast_${TIMESTAMP}"
    run_experiment "$run_name" "$base_mode" "$base_encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--use_fe --use_mixup"
    
    local run_name="ablation_with_contrast_${TIMESTAMP}"
    run_experiment "$run_name" "$base_mode" "$base_encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--use_fe --use_mixup --lambda_contrast 0.1"
}

# Hyperparameter experiments
run_hyperparameter_experiments() {
    log_info "Starting hyperparameter experiments..."
    
    local mode="separate"
    local encoder="resmlp_moe"
    
    # Learning rate experiments
    local lrs=("0.0001" "0.001" "0.01")
    for lr in "${lrs[@]}"; do
        local run_name="hyper_lr_${lr}_${TIMESTAMP}"
        run_experiment "$run_name" "$mode" "$encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$lr" "--use_fe"
    done
    
    # Batch size experiments
    local batch_sizes=("16" "32" "64")
    for batch_size in "${batch_sizes[@]}"; do
        local run_name="hyper_bs_${batch_size}_${TIMESTAMP}"
        run_experiment "$run_name" "$mode" "$encoder" "" "" "$BASE_EPOCHS" "$batch_size" "$BASE_LR" "--use_fe"
    done
    
    # Hidden dimension experiments
    local hidden_dims=("32" "64" "128")
    for hidden in "${hidden_dims[@]}"; do
        local run_name="hyper_hidden_${hidden}_${TIMESTAMP}"
        run_experiment "$run_name" "$mode" "$encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--use_fe --hidden $hidden"
    done
}

# CNN-specific experiments
run_cnn_experiments() {
    log_info "Starting CNN-specific experiments..."
    
    local mode="separate"
    local encoder="cnn"
    
    # Kernel size experiments
    local kernel_sizes=("3" "5" "7")
    for kernel_size in "${kernel_sizes[@]}"; do
        local run_name="cnn_kernel_${kernel_size}_${TIMESTAMP}"
        run_experiment "$run_name" "$mode" "$encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--kernel_size $kernel_size"
    done
    
    # Number of filters experiments
    local num_filters=("32" "64" "128")
    for filters in "${num_filters[@]}"; do
        local run_name="cnn_filters_${filters}_${TIMESTAMP}"
        run_experiment "$run_name" "$mode" "$encoder" "" "" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$BASE_LR" "--num_filters $filters"
    done
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local experiment_type="${1:-basic}"
    
    # Initialize counters
    total_experiments=0
    successful_experiments=0
    failed_experiments=0
    
    # Create directories
    mkdir -p "$EXPERIMENT_DIR"
    mkdir -p "$LOG_DIR"
    
    log_info "Starting PK/PD modeling experiments"
    log_info "Experiment type: $experiment_type"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Base configuration: epochs=$BASE_EPOCHS, batch_size=$BASE_BATCH_SIZE, lr=$BASE_LR"
    
    # Redirect output to log file
    local log_file="$LOG_DIR/experiment_${experiment_type}_${TIMESTAMP}.log"
    exec > >(tee -a "$log_file")
    exec 2>&1
    
    log_info "Logging to: $log_file"
    
    # Run experiments based on type
    case "$experiment_type" in
        "basic")
            run_basic_experiments
            ;;
        "advanced")
            run_advanced_experiments
            ;;
        "ablation")
            run_ablation_experiments
            ;;
        "hyperparameter")
            run_hyperparameter_experiments
            ;;
        "cnn")
            run_cnn_experiments
            ;;
        "all")
            run_basic_experiments
            run_advanced_experiments
            run_ablation_experiments
            run_hyperparameter_experiments
            run_cnn_experiments
            ;;
        *)
            log_error "Unknown experiment type: $experiment_type"
            log_info "Available types: basic, advanced, ablation, hyperparameter, cnn, all"
            exit 1
            ;;
    esac
    
    # Create summary
    create_summary "$experiment_type"
    
    # Final report
    log_info "Experiment completed!"
    log_info "Total experiments: $total_experiments"
    log_success "Successful: $successful_experiments"
    if [[ $failed_experiments -gt 0 ]]; then
        log_warning "Failed: $failed_experiments"
    fi
    log_info "Results saved to: $EXPERIMENT_DIR"
    log_info "Logs saved to: $LOG_DIR"
}

# =============================================================================
# Script Execution
# =============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
