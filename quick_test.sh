#!/bin/bash

# =============================================================================
# Quick Test Script for PK/PD Modeling
# =============================================================================
# This script runs a quick test of different mode-encoder combinations
# Usage: ./quick_test.sh
# =============================================================================

set -e

# Configuration
EPOCHS=5
BATCH_SIZE=16
TIMESTAMP=$(date +"%y%m%d_%H%M%S")

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Quick test function
quick_test() {
    local run_name="$1"
    local mode="$2"
    local encoder="$3"
    local extra_args="$4"
    
    log_info "Testing: $run_name (Mode: $mode, Encoder: $encoder)"
    
    local cmd="python main.py --run_name $run_name --mode $mode --encoder $encoder --epochs $EPOCHS --batch_size $BATCH_SIZE"
    
    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi
    
    if eval "$cmd"; then
        log_success "✓ $run_name completed"
        return 0
    else
        echo "✗ $run_name failed"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting quick tests..."
    log_info "Configuration: epochs=$EPOCHS, batch_size=$BATCH_SIZE"
    
    local success_count=0
    local total_count=0
    
    # Test basic combinations
    local tests=(
        "quick_mlp_separate_${TIMESTAMP} separate mlp"
        "quick_resmlp_joint_${TIMESTAMP} joint resmlp"
        "quick_cnn_shared_${TIMESTAMP} shared cnn"
        "quick_moe_dual_stage_${TIMESTAMP} dual_stage moe"
        "quick_resmlp_moe_integrated_${TIMESTAMP} integrated resmlp_moe"
    )
    
    for test in "${tests[@]}"; do
        read -r run_name mode encoder <<< "$test"
        total_count=$((total_count + 1))
        
        if quick_test "$run_name" "$mode" "$encoder"; then
            success_count=$((success_count + 1))
        fi
        
        echo "---"
    done
    
    # Test with feature engineering
    log_info "Testing with feature engineering..."
    total_count=$((total_count + 1))
    if quick_test "quick_fe_test_${TIMESTAMP}" "separate" "mlp" "--use_fe"; then
        success_count=$((success_count + 1))
    fi
    
    # Test with mixup
    log_info "Testing with mixup..."
    total_count=$((total_count + 1))
    if quick_test "quick_mixup_test_${TIMESTAMP}" "separate" "resmlp" "--use_mixup"; then
        success_count=$((success_count + 1))
    fi
    
    # Test different PK/PD encoders
    log_info "Testing different PK/PD encoders..."
    total_count=$((total_count + 1))
    if quick_test "quick_mixed_encoders_${TIMESTAMP}" "joint" "" "--encoder_pk mlp --encoder_pd cnn"; then
        success_count=$((success_count + 1))
    fi
    
    # Final report
    echo "=========================================="
    log_info "Quick test completed!"
    log_info "Total tests: $total_count"
    log_success "Successful: $success_count"
    
    if [[ $success_count -eq $total_count ]]; then
        log_success "All tests passed! 🎉"
    else
        local failed=$((total_count - success_count))
        echo "Failed: $failed"
    fi
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
