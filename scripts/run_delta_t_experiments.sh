#!/bin/bash

# ==============================================================================
# Simple Delta_t Ablation Script
# ==============================================================================
# This script runs train_embedding.py with different delta_t values
# Delta_t values: 1, 2, 4, 8, 16, 32, 64, 128
# Usage: 
#   bash run_delta_t_experiments.sh              # Default: no detach
#   bash run_delta_t_experiments.sh detach       # With detach
#   bash run_delta_t_experiments.sh no-detach    # Explicitly no detach
# ==============================================================================

# Parse detach argument
DETACH_FLAG=""
if [ $# -gt 0 ]; then
    if [ "$1" = "detach" ]; then
        DETACH_FLAG="--detach_embedding_from_policy"
        echo "Running experiments WITH detach_embedding_from_policy"
    elif [ "$1" = "no-detach" ]; then
        DETACH_FLAG=""
        echo "Running experiments WITHOUT detach_embedding_from_policy"
    else
        echo "Unknown argument: $1"
        echo "Usage: bash $0 [detach|no-detach]"
        exit 1
    fi
else
    echo "Running experiments WITHOUT detach_embedding_from_policy (default)"
fi

# Delta_t values to test
DELTA_T_VALUES=(1 2 4 8 16 32 64 128)

# Loop through each delta_t value and run training
for delta_t in "${DELTA_T_VALUES[@]}"; do
    echo "=========================================="
    echo "Running training with delta_t=${delta_t}"
    if [ -n "$DETACH_FLAG" ]; then
        echo "With detach_embedding_from_policy=True"
    else
        echo "With detach_embedding_from_policy=False"
    fi
    echo "=========================================="
    
    python train_embedding.py --delta_t ${delta_t} ${DETACH_FLAG}
    
    echo ""
    echo "Completed delta_t=${delta_t}"
    echo ""
    sleep 5  # Small delay between runs
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
