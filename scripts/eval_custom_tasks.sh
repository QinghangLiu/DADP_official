#!/bin/bash

# Script to evaluate multiple models on custom tasks
# Usage: bash eval_custom_tasks.sh

set -e  # Exit on error

# Activate conda environment
source /home/anaconda3/etc/profile.d/conda.sh
conda activate dadp

# Configuration
ENV_NAME="RandomHalfCheetah-v0"
DATASET="RandomHalfCheetah/82dynamics-v7"
DADP_CHECKPOINT="./dadp/embedding/logs/transformer/exp_halfcheetah_28/best_model.zip"
DEVICE="cuda:0"
NUM_ENVS=10
NUM_EPISODES=5

# Define your four models here
# Format: "model_name:model_directory"
MODELS=(
    "guide_0.01:/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_halfcheetah_28_0.01_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.01_noisemixed_ddim"
    "guide_0.05:/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_halfcheetah_28_0.05_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim"
    "guide_0.5:/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_halfcheetah_28_0.5_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.5_noisemixed_ddim"
    "guide_1:/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_halfcheetah_28_1_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide1_noisemixed_ddim"
)

# Create results directory
RESULTS_DIR="./eval_results/custom_tasks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Save configuration
cat > "$RESULTS_DIR/config.txt" << EOF
Evaluation Configuration
========================
Environment: $ENV_NAME
Dataset: $DATASET
DADP Checkpoint: $DADP_CHECKPOINT
Device: $DEVICE
Number of Environments: $NUM_ENVS
Number of Episodes: $NUM_EPISODES
Custom Task Generation: ENABLED
Models to Evaluate: ${#MODELS[@]}
EOF

echo "=========================================="
echo "Custom Task Evaluation Script"
echo "=========================================="
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Loop through each model
for i in "${!MODELS[@]}"; do
    MODEL_INFO="${MODELS[$i]}"
    IFS=':' read -r MODEL_NAME MODEL_DIR <<< "$MODEL_INFO"
    
    echo ""
    echo "=========================================="
    echo "Evaluating Model $((i+1))/${#MODELS[@]}: $MODEL_NAME"
    echo "=========================================="
    echo "Model Directory: $MODEL_DIR"
    echo ""
    
    # Check if directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "WARNING: Model directory not found: $MODEL_DIR"
        echo "Skipping this model..."
        continue
    fi
    
    # Find the latest planner checkpoint
    PLANNER_CKPT=$(find "$MODEL_DIR" -name "planner_*.pt" -type f | sort -V | tail -n 1)
    
    if [ -z "$PLANNER_CKPT" ]; then
        echo "WARNING: No planner checkpoint found in: $MODEL_DIR"
        echo "Skipping this model..."
        continue
    fi
    
    echo "Found checkpoint: $PLANNER_CKPT"
    
    # Extract pipeline name from directory (prefix before first _H… to match make_save_path)
    BASENAME=$(basename "$MODEL_DIR")
    PIPELINE_NAME=${BASENAME%%_H*}
    if [ -z "$PIPELINE_NAME" ]; then
        PIPELINE_NAME="$BASENAME"
    fi
    
    # Create model-specific results directory
    MODEL_RESULTS_DIR="$RESULTS_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_RESULTS_DIR"

    # Run evaluation
    python train_diffusion.py \
        --mode inference \
        --env_name "$ENV_NAME" \
        --dataset "$DATASET" \
        --dadp_checkpoint_path "$DADP_CHECKPOINT" \
        --pipeline_name "$PIPELINE_NAME" \
        --name "custom_task_eval_$MODEL_NAME" \
        --device "$DEVICE" \
        --num_envs "$NUM_ENVS" \
        --num_episodes "$NUM_EPISODES" \
        --customize_task True \
        --noise_type mixed_ddim \
        --pipeline_type joint \
        --condition False \
        --enable_wandb False \
        2>&1 | tee "$MODEL_RESULTS_DIR/evaluation_log.txt"

    # Copy any existing result artifacts from the model directory
    if [ -d "$MODEL_DIR" ]; then
        cp "$MODEL_DIR"/*.json "$MODEL_RESULTS_DIR/" 2>/dev/null || true
        cp "$MODEL_DIR"/*.txt "$MODEL_RESULTS_DIR/" 2>/dev/null || true
    fi

    echo ""
    echo "Model $MODEL_NAME evaluation completed!"
    echo "Results saved to: $MODEL_RESULTS_DIR"
    echo ""
done

echo ""
echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Generate summary report
echo "Generating summary report..."
python << 'PYTHON_SCRIPT'
import os
import json
import sys
from pathlib import Path

results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
summary = []

for model_dir in sorted(Path(results_dir).glob("model*")):
    if not model_dir.is_dir():
        continue
    
    model_name = model_dir.name
    json_files = list(model_dir.glob("*.json"))
    
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            overall_stats = data.get('overall_stats', {})
            
            summary.append({
                'model': model_name,
                'mean_reward': overall_stats.get('mean_episode_reward', 0),
                'std_reward': overall_stats.get('std_episode_reward', 0),
                'mean_success': overall_stats.get('mean_success_rate', 0),
                'num_samples': overall_stats.get('num_episode_samples', 0)
            })

# Save summary
summary_file = os.path.join(results_dir, 'summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary table
print("\n" + "="*80)
print("EVALUATION SUMMARY - CUSTOM TASKS")
print("="*80)
print(f"{'Model':<20} {'Mean Reward':<15} {'Std':<12} {'Success Rate':<15} {'Samples':<10}")
print("-"*80)
for item in summary:
    print(f"{item['model']:<20} {item['mean_reward']:>12.4f}   {item['std_reward']:>10.4f}   {item['mean_success']:>12.4f}   {item['num_samples']:>8}")
print("="*80)

PYTHON_SCRIPT

python -c "
import sys
results_dir = '$RESULTS_DIR'
exec(open('$0').read().split('PYTHON_SCRIPT')[1].split('PYTHON_SCRIPT')[0])
" "$RESULTS_DIR"

echo ""
echo "Summary saved to: $RESULTS_DIR/summary.json"
