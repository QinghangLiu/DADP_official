#!/usr/bin/env bash
set -euo pipefail

# Script to train embeddings on all six environments in parallel using tmux
# Based on run_multi_seed.sh pattern

ROOT_DIR="./"
SESSION_NAME="embedding_training"
GPUS=(0 1)  # Two GPUs for parallel training
PYTHON_CMD="/home/pengcheng/anaconda3/bin/conda run -n dadp310 python"
PYTHON_RUN="stdbuf -oL -eL ${PYTHON_CMD}"
CONFIG_DIR="${ROOT_DIR}/scripts/configs"

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it first:"
    echo "  sudo apt-get install jq  # On Ubuntu/Debian"
    echo "  brew install jq          # On macOS"
    exit 1
fi

# Environment config files
ENV_CONFIGS=(
  "ant_config.json"
  # "door_config.json"
  "halfcheetah_config.json"
  "hopper_config.json"
  # "relocate_config.json"
  # "walker_config.json"
)

# Common arguments
# Note: Boolean flags use action="store_true", only include them if you want to enable them
# Default-true flags (cross_prediction, detach_embedding_for_state/policy, norm_z) are enabled by default
COMMON_ARGS="--wandb_project walker27 --observation_function mask_dimensions --observation_noise_std 0.1 --observation_mask_dims 0 1 --history 16 --min_visible_length 16 --delta_t 1 --inverse_loss_weight 1.0 --forward_loss_weight 1.0 --state_loss_weight 1.0 --factor_loss_weight 1.0 --policy_loss_weight 1.0 --intra_traj_consistency_loss_weight 0.0 --inter_traj_consistency_loss_weight 0.0 --d_model 256 --n_layer 4 --head_hidden 256 --n_head 8 --d_ff 1024 --dropout 0.1 --adaptive_pooling_heads 8 --adaptive_pooling_dropout 0.1 --pos_encoding_max_len 5000 --learning_rate 0.0003 --num_epochs 10 --batch_size 128 --window_size 2 --eval_interval 1 --train_split 0.8 --device cuda:0 --seed 42 --save_checkpoint_epochs 10"

# Kill existing session if it exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Killing existing tmux session: ${SESSION_NAME}"
  tmux kill-session -t "${SESSION_NAME}"
fi

# Create session with one window and split into 2 panes
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" -n "parallel_embedding"
tmux split-window -h -t "${SESSION_NAME}:parallel_embedding"
tmux select-layout -t "${SESSION_NAME}:parallel_embedding" tiled

# Initialize each pane
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  echo "Setting up pane ${i} for GPU ${gpu}"
  tmux send-keys -t "${SESSION_NAME}:parallel_embedding.${i}" "cd ${ROOT_DIR}" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_embedding.${i}" "echo 'Using conda env: dadp310 via conda run'" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_embedding.${i}" "export PYTHONPATH=${ROOT_DIR}" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_embedding.${i}" "echo '========================================'" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_embedding.${i}" "echo 'Running on GPU ${gpu} in pane ${i}'" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_embedding.${i}" "echo '========================================'" C-m
done

# Function to queue a job to a specific pane
queue_job() {
  local pane_idx="$1"
  shift
  tmux send-keys -t "${SESSION_NAME}:parallel_embedding.${pane_idx}" "$*" C-m
}

# Distribute environments across panes
job_idx=0

for config_file in "${ENV_CONFIGS[@]}"; do
  config_path="${CONFIG_DIR}/${config_file}"
  
  if [[ ! -f "${config_path}" ]]; then
    echo "Warning: Config file not found: ${config_path}"
    continue
  fi
  
  # Parse JSON config using jq
  env_name=$(jq -r '.env_name' "${config_path}")
  dataset_name=$(jq -r '.dataset_name' "${config_path}")
  embedding_size=$(jq -r '.embedding_size' "${config_path}")
  train_task_ids=$(jq -r '.train_task_ids | join(" ")' "${config_path}")
  test_task_ids=$(jq -r '.test_task_ids | join(" ")' "${config_path}")
  log_dir=$(jq -r '.log_dir' "${config_path}")
  state_mean=$(jq -r 'if .state_mean then (.state_mean | join(" ")) else "" end' "${config_path}")
  state_std=$(jq -r 'if .state_std then (.state_std | join(" ")) else "" end' "${config_path}")
  
  pane_idx=$((job_idx % ${#GPUS[@]}))
  gpu="${GPUS[$pane_idx]}"
  
  echo "Queueing ${env_name} to pane ${pane_idx} (GPU ${gpu})"
  
  # Queue the training command
  queue_job "${pane_idx}" "echo ''"
  queue_job "${pane_idx}" "echo '========================================'"
  queue_job "${pane_idx}" "echo 'Training ${env_name}'"
  queue_job "${pane_idx}" "echo 'Dataset: ${dataset_name}'"
  queue_job "${pane_idx}" "echo 'Embedding size: ${embedding_size}'"
  queue_job "${pane_idx}" "echo 'GPU: ${gpu}'"
  queue_job "${pane_idx}" "echo '========================================'"
  
  mean_args=""
  std_args=""
  if [[ -n "${state_mean}" ]]; then
    mean_args="--state_mean ${state_mean}"
  fi
  if [[ -n "${state_std}" ]]; then
    std_args="--state_std ${state_std}"
  fi

  escaped_log_dir=$(printf '%q' "${log_dir}")
  log_file="${log_dir}/tmux_${env_name}.log"
  escaped_log_file=$(printf '%q' "${log_file}")

  queue_job "${pane_idx}" "mkdir -p ${escaped_log_dir}"

  cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_RUN} train_embedding.py --dataset_name \"${dataset_name}\" --embedding_size ${embedding_size} --train_task_ids ${train_task_ids} --test_task_ids ${test_task_ids} --log_dir \"${log_dir}\" ${mean_args} ${std_args} ${COMMON_ARGS}"
  
  queue_job "${pane_idx}" "${cmd}"
  queue_job "${pane_idx}" "echo 'Finished training ${env_name}'"
  
  job_idx=$((job_idx + 1))
done

# Add completion messages
for i in "${!GPUS[@]}"; do
  queue_job "${i}" "echo ''"
  queue_job "${i}" "echo '========================================'"
  queue_job "${i}" "echo 'All jobs completed on pane ${i}'"
  queue_job "${i}" "echo '========================================'"
done

echo ""
echo "=========================================="
echo "All jobs queued successfully!"
echo "=========================================="
echo "Attach to the session with:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "Detach from session: Ctrl+B then D"
echo "Kill session: tmux kill-session -t ${SESSION_NAME}"
echo "=========================================="
