#!/usr/bin/env bash

# Sequential evaluation helper.
# Usage: ./scripts/run_sequential_eval.sh
# Customize the per-run arrays below; each index corresponds to one run.
# Unlike run_parallel_eval.sh (which spawns tmux panes), this script executes
# each configuration one after another in the current shell.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

# Optionally override via environment variable, e.g.
#   EVAL_TASK_IDS="0,1,2" ./scripts/run_sequential_eval.sh
# so you do not need to edit this file when trying different task subsets.
EVAL_TASK_IDS=${EVAL_TASK_IDS:-"0,1,2,3,4,10,30,50,70,81"}
TRAINING_TASK_IDS=${TRAINING_TASK_IDS:-$(seq -s, 5 81)}
TEST_TASK_IDS=${TEST_TASK_IDS:-$(seq -s, 0 4)}

# Define the runs (update/add/remove entries as needed).
# Each index across PIPELINES/DEVICES/... represents a single call to train_diffusion.py.
PIPELINES=(
  "exp_walker_82_long_horizon"
  "exp_walker_82_long_horizon"
)
NAMES=(
  "exp_walker_82_long_horizon"
  "exp_walker_82_long_horizon"
)
DEVICES=("cuda:2" "cuda:2")
NOISES=("embedding_guided" "embedding_guided")
CONDITIONS=("False" "False")
DADP_CKPTS=(
  "./dadp/embedding/logs/transformer/exp_walker_82/best_model.zip"
  "./dadp/embedding/logs/transformer/exp_walker_82/best_model.zip"
)
MODES=("train" "inference")
PIPELINE_TYPES=("joint" "joint")
DATASET_NAMES=(
  "RandomWalker2d/82dynamics-v7"
  "RandomWalker2d/82dynamics-v7"
)

NUM_RUNS=${#PIPELINES[@]}
if [[ $NUM_RUNS -eq 0 ]]; then
  echo "No runs configured. Edit run_sequential_eval.sh to add at least one entry." >&2
  exit 1
fi

if [[ ${#NAMES[@]} -ne $NUM_RUNS || ${#DEVICES[@]} -ne $NUM_RUNS || ${#NOISES[@]} -ne $NUM_RUNS || \
  ${#CONDITIONS[@]} -ne $NUM_RUNS || ${#DADP_CKPTS[@]} -ne $NUM_RUNS || ${#MODES[@]} -ne $NUM_RUNS || \
  ${#PIPELINE_TYPES[@]} -ne $NUM_RUNS || ${#DATASET_NAMES[@]} -ne $NUM_RUNS ]]; then
  echo "Configuration arrays differ in length. Please keep them aligned." >&2
  exit 1
fi

# Ensure the conda environment is available if the script is invoked from a login shell.
# When you run this script from cron/ssh/etc., your shell might not have conda initialized,
# so we source the hook explicitly and activate the desired environment.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate dadp
else
  echo "conda not found in PATH; make sure the desired environment is active before running." >&2
fi

for ((i = 0; i < NUM_RUNS; i++)); do
  RUN_LABEL=$((i + 1))
  PIPELINE_NAME=${PIPELINES[$i]}
  RUN_NAME=${NAMES[$i]}
  DEVICE=${DEVICES[$i]}
  NOISE=${NOISES[$i]}
  CONDITION=${CONDITIONS[$i]}
  DADP_CKPT=${DADP_CKPTS[$i]}
  MODE=${MODES[$i]}
  PIPELINE_TYPE=${PIPELINE_TYPES[$i]}
  DATASET_NAME=${DATASET_NAMES[$i]}

  echo
  echo "================ RUN ${RUN_LABEL}/${NUM_RUNS}: ${PIPELINE_NAME} ================"
  echo "Device       : ${DEVICE}"
  echo "Noise type   : ${NOISE}"
  echo "Condition    : ${CONDITION}"
  echo "DADP Checkpt : ${DADP_CKPT}"
  echo "Run name     : ${RUN_NAME}"
  echo "Mode         : ${MODE}"
  echo "Pipeline Type: ${PIPELINE_TYPE}"
  echo "Dataset Name : ${DATASET_NAME}"
  echo "Eval tasks   : ${EVAL_TASK_IDS}"
  echo "Start time   : $(date)"

  # Execute the actual evaluation run. Feel free to add/remove CLI args here
  # (e.g., --num_episodes, --planner_ckpt, etc.).
  python train_diffusion.py \
    --pipeline_name "${PIPELINE_NAME}" \
    --name "${RUN_NAME}" \
    --device "${DEVICE}" \
    --noise_type "${NOISE}" \
    --condition "${CONDITION}" \
    --dadp_checkpoint_path "${DADP_CKPT}" \
    --eval_task_ids "${EVAL_TASK_IDS}" \
    --training_task_ids "${TRAINING_TASK_IDS}" \
    --test_task_ids "${TEST_TASK_IDS}" \
    --mode "${MODE}" \
    --pipeline_type "${PIPELINE_TYPE}" \
    --dataset "${DATASET_NAME}"

  echo "Completed run ${RUN_LABEL}/${NUM_RUNS} at $(date)"
  echo "--------------------------------------------------------------"

done

echo "All ${NUM_RUNS} runs finished."
