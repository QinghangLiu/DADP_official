#!/usr/bin/env bash
set -euo pipefail
# Run inference on selected configs (id + optional ood) in parallel panes.
# Usage:
#   GPUS="0 1" SEED=1 DRYRUN=0 ./scripts/test_ckpt_latest.sh [ROOT_RESULTS_DIR] [EVAL_OUT_DIR]
# Defaults:
#   ROOT_RESULTS_DIR: /home/qinghang/DomainAdaptiveDiffusionPolicy/results
#   EVAL_OUT_DIR:     ${ROOT_RESULTS_DIR}/eval_ckpt_latest_$(date +%Y%m%d_%H%M%S)
# Env overrides:
#   PYTHON: Python executable (default: python)
#   DEVICE: Device string passed to launch_from_config.py (default: cuda:0)
#   SEED:   Seed for inference (default: 1)
#   DRYRUN: If set to 1, only print commands without executing
#   CONFIG_NAME: Config filename to use (default: args.json)
#   GPUS: space-separated list of GPU indices (default: "0 1")
#   OOD: if set to 1, also run --eval_task_mode ood
#   ENV_ACTIVATE: command to activate env (default: source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp)
#   PROJ_ROOT: project path for PYTHONPATH (default: /home/qinghang/DomainAdaptiveDiffusionPolicy)

ROOT_DIR="${1:-/home/qinghang/DomainAdaptiveDiffusionPolicy/results}"
EVAL_ROOT_INPUT="${2:-}"  # optional second arg
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_EVAL_ROOT="${ROOT_DIR}/eval_ckpt_latest_${TIMESTAMP}"
EVAL_ROOT=${EVAL_ROOT_INPUT:-${DEFAULT_EVAL_ROOT}}

PROJ_ROOT=${PROJ_ROOT:-/home/qinghang/DomainAdaptiveDiffusionPolicy}
PYTHON_BIN=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-0}
DRYRUN=${DRYRUN:-0}
CONFIG_NAME=${CONFIG_NAME:-args.json}
GPUS_ARR=(${GPUS:-0 1})
RUN_OOD=${OOD:-1}
ENV_ACTIVATE=${ENV_ACTIVATE:-"source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp"}
SESSION_NAME="test_ckpt_latest"

mkdir -p "${EVAL_ROOT}"

# Explicitly defined target configurations (as requested)
declare -A TARGET_CONFIGS=(
  [ant]="${PROJ_ROOT}/results/exp_ant_28_reproduce_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim/RandomAnt-v0/RandomAnt/82dynamics-v7/args.json"
  [halfcheetah]="${PROJ_ROOT}/results/exp_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  [relocate]="${PROJ_ROOT}/results/exp_relocate_3_shrink_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/relocate-shrink-finger-medium-v0/Adroit/relocate_shrink_combined-v0/args.json"
  [walker]="${PROJ_ROOT}/results/exp_walker_28(2)_predict_mixddim_long_horizon_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  [hopper]="${PROJ_ROOT}/results/exp_hopper_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHopper-v0/RandomHopper/82dynamics-v7/args.json"
  [door]="${PROJ_ROOT}/results/exp_door_3_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim/door-shrink-finger-medium-v0/Adroit/door_shrink_combined-v0/args.json"
)

CONFIGS=()
# Iterate over sorted keys for deterministic order
for key in $(echo "${!TARGET_CONFIGS[@]}" | tr ' ' '\n' | sort); do
  cfg="${TARGET_CONFIGS[$key]}"
  if [[ -f "${cfg}" ]]; then
    CONFIGS+=("${cfg}")
  else
    echo "Warning: Config not found at ${cfg}" >&2
  fi
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs found for the defined targets." >&2
  exit 1
fi

echo "Found ${#CONFIGS[@]} configs. Writing evals to ${EVAL_ROOT}".

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi
tmux new-session -d -s "${SESSION_NAME}" -n "runs"

# Split panes equal to number of GPUs
pane_count=${#GPUS_ARR[@]}
for ((i=1; i<pane_count; i++)); do
  tmux split-window -t "${SESSION_NAME}:runs" -h
done
tmux select-layout -t "${SESSION_NAME}:runs" tiled

queue_job() {
  local pane_idx="$1"; shift
  tmux send-keys -t "${SESSION_NAME}:runs.${pane_idx}" "$*" C-m
}

job_idx=0
for cfg in "${CONFIGS[@]}"; do
  dir=$(dirname "${cfg}")

  rel_path=${dir#${ROOT_DIR}}
  out_dir="${EVAL_ROOT}${rel_path}"
  mkdir -p "${out_dir}"

  gpu_idx=${GPUS_ARR[$((job_idx % pane_count))]}
  pane_target=$((job_idx % pane_count))

  base_cmd_prefix="cd ${PROJ_ROOT} && ${ENV_ACTIVATE} && PYTHONPATH=${PROJ_ROOT}:\$PYTHONPATH CUDA_VISIBLE_DEVICES=${gpu_idx} ${PYTHON_BIN} scripts/launch_from_config.py --config \"${cfg}\" --mode inference --device ${DEVICE} --seed ${SEED} --enable_wandb false"

  cmd_id="${base_cmd_prefix} --eval_task_mode id --eval_out_dir \"${out_dir}\""
  printf '\n[QUEUE][GPU %s] %s\n' "${gpu_idx}" "${cmd_id}"
  if [[ "${DRYRUN}" != "1" ]]; then
    queue_job "${pane_target}" "${cmd_id}"
  fi

  if [[ "${RUN_OOD}" == "1" ]]; then
    cmd_ood="${base_cmd_prefix} --eval_task_mode ood --eval_out_dir \"${out_dir}\""
    printf '[QUEUE][GPU %s] %s\n' "${gpu_idx}" "${cmd_ood}"
    if [[ "${DRYRUN}" != "1" ]]; then
      queue_job "${pane_target}" "${cmd_ood}"
    fi
  fi

  queue_job "${pane_target}"
  job_idx=$((job_idx + 1))
done

echo "Queued all jobs. Attach with: tmux attach -t ${SESSION_NAME}"
