#!/usr/bin/env bash
set -euo pipefail

# Train then evaluate (ID + OOD) for one or more configs, running 5 episodes by default.
# Usage:
#   ./scripts/train_and_eval_id_ood.sh [CONFIG1 args.json CONFIG2 args.json ...] [--eval-out DIR]
# If no configs are passed, a default set of six env configs is used (ant, halfcheetah, hopper, walker, door, relocate).
# Env overrides:
#   PROJ_ROOT   : project root (default: /home/qinghang/DomainAdaptiveDiffusionPolicy)
#   PYTHON      : python executable (default: python)
#   DEVICE      : device string (default: cuda:0)
#   SEED        : seed for both train and eval (default: 0)
#   CONDITION   : force --condition flag (default: true)
#   PRECOLLECT_EPISODES: embedding pre-collection episodes for eval (default: 0)
#   NUM_EPISODES: eval episodes (default: 5)
#   RUN_ROOT    : training save dir root (default: ${PROJ_ROOT}/results/train_eval_${TIMESTAMP})
#   EVAL_ROOT   : base eval dir (default: ${RUN_ROOT}/eval)
#   GPUS        : space-separated GPU indices for tmux panes (default: "0 1")
#   SESSION_NAME: tmux session name (default: train_eval_id_ood)
#   ENV_ACTIVATE: command to activate env (default: source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp)
#   DRYRUN      : if 1, only print commands

# Parse optional trailing --eval-out DIR
EVAL_ROOT_INPUT=""
CONFIGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-out)
      EVAL_ROOT_INPUT="$2"; shift 2;;
    *)
      CONFIGS+=("$1"); shift 1;;
  esac
done

PROJ_ROOT=${PROJ_ROOT:-/home/qinghang/DomainAdaptiveDiffusionPolicy}
PYTHON_BIN=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-0}
CONDITION=${CONDITION:-true}
PRECOLLECT_EPISODES=${PRECOLLECT_EPISODES:-0}

GPUS=(${GPUS:-2})
SESSION_NAME=${SESSION_NAME:-train_eval_id_ood}
ENV_ACTIVATE=${ENV_ACTIVATE:-"source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp"}
DRYRUN=${DRYRUN:-0}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT_DEFAULT="${PROJ_ROOT}/results/condition_and_guidance_train_eval_${TIMESTAMP}"
RUN_ROOT=${RUN_ROOT:-${RUN_ROOT_DEFAULT}}
EVAL_ROOT=${EVAL_ROOT_INPUT:-${RUN_ROOT}/eval}

mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}"

cd "${PROJ_ROOT}"
eval "${ENV_ACTIVATE}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

sanitize_name() {
  local path="$1"
  path="${path%/}"
  local base="${path##*/}" 
  base="${base%.json}"
  base="${base//[^a-zA-Z0-9_.-]/_}"
  echo "${base}"
}

declare -A TARGET_CONFIGS_DEFAULT=(
  # [ant]="${PROJ_ROOT}/results/exp_ant_28_reproduce_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim/RandomAnt-v0/RandomAnt/82dynamics-v7/args.json"
  # [halfcheetah]="${PROJ_ROOT}/results/exp_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  # [hopper]="${PROJ_ROOT}/results/exp_hopper_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHopper-v0/RandomHopper/82dynamics-v7/args.json"
  [walker]="${PROJ_ROOT}/results/exp_walker_28(2)_predict_mixddim_long_horizon_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  # [door]="${PROJ_ROOT}/results/exp_door_3_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim/door-shrink-finger-medium-v0/Adroit/door_shrink_combined-v0/args.json"
  [relocate]="${PROJ_ROOT}/results/exp_relocate_3_shrink_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/relocate-shrink-finger-medium-v0/Adroit/relocate_shrink_combined-v0/args.json"
)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs provided; using default six env configs." >&2
  for key in $(echo "${!TARGET_CONFIGS_DEFAULT[@]}" | tr ' ' '\n' | sort); do
    cfg="${TARGET_CONFIGS_DEFAULT[$key]}"
    if [[ -f "${cfg}" ]]; then
      CONFIGS+=("${cfg}")
    else
      echo "Warning: missing default config for ${key}: ${cfg}" >&2
    fi
  done
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs to run." >&2
  exit 1
fi

# Setup tmux session with one pane per GPU for parallel training/eval
pane_count=${#GPUS[@]}
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi
tmux new-session -d -s "${SESSION_NAME}" -n "runs"
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
  if [[ ! -f "${cfg}" ]]; then
    echo "Warning: config not found, skipping: ${cfg}" >&2
    continue
  fi

  name=$(sanitize_name "${cfg}")
  run_dir="${RUN_ROOT}/${name}"
  eval_dir="${EVAL_ROOT}/${name}"
  mkdir -p "${run_dir}" "${eval_dir}"

  gpu_idx=${GPUS[$((job_idx % pane_count))]}
  pane_target=$((job_idx % pane_count))
  base_prefix="cd ${PROJ_ROOT} && ${ENV_ACTIVATE} && PYTHONPATH=${PROJ_ROOT}:\$PYTHONPATH CUDA_VISIBLE_DEVICES=${gpu_idx}"

  train_cmd="${base_prefix} ${PYTHON_BIN} scripts/launch_from_config.py --config \"${cfg}\" --mode train --device ${DEVICE} --seed ${SEED} --condition ${CONDITION} --save_dir \"${run_dir}\""

  eval_id_cmd="${base_prefix} ${PYTHON_BIN} scripts/launch_from_config.py --config \"${cfg}\" --mode inference --device ${DEVICE} --seed ${SEED} --condition ${CONDITION} --precollect_episodes ${PRECOLLECT_EPISODES} --eval_task_mode id --eval_out_dir \"${eval_dir}\" --save_dir \"${run_dir}\""

  eval_ood_cmd="${base_prefix} ${PYTHON_BIN} scripts/launch_from_config.py --config \"${cfg}\" --mode inference --device ${DEVICE} --seed ${SEED} --condition ${CONDITION} --precollect_episodes ${PRECOLLECT_EPISODES} --eval_task_mode ood  --eval_out_dir \"${eval_dir}\" --save_dir \"${run_dir}\""

  job_cmd="${train_cmd} && ${eval_id_cmd} && ${eval_ood_cmd}"

  echo "[QUEUE][GPU ${gpu_idx}][pane ${pane_target}][${name}] ${job_cmd}";
  if [[ "${DRYRUN}" != "1" ]]; then
    queue_job "${pane_target}" "${job_cmd}"
  fi

  job_idx=$((job_idx + 1))
done

echo "Queued all jobs in tmux session ${SESSION_NAME}. Attach with: tmux attach -t ${SESSION_NAME}"
echo "Outputs under ${RUN_ROOT}; eval JSON copies under ${EVAL_ROOT}."
