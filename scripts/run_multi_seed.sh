#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="./"
SESSION_NAME="multi_seed_runs"
RUN_ROOT="${ROOT_DIR}/multi_seed_runs/$(date +%Y%m%d_%H%M%S)"
# RUN_ROOT="${ROOT_DIR}/multi_seed_runs/20260210_132141"
EVAL_ROOT="${RUN_ROOT}/eval_results"
SEEDS=(0 1)
GPUS=(1 )
PYTHON_CMD="/home/pengcheng/anaconda3/bin/conda run -n dadp310 python"
PYTHON_RUN="stdbuf -oL -eL ${PYTHON_CMD}"

# Base configs for DADP
declare -A DADP_CONFIGS=(
  [ant]="${ROOT_DIR}/config/ant_args.json"
  [halfcheetah]="${ROOT_DIR}/config/halfcheetah_args.json"
  [walker]="${ROOT_DIR}/config/walker_args.json"
  [hopper]="${ROOT_DIR}/config/hopper_args.json"
  [door]="${ROOT_DIR}/config/door_args.json"
  [relocate]="${ROOT_DIR}/config/relocate_args.json")
# Base configs for DV
declare -A DV_CONFIGS=(
  # [ant]="${ROOT_DIR}/results/exp_dv_ant_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomAnt-v0/RandomAnt/82dynamics-v7/args.json"
  # [halfcheetah]="${ROOT_DIR}/results/exp_dv_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  # [walker]="${ROOT_DIR}/results/exp_dv_walker_28(2)_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  # [hopper]="${ROOT_DIR}/results/exp_dv_hopper_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHopper-v0/RandomHopper/82dynamics-v7/args.json"
  # [door]="${ROOT_DIR}/results/exp_dv_door_3_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/door-shrink-finger-medium-v0/Adroit/door_shrink_combined-v0/args.json"
  # [relocate]="${ROOT_DIR}/results/exp_dv_relocate_3_shrink_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/relocate-shrink-finger-medium-v0/Adroit/relocate_shrink_combined-v0/args.json"
)

OOD_ENV_KEYS=(ant halfcheetah walker hopper door relocate)

mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}"

for cfg in "${DADP_CONFIGS[@]}" "${DV_CONFIGS[@]}"; do
  if [ ! -f "${cfg}" ]; then
    echo "Missing config: ${cfg}" >&2
    exit 1
  fi
done

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi

# Create session with one window and split into 2 panes
tmux new-session -d -s "${SESSION_NAME}" -n "parallel_runs"
tmux split-window -h -t "${SESSION_NAME}:parallel_runs"
tmux select-layout -t "${SESSION_NAME}:parallel_runs" tiled

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  tmux send-keys -t "${SESSION_NAME}:parallel_runs.${i}" "cd ${ROOT_DIR}" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_runs.${i}" "echo 'Using conda env: dadp310 via conda run'" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_runs.${i}" "export PYTHONPATH=${ROOT_DIR}" C-m
  tmux send-keys -t "${SESSION_NAME}:parallel_runs.${i}" "echo Running on GPU ${gpu} in pane ${i}" C-m
done

ood_supported() {
  local key="$1"
  for env_key in "${OOD_ENV_KEYS[@]}"; do
    if [[ "${env_key}" == "${key}" ]]; then
      return 0
    fi
  done
  return 1
}

queue_job() {
  local pane_idx="$1"
  shift
  tmux send-keys -t "${SESSION_NAME}:parallel_runs.${pane_idx}" "$*" C-m
}

job_idx=0

# Narrowed scope to rerun only the problematic tasks (missing ckpts: relocate, door, walker; DV halfcheetah config fixed to task 28)
# env_keys=(halfcheetah ant walker hopper door relocate)
# variants=(dadp dv)
variants=(dadp dv)
dadp_env_keys=(relocate)
dv_env_keys=()

for variant in "${variants[@]}"; do
  if [[ "${variant}" == "dadp" ]]; then
    env_keys=("${dadp_env_keys[@]}")
  else
    env_keys=("${dv_env_keys[@]}")
  fi

  for env_key in "${env_keys[@]}"; do
    if [[ "${variant}" == "dadp" ]]; then
      cfg="${DADP_CONFIGS[${env_key}]}"
    else
      cfg="${DV_CONFIGS[${env_key}]}"
    fi

    for seed in "${SEEDS[@]}"; do
      pane_idx=$((job_idx % ${#GPUS[@]}))
      gpu="${GPUS[$pane_idx]}"
      job_idx=$((job_idx + 1))
      suffix="_${variant}_s${seed}"
      base_cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_RUN} scripts/launch_from_config.py --config \"${cfg}\" --seed ${seed} --device cuda:0 --save_dir \"${RUN_ROOT}/${variant}\" --pipeline_suffix=\"${suffix}\" --enable_wandb false"

      queue_job "${pane_idx}" "${base_cmd} --mode train"
      queue_job "${pane_idx}" "${base_cmd} --mode inference --eval_task_mode id --eval_out_dir ${EVAL_ROOT} --planner_ckpt latest"
      if ood_supported "${env_key}"; then
        queue_job "${pane_idx}" "${base_cmd} --mode inference --eval_task_mode ood --eval_out_dir ${EVAL_ROOT} --planner_ckpt latest"
      fi
      queue_job "${pane_idx}" "echo Finished ${variant} ${env_key} seed ${seed}"
    done
  done
done

echo "Queued all jobs. Attach with: tmux attach -t ${SESSION_NAME}"
