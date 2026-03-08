#!/usr/bin/env bash
set -euo pipefail

# Launch inference for the ablation set (10 configs: walker + halfcheetah) using latest ckpts.
# Defaults can be overridden via env vars:
#   PROJ_ROOT   - project root (default: /home/qinghang/DomainAdaptiveDiffusionPolicy)
#   RESULTS_ROOT- ablation results root (default: ${PROJ_ROOT}/results/ablation)
#   EVAL_ROOT   - output root (default: ${RESULTS_ROOT}/eval_ablation_latest_<ts>)
#   GPUS        - space-separated GPU list (default: "0 1")
#   OOD         - if 1, also run --eval_task_mode ood (default: 0)
#   SEED        - eval seed (default: 1)
#   DEVICE      - torch device arg (default: cuda:0)
#   PYTHON      - python executable (default: python)
#   ENV_ACTIVATE- activation command (default: source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp)
#   SESSION_NAME- tmux session name (default: ablation_latest)
#   DRYRUN      - if 1, print only

PROJ_ROOT=${PROJ_ROOT:-/home/qinghang/DomainAdaptiveDiffusionPolicy}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJ_ROOT}/results/ablation}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT=${EVAL_ROOT:-${RESULTS_ROOT}/eval_ablation_latest_${TIMESTAMP}}

PYTHON_BIN=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-0}
DRYRUN=${DRYRUN:-0}
RUN_OOD=${OOD:-1}
ENV_ACTIVATE=${ENV_ACTIVATE:-"source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp"}
SESSION_NAME=${SESSION_NAME:-ablation_latest}

GPUS_STR=${GPUS:-"0 1"}
IFS=' ' read -r -a GPUS_ARR <<< "${GPUS_STR}"
pane_count=${#GPUS_ARR[@]}
if (( pane_count == 0 )); then
  echo "No GPUs specified via GPUS env." >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}"

CONFIGS=(
  "${RESULTS_ROOT}/exp_dv_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  "${RESULTS_ROOT}/exp_halfcheetah_28_condition_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  "${RESULTS_ROOT}/exp_halfcheetah_28_dt1_condition_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  "${RESULTS_ROOT}/exp_halfcheetah_28_mixedddim_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  "${RESULTS_ROOT}/hc/exp_halfcheetah_28_hc_sup_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide1_noisemixed_ddim/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  "${RESULTS_ROOT}/exp_dv_walker_28(2)_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  "${RESULTS_ROOT}/exp_walker_28(2)_condition_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  "${RESULTS_ROOT}/exp_walker_28(2)_dt1_condition_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  "${RESULTS_ROOT}/exp_walker_28(2)_mixedddim_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  "${RESULTS_ROOT}/walker/exp_walker_28(2)_predict_mixddim_long_horizon_walker_sup_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
)

VALID_CONFIGS=()
for cfg in "${CONFIGS[@]}"; do
  if [[ -f "${cfg}" ]]; then
    VALID_CONFIGS+=("${cfg}")
  else
    echo "Missing config: ${cfg}" >&2
  fi
done

if [[ ${#VALID_CONFIGS[@]} -eq 0 ]]; then
  echo "No configs found under ${RESULTS_ROOT}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi

tmux new-session -d -s "${SESSION_NAME}" -n "runs"
for ((i=1; i<pane_count; i++)); do
  tmux split-window -t "${SESSION_NAME}:runs" -h
done
tmux select-layout -t "${SESSION_NAME}:runs" tiled

for idx in "${!GPUS_ARR[@]}"; do
  gpu="${GPUS_ARR[$idx]}"
  tmux send-keys -t "${SESSION_NAME}:runs.${idx}" "cd ${PROJ_ROOT}" C-m
  tmux send-keys -t "${SESSION_NAME}:runs.${idx}" "${ENV_ACTIVATE}" C-m
  tmux send-keys -t "${SESSION_NAME}:runs.${idx}" "export PYTHONPATH=${PROJ_ROOT}:\$PYTHONPATH" C-m
  tmux send-keys -t "${SESSION_NAME}:runs.${idx}" "echo Using GPU ${gpu} in pane ${idx}" C-m
done

queue_job() { local pane="$1"; shift; tmux send-keys -t "${SESSION_NAME}:runs.${pane}" "$*" C-m; }

job_idx=0
for cfg in "${VALID_CONFIGS[@]}"; do
  dir=$(dirname "${cfg}")
  rel="${dir#${RESULTS_ROOT}}"
  out_dir="${EVAL_ROOT}${rel}"
  mkdir -p "${out_dir}"

  gpu="${GPUS_ARR[$((job_idx % pane_count))]}"
  pane_target=$((job_idx % pane_count))

  base_cmd="cd ${PROJ_ROOT} && ${ENV_ACTIVATE} && PYTHONPATH=${PROJ_ROOT}:\$PYTHONPATH CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_BIN} scripts/launch_from_config.py --config \"${cfg}\" --mode inference --device ${DEVICE} --seed ${SEED} --enable_wandb false --planner_ckpt latest --eval_out_dir \"${out_dir}\""

  cmd_id="${base_cmd} --eval_task_mode id"
  printf '[QUEUE][GPU %s] %s\n' "${gpu}" "${cmd_id}"
  if [[ "${DRYRUN}" != "1" ]]; then
    queue_job "${pane_target}" "${cmd_id}"
  fi

  if [[ "${RUN_OOD}" == "1" ]]; then
    cmd_ood="${base_cmd} --eval_task_mode ood"
    printf '[QUEUE][GPU %s] %s\n' "${gpu}" "${cmd_ood}"
    if [[ "${DRYRUN}" != "1" ]]; then
      queue_job "${pane_target}" "${cmd_ood}"
    fi
  fi

  queue_job "${pane_target}"
  job_idx=$((job_idx + 1))
done

echo "Queued ${#VALID_CONFIGS[@]} ablation jobs. Attach with: tmux attach -t ${SESSION_NAME}"
