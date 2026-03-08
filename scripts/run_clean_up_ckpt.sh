#!/usr/bin/env bash
# Batch the missing evals (Adroit + Mujoco seed0) using tmux; auto-detect or override GPUs.

set -euo pipefail

ROOT_DIR="/home/qinghang/DomainAdaptiveDiffusionPolicy"
ENV_ACTIVATE="source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp"
PY_CMD="python"
GPU_LIST=(0 1)

# Existing eval folder to place outputs
EVAL_OUT_DIR="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/eval_results_20260113_075121"
PLANNER_CKPT_OVERRIDE="latest"
POLICY_CKPT_OVERRIDE="latest"
CRITIC_CKPT_OVERRIDE="latest"

ADROIT_SEEDS=(1 2 3 4)

cycle_gpu() {
  local idx=$1
  local count=${#GPU_LIST[@]}
  echo -n "${GPU_LIST[$((idx % count))]}"
}

ensure_tmux_session() {
  local session_name=$1
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    tmux kill-session -t "${session_name}"
  fi

  tmux new-session -d -s "${session_name}" -n "jobs"
  local pane_count=${#GPU_LIST[@]}
  for ((i=1; i<pane_count; i++)); do
    tmux split-window -h -t "${session_name}:jobs"
  done
  tmux select-layout -t "${session_name}:jobs" tiled

  for i in $(seq 0 $((pane_count-1))); do
    tmux send-keys -t "${session_name}:jobs.${i}" "cd ${ROOT_DIR}" C-m
    tmux send-keys -t "${session_name}:jobs.${i}" "${ENV_ACTIVATE}" C-m
    tmux send-keys -t "${session_name}:jobs.${i}" "export PYTHONPATH=${ROOT_DIR}" C-m
    tmux send-keys -t "${session_name}:jobs.${i}" "echo Pane ${i} using GPU ${GPU_LIST[$i]}" C-m
  done
}

queue_job() {
  local session_name=$1
  local pane_idx=$2
  shift 2
  tmux send-keys -t "${session_name}:jobs.${pane_idx}" "$*" C-m
}

enqueue_id_ood() {
  local session_name=$1
  local job_idx=$2
  local cfg=$3
  local gpu
  gpu=$(cycle_gpu "${job_idx}")
  local pane_idx=$((job_idx % ${#GPU_LIST[@]}))

  queue_job "${session_name}" "${pane_idx}" "CUDA_VISIBLE_DEVICES=${gpu} ${PY_CMD} ${ROOT_DIR}/scripts/launch_from_config.py --config \"${cfg}\" --mode inference --enable_wandb false --device cuda:0 --eval_task_mode id --planner_ckpt \"${PLANNER_CKPT_OVERRIDE}\" --policy_ckpt \"${POLICY_CKPT_OVERRIDE}\" --critic_ckpt \"${CRITIC_CKPT_OVERRIDE}\" --eval_out_dir \"${EVAL_OUT_DIR}\""
  queue_job "${session_name}" "${pane_idx}" "CUDA_VISIBLE_DEVICES=${gpu} ${PY_CMD} ${ROOT_DIR}/scripts/launch_from_config.py --config \"${cfg}\" --mode inference --enable_wandb false --device cuda:0 --eval_task_mode ood --planner_ckpt \"${PLANNER_CKPT_OVERRIDE}\" --policy_ckpt \"${POLICY_CKPT_OVERRIDE}\" --critic_ckpt \"${CRITIC_CKPT_OVERRIDE}\" --eval_out_dir \"${EVAL_OUT_DIR}\""

  echo $((job_idx + 1))
}






run_mujoco_seed0() {
  local session_name=$1
  local job_idx=$2

  local cfg_ant="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/exp_ant_28_reproduce_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.05_noisemixed_ddim/RandomAnt-v0/RandomAnt/82dynamics-v7/args.json"
  local cfg_halfcheetah="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/exp_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  local cfg_hopper="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/exp_hopper_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHopper-v0/RandomHopper/82dynamics-v7/args.json"
  local cfg_walker="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/exp_walker_28(2)_predict_mixddim_long_horizon_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"
  local cfg_dv_ant="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/dv/exp_dv_ant_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomAnt-v0/RandomAnt/82dynamics-v7/args.json"
  local cfg_dv_halfcheetah="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/dv/exp_dv_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
  local cfg_dv_hopper="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/dv/exp_dv_hopper_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomHopper-v0/RandomHopper/82dynamics-v7/args.json"
  local cfg_dv_walker="${ROOT_DIR}/multi_seed_runs/clean_up_ckpt/dv/exp_dv_walker_28(2)_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"


  job_idx=$(enqueue_id_ood "${session_name}" "${job_idx}" "${cfg_halfcheetah}")
  job_idx=$(enqueue_id_ood "${session_name}" "${job_idx}" "${cfg_dv_ant}")
  job_idx=$(enqueue_id_ood "${session_name}" "${job_idx}" "${cfg_dv_halfcheetah}")
  job_idx=$(enqueue_id_ood "${session_name}" "${job_idx}" "${cfg_dv_hopper}")
  job_idx=$(enqueue_id_ood "${session_name}" "${job_idx}" "${cfg_dv_walker}")


  echo "${job_idx}"
}

main() {
  local session_name="clean_eval"
  ensure_tmux_session "${session_name}"

  local job_idx=0

  job_idx=$(run_mujoco_seed0 "${session_name}" "${job_idx}")

  echo "Dispatched Adroit + Mujoco jobs in tmux session: ${session_name}"
  echo "Outputs will be under: ${EVAL_OUT_DIR}"
  echo "Attach with: tmux attach -t ${session_name}"
}

main "$@"
