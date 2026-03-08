#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/qinghang/DomainAdaptiveDiffusionPolicy"
SESSION_NAME="supervised_comparison"
RUN_ROOT="${ROOT_DIR}/results/supervised_comparison_$(date +%Y%m%d_%H%M%S)"
EVAL_ROOT="${RUN_ROOT}/eval_results"

# Checkpoints
HC_CKPT="/home/qinghang/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/5090/hc_supervised/supervised.zip"
WALKER_CKPT="/home/qinghang/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/5090/walker_supervised/supervised.zip"

# Base Configs
HC_CFG="/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_halfcheetah_28_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomHalfCheetah-v0/RandomHalfCheetah/82dynamics-v7/args.json"
WALKER_CFG="/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_walker_28(2)_predict_mixddim_long_horizon_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/args.json"

mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}"

# tmux setup
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi

tmux new-session -d -s "${SESSION_NAME}" -n "run"
tmux split-window -h -t "${SESSION_NAME}:run"

# Commands for Pane 0 (HalfCheetah, GPU 2)
HC_CMD="cd ${ROOT_DIR} && source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp && export PYTHONPATH=${ROOT_DIR} && python scripts/launch_from_config.py --config \"${HC_CFG}\" --dadp_checkpoint_path \"${HC_CKPT}\" --device cuda:2 --mode train --save_dir \"${RUN_ROOT}/hc\" --pipeline_suffix \"_hc_sup\" --enable_wandb true && python scripts/launch_from_config.py --config \"${HC_CFG}\" --dadp_checkpoint_path \"${HC_CKPT}\" --device cuda:2 --mode inference --eval_task_mode id --eval_out_dir \"${EVAL_ROOT}\" --save_dir \"${RUN_ROOT}/hc\" --pipeline_suffix \"_hc_sup\" --enable_wandb false && python scripts/launch_from_config.py --config \"${HC_CFG}\" --dadp_checkpoint_path \"${HC_CKPT}\" --device cuda:2 --mode inference --eval_task_mode ood --eval_out_dir \"${EVAL_ROOT}\" --save_dir \"${RUN_ROOT}/hc\" --pipeline_suffix \"_hc_sup\" --enable_wandb false"

# Commands for Pane 1 (Walker, GPU 3)
WALKER_CMD="cd ${ROOT_DIR} && source /home/anaconda3/etc/profile.d/conda.sh && conda activate dadp && export PYTHONPATH=${ROOT_DIR} && python scripts/launch_from_config.py --config \"${WALKER_CFG}\" --dadp_checkpoint_path \"${WALKER_CKPT}\" --device cuda:3 --mode train --save_dir \"${RUN_ROOT}/walker\" --pipeline_suffix \"_walker_sup\" --enable_wandb true && python scripts/launch_from_config.py --config \"${WALKER_CFG}\" --dadp_checkpoint_path \"${WALKER_CKPT}\" --device cuda:3 --mode inference --eval_task_mode id --eval_out_dir \"${EVAL_ROOT}\" --save_dir \"${RUN_ROOT}/walker\" --pipeline_suffix \"_walker_sup\" --enable_wandb false && python scripts/launch_from_config.py --config \"${WALKER_CFG}\" --dadp_checkpoint_path \"${WALKER_CKPT}\" --device cuda:3 --mode inference --eval_task_mode ood --eval_out_dir \"${EVAL_ROOT}\" --save_dir \"${RUN_ROOT}/walker\" --pipeline_suffix \"_walker_sup\" --enable_wandb false"

tmux send-keys -t "${SESSION_NAME}:run.0" "${HC_CMD}" C-m
tmux send-keys -t "${SESSION_NAME}:run.1" "${WALKER_CMD}" C-m

echo "Started training/evaluation for HC on GPU 2 and Walker on GPU 3 in tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
