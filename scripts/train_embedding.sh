#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${ROOT_DIR}/config/embedding_configs"
CONDA_ENV="${CONDA_ENV:-dadp310}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"

if [[ -n "${CONDA_BIN}" ]]; then
  PYTHON_RUN=(stdbuf -oL -eL "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u)
else
  echo "Warning: 'conda' not found. Falling back to system python." >&2
  PYTHON_RUN=(stdbuf -oL -eL python -u)
fi

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_single_embedding.sh <env_key> --seed <int> --gpu <id> [options]
  scripts/run_single_embedding.sh --config <path> --seed <int> --gpu <id> [options]

Required:
  --seed <int>         Seed value
  --gpu <id>           GPU id for CUDA_VISIBLE_DEVICES

Select one of:
  <env_key>            One of: ant, halfcheetah, walker, hopper, door, relocate
  --config <path>      Path to embedding config JSON (overrides env key mapping)

Optional:
  --device <cuda:x>    Device passed to train_embedding.py (default: cuda:0)
  --num_epochs <int>   Override epochs (default: from common args = 10)
  --batch_size <int>   Override batch size (default: from common args = 128)
  --log_dir <path>     Override log_dir from config JSON

Env overrides:
  CONDA_ENV=<name>     Conda env name (default: dadp310)
  CONDA_BIN=<path>     Conda executable (default: from PATH)

Examples:
  bash scripts/run_single_embedding.sh ant --seed 0 --gpu 0
  bash scripts/run_single_embedding.sh walker --seed 1 --gpu 1 --num_epochs 20
USAGE
}

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required. Install it first (e.g., sudo apt-get install jq)." >&2
  exit 1
fi

CONFIG=""
ENV_KEY=""
SEED=""
GPU=""
DEVICE="cuda:0"
NUM_EPOCHS=""
BATCH_SIZE=""
LOG_DIR_OVERRIDE=""

if [[ $# -gt 0 && "${1}" != -* ]]; then
  ENV_KEY="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --log_dir) LOG_DIR_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${SEED}" || -z "${GPU}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ -z "${CONFIG}" ]]; then
  if [[ -z "${ENV_KEY}" ]]; then
    echo "Provide either an env key (e.g. ant) or --config <path>." >&2
    usage
    exit 1
  fi
  CONFIG="${CONFIG_DIR}/${ENV_KEY}_config.json"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

env_name="$(jq -r '.env_name' "${CONFIG}")"
dataset_name="$(jq -r '.dataset_name' "${CONFIG}")"
embedding_size="$(jq -r '.embedding_size' "${CONFIG}")"
train_task_ids="$(jq -r '.train_task_ids | join(" ")' "${CONFIG}")"
test_task_ids="$(jq -r '.test_task_ids | join(" ")' "${CONFIG}")"
log_dir="$(jq -r '.log_dir' "${CONFIG}")"

if [[ -n "${LOG_DIR_OVERRIDE}" ]]; then
  log_dir="${LOG_DIR_OVERRIDE}"
fi

readarray -t state_mean_arr < <(jq -r 'if .state_mean then .state_mean[] else empty end' "${CONFIG}")
readarray -t state_std_arr < <(jq -r 'if .state_std then .state_std[] else empty end' "${CONFIG}")

common_args=(
  --wandb_project walker27
  --observation_function mask_dimensions
  --observation_noise_std 0.1
  --observation_mask_dims 0 1
  --history 16
  --min_visible_length 16
  --delta_t 1
  --inverse_loss_weight 1.0
  --forward_loss_weight 1.0
  --state_loss_weight 1.0
  --factor_loss_weight 1.0
  --policy_loss_weight 1.0
  --intra_traj_consistency_loss_weight 0.0
  --inter_traj_consistency_loss_weight 0.0
  --d_model 256
  --n_layer 4
  --head_hidden 256
  --n_head 8
  --d_ff 1024
  --dropout 0.1
  --adaptive_pooling_heads 8
  --adaptive_pooling_dropout 0.1
  --pos_encoding_max_len 5000
  --learning_rate 0.0003
  --num_epochs 10
  --batch_size 128
  --window_size 2
  --eval_interval 1
  --train_split 0.8
  --save_checkpoint_epochs 10
)

if [[ -n "${NUM_EPOCHS}" ]]; then
  common_args+=(--num_epochs "${NUM_EPOCHS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  common_args+=(--batch_size "${BATCH_SIZE}")
fi

cmd=(
  "${PYTHON_RUN[@]}"
  "${ROOT_DIR}/train_embedding.py"
  --dataset_name "${dataset_name}"
  --embedding_size "${embedding_size}"
  --train_task_ids ${train_task_ids}
  --test_task_ids ${test_task_ids}
  --log_dir "${log_dir}"
  --device "${DEVICE}"
  --seed "${SEED}"
  "${common_args[@]}"
)

if [[ ${#state_mean_arr[@]} -gt 0 ]]; then
  cmd+=(--state_mean "${state_mean_arr[@]}")
fi
if [[ ${#state_std_arr[@]} -gt 0 ]]; then
  cmd+=(--state_std "${state_std_arr[@]}")
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG=${CONFIG}"
echo "ENV=${env_name} DATASET=${dataset_name}"
echo "CONDA_ENV=${CONDA_ENV} GPU=${GPU}"
echo "Launching embedding training..."

PYTHONUNBUFFERED=1 \
PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
"${cmd[@]}"
