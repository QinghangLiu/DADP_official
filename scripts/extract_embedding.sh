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
  scripts/extract_embedding.sh <env_key> --gpu <id> [options]
  scripts/extract_embedding.sh --config <path> --gpu <id> [options]

Required:
  --gpu <id>           GPU id for CUDA_VISIBLE_DEVICES

Select one of:
  <env_key>            One of: ant, halfcheetah, walker, hopper, door, relocate
  --config <path>      Path to embedding config JSON (overrides env key mapping)

Optional:
  --seed <int>         Seed value (default: 42)
  --device <cuda:x>    Device passed to eval_embedding.py (default: cuda:0)
  --batch_size <int>   Override batch size (default: 256)
  --checkpoint <path>  Override checkpoint path (default: log_dir/best_model.zip)

Env overrides:
  CONDA_ENV=<name>     Conda env name (default: dadp310)
  CONDA_BIN=<path>     Conda executable (default: from PATH)

Examples:
  bash scripts/extract_embedding.sh ant --gpu 0
  bash scripts/extract_embedding.sh walker --gpu 1 --batch_size 128
USAGE
}

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required. Install it first." >&2
  exit 1
fi

CONFIG=""
ENV_KEY=""
SEED="42"
GPU=""
DEVICE="cuda:0"
BATCH_SIZE="256"
CHECKPOINT_OVERRIDE=""

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
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${GPU}" ]]; then
  echo "Missing required arguments (--gpu)." >&2
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

dataset_name="$(jq -r '.dataset_name' "${CONFIG}")"
log_dir="$(jq -r '.log_dir' "${CONFIG}")"

readarray -t state_mean_arr < <(jq -r 'if .state_mean then .state_mean[] else empty end' "${CONFIG}")
readarray -t state_std_arr < <(jq -r 'if .state_std then .state_std[] else empty end' "${CONFIG}")

checkpoint_path="${LOG_DIR:-${log_dir}}/best_model.zip"
if [[ -n "${CHECKPOINT_OVERRIDE}" ]]; then
  checkpoint_path="${CHECKPOINT_OVERRIDE}"
fi

if [[ ! -f "${checkpoint_path}" ]]; then
  echo "Warning: Checkpoint not found at ${checkpoint_path}. Ensure it exists or provide --checkpoint." >&2
fi

cmd=(
  "${PYTHON_RUN[@]}"
  "${ROOT_DIR}/eval_embedding.py"
  --checkpoint_path "${checkpoint_path}"
  --dataset_name "${dataset_name}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ ${#state_mean_arr[@]} -gt 0 ]]; then
  cmd+=(--state_mean "${state_mean_arr[@]}")
fi
if [[ ${#state_std_arr[@]} -gt 0 ]]; then
  cmd+=(--state_std "${state_std_arr[@]}")
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG=${CONFIG}"
echo "DATASET=${dataset_name}"
echo "CHECKPOINT=${checkpoint_path}"
echo "CONDA_ENV=${CONDA_ENV} GPU=${GPU}"
echo "Launching embedding extraction..."

PYTHONUNBUFFERED=1 \
PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
"${cmd[@]}"
