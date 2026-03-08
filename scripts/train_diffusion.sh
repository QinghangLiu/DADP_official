#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
  scripts/run_single_seed.sh <env_key> --seed <int> --gpu <id> [options]
  scripts/run_single_seed.sh --config <path> --seed <int> --gpu <id> [options]

Required:
  --seed <int>         Seed value
  --gpu <id>           GPU id for CUDA_VISIBLE_DEVICES

Select one of:
  <env_key>            One of: ant, halfcheetah, walker, hopper, door, relocate
  --config <path>      Path to args.json (overrides env key mapping)

Optional:
  --mode <train|test|inference>   Run mode (default: train)
  --save_dir <path>              Root save dir (default: ./results)
  --suffix <text>                Suffix appended to pipeline_name/name
  --eval <none|id|ood|both>       Run inference evals after train (default: none)
  --planner_ckpt <name>          Planner checkpoint name (default: latest)
  --device <cuda:x>              Device string passed to script (default: cuda:0)
  --enable_wandb <true|false>    Toggle wandb (default: false)
  CONDA_ENV=<name>               Conda env name (default: dadp310)
  CONDA_BIN=<path|conda>         Conda executable (default: from PATH)

Examples:
  scripts/run_single_seed.sh ant --seed 0 --gpu 0 \
    --mode train --save_dir results --suffix _single

  scripts/run_single_seed.sh walker --seed 1 --gpu 1 \
    --eval both --save_dir results
USAGE
}

CONFIG=""
ENV_KEY=""
SEED=""
GPU=""
MODE="train"
SAVE_DIR="${ROOT_DIR}/results"
SUFFIX=""
EVAL_MODE="none"
PLANNER_CKPT="latest"
DEVICE="cuda:0"
ENABLE_WANDB="false"

declare -A DADP_CONFIGS=(
  [ant]="${ROOT_DIR}/config/diffusion_configs/ant_args.json"
  [halfcheetah]="${ROOT_DIR}/config/diffusion_configs/halfcheetah_args.json"
  [walker]="${ROOT_DIR}/config/diffusion_configs/walker_args.json"
  [hopper]="${ROOT_DIR}/config/diffusion_configs/hopper_args.json"
  [door]="${ROOT_DIR}/config/diffusion_configs/door_args.json"
  [relocate]="${ROOT_DIR}/config/diffusion_configs/relocate_args.json"
)

# Optional positional env key for simple commands, e.g. "scripts/run_single_seed.sh ant --seed 0 --gpu 0"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  ENV_KEY="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --save_dir) SAVE_DIR="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    --eval) EVAL_MODE="$2"; shift 2 ;;
    --planner_ckpt) PLANNER_CKPT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --enable_wandb) ENABLE_WANDB="$2"; shift 2 ;;
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

  CONFIG="${DADP_CONFIGS[${ENV_KEY}]:-}"

  if [[ -z "${CONFIG}" ]]; then
    echo "No config mapping for env='${ENV_KEY}'." >&2
    exit 1
  fi
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG=${CONFIG}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "GPU=${GPU} MODE=${MODE}"

run_launch() {
  local run_mode="$1"
  shift
  echo "Launching: CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_RUN[*]} ${ROOT_DIR}/scripts/launch_from_config.py --config ${CONFIG} --seed ${SEED} --device ${DEVICE} --save_dir ${SAVE_DIR} --pipeline_suffix=${SUFFIX} --enable_wandb ${ENABLE_WANDB} --mode ${run_mode} $*"
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_RUN[@]}" "${ROOT_DIR}/scripts/launch_from_config.py" \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --save_dir "${SAVE_DIR}" \
    --pipeline_suffix="${SUFFIX}" \
    --enable_wandb "${ENABLE_WANDB}" \
    --mode "${run_mode}" "$@"
}

# Run primary mode
run_launch "${MODE}"

# Optional inference evals
if [[ "${EVAL_MODE}" == "id" || "${EVAL_MODE}" == "both" ]]; then
  run_launch "inference" --eval_task_mode id --planner_ckpt "${PLANNER_CKPT}"
fi

if [[ "${EVAL_MODE}" == "ood" || "${EVAL_MODE}" == "both" ]]; then
  run_launch "inference" --eval_task_mode ood --planner_ckpt "${PLANNER_CKPT}"
fi
