#!/usr/bin/env bash
# Sweep script to run eval_diffusion_meta_dt_style.py across planner_ckpt values
# Usage examples:
#  ./scripts/eval_ckpt_sweep.sh --ckpts 100000,200000,300000 --env_type walker --dataset_name RandomWalker2d/50dynamics-v0 --dadp_checkpoint ./dadp/...zip
#  ./scripts/eval_ckpt_sweep.sh --range 100000:500000:100000 ...

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="$ROOT_DIR/eval_diffusion_meta_dt_style.py"

# Defaults (can be overridden by CLI)
# add timestamp to default out dir so each sweep run is separated
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$ROOT_DIR/results/eval_ckpt_sweep_${TIMESTAMP}"
REPEATS=1
SLEEP_BETWEEN=2

# CLI-only required args default to empty (you must pass them if needed)
ENV_TYPE=""
DATASET_NAME=""
DADP_CHECKPOINT=""
PLANNER_HORIZON=""
NOISE_TYPE=""
CONDITION=""
NUM_TASKS=""
NUM_TRAIN_TASKS=""
EVAL_ON_TRAIN_TASKS=""
NUM_EVAL_TRAIN_TASKS=""
DATA_QUALITY=""
DEVICE=""
NUM_EVAL_EPISODES=5

# Input ckpts: comma list or range syntax start:stop:step
CKPTS_LIST=""

print_usage(){
  cat <<EOF
Usage: $0 [options]
Options:
  --ckpts LIST            Comma-separated planner_ckpt values (e.g. 100000,200000,300000)
  --range START:STOP:STEP  Range of ckpts (inclusive start, exclusive stop) e.g. 100000:500000:100000
  --repeats N             Repeat each ckpt N times (default: ${REPEATS})
  --sleep S               Seconds to sleep between runs (default: ${SLEEP_BETWEEN})
  --out_dir PATH          Directory to store logs/results (default: ${OUT_DIR})

  (CLI-only args - must be passed here if they are not set elsewhere)
  --env_type TYPE
  --dataset_name NAME
  --dadp_checkpoint PATH
  --planner_horizon N
  --noise_type STR
  --condition BOOL        true/false
  --num_tasks N
  --num_train_tasks N
  --eval_on_train_tasks   flag (set to 'true' to evaluate on training tasks)
  --num_eval_train_tasks N
  --data_quality STR
  --device DEVICE
  --num_eval_episodes N
  --help                  Show this message

Example:
  $0 --ckpts 100000,200000,300000 --env_type walker --dataset_name RandomWalker2d/50dynamics-v0 \
     --dadp_checkpoint ./dadp/embedding/logs/transformer/exp_walker_medium/best_model.zip --device cpu
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpts)
      CKPTS_LIST="$2"; shift 2;;
    --range)
      RANGE_SPEC="$2"; shift 2;;
    --repeats)
      REPEATS="$2"; shift 2;;
    --sleep)
      SLEEP_BETWEEN="$2"; shift 2;;
    --out_dir)
      OUT_DIR="$2"; shift 2;;
    --env_type)
      ENV_TYPE="$2"; shift 2;;
    --dataset_name)
      DATASET_NAME="$2"; shift 2;;
    --dadp_checkpoint)
      DADP_CHECKPOINT="$2"; shift 2;;
    --planner_horizon)
      PLANNER_HORIZON="$2"; shift 2;;
    --noise_type)
      NOISE_TYPE="$2"; shift 2;;
    --condition)
      CONDITION="$2"; shift 2;;
    --num_tasks)
      NUM_TASKS="$2"; shift 2;;
    --num_train_tasks)
      NUM_TRAIN_TASKS="$2"; shift 2;;
    --eval_on_train_tasks)
      EVAL_ON_TRAIN_TASKS="$2"; shift 2;;
    --num_eval_train_tasks)
      NUM_EVAL_TRAIN_TASKS="$2"; shift 2;;
    --data_quality)
      DATA_QUALITY="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --num_eval_episodes)
      NUM_EVAL_EPISODES="$2"; shift 2;;
    --help)
      print_usage; exit 0;;
    *)
      echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

# build planner ckpt array
CKPTS=()
if [[ -n "$CKPTS_LIST" ]]; then
  IFS=',' read -r -a tmp <<< "$CKPTS_LIST"
  for v in "${tmp[@]}"; do
    CKPTS+=("$v")
  done
elif [[ -n "${RANGE_SPEC:-}" ]]; then
  IFS=':' read -r START STOP STEP <<< "$RANGE_SPEC"
  if [[ -z "$START" || -z "$STOP" || -z "$STEP" ]]; then
    echo "Invalid range specification. Use START:STOP:STEP."; exit 1
  fi
  for ((i=START; i<STOP; i+=STEP)); do
    CKPTS+=("$i")
  done
else
  echo "No ckpts provided. Use --ckpts or --range."; exit 1
fi

mkdir -p "$OUT_DIR"

echo "Starting planner_ckpt sweep: ${CKPTS[*]}"

for ckpt in "${CKPTS[@]}"; do
  for r in $(seq 1 "$REPEATS"); do
    ts=$(date +%Y%m%d-%H%M%S)
    run_dir="$OUT_DIR/ckpt_${ckpt}/run_${r}_${ts}"
    mkdir -p "$run_dir"
    log_file="$run_dir/run.log"

    echo "Running planner_ckpt=${ckpt} (repeat ${r}) -> $log_file"

    cmd=("$ROOT_DIR/venv/bin/python" "$EVAL_PY")
    # if no venv python exists, fall back to system python
    if [[ ! -x "${cmd[0]}" ]]; then
      cmd=("python" "$EVAL_PY")
    fi

    # Only pass the planner checkpoint and eval episodes —
    # rely on eval script defaults (in .py and JSON) for everything else.
    cmd+=(--planner_ckpt "$ckpt")
    cmd+=(--num_eval_episodes "$NUM_EVAL_EPISODES")

    # optional: other flags may be picked up from config JSON defaults

    echo "Command: ${cmd[*]}" | tee -a "$log_file"
    start_ts=$(date +%s)
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    end_ts=$(date +%s)
    duration=$((end_ts - start_ts))
    echo "Run finished in ${duration}s" | tee -a "$log_file"

    # small sleep
    sleep "$SLEEP_BETWEEN"
  done
done

echo "Sweep complete. Results saved to $OUT_DIR"
