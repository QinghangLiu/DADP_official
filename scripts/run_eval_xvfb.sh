#!/usr/bin/env bash
# Wrapper to run a Python script under Xvfb. Usage:
#   ./run_eval_xvfb.sh /path/to/script.py [args...]
# Note: ensure xvfb-run is installed on the system.
set -euo pipefail
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run not found. Install with: sudo apt-get install xvfb" >&2
  exit 1
fi
# Default screen size can be overridden by XVFB_SCREEN env var
SCREEN=${XVFB_SCREEN:-"-screen 0 1400x900x24"}
# Use the same Python executable as the environment running this script (if run via full path), otherwise rely on PATH
PYTHON=${PYTHON:-python}
# If first arg is a Python script, run under xvfb-run
if [ $# -lt 1 ]; then
  # Default to the repository's eval script when no args are provided
  SCRIPT="${PWD}/eval_diffusion_meta_dt_style.py"
  ARGS=("--env_type" "walker" "--plot" "True" "--num_eval_episodes" "1")
else
  SCRIPT="$1"
  shift
  ARGS=("$@")
fi

exec xvfb-run -s "$SCREEN" "$PYTHON" "$SCRIPT" "${ARGS[@]}"
