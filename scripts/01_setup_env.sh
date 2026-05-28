#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
MATLAB_ENGINE_DIR="${MATLAB_ENGINE_DIR:-}"

print_title "1/3 Prepare Python environment"

if [[ "$USE_VENV" == "1" ]]; then
  run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
  PYTHON_FOR_PROJECT="$(get_venv_python "$VENV_DIR")"
  echo "Virtual environment ready: $VENV_DIR"
else
  PYTHON_FOR_PROJECT="$PYTHON_BIN"
  echo "Skipping virtual environment; using Python: $PYTHON_FOR_PROJECT"
fi

print_title "2/3 Install Python dependencies"
run_cmd "$PYTHON_FOR_PROJECT" -m pip install --upgrade pip
run_cmd "$PYTHON_FOR_PROJECT" -m pip install -r "$PROJECT_ROOT/requirements.txt"

print_title "3/3 Install MATLAB Python Engine (optional)"
if [[ -n "$MATLAB_ENGINE_DIR" ]]; then
  if [[ -d "$MATLAB_ENGINE_DIR" ]]; then
    run_cmd "$PYTHON_FOR_PROJECT" -m pip install "$MATLAB_ENGINE_DIR"
    echo "MATLAB Python Engine installed."
  else
    echo "[Warn] MATLAB_ENGINE_DIR does not exist: $MATLAB_ENGINE_DIR"
  fi
else
  echo "MATLAB_ENGINE_DIR is not set; skipping MATLAB Engine installation."
  echo "To install it later, run for example:"
  echo "  MATLAB_ENGINE_DIR=/usr/local/MATLAB/R2024a/extern/engines/python bash scripts/01_setup_env.sh"
fi

echo
echo "Example for reusing the same Python:"
echo "  PYTHON_BIN=\"$PYTHON_FOR_PROJECT\" bash scripts/02_optimize.sh"
