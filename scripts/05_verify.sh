#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WORK_DIR="${WORK_DIR:-./opt_run/RL_Diffusion/T001}"

print_title "Run verification"

cmd=(
  "$PYTHON_BIN" "./run_project.py" "verify"
  --work_dir "$WORK_DIR"
)

run_cmd "${cmd[@]}"
