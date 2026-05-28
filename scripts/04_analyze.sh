#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

INPUT_ROOT="${INPUT_ROOT:-./opt_run/BENCH}"
OUTPUT_DIR="${OUTPUT_DIR:-./opt_run/BENCH/analysis}"

print_title "Analyze benchmark results"

cmd=(
  "$PYTHON_BIN" "./run_project.py" "analyze"
  --input_root "$INPUT_ROOT"
  --output_dir "$OUTPUT_DIR"
)

run_cmd "${cmd[@]}"
