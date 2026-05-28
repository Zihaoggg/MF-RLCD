#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

print_title() {
  printf '\n========== %s ==========\n' "$1"
}

show_cmd() {
  printf '[Command]'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  show_cmd "$@"
  (
    cd "$PROJECT_ROOT"
    "$@"
  )
}

get_venv_python() {
  local venv_dir="$1"
  if [[ -x "${venv_dir}/bin/python" ]]; then
    printf '%s\n' "${venv_dir}/bin/python"
    return 0
  fi
  if [[ -x "${venv_dir}/Scripts/python.exe" ]]; then
    printf '%s\n' "${venv_dir}/Scripts/python.exe"
    return 0
  fi
  return 1
}
