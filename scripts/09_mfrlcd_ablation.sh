#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_FILE="${TARGET_FILE:-./configs/mfrlcd_ablation_targets.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./opt_run/MFRLCD_ABLATION_9}"
FRAMEWORK_PATH="${FRAMEWORK_PATH:-./rl_diffusion_framework_v7.py}"
BASE_SEED="${BASE_SEED:-101}"
BUDGET_EVALS="${BUDGET_EVALS:-300}"
NUM_TASKS="${NUM_TASKS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_RESUME="${NO_RESUME:-0}"

ABLATIONS="${ABLATIONS:-full no_updates no_ppo no_diffusion no_refine}"

MFRLCD_PPO_MIN_BATCH="${MFRLCD_PPO_MIN_BATCH:-8}"
MFRLCD_UPDATE_EVERY_STEPS="${MFRLCD_UPDATE_EVERY_STEPS:-1}"
MFRLCD_DIFFUSION_TRAIN_EVERY="${MFRLCD_DIFFUSION_TRAIN_EVERY:-5}"
MFRLCD_DIFFUSION_STEPS="${MFRLCD_DIFFUSION_STEPS:-50}"
MFRLCD_DIFFUSION_TOP_FRAC="${MFRLCD_DIFFUSION_TOP_FRAC:-0.30}"
MFRLCD_COST_PER_EVAL="${MFRLCD_COST_PER_EVAL:-0.08}"
MFRLCD_COST_STEPS_SCALE="${MFRLCD_COST_STEPS_SCALE:-0.04}"
MFRLCD_LOCAL_REFINE_ROUNDS="${MFRLCD_LOCAL_REFINE_ROUNDS:-1}"
MFRLCD_LOCAL_REFINE_SCALE="${MFRLCD_LOCAL_REFINE_SCALE:-0.08}"
MFRLCD_LOCAL_REFINE_DECAY="${MFRLCD_LOCAL_REFINE_DECAY:-0.65}"
MFRLCD_DETERMINISTIC_POLICY="${MFRLCD_DETERMINISTIC_POLICY:-0}"

ABLATION_MANIFEST="${OUTPUT_ROOT}/ablation_runs.csv"
FINAL_ANALYSIS_DIR="${OUTPUT_ROOT}/analysis"

if [[ -f "$PROJECT_ROOT/${TARGET_FILE#./}" ]]; then
  TARGET_FILE_ABS="$PROJECT_ROOT/${TARGET_FILE#./}"
elif [[ -f "$TARGET_FILE" ]]; then
  TARGET_FILE_ABS="$TARGET_FILE"
else
  echo "[Error] Target list not found: $TARGET_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
printf 'variant,difficulty,target_id,target_name,E_target,sigma_y_target,Kt_target,seed,budget_evals,run_dir\n' > "$ABLATION_MANIFEST"

run_one_variant() {
  local variant="$1"
  local variant_root="${OUTPUT_ROOT}/${variant}"
  local variant_manifest="${variant_root}/paper_target_runs.csv"
  local analysis_dir="${variant_root}/analysis"

  mkdir -p "$variant_root"
  printf 'difficulty,target_id,target_name,E_target,sigma_y_target,Kt_target,seed,budget_evals,run_dir\n' > "$variant_manifest"

  print_title "Start ablation variant: ${variant}"

  local target_index=0
  while IFS=, read -r difficulty target_id target_name E_TARGET SIGMA_Y_TARGET KT_TARGET; do
    if [[ "$difficulty" == "difficulty" ]]; then
      continue
    fi

    target_index=$((target_index + 1))
    local seed=$((BASE_SEED + target_index - 1))
    local run_dir="${variant_root}/${difficulty}/${target_name}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$difficulty" "$target_id" "$target_name" "$E_TARGET" "$SIGMA_Y_TARGET" "$KT_TARGET" "$seed" "$BUDGET_EVALS" "$run_dir" \
      >> "$variant_manifest"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$variant" "$difficulty" "$target_id" "$target_name" "$E_TARGET" "$SIGMA_Y_TARGET" "$KT_TARGET" "$seed" "$BUDGET_EVALS" "$run_dir" \
      >> "$ABLATION_MANIFEST"

    if [[ "$SKIP_EXISTING" == "1" && -f "$run_dir/benchmark_results.json" ]]; then
      echo "[Skip] ${variant}/${target_name} already has benchmark_results.json"
      continue
    fi

    local enable_updates=1
    local ppo_min_batch="$MFRLCD_PPO_MIN_BATCH"
    local diffusion_train_every="$MFRLCD_DIFFUSION_TRAIN_EVERY"
    local local_refine_rounds="$MFRLCD_LOCAL_REFINE_ROUNDS"

    case "$variant" in
      full)
        ;;
      no_updates)
        enable_updates=0
        ;;
      no_ppo)
        enable_updates=1
        ppo_min_batch=1000000
        ;;
      no_diffusion)
        enable_updates=1
        diffusion_train_every=1000000
        ;;
      no_refine)
        enable_updates=1
        local_refine_rounds=0
        ;;
      *)
        echo "[Error] Unknown ablation variant: $variant"
        exit 1
        ;;
    esac

    print_title "Run ${variant} / ${target_name} (${difficulty})"

    cmd=(
      "$PYTHON_BIN" "-m" "benchmark.main"
      --framework_path "$FRAMEWORK_PATH"
      --work_dir "$run_dir"
      --seed "$seed"
      --E_target "$E_TARGET"
      --sigma_y_target "$SIGMA_Y_TARGET"
      --Kt_target "$KT_TARGET"
      --budget_evals "$BUDGET_EVALS"
      --num_tasks "$NUM_TASKS"
      --run_mfrlcd
      --mfrlcd_ppo_min_batch "$ppo_min_batch"
      --mfrlcd_update_every_steps "$MFRLCD_UPDATE_EVERY_STEPS"
      --mfrlcd_diffusion_train_every "$diffusion_train_every"
      --mfrlcd_diffusion_steps "$MFRLCD_DIFFUSION_STEPS"
      --mfrlcd_diffusion_top_frac "$MFRLCD_DIFFUSION_TOP_FRAC"
      --mfrlcd_cost_per_eval "$MFRLCD_COST_PER_EVAL"
      --mfrlcd_cost_steps_scale "$MFRLCD_COST_STEPS_SCALE"
      --mfrlcd_local_refine_rounds "$local_refine_rounds"
      --mfrlcd_local_refine_scale "$MFRLCD_LOCAL_REFINE_SCALE"
      --mfrlcd_local_refine_decay "$MFRLCD_LOCAL_REFINE_DECAY"
      --save_json
    )

    if [[ "$enable_updates" == "1" ]]; then
      cmd+=(--mfrlcd_enable_updates)
    fi

    if [[ "$MFRLCD_DETERMINISTIC_POLICY" == "1" ]]; then
      cmd+=(--mfrlcd_deterministic_policy)
    fi

    if [[ "$NO_RESUME" == "1" ]]; then
      cmd+=(--no_resume)
    fi

    run_cmd "${cmd[@]}"
  done < "$TARGET_FILE_ABS"

  print_title "Aggregate target results for ${variant}"
  run_cmd \
    "$PYTHON_BIN" -m benchmark.paper_analysis \
    --input_root "$variant_root" \
    --manifest "$variant_manifest" \
    --output_dir "$analysis_dir"
}

for variant in $ABLATIONS; do
  run_one_variant "$variant"
done

print_title "Aggregate all ablation variants"
run_cmd \
  "$PYTHON_BIN" -m benchmark.ablation_analysis \
  --input_root "$OUTPUT_ROOT" \
  --output_dir "$FINAL_ANALYSIS_DIR"

echo
echo "Ablation experiments finished."
echo "Manifest: $ABLATION_MANIFEST"
echo "Target comparison: $FINAL_ANALYSIS_DIR/ablation_target_summary.csv"
echo "Difficulty comparison: $FINAL_ANALYSIS_DIR/ablation_difficulty_summary.csv"
echo "Variant comparison: $FINAL_ANALYSIS_DIR/ablation_variant_summary.csv"
