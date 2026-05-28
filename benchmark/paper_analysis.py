from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    sanitized_rows: List[dict] = []
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        clean_row = {str(key): value for key, value in row.items() if key is not None}
        sanitized_rows.append(clean_row)
        for key in clean_row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sanitized_rows:
            writer.writerow(row)

def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default

def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default

def safe_bool(value) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None

def normalize_update_rows(rows: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key is not None}

        if clean.get("update_type") in {"ppo", "diffusion"}:
            normalized.append(clean)
            continue

        if clean.get("event") == "mfrlcd_config":
            normalized.append(clean)
            continue

        shifted_type = str(clean.get("enable_updates", "")).strip().lower()
        if shifted_type in {"ppo", "diffusion"}:
            normalized.append(
                {
                    "step": safe_int(clean.get("event")),
                    "update_type": shifted_type,
                    "applied": safe_bool(clean.get("ppo_min_batch")),
                    "buffer_size": safe_int(clean.get("update_every_steps")),
                    "train_steps": safe_int(clean.get("diffusion_train_every")),
                    "top_frac": safe_float(clean.get("diffusion_steps")),
                }
            )
            continue

        normalized.append(clean)
    return normalized

def first_success_eval(history: List[dict]) -> Optional[int]:
    for item in history:
        if bool(item.get("success", False)):
            return safe_int(item.get("eval"))
    return None

def best_reward_eval(history: List[dict], best_reward: float) -> Optional[int]:
    for item in history:
        reward = safe_float(item.get("reward"), default=None)
        if reward is None:
            continue
        if abs(reward - float(best_reward)) <= 1e-9:
            return safe_int(item.get("eval"))
    return None

def compute_errors(result: dict, target_row: dict) -> Dict[str, Optional[float]]:
    best_E = safe_float(result.get("best_E"))
    best_Sy = safe_float(result.get("best_Sy"))
    best_Kt = safe_float(result.get("best_Kt"))
    target_E = safe_float(target_row.get("E_target"))
    target_Sy = safe_float(target_row.get("sigma_y_target"))
    target_Kt = safe_float(target_row.get("Kt_target"))

    if None in {best_E, best_Sy, best_Kt, target_E, target_Sy, target_Kt}:
        return {
            "best_E_error": None,
            "best_sigma_y_error": None,
            "best_Kt_error": None,
            "best_error_sum": None,
        }

    eE = abs(best_E - target_E) / max(abs(target_E), 1e-12)
    eS = abs(best_Sy - target_Sy) / max(abs(target_Sy), 1e-12)
    eK = abs(best_Kt - target_Kt) / max(abs(target_Kt), 1e-12)
    return {
        "best_E_error": eE,
        "best_sigma_y_error": eS,
        "best_Kt_error": eK,
        "best_error_sum": eE + eS + eK,
    }

def load_manifest(manifest_path: Path) -> List[dict]:
    rows = read_csv_rows(manifest_path)
    for row in rows:
        row["run_dir"] = str(Path(row["run_dir"]).resolve())
        row["seed"] = safe_int(row.get("seed"))
        row["budget_evals"] = safe_int(row.get("budget_evals"))
        row["E_target"] = safe_float(row.get("E_target"))
        row["sigma_y_target"] = safe_float(row.get("sigma_y_target"))
        row["Kt_target"] = safe_float(row.get("Kt_target"))
    return rows

def collect_target_outputs(target_row: dict) -> dict:
    run_dir = Path(target_row["run_dir"])
    result_path = run_dir / "benchmark_results.json"
    root_events_path = run_dir / "records" / "events.jsonl"
    method_dir = run_dir / "T000_MF-RLCD"
    eval_path = method_dir / "records" / "evaluations.csv"
    diag_path = method_dir / "records" / "mfrlcd_diagnostics.csv"
    updates_path = method_dir / "records" / "mfrlcd_updates.csv"
    method_events_path = method_dir / "records" / "events.jsonl"

    results = read_json(result_path) if result_path.exists() else []
    result = results[0] if isinstance(results, list) and results else {}
    history = result.get("history", []) if isinstance(result, dict) else []
    eval_rows = read_csv_rows(eval_path)
    diag_rows = read_csv_rows(diag_path)
    update_rows = normalize_update_rows(read_csv_rows(updates_path))
    root_events = read_jsonl(root_events_path)
    method_events = read_jsonl(method_events_path)

    solver_event = None
    for item in root_events:
        if item.get("event_type") == "solver_result" and item.get("method") == "MF-RLCD":
            solver_event = item
            break

    ppo_updates = [row for row in update_rows if row.get("update_type") == "ppo" and safe_bool(row.get("applied"))]
    diffusion_updates = [row for row in update_rows if row.get("update_type") == "diffusion" and safe_bool(row.get("applied"))]

    diag_success_steps = [
        safe_int(row.get("step"))
        for row in diag_rows
        if safe_bool(row.get("step_best_success"))
    ]
    diag_success_steps = [step for step in diag_success_steps if step is not None]

    final_diag = diag_rows[-1] if diag_rows else {}
    global_best_step = None
    if diag_rows:
        final_global_best_reward = safe_float(final_diag.get("global_best_reward"))
        for row in diag_rows:
            if abs(safe_float(row.get("global_best_reward"), -1e30) - final_global_best_reward) <= 1e-9:
                global_best_step = safe_int(row.get("step"))
                break

    summary = {
        **target_row,
        "benchmark_result_path": str(result_path.resolve()) if result_path.exists() else "",
        "method_dir": str(method_dir.resolve()) if method_dir.exists() else "",
        "n_evals": safe_int(result.get("n_evals"), 0),
        "success": bool(result.get("success", False)),
        "best_reward": safe_float(result.get("best_reward")),
        "best_E": safe_float(result.get("best_E")),
        "best_Sy": safe_float(result.get("best_Sy")),
        "best_Kt": safe_float(result.get("best_Kt")),
        "first_success_eval": first_success_eval(history),
        "best_reward_eval": best_reward_eval(history, safe_float(result.get("best_reward"), -1e30)),
        "duration_sec": safe_float(None if solver_event is None else solver_event.get("duration_sec")),
        "ppo_update_count": len(ppo_updates),
        "diffusion_update_count": len(diffusion_updates),
        "num_diagnostic_steps": len(diag_rows),
        "num_success_steps": len(diag_success_steps),
        "first_success_step": diag_success_steps[0] if diag_success_steps else None,
        "global_best_reached_step": global_best_step,
        "final_global_best_reward": safe_float(final_diag.get("global_best_reward")),
        "final_global_best_error_sum": safe_float(final_diag.get("global_best_error_sum")),
        "final_step_best_reward": safe_float(final_diag.get("step_best_reward")),
        "final_step_best_error_sum": safe_float(final_diag.get("step_best_error_sum")),
        "evaluation_rows": len(eval_rows),
        "diagnostic_rows": len(diag_rows),
        "update_rows": len(update_rows),
        "root_event_rows": len(root_events),
        "method_event_rows": len(method_events),
    }
    summary.update(compute_errors(result, target_row))

    return {
        "summary": summary,
        "evaluations": eval_rows,
        "diagnostics": diag_rows,
        "updates": update_rows,
    }

def tag_rows(rows: List[dict], meta: dict, prefix_keys: Optional[List[str]] = None) -> List[dict]:
    prefix_keys = prefix_keys or [
        "difficulty",
        "target_id",
        "target_name",
        "E_target",
        "sigma_y_target",
        "Kt_target",
        "seed",
        "budget_evals",
        "run_dir",
    ]
    tagged = []
    for row in rows:
        tagged.append({key: meta.get(key) for key in prefix_keys} | row)
    return tagged

def summarize_by_difficulty(rows: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["difficulty"])].append(row)

    summary_rows = []
    for difficulty, items in sorted(grouped.items()):
        success_rate = sum(1 for item in items if bool(item["success"])) / max(1, len(items))
        error_sums = [item["best_error_sum"] for item in items if item["best_error_sum"] is not None]
        first_success_evals = [item["first_success_eval"] for item in items if item["first_success_eval"] is not None]
        summary_rows.append(
            {
                "difficulty": difficulty,
                "num_targets": len(items),
                "success_rate": success_rate,
                "avg_best_reward": sum(item["best_reward"] for item in items if item["best_reward"] is not None) / max(1, len(items)),
                "avg_best_error_sum": (sum(error_sums) / len(error_sums)) if error_sums else None,
                "avg_first_success_eval": (sum(first_success_evals) / len(first_success_evals)) if first_success_evals else None,
                "avg_n_evals": sum(item["n_evals"] for item in items) / max(1, len(items)),
                "avg_ppo_update_count": sum(item["ppo_update_count"] for item in items) / max(1, len(items)),
                "avg_diffusion_update_count": sum(item["diffusion_update_count"] for item in items) / max(1, len(items)),
                "avg_duration_sec": sum(item["duration_sec"] for item in items if item["duration_sec"] is not None) / max(1, len(items)),
            }
        )
    return summary_rows

def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate paper-ready MF-RLCD batch outputs.")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser

def main():
    args = build_parser().parse_args()
    input_root = Path(str(args.input_root).replace("\\", "/")).resolve()
    manifest_path = Path(str(args.manifest).replace("\\", "/")).resolve()
    output_dir = Path(str(args.output_dir).replace("\\", "/")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(manifest_path)
    target_summary_rows: List[dict] = []
    evaluation_rows: List[dict] = []
    diagnostic_rows: List[dict] = []
    update_rows: List[dict] = []

    for target_row in manifest_rows:
        outputs = collect_target_outputs(target_row)
        target_summary_rows.append(outputs["summary"])
        evaluation_rows.extend(tag_rows(outputs["evaluations"], outputs["summary"]))
        diagnostic_rows.extend(tag_rows(outputs["diagnostics"], outputs["summary"]))
        update_rows.extend(tag_rows(outputs["updates"], outputs["summary"]))

    difficulty_summary_rows = summarize_by_difficulty(target_summary_rows)

    write_csv(output_dir / "paper_target_summary.csv", target_summary_rows)
    write_csv(output_dir / "paper_difficulty_summary.csv", difficulty_summary_rows)
    write_csv(output_dir / "paper_evaluations.csv", evaluation_rows)
    write_csv(output_dir / "paper_diagnostics.csv", diagnostic_rows)
    write_csv(output_dir / "paper_updates.csv", update_rows)

    summary = {
        "input_root": str(input_root),
        "manifest": str(manifest_path),
        "num_targets": len(target_summary_rows),
        "difficulty_levels": sorted({row["difficulty"] for row in target_summary_rows}),
        "generated_files": [
            str((output_dir / "paper_target_summary.csv").resolve()),
            str((output_dir / "paper_difficulty_summary.csv").resolve()),
            str((output_dir / "paper_evaluations.csv").resolve()),
            str((output_dir / "paper_diagnostics.csv").resolve()),
            str((output_dir / "paper_updates.csv").resolve()),
        ],
    }
    with (output_dir / "paper_analysis_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"[PaperAnalysis] targets={len(target_summary_rows)} difficulties={len(summary['difficulty_levels'])}")
    print(f"[Saved] {(output_dir / 'paper_target_summary.csv').resolve()}")
    print(f"[Saved] {(output_dir / 'paper_difficulty_summary.csv').resolve()}")
    print(f"[Saved] {(output_dir / 'paper_evaluations.csv').resolve()}")
    print(f"[Saved] {(output_dir / 'paper_diagnostics.csv').resolve()}")
    print(f"[Saved] {(output_dir / 'paper_updates.csv').resolve()}")

if __name__ == "__main__":
    main()
