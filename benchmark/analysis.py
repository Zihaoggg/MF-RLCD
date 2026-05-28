from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

def method_safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

def find_first_success_eval(history: List[dict]) -> Optional[int]:
    for item in history or []:
        if bool(item.get("success", False)):
            return int(item.get("eval", 0))
    return None

def find_best_reward_eval(history: List[dict], best_reward: float) -> Optional[int]:
    for item in history or []:
        try:
            if abs(float(item.get("reward", -1e30)) - float(best_reward)) <= 1e-9:
                return int(item.get("eval", 0))
        except Exception:
            continue
    return None

def extract_target_and_errors(benchmark_dir: Path, result: dict) -> Dict[str, Optional[float]]:
    task_id = int(result["task_id"])
    method = str(result["method"])
    method_dir = benchmark_dir / f"T{task_id:03d}_{method_safe_name(method)}"
    eval_csv = method_dir / "records" / "evaluations.csv"
    rows = read_csv_rows(eval_csv)
    if not rows:
        return {
            "target_E": None,
            "target_sigma_y": None,
            "target_Kt": None,
            "best_E_error": None,
            "best_sigma_y_error": None,
            "best_Kt_error": None,
            "best_error_sum": None,
        }

    first = rows[0]
    try:
        target_E = float(first["target_E"])
        target_Sy = float(first["target_sigma_y"])
        target_Kt = float(first["target_Kt"])
    except Exception:
        return {
            "target_E": None,
            "target_sigma_y": None,
            "target_Kt": None,
            "best_E_error": None,
            "best_sigma_y_error": None,
            "best_Kt_error": None,
            "best_error_sum": None,
        }

    best_E = result.get("best_E")
    best_Sy = result.get("best_Sy")
    best_Kt = result.get("best_Kt")
    if best_E is None or best_Sy is None or best_Kt is None:
        return {
            "target_E": target_E,
            "target_sigma_y": target_Sy,
            "target_Kt": target_Kt,
            "best_E_error": None,
            "best_sigma_y_error": None,
            "best_Kt_error": None,
            "best_error_sum": None,
        }

    eE = abs(float(best_E) - target_E) / max(abs(target_E), 1e-12)
    eS = abs(float(best_Sy) - target_Sy) / max(abs(target_Sy), 1e-12)
    eK = abs(float(best_Kt) - target_Kt) / max(abs(target_Kt), 1e-12)
    return {
        "target_E": target_E,
        "target_sigma_y": target_Sy,
        "target_Kt": target_Kt,
        "best_E_error": eE,
        "best_sigma_y_error": eS,
        "best_Kt_error": eK,
        "best_error_sum": eE + eS + eK,
    }

def collect_result_records(input_root: Path) -> List[dict]:
    records = []
    for result_path in input_root.rglob("benchmark_results.json"):
        benchmark_dir = result_path.parent
        try:
            payload = read_json(result_path)
        except Exception as exc:
            print(f"[Warn] failed to read {result_path}: {exc}")
            continue
        if not isinstance(payload, list):
            continue

        for result in payload:
            history = result.get("history", [])
            target_info = extract_target_and_errors(benchmark_dir, result)
            record = {
                "benchmark_dir": str(benchmark_dir.resolve()),
                "benchmark_file": str(result_path.resolve()),
                "method": result.get("method"),
                "task_id": int(result.get("task_id", -1)),
                "seed": int(result.get("seed", -1)),
                "budget_evals": int(result.get("budget_evals", 0)),
                "n_evals": int(result.get("n_evals", 0)),
                "best_reward": float(result.get("best_reward", -1e30)),
                "best_E": result.get("best_E"),
                "best_Sy": result.get("best_Sy"),
                "best_Kt": result.get("best_Kt"),
                "success": bool(result.get("success", False)),
                "first_success_eval": find_first_success_eval(history),
                "best_reward_eval": find_best_reward_eval(history, float(result.get("best_reward", -1e30))),
                "history_len": len(history),
                **target_info,
            }
            records.append(record)
    return records

def summarize_by_method(records: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)

    rows = []
    for method, items in sorted(grouped.items()):
        success_count = sum(1 for item in items if item["success"])
        first_success = [item["first_success_eval"] for item in items if item["first_success_eval"] is not None]
        error_sums = [item["best_error_sum"] for item in items if item["best_error_sum"] is not None]
        rows.append(
            {
                "method": method,
                "num_records": len(items),
                "success_rate": success_count / max(1, len(items)),
                "avg_best_reward": sum(float(item["best_reward"]) for item in items) / max(1, len(items)),
                "avg_best_error_sum": (sum(error_sums) / len(error_sums)) if error_sums else None,
                "avg_n_evals": sum(int(item["n_evals"]) for item in items) / max(1, len(items)),
                "avg_first_success_eval": (sum(first_success) / len(first_success)) if first_success else None,
            }
        )
    return rows

def mfrlcd_pairwise_summary(records: List[dict]) -> List[dict]:
    grouped: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for record in records:
        key = (record["benchmark_dir"], record["task_id"])
        grouped[key][str(record["method"])] = record

    pairwise: Dict[str, dict] = {}
    for _, methods in grouped.items():
        mfrlcd = methods.get("MF-RLCD")
        if mfrlcd is None:
            continue
        for method, other in methods.items():
            if method == "MF-RLCD":
                continue
            row = pairwise.setdefault(
                method,
                {
                    "baseline_method": method,
                    "num_tasks": 0,
                    "mfrlcd_better_success": 0,
                    "mfrlcd_better_reward": 0,
                    "mfrlcd_better_error_sum": 0,
                    "mfrlcd_faster_first_success": 0,
                },
            )
            row["num_tasks"] += 1
            row["mfrlcd_better_success"] += int(bool(mfrlcd["success"]) and (not bool(other["success"])))
            row["mfrlcd_better_reward"] += int(float(mfrlcd["best_reward"]) > float(other["best_reward"]))
            if (mfrlcd["best_error_sum"] is not None) and (other["best_error_sum"] is not None):
                row["mfrlcd_better_error_sum"] += int(float(mfrlcd["best_error_sum"]) < float(other["best_error_sum"]))
            if (mfrlcd["first_success_eval"] is not None) and (other["first_success_eval"] is not None):
                row["mfrlcd_faster_first_success"] += int(int(mfrlcd["first_success_eval"]) < int(other["first_success_eval"]))

    summary_rows = []
    for _, row in sorted(pairwise.items()):
        num_tasks = max(1, int(row["num_tasks"]))
        summary_rows.append(
            {
                **row,
                "mfrlcd_better_success_rate": row["mfrlcd_better_success"] / num_tasks,
                "mfrlcd_better_reward_rate": row["mfrlcd_better_reward"] / num_tasks,
                "mfrlcd_better_error_sum_rate": row["mfrlcd_better_error_sum"] / num_tasks,
                "mfrlcd_faster_first_success_rate": row["mfrlcd_faster_first_success"] / num_tasks,
            }
        )
    return summary_rows

def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def build_parser():
    parser = argparse.ArgumentParser(description="Specialized benchmark analysis for MF-RLCD.")
    parser.add_argument("--input_root", type=str, default="./opt_run/BENCH")
    parser.add_argument("--output_dir", type=str, default="./opt_run/BENCH/analysis")
    return parser

def main():
    args = build_parser().parse_args()
    input_root = Path(str(args.input_root).replace("\\", "/"))
    output_dir = Path(str(args.output_dir).replace("\\", "/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_result_records(input_root)
    method_summary = summarize_by_method(records)
    pairwise_summary = mfrlcd_pairwise_summary(records)

    write_csv(output_dir / "benchmark_records.csv", records)
    write_csv(output_dir / "method_summary.csv", method_summary)
    write_csv(output_dir / "mfrlcd_pairwise_summary.csv", pairwise_summary)

    summary = {
        "input_root": str(input_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_result_records": len(records),
        "num_methods": len({record["method"] for record in records}),
        "methods": sorted({record["method"] for record in records}),
        "generated_files": [
            str((output_dir / "benchmark_records.csv").resolve()),
            str((output_dir / "method_summary.csv").resolve()),
            str((output_dir / "mfrlcd_pairwise_summary.csv").resolve()),
        ],
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if len(records) == 0:
        print(f"[Warn] no benchmark_results.json found under: {input_root.resolve()}")
        print("[Hint] On Linux/macOS, use forward slashes such as ./opt_run/BENCH")
    print(f"[Analysis] records={len(records)} methods={summary['num_methods']}")
    print(f"[Saved] {output_dir.resolve()}")

if __name__ == "__main__":
    main()
