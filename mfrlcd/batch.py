from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mfrlcd.configuration import config_fingerprint, parse_options
from mfrlcd.records import write_json


ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = {"name", "E_target", "sigma_y_target", "Kt_target"}


@dataclass(frozen=True)
class DesignTask:
    name: str
    E_target: float
    sigma_y_target: float
    Kt_target: float
    seed: int


def read_tasks(path, base_seed=42):
    tasks = []
    names = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not TARGET_COLUMNS <= set(columns) or set(columns) - TARGET_COLUMNS - {"seed"}:
            raise ValueError("Target CSV requires name,E_target,sigma_y_target,Kt_target and an optional seed column")
        for line, row in enumerate(reader, start=2):
            if None in row or any(row.get(name) is None for name in columns):
                raise ValueError(f"Malformed target row at line {line}")
            name = row["name"].strip()
            reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) or name.endswith(".") or name.split(".")[0].upper() in reserved:
                raise ValueError(f"Invalid design name at line {line}: {name}")
            if name.casefold() in names:
                raise ValueError(f"Duplicate design name: {name}")
            names.add(name.casefold())
            values = [float(row[key]) for key in ("E_target", "sigma_y_target", "Kt_target")]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"Target properties must be finite and positive at line {line}")
            seed = int(row["seed"]) if row.get("seed", "").strip() else base_seed + len(tasks)
            if not 0 <= seed < 2 ** 32:
                raise ValueError(f"Invalid seed at line {line}")
            tasks.append(DesignTask(name, *values, seed))
    if not tasks:
        raise ValueError("Target CSV contains no designs")
    return tasks


def task_arguments(task, config, output_root, history_root):
    arguments = ["--work_dir", str(output_root / task.name), "--history_root", str(history_root),
                 "--E_target", str(task.E_target), "--sigma_y_target", str(task.sigma_y_target),
                 "--Kt_target", str(task.Kt_target), "--seed", str(task.seed)]
    return (["--config", str(config)] if config else []) + arguments


def completed_run(run_dir, fingerprint):
    try:
        with (run_dir / "run_summary.json").open(encoding="utf-8") as source:
            summary = json.load(source)
        files = ("checkpoint_final.pt", "best_params.npy", "diff_replay.npz", "run_config.json")
        return (summary.get("status") == "completed" and summary.get("error") is None
                and summary.get("config_fingerprint") == fingerprint
                and all((run_dir / name).is_file() and (run_dir / name).stat().st_size for name in files))
    except (OSError, ValueError, AttributeError):
        return False


def run_design(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a sequence of microstructure design requests")
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output_root", type=Path, default=Path("outputs/designs"))
    parser.add_argument("--history_root", type=Path)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args(argv)
    config = args.config.expanduser().resolve() if args.config else None
    root = args.output_root.expanduser().resolve()
    history = args.history_root.expanduser().resolve() if args.history_root else root / "history"
    try:
        base_options = parse_options(["--config", str(config)] if config else [])
        tasks = read_tasks(args.targets.expanduser(), base_options.seed)
        planned = []
        for task in tasks:
            run_dir = root / task.name
            if (run_dir.resolve().parent != root or run_dir.resolve() == history
                    or run_dir.resolve() in history.parents or task.name.casefold() == "batch_summary.json"):
                raise ValueError(f"Design directory conflicts with output/history paths: {task.name}")
            options = task_arguments(task, config, root, history)
            fingerprint = config_fingerprint(parse_options(options))
            planned.append((task, options, fingerprint))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    root.mkdir(parents=True, exist_ok=True)
    rows = [{"name": task.name, "status": "pending", "config_fingerprint": fingerprint} for task, _, fingerprint in planned]
    manifest = root / "batch_summary.json"
    for index, (task, options, fingerprint) in enumerate(planned):
        run_dir = root / task.name
        row = rows[index]
        if args.skip_completed and completed_run(run_dir, fingerprint):
            row["status"] = "skipped"
            print(f"[{index + 1}/{len(planned)}] {task.name}: already completed")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            row["status"] = "running"
            write_json(manifest, {"designs": rows})
            print(f"[{index + 1}/{len(planned)}] {task.name}: running; output: {run_dir}", flush=True)
            try:
                code = run_design([sys.executable, "-u", str(ROOT / "run.py"), "optimize", *options], run_dir / "run.log")
                row["returncode"] = code
                row["status"] = "completed" if code == 0 and completed_run(run_dir, fingerprint) else "failed"
            except KeyboardInterrupt:
                row["status"] = "interrupted"
                write_json(manifest, {"designs": rows})
                return 130
            except OSError as exc:
                row.update(status="failed", error=str(exc))
            print(f"  {task.name}: {row['status']}", flush=True)
        write_json(manifest, {"designs": rows})
        if row["status"] == "failed" and not args.continue_on_error:
            break
    return int(any(row["status"] not in {"completed", "skipped"} for row in rows))
