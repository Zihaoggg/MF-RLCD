#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPTIMIZE_SCRIPT = ROOT / "rl_diffusion_framework_v7.py"
BENCHMARK_SCRIPT = ROOT / "mfrlcd_experiments.py"
VERIFY_SCRIPT = ROOT / "verification_experiment.py"
ANALYSIS_SCRIPT = ROOT / "benchmark_analysis.py"

DEFAULT_TARGETS = [
    ("Easy_2", 120000, 960, 1.25),
    ("Medium_1", 120000, 960, 1.20),
    ("Medium_2", 110000, 950, 1.25),
    ("Hard_1", 110000, 890, 1.19),
    ("Hard_2", 110000, 910, 1.20),
]

def run_command(args):
    print("[Command]", " ".join(str(x) for x in args))
    subprocess.run(args, cwd=str(ROOT), check=True)

def normalize_cli_path(path_str: str) -> str:
    if path_str is None:
        return path_str
    text = str(path_str)
    if "\\" in text:
        text = text.replace("\\", "/")
    return str(Path(text))

def add_target_args(parser: argparse.ArgumentParser):
    parser.add_argument("--E_target", type=float, default=110000)
    parser.add_argument("--sigma_y_target", type=float, default=910)
    parser.add_argument("--Kt_target", type=float, default=1.20)
    parser.add_argument("--work_dir", type=str, default=str(ROOT / "opt_run" / "RL_Diffusion"))

def handle_optimize(args):
    cmd = [
        sys.executable,
        str(OPTIMIZE_SCRIPT),
        "--work_dir", normalize_cli_path(args.work_dir),
        "--E_target", str(args.E_target),
        "--sigma_y_target", str(args.sigma_y_target),
        "--Kt_target", str(args.Kt_target),
        "--num_episodes", str(args.num_episodes),
        "--steps_per_episode", str(args.steps_per_episode),
    ]
    if args.no_resume:
        cmd.append("--no_resume")
    if getattr(args, "history_root", None):
        cmd.extend(["--history_root", normalize_cli_path(args.history_root)])
    run_command(cmd)

def handle_benchmark(args):
    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--framework_path", str(OPTIMIZE_SCRIPT),
        "--work_dir", normalize_cli_path(args.work_dir),
        "--E_target", str(args.E_target),
        "--sigma_y_target", str(args.sigma_y_target),
        "--Kt_target", str(args.Kt_target),
        "--budget_evals", str(args.budget_evals),
        "--num_tasks", str(args.num_tasks),
        "--task_mode", args.task_mode,
        "--save_json",
        "--run_mfrlcd",
    ]
    if args.run_diffusion_direct:
        cmd.append("--run_diffusion_direct")
    if args.run_cmaes:
        cmd.append("--run_cmaes")
    if args.run_rl_only:
        cmd.append("--run_rl_only")
    if getattr(args, "mfrlcd_enable_updates", False):
        cmd.append("--mfrlcd_enable_updates")
    if args.no_resume:
        cmd.append("--no_resume")
    run_command(cmd)

def handle_verify(args):
    cmd = [
        sys.executable,
        str(VERIFY_SCRIPT),
        "--work_dir", normalize_cli_path(args.work_dir),
    ]
    run_command(cmd)

def handle_analyze(args):
    cmd = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        "--input_root", normalize_cli_path(args.input_root),
        "--output_dir", normalize_cli_path(args.output_dir),
    ]
    run_command(cmd)

def handle_batch(args):
    base_out = Path(normalize_cli_path(args.base_out))
    base_out.mkdir(parents=True, exist_ok=True)
    for name, E, sigma_y, Kt in DEFAULT_TARGETS:
        out_dir = base_out / name
        cmd = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--framework_path", str(OPTIMIZE_SCRIPT),
            "--work_dir", str(out_dir),
            "--E_target", str(E),
            "--sigma_y_target", str(sigma_y),
            "--Kt_target", str(Kt),
            "--budget_evals", str(args.budget_evals),
            "--num_tasks", "0",
            "--task_mode", "jitter",
            "--save_json",
            "--run_mfrlcd",
        ]
        if args.run_diffusion_direct:
            cmd.append("--run_diffusion_direct")
        if args.run_cmaes:
            cmd.append("--run_cmaes")
        if args.run_rl_only:
            cmd.append("--run_rl_only")
        if args.no_resume:
            cmd.append("--no_resume")
        print(f"\n[Batch] {name} -> {out_dir}")
        run_command(cmd)

def build_parser():
    parser = argparse.ArgumentParser(description="Unified project runner for RL microstructure design.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize_parser = subparsers.add_parser("optimize", help="Run online RL-guided optimization on one target.")
    add_target_args(optimize_parser)
    optimize_parser.add_argument("--num_episodes", type=int, default=300)
    optimize_parser.add_argument("--steps_per_episode", type=int, default=10)
    optimize_parser.add_argument("--history_root", type=str, default=None)
    optimize_parser.add_argument("--no_resume", action="store_true")
    optimize_parser.set_defaults(func=handle_optimize)

    train_parser = subparsers.add_parser("train", help="Legacy alias of optimize.")
    add_target_args(train_parser)
    train_parser.add_argument("--num_episodes", type=int, default=300)
    train_parser.add_argument("--steps_per_episode", type=int, default=10)
    train_parser.add_argument("--history_root", type=str, default=None)
    train_parser.add_argument("--no_resume", action="store_true")
    train_parser.set_defaults(func=handle_optimize)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run MF-RLCD benchmark on one target suite.")
    add_target_args(benchmark_parser)
    benchmark_parser.add_argument("--budget_evals", type=int, default=80)
    benchmark_parser.add_argument("--num_tasks", type=int, default=0)
    benchmark_parser.add_argument("--task_mode", type=str, default="jitter")
    benchmark_parser.add_argument("--run_diffusion_direct", action="store_true")
    benchmark_parser.add_argument("--run_cmaes", action="store_true")
    benchmark_parser.add_argument("--run_rl_only", action="store_true")
    benchmark_parser.add_argument("--mfrlcd_enable_updates", action="store_true")
    benchmark_parser.add_argument("--no_resume", action="store_true")
    benchmark_parser.set_defaults(func=handle_benchmark)

    verify_parser = subparsers.add_parser("verify", help="Run verification experiment from a trained work_dir.")
    verify_parser.add_argument("--work_dir", type=str, default=str(ROOT / "opt_run" / "RL_Diffusion"))
    verify_parser.set_defaults(func=handle_verify)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results with a MF-RLCD-focused summary.")
    analyze_parser.add_argument("--input_root", type=str, default=str(ROOT / "opt_run" / "BENCH"))
    analyze_parser.add_argument("--output_dir", type=str, default=str(ROOT / "opt_run" / "BENCH" / "analysis"))
    analyze_parser.set_defaults(func=handle_analyze)

    batch_parser = subparsers.add_parser("batch", help="Run the default target batch.")
    batch_parser.add_argument("--base_out", type=str, default=str(ROOT / "opt_run" / "RL_Diffusion"))
    batch_parser.add_argument("--budget_evals", type=int, default=80)
    batch_parser.add_argument("--run_diffusion_direct", action="store_true")
    batch_parser.add_argument("--run_cmaes", action="store_true")
    batch_parser.add_argument("--run_rl_only", action="store_true")
    batch_parser.add_argument("--no_resume", action="store_true")
    batch_parser.set_defaults(func=handle_batch)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
