from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import torch

from experiment_tracking import ExperimentTracker, build_metric_snapshot
from benchmark.solvers import (
    CMAESSolver,
    DiffusionDirectSolver,
    MFRLCDSolver,
    RLOnlySolver,
    SolveResult,
    json_default,
    load_framework_module,
    make_task_suite,
    reset_env_workdir,
    set_global_seed,
)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, default="./opt_run/BENCH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true", help="do not load checkpoint_final.pt / diff_replay.npz")
    parser.add_argument("--E_target", type=float, default=110000.0)
    parser.add_argument("--sigma_y_target", type=float, default=910.0)
    parser.add_argument("--Kt_target", type=float, default=1.25)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--task_mode", type=str, default="jitter", choices=["jitter", "random"])
    parser.add_argument("--budget_evals", type=int, default=80)
    parser.add_argument("--run_mfrlcd", action="store_true")
    parser.add_argument("--run_diffusion_direct", action="store_true")
    parser.add_argument("--run_cmaes", action="store_true")
    parser.add_argument("--run_rl_only", action="store_true")
    parser.add_argument("--dd_K", type=int, default=4)
    parser.add_argument("--dd_tau", type=float, default=1.0)
    parser.add_argument("--dd_steps", type=int, default=600)
    parser.add_argument("--mfrlcd_enable_updates", action="store_true", help="enable online PPO and diffusion updates during MF-RLCD benchmark")
    parser.add_argument("--mfrlcd_deterministic_policy", action="store_true")
    parser.add_argument("--mfrlcd_ppo_min_batch", type=int, default=256)
    parser.add_argument("--mfrlcd_update_every_steps", type=int, default=1)
    parser.add_argument("--mfrlcd_diffusion_train_every", type=int, default=5)
    parser.add_argument("--mfrlcd_diffusion_steps", type=int, default=50)
    parser.add_argument("--mfrlcd_diffusion_top_frac", type=float, default=0.30)
    parser.add_argument("--mfrlcd_cost_per_eval", type=float, default=0.08)
    parser.add_argument("--mfrlcd_cost_steps_scale", type=float, default=0.04)
    parser.add_argument("--mfrlcd_local_refine_rounds", type=int, default=2)
    parser.add_argument("--mfrlcd_local_refine_scale", type=float, default=0.08)
    parser.add_argument("--mfrlcd_local_refine_decay", type=float, default=0.65)
    parser.add_argument("--cma_popsize", type=int, default=32)
    parser.add_argument("--cma_sigma0", type=float, default=1.0)
    parser.add_argument("--rl_update_every", type=int, default=64)
    parser.add_argument("--rl_cost_per_eval", type=float, default=0.0)
    parser.add_argument("--rl_deterministic_policy", action="store_true")
    parser.add_argument("--save_json", action="store_true", default=True)
    return parser

def main():
    args = build_parser().parse_args()
    set_global_seed(args.seed)
    fw = load_framework_module(args.framework_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_tracker = ExperimentTracker(work_dir)

    print(f"[Device] {device}")
    print(f"[WorkDir] {work_dir.resolve()}")

    base_target = fw.TargetProperties(
        E_target=float(args.E_target),
        sigma_y_target=float(args.sigma_y_target),
        Kt_target=float(args.Kt_target),
        tolerance={"E": 0.05, "sigma_y": 0.05, "Kt": 0.10},
    )
    tasks = make_task_suite(fw, base_target, int(args.num_tasks), args.task_mode, args.seed + 1007)
    print(f"[Tasks] num_tasks={len(tasks)} mode={args.task_mode}")
    for i, target in enumerate(tasks[:5]):
        print(f"  - task{i}: E={target.E_target:.1f}, Sy={target.sigma_y_target:.1f}, Kt={target.Kt_target:.3f}")

    print("\n[Init] Loading pretrained GraphSAGE...")
    try:
        gs_predictor = fw.GraphSAGEPredictor(
            checkpoint_path="./models/checkpoints/checkpoint.state_dict.pth",
            label_norm_path="./models/checkpoints/norm.npz",
            max_node_num=300,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        gnn_model = gs_predictor
        print("GraphSAGE loaded")
    except Exception as e:
        print(f"Failed to load GraphSAGE: {e}")
        traceback.print_exc()
        return

    print("\n[Init] Loading pretrained cGAN...")
    try:
        tf.get_logger().setLevel("ERROR")
        gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        generator = fw.Generator()
        discriminator = fw.Discriminator()
        checkpoint_dir = "./models/training_checkpoints"
        ckpt = tf.train.Checkpoint(generator_optimizer=gen_opt, discriminator_optimizer=disc_opt, generator=generator, discriminator=discriminator)
        ckpt.restore(checkpoint_dir + "/ckpt-3")
        print("cGAN loaded")
    except Exception as e:
        print(f"Failed to load cGAN: {e}")
        traceback.print_exc()
        return

    print("\n[Init] Starting MATLAB engine...")
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        print("MATLAB engine started")
    except Exception as e:
        print(f"Failed to start MATLAB engine: {e}")
        traceback.print_exc()
        return

    env = fw.MicrostructureEnvironment(
        gnn_model=gnn_model,
        cgan_generator=generator,
        target_properties=base_target,
        matlab_engine=eng,
        work_dir=str(work_dir / "ENV_TMP"),
    )

    z_dim = int(env.max_n_priorBeta * 6 + 2)
    diffusion_model = fw.ConditionalDiffusionModel(param_dim=z_dim, condition_dim=3, hidden_dim=512).to(device)
    mfrlcd_agent = fw.PPOAgent(state_dim=9, action_dim=3, hidden_dim=256).to(device)
    trainer = fw.RLDiffusionTrainer(env, diffusion_model, mfrlcd_agent, device=device)

    ckpt_path = work_dir / "checkpoint_final.pt"
    replay_path = work_dir / "diff_replay.npz"
    if (not args.no_resume) and ckpt_path.exists():
        try:
            trainer.load_checkpoint(str(ckpt_path))
            print(f"\n[Resume] loaded {ckpt_path.name}")
        except Exception as e:
            print(f"[Resume] failed: {e}")
            traceback.print_exc()
    if hasattr(trainer, "diff_replay") and replay_path.exists():
        try:
            trainer.diff_replay.load_npz(str(replay_path), merge=True)
            print(f"[Resume] loaded {replay_path.name} (merge=True)")
        except Exception as e:
            print(f"[Replay] load failed: {e}")
            traceback.print_exc()

    solvers = []
    if args.run_mfrlcd:
        solvers.append(("MF-RLCD", MFRLCDSolver(trainer, device)))
    if args.run_diffusion_direct:
        dd_model = fw.ConditionalDiffusionModel(param_dim=z_dim, condition_dim=3, hidden_dim=512).to(device)
        dd_model.load_state_dict(trainer.diffusion_model.state_dict(), strict=True)
        solvers.append(("Diffusion-Direct", DiffusionDirectSolver(dd_model, device)))
    if args.run_cmaes:
        solvers.append(("CMA-ES", CMAESSolver(z_dim=z_dim)))
    if args.run_rl_only:
        rl_only_agent = fw.PPOAgent(state_dim=9, action_dim=z_dim, hidden_dim=256).to(device)
        solvers.append(("RL-only(PPO)", RLOnlySolver(rl_only_agent, device)))

    if not solvers:
        print("\n[Warn] No methods selected. Use --run_mfrlcd / --run_diffusion_direct / --run_cmaes / --run_rl_only.")
        return

    all_results: List[SolveResult] = []
    for task_id, target in enumerate(tasks):
        print("\n" + "=" * 90)
        print(f"[Task {task_id}] Target: E={target.E_target:.1f}, Sy={target.sigma_y_target:.1f}, Kt={target.Kt_target:.3f}")
        print("=" * 90)

        for name, solver in solvers:
            safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            method_dir = work_dir / f"T{task_id:03d}_{safe_name}"
            reset_env_workdir(env, method_dir)
            env.set_target(target)
            run_seed = int(args.seed + 10000 * task_id + 17)
            print(f"\n[{name}] seed={run_seed} budget_evals={args.budget_evals} work={method_dir.name}")

            t0 = time.perf_counter()

            if name == "MF-RLCD":
                res = solver.solve(
                    env=env,
                    target=target,
                    task_id=task_id,
                    seed=run_seed,
                    budget_evals=int(args.budget_evals),
                    enable_updates=bool(args.mfrlcd_enable_updates),
                    ppo_min_batch=int(args.mfrlcd_ppo_min_batch),
                    update_every_steps=int(args.mfrlcd_update_every_steps),
                    diffusion_train_every=int(args.mfrlcd_diffusion_train_every),
                    diffusion_steps=int(args.mfrlcd_diffusion_steps),
                    diffusion_top_frac=float(args.mfrlcd_diffusion_top_frac),
                    cost_per_eval=float(args.mfrlcd_cost_per_eval),
                    cost_steps_scale=float(args.mfrlcd_cost_steps_scale),
                    deterministic_policy=bool(args.mfrlcd_deterministic_policy),
                    local_refine_rounds=int(args.mfrlcd_local_refine_rounds),
                    local_refine_scale=float(args.mfrlcd_local_refine_scale),
                    local_refine_decay=float(args.mfrlcd_local_refine_decay),
                )
            elif name == "Diffusion-Direct":
                res = solver.solve(env=env, target=target, task_id=task_id, seed=run_seed, budget_evals=int(args.budget_evals), K=int(args.dd_K), tau=float(args.dd_tau), stepsN=int(args.dd_steps))
            elif name == "CMA-ES":
                res = solver.solve(env=env, target=target, task_id=task_id, seed=run_seed, budget_evals=int(args.budget_evals), popsize=int(args.cma_popsize), sigma0=float(args.cma_sigma0))
            else:
                res = solver.solve(env=env, target=target, task_id=task_id, seed=run_seed, budget_evals=int(args.budget_evals), update_every=int(args.rl_update_every), cost_per_eval=float(args.rl_cost_per_eval), deterministic_policy=bool(args.rl_deterministic_policy))

            dt = time.perf_counter() - t0

            all_results.append(res)
            snapshot = build_metric_snapshot(target, res.best_E, res.best_Sy, res.best_Kt, lambda E, Sy, Kt: env.compute_reward(E, Sy, Kt))
            benchmark_tracker.log_event(
                "solver_result",
                {
                    "method": name,
                    "task_id": int(task_id),
                    "seed": int(run_seed),
                    "duration_sec": float(dt),
                    "best_reward": float(res.best_reward),
                    "success": bool(res.success),
                    "best_E": res.best_E,
                    "best_Sy": res.best_Sy,
                    "best_Kt": res.best_Kt,
                    "E_error": snapshot.E_error,
                    "Sy_error": snapshot.sigma_y_error,
                    "Kt_error": snapshot.Kt_error,
                    "n_evals": int(res.n_evals),
                },
            )
            print(
                f"[{name}] done in {dt:.1f}s | best_reward={res.best_reward:.4f} | "
                f"success={res.success} | "
                f"E_err={None if snapshot.E_error is None else round(snapshot.E_error * 100, 2)}% "
                f"Sy_err={None if snapshot.sigma_y_error is None else round(snapshot.sigma_y_error * 100, 2)}% "
                f"Kt_err={None if snapshot.Kt_error is None else round(snapshot.Kt_error * 100, 2)}% | "
                f"E={None if res.best_E is None else round(res.best_E, 2)} "
                f"Sy={None if res.best_Sy is None else round(res.best_Sy, 2)} "
                f"Kt={None if res.best_Kt is None else round(res.best_Kt, 4)}"
            )

    print("\n" + "#" * 90)
    print("[Summary] success-rate / avg best-reward (over tasks)")
    print("#" * 90)

    by_method: Dict[str, List[SolveResult]] = {}
    for result in all_results:
        by_method.setdefault(result.method, []).append(result)
    for method, results in by_method.items():
        succ_rate = float(sum(1.0 if x.success else 0.0 for x in results) / max(1, len(results)))
        avg_best = float(sum(x.best_reward for x in results) / max(1, len(results)))
        print(f"- {method:16s} | success_rate={succ_rate * 100:6.2f}% | avg_best_reward={avg_best:.4f}")

    if args.save_json:
        out_path = work_dir / "benchmark_results.json"
        payload = [asdict(r) for r in all_results]
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=json_default)
        print(f"\n[Saved] {out_path.resolve()}")
        benchmark_tracker.write_summary(
            {
                "framework_path": str(Path(args.framework_path).resolve()),
                "work_dir": str(work_dir.resolve()),
                "num_tasks": int(len(tasks)),
                "methods": list(by_method.keys()),
                "results_path": str(out_path.resolve()),
            }
        )

    print("\n[Done]")

if __name__ == "__main__":
    main()
