from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import matlab.engine
import tensorflow as tf
import torch

from rlmsdesign.env import Generator, Discriminator, GraphSAGEPredictor, MicrostructureEnvironment, set_seed
from rlmsdesign.models import ConditionalDiffusionModel, PPOAgent
from rlmsdesign.targets import (
    TargetPoolManager,
    TargetProperties,
    auto_warmstart_from_history,
    discover_history_runs,
    find_similar_target_in_pool,
    make_mixed_target_sampler,
    merge_replay_from_history,
    rank_history_runs,
    update_two_level_experience_indexes,
)
from rlmsdesign.trainer import RLDiffusionTrainer

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--E_target", type=float, default=110000.0)
    parser.add_argument("--sigma_y_target", type=float, default=890.0)
    parser.add_argument("--Kt_target", type=float, default=1.19)
    parser.add_argument("--work_dir", type=str, default="./opt_run/RL_Diffusion_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--history_root", type=str, default=None)
    parser.add_argument("--history_topk", type=int, default=5)
    parser.add_argument("--history_replay_runs", type=int, default=3)
    parser.add_argument("--p_current", type=float, default=0.55)
    parser.add_argument("--p_pool", type=float, default=0.30)
    parser.add_argument("--p_jitter", type=float, default=0.15)
    parser.add_argument("--jitter_E", type=float, default=0.02)
    parser.add_argument("--jitter_Sy", type=float, default=0.02)
    parser.add_argument("--jitter_Kt", type=float, default=0.03)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--steps_per_episode", type=int, default=10)
    parser.add_argument("--update_every_episodes", type=int, default=1)
    parser.add_argument("--ppo_min_batch", type=int, default=256)
    parser.add_argument("--diffusion_train_every", type=int, default=5)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_top_frac", type=float, default=0.30)
    parser.add_argument("--warmstart_topn", type=int, default=3)
    parser.add_argument("--warmstart_verify_repeats", type=int, default=2)
    return parser

def main():
    print("=" * 70)
    print("RL-guided Conditional Diffusion for Online Microstructure Optimization")
    print("=" * 70)

    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    target = TargetProperties(
        E_target=float(args.E_target),
        sigma_y_target=float(args.sigma_y_target),
        Kt_target=float(args.Kt_target),
        tolerance={"E": 0.05, "sigma_y": 0.05, "Kt": 0.10},
    )
    print("\nOptimization target:")
    print(f"  E={target.E_target:.1f} MPa, Sy={target.sigma_y_target:.1f} MPa, Kt={target.Kt_target:.3f}")

    pool_path = work_dir / "target_pool.json"
    pool_mgr = TargetPoolManager(pool_path, max_size=2000)
    pool_mgr.load()

    similar_ref = find_similar_target_in_pool(pool_mgr, target, rel_each=0.05)
    pool_hit = similar_ref is not None
    if pool_hit:
        print("[TargetPool] nearby target found in local pool; warm-start will be attempted.")
    else:
        print("[TargetPool] no nearby target in local pool; checking broader history next.")
    pool_mgr.add(target, save=True)

    history_root = Path(args.history_root) if args.history_root else work_dir.parent
    index_path = history_root / "experience_index.json"
    hq_index_path = history_root / "experience_index_hq.json"
    history_runs = rank_history_runs(
        discover_history_runs(history_root, exclude_dir=work_dir, max_runs=400, prefer_high_quality=True),
        target,
    )
    if history_runs:
        print(f"[History] discovered {len(history_runs)} reusable past runs under {history_root}")
        if hq_index_path.exists():
            print(f"[History] preferred HQ index: {hq_index_path}")
        elif index_path.exists():
            print(f"[History] loaded from global index: {index_path}")
        for idx, item in enumerate(history_runs[: min(3, len(history_runs))], start=1):
            hist_target = item["target"]
            print(
                f"  {idx}. {item['run_dir']} | "
                f"distance={item['distance'] * 100:.2f}% | "
                f"success={item['best_success']} | "
                f"target=({hist_target.E_target:.0f}, {hist_target.sigma_y_target:.0f}, {hist_target.Kt_target:.3f})"
            )
    else:
        print(f"[History] no reusable past runs discovered under {history_root}")

    target_sampler = make_mixed_target_sampler(
        pool_mgr,
        current_target=target,
        seed=args.seed + 123,
        p_current=args.p_current,
        p_pool=args.p_pool,
        p_jitter=args.p_jitter,
        jitter_rel_E=args.jitter_E,
        jitter_rel_Sy=args.jitter_Sy,
        jitter_rel_Kt=args.jitter_Kt,
    )

    print("\nLoading pretrained models...")
    try:
        gs_predictor = GraphSAGEPredictor(
            checkpoint_path="./models/checkpoints/checkpoint.state_dict.pth",
            label_norm_path="./models/checkpoints/norm.npz",
            max_node_num=300,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        gnn_model = gs_predictor
        print("Loaded GraphSAGE")
    except Exception as exc:
        print(f"Failed to load GraphSAGE: {exc}")
        traceback.print_exc()
        return

    try:
        tf.get_logger().setLevel("ERROR")
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        generator = Generator()
        discriminator = Discriminator()
        checkpoint_dir = "./models/training_checkpoints"
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator,
        )
        checkpoint.restore(checkpoint_dir + "/ckpt-3")
        print("Loaded cGAN")
    except Exception as exc:
        print(f"Failed to load cGAN: {exc}")
        traceback.print_exc()
        return

    print("\nStarting MATLAB engine...")
    try:
        eng = matlab.engine.start_matlab()
        print("MATLAB engine started")
    except Exception as exc:
        print(f"Failed to start MATLAB engine: {exc}")
        print("Please ensure the MATLAB Python engine is installed.")
        traceback.print_exc()
        return

    env = MicrostructureEnvironment(
        gnn_model=gnn_model,
        cgan_generator=generator,
        target_properties=target,
        matlab_engine=eng,
        work_dir=str(work_dir),
    )

    diffusion_model = ConditionalDiffusionModel(
        param_dim=env.max_n_priorBeta * 6 + 2,
        condition_dim=3,
        hidden_dim=512,
    )
    rl_agent = PPOAgent(state_dim=9, action_dim=3, hidden_dim=256)
    trainer = RLDiffusionTrainer(env=env, diffusion_model=diffusion_model, rl_agent=rl_agent, device=device)

    ckpt_path = work_dir / "checkpoint_final.pt"
    replay_path = work_dir / "diff_replay.npz"

    if (not args.no_resume) and ckpt_path.exists():
        try:
            last_ep = trainer.load_checkpoint(str(ckpt_path))
            print(f"\n[Resume] loaded checkpoint_final.pt (episode={last_ep})")
            trainer.episode_rewards = []
            trainer.episode_properties = []
            trainer._no_improve = 0
            trainer._steps_done = 0
            if hasattr(trainer, "rl_buffer") and hasattr(trainer.rl_buffer, "clear"):
                trainer.rl_buffer.clear()
            if hasattr(env, "eval_counter"):
                env.eval_counter = 0
        except Exception as exc:
            print(f"[Resume] failed: {exc}")
            traceback.print_exc()

    if hasattr(trainer, "diff_replay") and replay_path.exists():
        try:
            trainer.diff_replay.load_npz(str(replay_path), merge=True)
        except Exception as exc:
            print(f"[Replay] local load failed: {exc}")

    if history_runs:
        merge_replay_from_history(
            trainer,
            history_runs,
            max_histories=int(args.history_replay_runs),
            verbose=True,
        )

    try:
        auto_warmstart_from_history(
            trainer,
            target,
            enable=bool(pool_hit or len(history_runs) > 0),
            history_runs=history_runs[: max(0, int(args.history_topk))],
            topn=int(args.warmstart_topn),
            verify_repeats=int(args.warmstart_verify_repeats),
            verbose=True,
        )
    except Exception as exc:
        print(f"[WarmStart] failed: {exc}")
        traceback.print_exc()
        trainer.best_reward = -float("inf")
        trainer.best_params = None

    print("\nOptimization search...")
    print("-" * 70)
    try:
        trainer.optimize(
            num_episodes=int(args.num_episodes),
            steps_per_episode=int(args.steps_per_episode),
            target_sampler=target_sampler,
            update_every_episodes=int(args.update_every_episodes),
            ppo_min_batch=int(args.ppo_min_batch),
            diffusion_train_every=int(args.diffusion_train_every),
            diffusion_steps=int(args.diffusion_steps),
            diffusion_top_frac=float(args.diffusion_top_frac),
        )
        trainer.verify_best(repeats=3, verbose=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as exc:
        print(f"\nOptimization error: {exc}")
        traceback.print_exc()
    finally:
        print("\nSaving final artifacts...")
        try:
            trainer.save_checkpoint("final")
            print(f"saved: {ckpt_path}")
        except Exception as exc:
            print(f"save checkpoint failed: {exc}")
        try:
            pool_mgr.save()
            print(f"saved: {pool_path}")
        except Exception as exc:
            print(f"save target_pool failed: {exc}")
        try:
            if hasattr(trainer, "diff_replay"):
                trainer.diff_replay.save_npz(str(replay_path), max_save=120000)
                print(f"saved: {replay_path}")
        except Exception as exc:
            print(f"save diff_replay failed: {exc}")

        E = sigma_y = Kt = None
        try:
            if trainer.best_params is not None:
                env.set_target(target)
                E, sigma_y, Kt = env.evaluate_params(trainer.best_params, metadata={"phase": "final_best"})
                if E is not None:
                    print("\nBest params (evaluated on current target):")
                    print(f"  E   = {E/1000:.2f} GPa   (target {target.E_target/1000:.2f} GPa)")
                    print(f"  Sy  = {sigma_y:.2f} MPa  (target {target.sigma_y_target:.2f} MPa)")
                    print(f"  Kt  = {Kt:.4f}      (target {target.Kt_target:.4f})")
                import numpy as np
                np.save(work_dir / "best_params.npy", trainer.best_params)
                print(f"saved: {work_dir / 'best_params.npy'}")
        except Exception as exc:
            print(f"evaluate/save best_params failed: {exc}")

        try:
            best_snapshot = env.summarize_metrics(E, sigma_y, Kt)
            env.tracker.write_summary({
                "work_dir": str(work_dir.resolve()),
                "target": {"E": float(target.E_target), "sigma_y": float(target.sigma_y_target), "Kt": float(target.Kt_target)},
                "history_root": str(history_root.resolve()),
                "history_candidates": int(len(history_runs)),
                "best_reward": float(trainer.best_reward),
                "best_success": bool(best_snapshot.success),
                "best_E": best_snapshot.E,
                "best_sigma_y": best_snapshot.sigma_y,
                "best_Kt": best_snapshot.Kt,
                "best_E_error": best_snapshot.E_error,
                "best_sigma_y_error": best_snapshot.sigma_y_error,
                "best_Kt_error": best_snapshot.Kt_error,
                "num_evaluations": int(env.eval_counter),
                "num_episode_records": int(len(trainer.episode_rewards)),
                "diffusion_replay_size": int(len(trainer.diff_replay)) if hasattr(trainer, "diff_replay") else 0,
            })
        except Exception as exc:
            print(f"save run summary failed: {exc}")

        try:
            best_error_sum = None
            if "best_snapshot" in locals():
                if (best_snapshot.E_error is not None) and (best_snapshot.sigma_y_error is not None) and (best_snapshot.Kt_error is not None):
                    best_error_sum = float(best_snapshot.E_error + best_snapshot.sigma_y_error + best_snapshot.Kt_error)

            update_two_level_experience_indexes(
                history_root,
                {
                    "run_dir": work_dir.resolve(),
                    "summary_path": (work_dir / "run_summary.json").resolve(),
                    "target": target,
                    "best_reward": float(trainer.best_reward),
                    "best_success": bool(best_snapshot.success) if "best_snapshot" in locals() else False,
                    "best_error_sum": best_error_sum,
                    "best_params_path": (work_dir / "best_params.npy").resolve(),
                    "replay_path": replay_path.resolve(),
                    "checkpoint_path": ckpt_path.resolve(),
                },
                max_global_runs=1000,
                max_hq_runs=300,
            )
            print(f"updated global history index: {index_path}")
            print(f"updated HQ history index: {hq_index_path}")
        except Exception as exc:
            print(f"update history indexes failed: {exc}")

        try:
            eng.quit()
        except Exception:
            pass

        print("\nDone.")
