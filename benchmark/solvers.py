from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rlmsdesign.local_search import refine_latent_candidate

def load_framework_module(framework_path: str):
    framework_path = str(framework_path)
    path = Path(framework_path)
    if not path.exists():
        raise FileNotFoundError(f"--framework_path not found: {path.resolve()}")
    spec = importlib.util.spec_from_file_location("mfrlcd_framework", framework_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load framework spec from: {framework_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mfrlcd_framework"] = module
    spec.loader.exec_module(module)
    return module

def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reset_env_workdir(env, work_dir: Path):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(env, "set_work_dir"):
        env.set_work_dir(work_dir)
    else:
        env.work_dir = work_dir
        if hasattr(env, "eval_counter"):
            env.eval_counter = 0

def is_success(E: Optional[float], Sy: Optional[float], Kt: Optional[float], target) -> bool:
    if E is None or Sy is None or Kt is None:
        return False
    eE = abs(E - target.E_target) / target.E_target
    eS = abs(Sy - target.sigma_y_target) / target.sigma_y_target
    eK = abs(Kt - target.Kt_target) / (abs(target.Kt_target) if abs(target.Kt_target) > 1e-12 else 1.0)
    return (eE < target.tolerance["E"]) and (eS < target.tolerance["sigma_y"]) and (eK < target.tolerance["Kt"])

def z_to_params_vector(z: np.ndarray, env) -> np.ndarray:
    max_n = int(env.max_n_priorBeta)
    candidates = list(env.beta_candidates)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    need = max_n * 6 + 2
    if z.shape[0] < need:
        z_pad = np.zeros((need,), dtype=np.float64)
        z_pad[: z.shape[0]] = z
        z = z_pad

    def z2u(x):
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * (x ** 3))))

    z_angles = z[: max_n * 3]
    u_angles = z2u(z_angles)
    angles = np.empty((max_n * 3,), dtype=np.float64)
    angles[0:max_n] = (u_angles[0:max_n] * 360.0) % 360.0
    angles[max_n:2 * max_n] = u_angles[max_n:2 * max_n] * 180.0
    angles[2 * max_n:3 * max_n] = (u_angles[2 * max_n:3 * max_n] * 360.0) % 360.0

    z_seeds = z[max_n * 3 : max_n * 6]
    u_seeds = z2u(z_seeds)
    seeds = 1 + np.floor(u_seeds * 29.0).astype(np.int64)
    seeds = np.clip(seeds, 1, 29).astype(np.int64)

    z_lam = z[max_n * 6]
    lam_ratio = float(0.1 + float(z2u(z_lam)) * (4.0 - 0.1))

    z_n = z[max_n * 6 + 1]
    u_n = float(z2u(z_n))
    idx = int(np.floor(u_n * len(candidates)))
    idx = int(np.clip(idx, 0, len(candidates) - 1))
    n_priorBeta = int(candidates[idx])

    return np.concatenate([angles, seeds.astype(np.float64), [lam_ratio], [float(n_priorBeta)]])

@dataclass
class SolveResult:
    method: str
    task_id: int
    seed: int
    budget_evals: int
    n_evals: int
    best_reward: float
    best_E: Optional[float]
    best_Sy: Optional[float]
    best_Kt: Optional[float]
    success: bool
    best_params_vector: Optional[List[float]]
    history: List[Dict[str, Any]]

class DiffusionDirectSolver:
    def __init__(self, diffusion_model: nn.Module, device: torch.device):
        self.diffusion_model = diffusion_model.to(device)
        self.device = device

    @torch.no_grad()
    def solve(self, env, target, task_id: int, seed: int, budget_evals: int, *, K: int = 4, tau: float = 1.0, stepsN: int = 600, deterministic: bool = False) -> SolveResult:
        env.set_target(target)
        set_global_seed(seed)
        self.diffusion_model.eval()
        cond = torch.FloatTensor(target.normalize()).to(self.device)

        n_evals = 0
        best = dict(reward=-1e30, E=None, Sy=None, Kt=None, params=None, succ=False, source=None)
        hist = []

        while n_evals < budget_evals:
            remain = budget_evals - n_evals
            K_now = int(min(K, remain))
            z_batch = self.diffusion_model.sample(cond, num_samples=K_now, num_steps=int(stepsN), noise_scale=float(tau))
            z_batch = z_batch.detach().cpu().numpy()

            for i in range(K_now):
                z = z_batch[i]
                params_vec = z_to_params_vector(z, env)
                E, Sy, Kt = env.evaluate_params(
                    params_vec,
                    metadata={"phase": "benchmark", "method": "Diffusion-Direct", "task_id": int(task_id), "seed": int(seed), "eval_index": int(n_evals + 1)},
                )
                reward = float(env.compute_reward(E, Sy, Kt)) if (E is not None and Sy is not None and Kt is not None) else -5.0
                succ = is_success(E, Sy, Kt, target)
                n_evals += 1
                if reward > best["reward"]:
                    best.update(reward=reward, E=E, Sy=Sy, Kt=Kt, params=params_vec, succ=succ)
                hist.append({"eval": n_evals, "reward": reward, "success": bool(succ), "best_reward": float(best["reward"]), "best_success": bool(best["succ"])})
                if n_evals >= budget_evals:
                    break

        return SolveResult("Diffusion-Direct", task_id, seed, int(budget_evals), int(n_evals), float(best["reward"]), best["E"], best["Sy"], best["Kt"], bool(best["succ"]), None if best["params"] is None else best["params"].astype(float).tolist(), hist)

class CMAESSolver:
    def __init__(self, z_dim: int):
        self.z_dim = int(z_dim)

    def solve(self, env, target, task_id: int, seed: int, budget_evals: int, *, popsize: int = 32, sigma0: float = 1.0, c_cov: float = 0.2, success_ema: float = 0.0, sigma_min: float = 0.10, sigma_max: float = 3.0) -> SolveResult:
        env.set_target(target)
        rng = np.random.default_rng(int(seed))
        lam = int(max(8, popsize))
        dim = self.z_dim
        mu = max(2, lam // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = (w / (w.sum() + 1e-12)).astype(np.float64)

        mean = np.zeros((dim,), dtype=np.float64)
        diagC = np.ones((dim,), dtype=np.float64)
        sigma = float(sigma0)

        n_evals = 0
        best = dict(reward=-1e30, E=None, Sy=None, Kt=None, params=None, succ=False)
        hist = []

        while n_evals < budget_evals:
            remain = budget_evals - n_evals
            lam_now = int(min(lam, remain))
            eps = rng.standard_normal(size=(lam_now, dim)).astype(np.float64)
            z_pop = mean[None, :] + sigma * eps * np.sqrt(diagC[None, :])
            fit = np.zeros((lam_now,), dtype=np.float64)

            for i in range(lam_now):
                params_vec = z_to_params_vector(z_pop[i], env)
                E, Sy, Kt = env.evaluate_params(
                    params_vec,
                    metadata={"phase": "benchmark", "method": "CMA-ES", "task_id": int(task_id), "seed": int(seed), "eval_index": int(n_evals + 1)},
                )
                reward = float(env.compute_reward(E, Sy, Kt)) if (E is not None and Sy is not None and Kt is not None) else -5.0
                succ = is_success(E, Sy, Kt, target)
                fit[i] = reward
                n_evals += 1
                if reward > best["reward"]:
                    best.update(reward=reward, E=E, Sy=Sy, Kt=Kt, params=params_vec, succ=succ)
                hist.append({"eval": n_evals, "reward": float(reward), "success": bool(succ), "best_reward": float(best["reward"]), "best_success": bool(best["succ"]), "sigma": float(sigma)})
                if n_evals >= budget_evals:
                    break

            if lam_now >= mu:
                idx = np.argsort(-fit)
                top = z_pop[idx[:mu]]
                old_mean = mean.copy()
                mean = (w[:, None] * top).sum(axis=0)
                y = (top - old_mean[None, :]) / (sigma + 1e-12)
                diagC = (1.0 - c_cov) * diagC + c_cov * (w[:, None] * (y ** 2)).sum(axis=0)
                diagC = np.clip(diagC, 1e-6, 1e6)
                improved = 1.0 if (best["reward"] >= float(np.max(fit)) - 1e-12) else 0.0
                success_ema = 0.9 * float(success_ema) + 0.1 * float(improved)
                sigma *= float(np.exp((success_ema - 0.2) / 0.3 * 0.05))
                sigma = float(np.clip(sigma, sigma_min, sigma_max))

        return SolveResult("CMA-ES", task_id, seed, int(budget_evals), int(n_evals), float(best["reward"]), best["E"], best["Sy"], best["Kt"], bool(best["succ"]), None if best["params"] is None else best["params"].astype(float).tolist(), hist)

class SimpleBuffer:
    def __init__(self):
        self.data = []

    def push(self, **kwargs):
        self.data.append(kwargs)

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)

    def collate(self, device: torch.device):
        states = torch.stack([torch.as_tensor(d["state"], dtype=torch.float32) for d in self.data]).to(device)
        actions = torch.stack([torch.as_tensor(d["action"], dtype=torch.float32) for d in self.data]).to(device)
        rewards = torch.as_tensor([d["reward"] for d in self.data], dtype=torch.float32).to(device)
        dones = torch.as_tensor([d["done"] for d in self.data], dtype=torch.float32).to(device)
        log_probs = torch.stack([d["log_prob"] for d in self.data]).to(device)
        values = torch.stack([d["value"] for d in self.data]).to(device).squeeze(-1)
        return dict(states=states, actions=actions, rewards=rewards, dones=dones, log_probs=log_probs, values=values)

class RLOnlySolver:
    def __init__(self, ppo_agent: nn.Module, device: torch.device):
        self.agent = ppo_agent.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=3e-4)
        self.buf = SimpleBuffer()

    def _ppo_update(self, gamma=0.99, clip_ratio=0.2, ppo_epochs=5):
        batch = self.buf.collate(self.device)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        old_logp = batch["log_probs"].detach()
        values = batch["values"].detach()

        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
            if d > 0.5:
                G = 0.0
            G = float(r) + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.detach()

        for _ in range(int(ppo_epochs)):
            mean, std, vpred = self.agent(states)
            dist = Normal(mean, std)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(vpred.squeeze(-1), returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.opt.step()
        self.buf.clear()

    def solve(self, env, target, task_id: int, seed: int, budget_evals: int, *, update_every: int = 64, cost_per_eval: float = 0.0, deterministic_policy: bool = False) -> SolveResult:
        env.set_target(target)
        set_global_seed(seed)
        self.agent.train()

        n_evals = 0
        best = dict(reward=-1e30, E=None, Sy=None, Kt=None, params=None, succ=False)
        hist = []
        last_metrics = dict(E=None, Sy=None, Kt=None)
        step = 0

        while n_evals < budget_evals:
            if last_metrics["E"] is None:
                state_np = env.get_state(target.E_target, target.sigma_y_target, target.Kt_target)
            else:
                state_np = env.get_state(last_metrics["E"], last_metrics["Sy"], last_metrics["Kt"])
            state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            out = self.agent.get_action(state_t, deterministic=bool(deterministic_policy))
            if deterministic_policy:
                action_t, value_t = out
                logp_t = torch.zeros((1,), device=self.device)
            else:
                action_t, logp_t, value_t = out

            z = action_t.squeeze(0).detach().cpu().numpy()
            params_vec = z_to_params_vector(z, env)
            E, Sy, Kt = env.evaluate_params(
                params_vec,
                metadata={"phase": "benchmark", "method": "RL-only(PPO)", "task_id": int(task_id), "seed": int(seed), "eval_index": int(n_evals + 1)},
            )
            reward_raw = float(env.compute_reward(E, Sy, Kt)) if (E is not None and Sy is not None and Kt is not None) else -5.0
            succ = is_success(E, Sy, Kt, target)
            reward = float(reward_raw - float(cost_per_eval))
            n_evals += 1
            step += 1

            if reward_raw > best["reward"]:
                best.update(reward=reward_raw, E=E, Sy=Sy, Kt=Kt, params=params_vec, succ=succ)
            if E is not None:
                last_metrics.update(E=E, Sy=Sy, Kt=Kt)

            hist.append({"eval": n_evals, "reward": float(reward_raw), "success": bool(succ), "best_reward": float(best["reward"]), "best_success": bool(best["succ"])})
            self.buf.push(state=state_np, action=action_t.squeeze(0).detach().cpu(), reward=reward, done=(n_evals >= budget_evals), log_prob=logp_t.detach().cpu(), value=value_t.detach().cpu())
            if (step % int(max(1, update_every)) == 0) and (len(self.buf) >= int(max(8, update_every // 2))):
                self._ppo_update()

        if len(self.buf) > 0:
            self._ppo_update()

        return SolveResult("RL-only(PPO)", task_id, seed, int(budget_evals), int(n_evals), float(best["reward"]), best["E"], best["Sy"], best["Kt"], bool(best["succ"]), None if best["params"] is None else best["params"].astype(float).tolist(), hist)

class MFRLCDSolver:
    def __init__(self, trainer, device: torch.device):
        self.trainer = trainer
        self.device = device

    @staticmethod
    def _log_diag(env, row: Dict[str, Any]):
        tracker = getattr(env, "tracker", None)
        if tracker is not None and hasattr(tracker, "log_diagnostic"):
            tracker.log_diagnostic(row)

    @staticmethod
    def _log_update(env, row: Dict[str, Any]):
        tracker = getattr(env, "tracker", None)
        if tracker is not None and hasattr(tracker, "log_update"):
            tracker.log_update(row)

    @torch.no_grad()
    def solve(self, env, target, task_id: int, seed: int, budget_evals: int, *, enable_updates: bool = True, ppo_min_batch: int = 256, update_every_steps: int = 1, diffusion_train_every: int = 5, diffusion_steps: int = 50, diffusion_top_frac: float = 0.30, cost_per_eval: float = 0.08, cost_steps_scale: float = 0.04, deterministic_policy: bool = False, local_refine_rounds: int = 2, local_refine_scale: float = 0.08, local_refine_decay: float = 0.65) -> SolveResult:
        set_global_seed(seed)
        env.set_target(target)
        self.trainer.best_reward = -float("inf")
        self.trainer.best_params = None
        if hasattr(self.trainer, "rl_buffer"):
            self.trainer.rl_buffer.clear()

        cond = torch.FloatTensor(target.normalize()).to(self.device)
        n_evals = 0
        step = 0
        last_metrics = dict(E=None, Sy=None, Kt=None)
        best = dict(reward=-1e30, E=None, Sy=None, Kt=None, params=None, succ=False)
        hist = []
        prev_error_sum = None
        global_best_error_sum = None
        global_best_z = None
        self._log_update(
            env,
            {
                "task_id": int(task_id),
                "seed": int(seed),
                "event": "mfrlcd_config",
                "enable_updates": bool(enable_updates),
                "ppo_min_batch": int(ppo_min_batch),
                "update_every_steps": int(update_every_steps),
                "diffusion_train_every": int(diffusion_train_every),
                "diffusion_steps": int(diffusion_steps),
                "diffusion_top_frac": float(diffusion_top_frac),
                "deterministic_policy": bool(deterministic_policy),
                "local_refine_rounds": int(local_refine_rounds),
                "local_refine_scale": float(local_refine_scale),
                "local_refine_decay": float(local_refine_decay),
            },
        )

        while n_evals < budget_evals:
            step += 1
            if last_metrics["E"] is None:
                state_np = env.get_state(target.E_target, target.sigma_y_target, target.Kt_target)
            else:
                state_np = env.get_state(last_metrics["E"], last_metrics["Sy"], last_metrics["Kt"])
            state_t = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)

            action_out = self.trainer.rl_agent.get_action(state_t, deterministic=bool(deterministic_policy))
            if deterministic_policy:
                action_t, value_t = action_out
                logp_t = torch.zeros((1,), device=self.device)
            else:
                action_t, logp_t, value_t = action_out

            K, tau, stepsN = self.trainer._decode_sampling_controls(action_t)
            remain = budget_evals - n_evals
            K = int(min(K, remain))
            z_batch = self.trainer.diffusion_model.sample(cond, num_samples=int(K), num_steps=int(stepsN), noise_scale=float(tau))
            z_batch = z_batch.detach().cpu().numpy()

            step_best = dict(reward=-1e30, E=None, Sy=None, Kt=None, params=None, z=None, succ=False, source=None)
            step_eval_count = 0
            ppo_buffer_before = len(self.trainer.rl_buffer) if hasattr(self.trainer, "rl_buffer") else 0
            replay_size_before = len(self.trainer.diff_replay) if hasattr(self.trainer, "diff_replay") else 0

            for i in range(int(K)):
                z = z_batch[i]
                params_vec = z_to_params_vector(z, env)
                E, Sy, Kt = env.evaluate_params(
                    params_vec,
                    metadata={"phase": "benchmark", "method": "MF-RLCD", "task_id": int(task_id), "seed": int(seed), "eval_index": int(n_evals + 1), "step": int(step), "candidate_index": int(i)},
                )
                error_info = env.compute_error_components(E, Sy, Kt) if (E is not None and Sy is not None and Kt is not None) else None
                error_sum = None if error_info is None else error_info["error_sum"]
                reward_raw = float(env.compute_reward(E, Sy, Kt, prev_error_sum=prev_error_sum, best_error_sum=global_best_error_sum)) if (E is not None and Sy is not None and Kt is not None) else -5.0
                succ = is_success(E, Sy, Kt, target)
                n_evals += 1
                step_eval_count += 1
                if hasattr(self.trainer, "diff_replay"):
                    try:
                        self.trainer.diff_replay.push(z=z, cond=cond.detach().cpu().numpy(), reward=reward_raw, success=succ, error_sum=error_sum)
                    except Exception:
                        pass
                if reward_raw > step_best["reward"]:
                    step_best.update(reward=reward_raw, E=E, Sy=Sy, Kt=Kt, params=params_vec, z=z, succ=succ, source="MF-RLCD")
                if reward_raw > best["reward"]:
                    best.update(reward=reward_raw, E=E, Sy=Sy, Kt=Kt, params=params_vec, succ=succ, source="MF-RLCD")
                    global_best_z = np.asarray(z, dtype=np.float64).copy()
                if (error_sum is not None) and ((global_best_error_sum is None) or (error_sum < global_best_error_sum)):
                    global_best_error_sum = float(error_sum)
                hist.append({"eval": n_evals, "reward": float(reward_raw), "success": bool(succ), "best_reward": float(best["reward"]), "best_success": bool(best["succ"]), "K": int(K), "tau": float(tau), "stepsN": int(stepsN)})
                if n_evals >= budget_evals:
                    break

            remain_after_batch = budget_evals - n_evals
            refine_budget = int(min(max(0, local_refine_rounds), max(0, remain_after_batch)))
            if refine_budget > 0 and step_best["z"] is not None:
                def _refine_eval(z_refined, round_idx):
                    nonlocal n_evals, step_eval_count, step_best, best, global_best_error_sum, global_best_z
                    params_refined = z_to_params_vector(z_refined, env)
                    eval_meta = {"phase": "benchmark", "method": "MF-RLCD-local-refine", "task_id": int(task_id), "seed": int(seed), "eval_index": int(n_evals + 1), "step": int(step), "local_round": int(round_idx)}
                    E_ref, Sy_ref, Kt_ref = env.evaluate_params(params_refined, metadata=eval_meta)
                    error_info_ref = env.compute_error_components(E_ref, Sy_ref, Kt_ref) if (E_ref is not None and Sy_ref is not None and Kt_ref is not None) else None
                    error_sum_ref = None if error_info_ref is None else error_info_ref["error_sum"]
                    reward_ref = float(env.compute_reward(E_ref, Sy_ref, Kt_ref, prev_error_sum=prev_error_sum, best_error_sum=global_best_error_sum)) if (E_ref is not None and Sy_ref is not None and Kt_ref is not None) else -5.0
                    succ_ref = is_success(E_ref, Sy_ref, Kt_ref, target)
                    n_evals += 1
                    step_eval_count += 1
                    if hasattr(self.trainer, "diff_replay"):
                        try:
                            self.trainer.diff_replay.push(z=z_refined, cond=cond.detach().cpu().numpy(), reward=reward_ref, success=succ_ref, error_sum=error_sum_ref)
                        except Exception:
                            pass
                    payload = {"reward": reward_ref, "E": E_ref, "Sy": Sy_ref, "Kt": Kt_ref, "params": params_refined, "z": z_refined, "succ": succ_ref, "source": "MF-RLCD-local-refine"}
                    if reward_ref > step_best["reward"]:
                        step_best.update(payload)
                    if reward_ref > best["reward"]:
                        best.update(payload)
                        global_best_z = np.asarray(z_refined, dtype=np.float64).copy()
                    if (error_sum_ref is not None) and ((global_best_error_sum is None) or (error_sum_ref < global_best_error_sum)):
                        global_best_error_sum = float(error_sum_ref)
                    hist.append({"eval": n_evals, "reward": float(reward_ref), "success": bool(succ_ref), "best_reward": float(best["reward"]), "best_success": bool(best["succ"]), "K": int(K), "tau": float(tau), "stepsN": int(stepsN), "local_refine": True, "local_round": int(round_idx)})
                    return reward_ref, succ_ref, payload

                refine_result = refine_latent_candidate(
                    step_best["z"],
                    _refine_eval,
                    budget=refine_budget,
                    seed=seed + task_id * 1000 + step,
                    init_scale=float(local_refine_scale),
                    decay=float(local_refine_decay),
                    base_result=(step_best["reward"], step_best["succ"], dict(step_best)),
                )
                if refine_result["reward"] > step_best["reward"]:
                    step_best.update(refine_result["payload"])
                if refine_result["reward"] > best["reward"]:
                    best.update(refine_result["payload"])
                print(f"[MF-RLCD][LocalRefine] step={step} rounds={refine_budget} best_reward={refine_result['reward']:.3f} success={refine_result['success']}")

            remain_after_local = budget_evals - n_evals
            global_refine_budget = int(min(max(0, local_refine_rounds), max(0, remain_after_local)))
            if global_refine_budget > 0 and global_best_z is not None:
                def _global_refine_eval(z_refined, round_idx):
                    nonlocal n_evals, step_eval_count, step_best, best, global_best_z, global_best_error_sum
                    params_refined = z_to_params_vector(z_refined, env)
                    eval_meta = {"phase": "benchmark", "method": "MF-RLCD-best-so-far-refine", "task_id": int(task_id), "seed": int(seed), "eval_index": int(n_evals + 1), "step": int(step), "global_round": int(round_idx)}
                    E_ref, Sy_ref, Kt_ref = env.evaluate_params(params_refined, metadata=eval_meta)
                    error_info_ref = env.compute_error_components(E_ref, Sy_ref, Kt_ref) if (E_ref is not None and Sy_ref is not None and Kt_ref is not None) else None
                    error_sum_ref = None if error_info_ref is None else error_info_ref["error_sum"]
                    reward_ref = float(env.compute_reward(E_ref, Sy_ref, Kt_ref, prev_error_sum=prev_error_sum, best_error_sum=global_best_error_sum)) if (E_ref is not None and Sy_ref is not None and Kt_ref is not None) else -5.0
                    succ_ref = is_success(E_ref, Sy_ref, Kt_ref, target)
                    n_evals += 1
                    step_eval_count += 1
                    if hasattr(self.trainer, "diff_replay"):
                        try:
                            self.trainer.diff_replay.push(z=z_refined, cond=cond.detach().cpu().numpy(), reward=reward_ref, success=succ_ref, error_sum=error_sum_ref)
                        except Exception:
                            pass
                    payload = {"reward": reward_ref, "E": E_ref, "Sy": Sy_ref, "Kt": Kt_ref, "params": params_refined, "z": z_refined, "succ": succ_ref, "source": "MF-RLCD-best-so-far-refine"}
                    if reward_ref > step_best["reward"]:
                        step_best.update(payload)
                    if reward_ref > best["reward"]:
                        best.update(payload)
                        global_best_z = np.asarray(z_refined, dtype=np.float64).copy()
                    if (error_sum_ref is not None) and ((global_best_error_sum is None) or (error_sum_ref < global_best_error_sum)):
                        global_best_error_sum = float(error_sum_ref)
                    hist.append({"eval": n_evals, "reward": float(reward_ref), "success": bool(succ_ref), "best_reward": float(best["reward"]), "best_success": bool(best["succ"]), "K": int(K), "tau": float(tau), "stepsN": int(stepsN), "global_best_refine": True, "global_round": int(round_idx)})
                    return reward_ref, succ_ref, payload

                global_refine_result = refine_latent_candidate(
                    global_best_z,
                    _global_refine_eval,
                    budget=global_refine_budget,
                    seed=seed + task_id * 2000 + step,
                    init_scale=float(local_refine_scale * 0.75),
                    decay=float(local_refine_decay),
                    base_result=(best["reward"], best["succ"], dict(best)),
                )
                if global_refine_result["reward"] > best["reward"]:
                    best.update(global_refine_result["payload"])
                    global_best_z = np.asarray(global_refine_result["z"], dtype=np.float64).copy()
                print(f"[MF-RLCD][BestSoFarRefine] step={step} rounds={global_refine_budget} best_reward={global_refine_result['reward']:.3f} success={global_refine_result['success']}")

            step_cost = float(cost_per_eval) * float(step_eval_count) + float(cost_steps_scale) * float(stepsN / self.trainer.diffusion_model.num_timesteps)
            rl_reward = float(step_best["reward"] - step_cost)
            if step_best["E"] is not None:
                last_metrics.update(E=step_best["E"], Sy=step_best["Sy"], Kt=step_best["Kt"])
                step_error_info = env.compute_error_components(step_best["E"], step_best["Sy"], step_best["Kt"])
                prev_error_sum = None if step_error_info is None else step_error_info["error_sum"]
                next_state_np = env.get_state(step_best["E"], step_best["Sy"], step_best["Kt"])
            else:
                next_state_np = env.get_state(100000, 900, 1.5)

            done = n_evals >= budget_evals
            try:
                self.trainer.rl_buffer.push(
                    state_np,
                    action_t.squeeze(0).detach().cpu().numpy(),
                    rl_reward,
                    next_state_np,
                    bool(done),
                    log_prob=logp_t.detach().cpu() if logp_t is not None else None,
                    value=value_t.detach().cpu() if value_t is not None else None,
                    params=None,
                    cond=None,
                    success=None,
                )
            except Exception:
                pass

            ppo_update_applied = False
            diffusion_update_applied = False
            ppo_update_reason = "disabled"
            diffusion_update_reason = "disabled"

            if enable_updates:
                ppo_update_reason = "step_gate"
                diffusion_update_reason = "step_gate"

                if (step % int(max(1, update_every_steps)) == 0) and hasattr(self.trainer, "rl_buffer"):
                    if len(self.trainer.rl_buffer) >= int(ppo_min_batch):
                        ppo_update_applied = True
                        ppo_update_reason = "applied"
                        ppo_used = len(self.trainer.rl_buffer)
                        with torch.enable_grad():
                            self.trainer._update_rl_onpolicy()
                        self.trainer.rl_buffer.clear()
                        self._log_update(
                            env,
                            {
                                "task_id": int(task_id),
                                "seed": int(seed),
                                "step": int(step),
                                "update_type": "ppo",
                                "applied": True,
                                "buffer_size": int(ppo_used),
                                "min_batch": int(ppo_min_batch),
                                "update_every_steps": int(update_every_steps),
                            },
                        )
                    else:
                        ppo_update_reason = "buffer_too_small"

                if (step % int(max(1, diffusion_train_every)) == 0) and hasattr(self.trainer, "diff_replay"):
                    if len(self.trainer.diff_replay) >= 4:
                        diffusion_update_applied = True
                        diffusion_update_reason = "applied"
                        replay_used = len(self.trainer.diff_replay)
                        with torch.enable_grad():
                            self.trainer._train_diffusion_online(num_steps=int(diffusion_steps), top_frac=float(diffusion_top_frac))
                        self._log_update(
                            env,
                            {
                                "task_id": int(task_id),
                                "seed": int(seed),
                                "step": int(step),
                                "update_type": "diffusion",
                                "applied": True,
                                "replay_size": int(replay_used),
                                "num_steps": int(diffusion_steps),
                                "top_frac": float(diffusion_top_frac),
                                "train_every_steps": int(diffusion_train_every),
                            },
                        )
                    else:
                        diffusion_update_reason = "replay_too_small"

            ppo_buffer_after = len(self.trainer.rl_buffer) if hasattr(self.trainer, "rl_buffer") else 0
            replay_size_after = len(self.trainer.diff_replay) if hasattr(self.trainer, "diff_replay") else 0
            step_error_info = None
            if step_best["E"] is not None and step_best["Sy"] is not None and step_best["Kt"] is not None:
                step_error_info = env.compute_error_components(step_best["E"], step_best["Sy"], step_best["Kt"])

            self._log_diag(
                env,
                {
                    "task_id": int(task_id),
                    "seed": int(seed),
                    "step": int(step),
                    "enable_updates": bool(enable_updates),
                    "deterministic_policy": bool(deterministic_policy),
                    "K": int(K),
                    "tau": float(tau),
                    "stepsN": int(stepsN),
                    "evals_this_step": int(step_eval_count),
                    "n_evals_total": int(n_evals),
                    "local_refine_rounds_used": int(refine_budget if step_best["z"] is not None else 0),
                    "global_refine_rounds_used": int(global_refine_budget if global_best_z is not None else 0),
                    "step_best_source": step_best.get("source"),
                    "step_best_reward": float(step_best["reward"]),
                    "step_best_success": bool(step_best["succ"]),
                    "step_best_E": step_best["E"],
                    "step_best_Sy": step_best["Sy"],
                    "step_best_Kt": step_best["Kt"],
                    "step_best_E_error": None if step_error_info is None else step_error_info["E_error"],
                    "step_best_sigma_y_error": None if step_error_info is None else step_error_info["sigma_y_error"],
                    "step_best_Kt_error": None if step_error_info is None else step_error_info["Kt_error"],
                    "step_best_error_sum": None if step_error_info is None else step_error_info["error_sum"],
                    "prev_error_sum_after_step": None if prev_error_sum is None else float(prev_error_sum),
                    "global_best_source": best.get("source"),
                    "global_best_reward": float(best["reward"]),
                    "global_best_success": bool(best["succ"]),
                    "global_best_error_sum": None if global_best_error_sum is None else float(global_best_error_sum),
                    "ppo_buffer_before": int(ppo_buffer_before),
                    "ppo_buffer_after": int(ppo_buffer_after),
                    "replay_size_before": int(replay_size_before),
                    "replay_size_after": int(replay_size_after),
                    "ppo_update_applied": bool(ppo_update_applied),
                    "ppo_update_reason": ppo_update_reason,
                    "diffusion_update_applied": bool(diffusion_update_applied),
                    "diffusion_update_reason": diffusion_update_reason,
                },
            )

        return SolveResult("MF-RLCD", task_id, seed, int(budget_evals), int(n_evals), float(best["reward"]), best["E"], best["Sy"], best["Kt"], bool(best["succ"]), None if best["params"] is None else best["params"].astype(float).tolist(), hist)

def make_task_suite(fw, base_target, num_tasks: int, task_mode: str, seed: int):
    rng = np.random.default_rng(int(seed))
    tasks = []
    if int(num_tasks) <= 1:
        tasks.append(base_target)
        return tasks

    E_range = (90000.0, 120000.0)
    Sy_range = (700.0, 1150.0)
    Kt_range = (1.05, 1.60)

    def clip(x, a, b):
        return float(np.clip(float(x), float(a), float(b)))

    for _ in range(int(num_tasks)):
        if task_mode == "random":
            E = float(rng.uniform(*E_range))
            Sy = float(rng.uniform(*Sy_range))
            Kt = float(rng.uniform(*Kt_range))
        else:
            E = clip(base_target.E_target * (1.0 + rng.normal(0.0, 0.02)), *E_range)
            Sy = clip(base_target.sigma_y_target * (1.0 + rng.normal(0.0, 0.02)), *Sy_range)
            Kt = clip(base_target.Kt_target * (1.0 + rng.normal(0.0, 0.03)), *Kt_range)
        tasks.append(fw.TargetProperties(E_target=E, sigma_y_target=Sy, Kt_target=Kt, tolerance=dict(base_target.tolerance)))
    return tasks

def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
