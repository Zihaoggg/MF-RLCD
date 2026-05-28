from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlmsdesign.models import DiffusionReplay, ExperienceBuffer

class RLDiffusionTrainer:

    def __init__(self, env, diffusion_model, rl_agent, device="cuda"):
        self.env = env
        self.diffusion_model = diffusion_model.to(device)
        self.rl_agent = rl_agent.to(device)
        self.device = device

        self.diffusion_optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=1e-4, weight_decay=0.01)
        self.rl_optimizer = torch.optim.Adam(rl_agent.parameters(), lr=3e-4)

        self.rl_buffer = ExperienceBuffer(capacity=50000)
        self.diff_replay = DiffusionReplay(capacity=200000)

        self.best_reward = -float("inf")
        self.best_params = None

        self.episode_rewards = []
        self.episode_properties = []

        self._no_improve = 0
        self._steps_done = 0

    def _decode_sampling_controls(self, action_tensor):

        a = action_tensor.squeeze()
        a0 = torch.sigmoid(a[0]).item()
        a1 = torch.sigmoid(a[1]).item()
        a2 = torch.sigmoid(a[2]).item()

        k_choices = [1, 2, 4, 8]
        idx = int(np.clip(np.floor(a0 * len(k_choices)), 0, len(k_choices) - 1))
        K = int(k_choices[idx])

        tau = float(0.6 + a1 * (1.6 - 0.6))
        ratio = float(0.2 + a2 * 0.8)
        steps = int(np.clip(round(self.diffusion_model.num_timesteps * ratio), 200, self.diffusion_model.num_timesteps))
        return K, tau, steps

    def _is_success(self, E, sigma_y, Kt):
        t = self.env.target
        if E is None or sigma_y is None or Kt is None:
            return False
        E_err = abs(E - t.E_target) / t.E_target
        Sy_err = abs(sigma_y - t.sigma_y_target) / t.sigma_y_target
        Kt_err = abs(Kt - t.Kt_target) / abs(t.Kt_target)
        return (E_err < t.tolerance["E"]) and (Sy_err < t.tolerance["sigma_y"]) and (Kt_err < t.tolerance["Kt"])

    def optimize(
        self,
        num_episodes=500,
        steps_per_episode=10,
        target_sampler=None,
        update_every_episodes=1,
        ppo_min_batch=256,
        diffusion_train_every=5,
        diffusion_steps=200,
        diffusion_top_frac=0.30,
        cost_per_eval=0.08,
        cost_steps_scale=0.04,
        early_stop_patience=100,
        min_steps_before_stop=150,
        improve_eps=0.005,
        log_interval=10,
    ):
        for episode in range(num_episodes):
            if target_sampler is not None:
                self.env.set_target(target_sampler())

            condition = torch.FloatTensor(self.env.target.normalize()).to(self.device)
            episode_reward = 0.0
            last_metrics = dict(E=None, Sy=None, Kt=None)
            prev_error_sum = None
            best_error_sum = None

            for step in range(steps_per_episode):
                self._steps_done += 1

                if last_metrics["E"] is None:
                    current_state = self.env.get_state(
                        self.env.target.E_target,
                        self.env.target.sigma_y_target,
                        self.env.target.Kt_target,
                    )
                else:
                    current_state = self.env.get_state(last_metrics["E"], last_metrics["Sy"], last_metrics["Kt"])

                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, log_prob, value = self.rl_agent.get_action(state_tensor)

                K, tau, stepsN = self._decode_sampling_controls(action)

                with torch.no_grad():
                    z_batch = self.diffusion_model.sample(condition, num_samples=K, num_steps=stepsN, noise_scale=tau)

                best = {
                    "reward": -1e9,
                    "E": None,
                    "Sy": None,
                    "Kt": None,
                    "params_np": None,
                    "z": None,
                    "success": False,
                }

                for i in range(K):
                    z_i = z_batch[i].detach().cpu().numpy()
                    params_np = self._denormalize_params(z_i)
                    eval_meta = {
                        "phase": "optimize",
                        "episode": int(episode),
                        "step": int(step),
                        "candidate_index": int(i),
                        "K": int(K),
                        "tau": float(tau),
                        "sampling_steps": int(stepsN),
                    }
                    E, sigma_y, Kt = self.env.evaluate_params(params_np, metadata=eval_meta)

                    if (E is None) or (sigma_y is None) or (Kt is None):
                        reward_raw = -5.0
                        succ = False
                        error_sum = None
                    else:
                        errors = self.env.compute_error_components(E, sigma_y, Kt)
                        error_sum = errors["error_sum"]
                        reward_raw = float(
                            self.env.compute_reward(
                                E,
                                sigma_y,
                                Kt,
                                prev_error_sum=prev_error_sum,
                                best_error_sum=best_error_sum,
                            )
                        )
                        succ = self._is_success(E, sigma_y, Kt)

                    self.diff_replay.push(
                        z=z_i,
                        cond=condition.detach().cpu().numpy(),
                        reward=reward_raw,
                        success=succ,
                        error_sum=error_sum,
                    )

                    if reward_raw > best["reward"]:
                        best.update(
                            {
                                "reward": reward_raw,
                                "E": E,
                                "Sy": sigma_y,
                                "Kt": Kt,
                                "params_np": params_np,
                                "z": z_i,
                                "success": succ,
                            }
                        )

                step_cost = cost_per_eval * float(K) + cost_steps_scale * float(stepsN / self.diffusion_model.num_timesteps)
                rl_reward = float(best["reward"] - step_cost)
                best_snapshot = self.env.summarize_metrics(best["E"], best["Sy"], best["Kt"])
                episode_reward += rl_reward

                if best["E"] is not None:
                    last_metrics.update(E=best["E"], Sy=best["Sy"], Kt=best["Kt"])
                    next_state = self.env.get_state(best["E"], best["Sy"], best["Kt"])
                    best_errors = self.env.compute_error_components(best["E"], best["Sy"], best["Kt"])
                    prev_error_sum = None if best_errors is None else best_errors["error_sum"]
                    if (best_errors is not None) and ((best_error_sum is None) or (best_errors["error_sum"] < best_error_sum)):
                        best_error_sum = best_errors["error_sum"]
                else:
                    next_state = self.env.get_state(100000, 900, 1.5)

                done = step == steps_per_episode - 1

                self.rl_buffer.push(
                    current_state,
                    action.squeeze().cpu().numpy(),
                    rl_reward,
                    next_state,
                    done,
                    log_prob.cpu() if log_prob is not None else None,
                    value.squeeze().cpu() if value is not None else None,
                    params=None,
                    cond=None,
                    success=None,
                )

                if best["reward"] > (self.best_reward + improve_eps) and (best["params_np"] is not None):
                    self.best_reward = float(best["reward"])
                    self.best_params = best["params_np"]
                    self._no_improve = 0
                    print(
                        f"\nNew best! Ep {episode} Step {step} | "
                        f"Reward={best['reward']:.3f} | "
                        f"E={best['E']}, Sy={best['Sy']}, Kt={best['Kt']} | "
                        f"success={best['success']}"
                    )
                    self.verify_best(repeats=2, verbose=True)
                else:
                    self._no_improve += 1

                print(
                    f"[Optimize] ep={episode:04d} step={step:02d} | "
                    f"K={K} tau={tau:.3f} steps={stepsN} | "
                    f"best_reward={best['reward']:.3f} rl_reward={rl_reward:.3f} step_cost={step_cost:.3f} | "
                    f"success={best_snapshot.success} | "
                    f"E_err={None if best_snapshot.E_error is None else round(best_snapshot.E_error * 100, 2)}% "
                    f"Sy_err={None if best_snapshot.sigma_y_error is None else round(best_snapshot.sigma_y_error * 100, 2)}% "
                    f"Kt_err={None if best_snapshot.Kt_error is None else round(best_snapshot.Kt_error * 100, 2)}%"
                )

                if (self._steps_done >= min_steps_before_stop) and (self._no_improve >= early_stop_patience):
                    print(f"\nNo improvement for {self._no_improve} steps. Early stop after {self._steps_done} total steps.")
                    self.save_checkpoint(f"earlystop_noimprove_ep{episode:03d}_step{step:03d}")
                    return

            self.episode_rewards.append(float(episode_reward))
            if last_metrics["E"] is not None:
                self.episode_properties.append([last_metrics["E"], last_metrics["Sy"], last_metrics["Kt"]])
            episode_snapshot = self.env.summarize_metrics(last_metrics["E"], last_metrics["Sy"], last_metrics["Kt"])
            self.env.tracker.log_episode(
                {
                    "episode": int(episode),
                    "episode_reward": float(episode_reward),
                    "best_reward_global": float(self.best_reward),
                    "last_E": episode_snapshot.E,
                    "last_sigma_y": episode_snapshot.sigma_y,
                    "last_Kt": episode_snapshot.Kt,
                    "last_E_error": episode_snapshot.E_error,
                    "last_sigma_y_error": episode_snapshot.sigma_y_error,
                    "last_Kt_error": episode_snapshot.Kt_error,
                    "last_success": bool(episode_snapshot.success),
                    "target_E": float(self.env.target.E_target),
                    "target_sigma_y": float(self.env.target.sigma_y_target),
                    "target_Kt": float(self.env.target.Kt_target),
                }
            )

            if (episode % int(update_every_episodes) == 0) and (len(self.rl_buffer) >= int(ppo_min_batch)):
                self._update_rl_onpolicy()
                self.rl_buffer.clear()

            if (episode % int(diffusion_train_every) == 0) and (len(self.diff_replay) >= 4):
                self._train_diffusion_online(num_steps=int(diffusion_steps), top_frac=float(diffusion_top_frac))

            if episode % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:]) if len(self.episode_rewards) >= log_interval else episode_reward
                print(
                    f"Episode {episode} | Avg RL-reward: {avg_reward:.3f} | "
                    f"Best (raw) reward: {self.best_reward:.3f} | "
                    f"Target: E={self.env.target.E_target:.1f}, Sy={self.env.target.sigma_y_target:.1f}, Kt={self.env.target.Kt_target:.3f} | "
                    f"NoImprove={self._no_improve}"
                )

    def train(self, *args, **kwargs):

        return self.optimize(*args, **kwargs)

    def _update_rl_onpolicy(self, gamma=0.99, clip_ratio=0.2, ppo_epochs=10):
        batch = self.rl_buffer.get_all()
        device = self.device

        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        old_log_prob = batch["log_probs"].to(device)
        values = batch["values"].to(device).squeeze()

        rewards = batch["rewards"].cpu().numpy().tolist()
        dones = batch["dones"].cpu().numpy().tolist()

        returns_list = []
        G = 0.0
        for reward_item, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0.0
            G = float(reward_item) + gamma * G
            returns_list.insert(0, G)
        returns = torch.tensor(returns_list, dtype=torch.float32, device=device)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()

        for _ in range(int(ppo_epochs)):
            log_probs_new, values_new, entropy = self.rl_agent.evaluate_actions(states, actions)

            ratio = torch.exp(log_probs_new - old_log_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_new.squeeze(), returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            self.rl_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rl_agent.parameters(), 0.5)
            self.rl_optimizer.step()

    def _train_diffusion_online(self, num_steps=200, top_frac=0.30):
        items = self.diff_replay.sample_top(max_items=8000, top_frac=top_frac)
        if len(items) < 4:
            return

        Z = torch.FloatTensor(np.stack([item["z"] for item in items])).to(self.device)
        C = torch.FloatTensor(np.stack([item["cond"] for item in items])).to(self.device)
        R = torch.FloatTensor([item["reward"] for item in items]).to(self.device)
        Q = torch.FloatTensor([float(item.get("quality_score", item["reward"])) for item in items]).to(self.device)

        w = (Q - Q.min()) / (Q.max() - Q.min() + 1e-8)
        w = 0.1 + w

        self.diffusion_model.train()
        N = Z.shape[0]
        for _ in range(int(num_steps)):
            bs = min(64, N)
            idx = torch.randint(0, N, (bs,), device=self.device)

            x0 = Z[idx]
            cond = C[idx]
            t = torch.randint(0, self.diffusion_model.num_timesteps, (bs,), device=self.device)

            noise = torch.randn_like(x0)
            xt = self.diffusion_model.q_sample(x0, t, noise)
            pred = self.diffusion_model(xt, t, cond)

            loss_vec = F.mse_loss(pred, noise, reduction="none").mean(dim=1)
            loss = (loss_vec * w[idx]).mean()

            self.diffusion_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
            self.diffusion_optimizer.step()

        self.diffusion_model.eval()

    def verify_best(self, repeats: int = 5, verbose: bool = True) -> bool:

        if getattr(self, "best_params", None) is None:
            if verbose:
                print("[Verify] best_params is not available yet.")
            return False

        Es, Sy, Ks = [], [], []
        for _ in range(max(1, repeats)):
            E, s, k = self.env.evaluate_params(self.best_params)
            if (E is None) or (s is None) or (k is None):
                continue
            Es.append(E)
            Sy.append(s)
            Ks.append(k)

        if len(Es) == 0:
            if verbose:
                print("[Verify] best_params evaluation failed for all repeats.")
            return False

        mE, mS, mK = float(np.mean(Es)), float(np.mean(Sy)), float(np.mean(Ks))
        t = self.env.target
        eE = abs(mE - t.E_target) / t.E_target
        eS = abs(mS - t.sigma_y_target) / t.sigma_y_target
        eK = abs(mK - t.Kt_target) / abs(t.Kt_target)

        ok = (eE < t.tolerance["E"]) and (eS < t.tolerance["sigma_y"]) and (eK < t.tolerance["Kt"])

        if verbose:
            print(
                f"[Best Verify] "
                f"E={mE:.1f} (err {eE*100:.2f}%), "
                f"Sy={mS:.1f} (err {eS*100:.2f}%), "
                f"Kt={mK:.3f} (err {eK*100:.2f}%) -> success: {ok}"
            )
            print(f"  Std: E±{np.std(Es):.1f}, Sy±{np.std(Sy):.1f}, Kt±{np.std(Ks):.3f} over {len(Es)} evaluations")

        self.best_verify = {
            "E_mean": mE,
            "sigma_y_mean": mS,
            "Kt_mean": mK,
            "E_err": eE,
            "sigma_y_err": eS,
            "Kt_err": eK,
            "n_eval": len(Es),
            "meets_tolerance": ok,
        }
        return ok

    def _denormalize_params(self, params):

        max_n = self.env.max_n_priorBeta
        candidates = self.env.beta_candidates
        params = np.asarray(params, dtype=np.float64)

        def z2u(z):
            z = np.asarray(z, dtype=np.float64)
            return 0.5 * (1.0 + np.tanh(0.7978845608028654 * (z + 0.044715 * (z ** 3))))

        z_angles = params[: max_n * 3]
        u_angles = z2u(z_angles)
        angles = np.empty(max_n * 3, dtype=np.float64)
        angles[0:max_n] = (u_angles[0:max_n] * 360.0) % 360.0
        angles[max_n:2 * max_n] = u_angles[max_n:2 * max_n] * 180.0
        angles[2 * max_n:3 * max_n] = (u_angles[2 * max_n:3 * max_n] * 360.0) % 360.0

        z_seeds = params[max_n * 3:max_n * 6]
        u_seeds = z2u(z_seeds)
        seeds = 1 + np.floor(u_seeds * 29.0).astype(np.int64)
        seeds = np.clip(seeds, 1, 29).astype(np.int64)

        z_lam = params[max_n * 6]
        u_lam = float(z2u(z_lam))
        lam_ratio = float(0.1 + u_lam * (4.0 - 0.1))

        z_n = params[max_n * 6 + 1] if len(params) > (max_n * 6 + 1) else 0.0
        u_n = float(z2u(z_n))
        idx = int(np.floor(u_n * len(candidates)))
        idx = int(np.clip(idx, 0, len(candidates) - 1))
        n_priorBeta = int(candidates[idx])

        return np.concatenate([angles, seeds.astype(np.float64), [lam_ratio], [float(n_priorBeta)]])

    def save_checkpoint(self, episode):

        save_dir = Path(self.env.work_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"checkpoint_{episode}.pt" if isinstance(episode, str) else f"checkpoint_ep{episode:04d}.pt"
        save_path = save_dir / filename

        checkpoint_data = {
            "episode": episode if not isinstance(episode, str) else -1,
            "best_reward": float(self.best_reward) if self.best_reward != -float("inf") else -1000.0,
            "episode_rewards": [float(r) for r in self.episode_rewards] if self.episode_rewards else [],
            "episode_properties": self.episode_properties if self.episode_properties else [],
        }

        try:
            checkpoint_data["diffusion_state"] = self.diffusion_model.state_dict()
            checkpoint_data["rl_state"] = self.rl_agent.state_dict()
            checkpoint_data["diffusion_optimizer"] = self.diffusion_optimizer.state_dict()
            checkpoint_data["rl_optimizer"] = self.rl_optimizer.state_dict()
        except Exception as exc:
            print(f"Warning: failed to store model state: {exc}")

        if self.best_params is not None:
            try:
                checkpoint_data["best_params"] = self.best_params.tolist() if hasattr(self.best_params, "tolist") else list(self.best_params)
            except Exception:
                checkpoint_data["best_params"] = None
        else:
            checkpoint_data["best_params"] = None

        try:
            torch.save(checkpoint_data, str(save_path), _use_new_zipfile_serialization=False)
            print(f"Checkpoint saved to {save_path}")
            return True
        except Exception as exc:
            print(f"Failed to save checkpoint with torch.save: {exc}")
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except Exception:
                pass
            try:
                import pickle

                pickle_path = save_path.with_suffix(".pkl")
                with open(pickle_path, "wb") as fh:
                    pickle.dump(checkpoint_data, fh)
                print(f"Checkpoint saved as pickle to {pickle_path}")
                return True
            except Exception as exc2:
                print(f"Also failed to save as pickle: {exc2}")
                return False

                try:
                    json_path = save_path.with_suffix(".json")
                    basic_data = {
                        "episode": checkpoint_data["episode"],
                        "best_reward": checkpoint_data["best_reward"],
                        "episode_rewards": checkpoint_data["episode_rewards"],
                        "best_params": checkpoint_data["best_params"],
                    }
                    with open(json_path, "w", encoding="utf-8") as fh:
                        json.dump(basic_data, fh)
                    print(f"Basic data saved as JSON to {json_path}")
                    return True
                except Exception as exc3:
                    print(f"Failed to save even basic data: {exc3}")
                    return False

    def load_checkpoint(self, path):

        checkpoint = torch.load(path, map_location=self.device)

        if "diffusion_state" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_state"])
        if "rl_state" in checkpoint:
            self.rl_agent.load_state_dict(checkpoint["rl_state"])
        if "diffusion_optimizer" in checkpoint:
            self.diffusion_optimizer.load_state_dict(checkpoint["diffusion_optimizer"])
        if "rl_optimizer" in checkpoint:
            self.rl_optimizer.load_state_dict(checkpoint["rl_optimizer"])

        self.best_reward = float(checkpoint.get("best_reward", -float("inf")))
        bp = checkpoint.get("best_params", None)
        self.best_params = None if bp is None else np.asarray(bp, dtype=np.float64)

        self.episode_rewards = list(checkpoint.get("episode_rewards", []))
        self.episode_properties = list(checkpoint.get("episode_properties", []))
        return int(checkpoint.get("episode", 0))
