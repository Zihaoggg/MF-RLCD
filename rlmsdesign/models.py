from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, param_dim=151, condition_dim=3, time_dim=64, hidden_dim=512):
        super().__init__()
        self.param_dim = param_dim
        self.condition_dim = condition_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        input_dim = param_dim + time_dim + hidden_dim // 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, param_dim),
        )

        self.num_timesteps = 1000
        self.register_buffer("betas", torch.linspace(0.0001, 0.02, self.num_timesteps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))

    def forward(self, x, t, condition):
        t_emb = self.time_encoder(t.float().unsqueeze(-1))
        c_emb = self.condition_encoder(condition)
        return self.network(torch.cat([x, t_emb, c_emb], dim=-1))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, condition, clip_denoised=True):
        model_out = self(x, t, condition)
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        while len(betas_t.shape) < len(x.shape):
            betas_t = betas_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.unsqueeze(-1)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_out / sqrt_one_minus_alphas_cumprod_t)
        if clip_denoised:
            model_mean = torch.clamp(model_mean, -1, 1)
        if t[0] > 0:
            alphas_cumprod_prev_t = self.alphas_cumprod_prev[t]
            alphas_cumprod_t = self.alphas_cumprod[t]
            while len(alphas_cumprod_prev_t.shape) < len(x.shape):
                alphas_cumprod_prev_t = alphas_cumprod_prev_t.unsqueeze(-1)
                alphas_cumprod_t = alphas_cumprod_t.unsqueeze(-1)
            posterior_variance = betas_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
            return model_mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
        return model_mean

    @torch.no_grad()
    def sample(self, condition, num_samples=1, guidance_scale=1.0, num_steps=None, noise_scale=1.0):
        device = next(self.parameters()).device
        batch_size = int(num_samples)
        x = noise_scale * torch.randn(batch_size, self.param_dim, device=device)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        if condition.shape[0] == 1 and batch_size > 1:
            condition = condition.repeat(batch_size, 1)
        T = self.num_timesteps if (num_steps is None) else int(num_steps)
        T = max(1, min(T, self.num_timesteps))
        start = self.num_timesteps - 1
        end = self.num_timesteps - T
        for t in range(start, end, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, condition)
        return x

class PPOAgent(nn.Module):
    def __init__(self, state_dim=9, action_dim=3, hidden_dim=256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.actor_log_std)
        value = self.critic(state)
        return action_mean, action_std, value

    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self.forward(state)
        if deterministic:
            return action_mean, value
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, states, actions):
        action_mean, action_std, values = self.forward(states)
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values, entropy

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, log_prob=None, value=None, params=None, cond=None, success=None):
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "log_prob": log_prob,
            "value": value,
            "params": None if params is None else np.asarray(params, dtype=np.float32),
            "cond": None if cond is None else np.asarray(cond, dtype=np.float32),
            "success": bool(success) if success is not None else None,
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return self._collate(batch)

    def get_all(self):
        return self._collate(list(self.buffer))

    def _collate(self, batch):
        states = torch.stack([torch.FloatTensor(exp["state"]) for exp in batch])
        actions = torch.stack([torch.FloatTensor(exp["action"]) for exp in batch])
        rewards = torch.FloatTensor([exp["reward"] for exp in batch])
        next_states = torch.stack([
            torch.FloatTensor(exp["state"] if exp["next_state"] is None else exp["next_state"])
            for exp in batch
        ])
        dones = torch.BoolTensor([exp["done"] for exp in batch])
        result = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones}
        if batch[0]["log_prob"] is not None:
            result["log_probs"] = torch.stack([exp["log_prob"] for exp in batch]).float()
        if batch[0]["value"] is not None:
            result["values"] = torch.stack([exp["value"] for exp in batch]).float()
        if batch[0].get("params", None) is not None:
            result["params"] = torch.stack([torch.FloatTensor(exp["params"]) for exp in batch])
        if batch[0].get("cond", None) is not None:
            result["cond"] = torch.stack([torch.FloatTensor(exp["cond"]) for exp in batch])
        if batch[0].get("success", None) is not None:
            result["success"] = torch.BoolTensor([bool(exp["success"]) for exp in batch])
        return result

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class DiffusionReplay:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, z, cond, reward, success, error_sum=None, quality_score=None, near_success_margin=0.20):
        error_sum = None if error_sum is None else float(error_sum)
        reward = float(reward)
        success = bool(success)
        if quality_score is None:
            if error_sum is None:
                quality_score = reward
            else:
                quality_score = reward + 1.5 / (1.0 + 8.0 * error_sum)
        quality_score = float(quality_score)

        keep = success
        if (not keep) and (error_sum is not None) and (error_sum <= float(near_success_margin)):
            keep = True
        if (not keep) and (quality_score >= 1.5):
            keep = True
        if not keep:
            return
        self.buffer.append({
            "z": np.asarray(z, dtype=np.float32),
            "cond": np.asarray(cond, dtype=np.float32),
            "reward": reward,
            "success": success,
            "error_sum": error_sum,
            "quality_score": quality_score,
        })

    def __len__(self):
        return len(self.buffer)

    def sample_top(self, max_items=5000, top_frac=0.30):
        items = list(self.buffer)
        if len(items) == 0:
            return []
        items.sort(key=lambda e: (float(e.get("quality_score", e["reward"])), float(e["reward"])), reverse=True)
        k = max(1, int(len(items) * float(top_frac)))
        return items[: min(k, int(max_items))]

    def save_npz(self, path: str, max_save: int = 120000):
        items = list(self.buffer)
        if len(items) == 0:
            return
        items.sort(key=lambda e: e["reward"])
        items = items[-min(int(max_save), len(items)) :]
        Z = np.stack([e["z"] for e in items]).astype(np.float32)
        C = np.stack([e["cond"] for e in items]).astype(np.float32)
        R = np.array([e["reward"] for e in items], dtype=np.float32)
        S = np.array([e["success"] for e in items], dtype=np.bool_)
        E = np.array([
            np.nan if e.get("error_sum", None) is None else float(e["error_sum"])
            for e in items
        ], dtype=np.float32)
        Q = np.array([float(e.get("quality_score", e["reward"])) for e in items], dtype=np.float32)
        np.savez_compressed(path, z=Z, cond=C, reward=R, success=S, error_sum=E, quality_score=Q)

    def load_npz(self, path: str, merge: bool = True):
        p = str(path)
        try:
            data = np.load(p, allow_pickle=False)
            Z = data["z"]
            C = data["cond"]
            R = data["reward"]
            S = data["success"] if "success" in data.files else np.ones((Z.shape[0],), dtype=np.bool_)
            E = data["error_sum"] if "error_sum" in data.files else np.full((Z.shape[0],), np.nan, dtype=np.float32)
            Q = data["quality_score"] if "quality_score" in data.files else R
            if not merge:
                self.buffer.clear()
            for i in range(Z.shape[0]):
                err_i = None if np.isnan(E[i]) else float(E[i])
                self.push(Z[i], C[i], float(R[i]), bool(S[i]), error_sum=err_i, quality_score=float(Q[i]))
            print(f"[DiffusionReplay] loaded {Z.shape[0]} samples from {p}")
        except Exception as e:
            print(f"[DiffusionReplay] load failed: {e}")
