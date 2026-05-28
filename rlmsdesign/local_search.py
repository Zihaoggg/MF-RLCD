from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

def refine_latent_candidate(
    base_z,
    evaluator: Callable[[np.ndarray, int], Tuple[float, bool, Dict]],
    *,
    budget: int = 3,
    seed: int = 0,
    init_scale: float = 0.10,
    decay: float = 0.65,
    base_result=None,
):
    base_z = np.asarray(base_z, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    if base_result is None:
        best_reward, best_success, best_payload = evaluator(base_z.copy(), -1)
    else:
        best_reward, best_success, best_payload = base_result
    best_z = base_z.copy()
    scale = float(init_scale)
    history = []

    for round_idx in range(int(max(0, budget))):
        candidate = best_z + rng.normal(0.0, scale, size=best_z.shape)
        reward, success, payload = evaluator(candidate, round_idx)
        history.append({
            "round": int(round_idx),
            "scale": float(scale),
            "reward": float(reward),
            "success": bool(success),
        })
        if reward > best_reward:
            best_reward = float(reward)
            best_success = bool(success)
            best_payload = payload
            best_z = candidate
        scale *= float(decay)

    return {
        "z": best_z,
        "reward": float(best_reward),
        "success": bool(best_success),
        "payload": best_payload,
        "history": history,
    }
