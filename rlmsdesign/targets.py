from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

@dataclass
class TargetProperties:
    E_target: float = 110000
    sigma_y_target: float = 866
    Kt_target: float = 1.19
    tolerance: Dict[str, float] = field(default_factory=lambda: {
        "E": 0.05,
        "sigma_y": 0.05,
        "Kt": 0.1,
    })

    def normalize(self):
        return np.array([
            self.E_target / 120000,
            self.sigma_y_target / 1200,
            self.Kt_target / 2.0,
        ])

class TargetPoolManager:
    def __init__(self, pool_path: Path, max_size: int = 2000):
        self.pool_path = Path(pool_path)
        self.max_size = int(max_size)
        self._targets = []
        self._keyset = set()

    @staticmethod
    def _key(t: "TargetProperties") -> str:
        return (
            f"{round(t.E_target, 1)}|{round(t.sigma_y_target, 1)}|{round(t.Kt_target, 4)}|"
            f"{round(t.tolerance.get('E', 0.05), 4)}|{round(t.tolerance.get('sigma_y', 0.05), 4)}|"
            f"{round(t.tolerance.get('Kt', 0.1), 4)}"
        )

    @staticmethod
    def _to_dict(t: "TargetProperties") -> dict:
        return {
            "E_target": float(t.E_target),
            "sigma_y_target": float(t.sigma_y_target),
            "Kt_target": float(t.Kt_target),
            "tolerance": dict(t.tolerance),
        }

    @staticmethod
    def _from_dict(d: dict) -> "TargetProperties":
        return TargetProperties(
            E_target=float(d["E_target"]),
            sigma_y_target=float(d["sigma_y_target"]),
            Kt_target=float(d["Kt_target"]),
            tolerance=dict(d.get("tolerance", {"E": 0.05, "sigma_y": 0.05, "Kt": 0.1})),
        )

    def load(self):
        if not self.pool_path.exists():
            return
        try:
            data = json.load(self.pool_path.open("r", encoding="utf-8"))
            for td in data.get("targets", []):
                self.add(self._from_dict(td), save=False)
        except Exception as exc:
            print(f"[TargetPoolManager] load failed: {exc}")

    def save(self):
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "targets": [self._to_dict(t) for t in self._targets[-self.max_size:]],
        }
        json.dump(data, self.pool_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    def add(self, t: "TargetProperties", save: bool = True):
        key = self._key(t)
        if key in self._keyset:
            return
        self._keyset.add(key)
        self._targets.append(t)
        if len(self._targets) > self.max_size:
            dropped = self._targets.pop(0)
            self._keyset.discard(self._key(dropped))
        if save:
            self.save()

    def all(self):
        return list(self._targets)

    def sample(self, rng: np.random.Generator) -> "TargetProperties":
        if not self._targets:
            raise RuntimeError("Target pool is empty")
        return self._targets[int(rng.integers(len(self._targets)))]

def _clip(v, lo, hi):
    return float(max(lo, min(hi, v)))

def target_error_vector(a: "TargetProperties", b: "TargetProperties"):
    return {
        "E": abs(float(a.E_target) - float(b.E_target)) / max(abs(float(b.E_target)), 1e-12),
        "sigma_y": abs(float(a.sigma_y_target) - float(b.sigma_y_target)) / max(abs(float(b.sigma_y_target)), 1e-12),
        "Kt": abs(float(a.Kt_target) - float(b.Kt_target)) / max(abs(float(b.Kt_target)), 1e-12),
    }

def target_distance(a: "TargetProperties", b: "TargetProperties", weights=(1.0, 1.0, 1.0)) -> float:
    err = target_error_vector(a, b)
    wE, wSy, wKt = weights
    return float(wE * err["E"] + wSy * err["sigma_y"] + wKt * err["Kt"])

def make_mixed_target_sampler(
    pool_mgr: TargetPoolManager,
    current_target: "TargetProperties",
    *,
    seed: int = 1234,
    p_current: float = 0.55,
    p_pool: float = 0.30,
    p_jitter: float = 0.15,
    jitter_rel_E: float = 0.02,
    jitter_rel_Sy: float = 0.02,
    jitter_rel_Kt: float = 0.03,
    E_range=(90000.0, 120000.0),
    Sy_range=(700.0, 1150.0),
    Kt_range=(1.05, 1.60),
):
    rng = np.random.default_rng(seed)
    ps = np.array([p_current, p_pool, p_jitter], dtype=np.float64)
    ps = ps / (ps.sum() + 1e-12)

    def _jitter(base: "TargetProperties") -> "TargetProperties":
        E = base.E_target * (1.0 + rng.normal(0.0, jitter_rel_E))
        Sy = base.sigma_y_target * (1.0 + rng.normal(0.0, jitter_rel_Sy))
        Kt = base.Kt_target * (1.0 + rng.normal(0.0, jitter_rel_Kt))
        return TargetProperties(
            E_target=_clip(E, *E_range),
            sigma_y_target=_clip(Sy, *Sy_range),
            Kt_target=_clip(Kt, *Kt_range),
            tolerance=dict(base.tolerance),
        )

    def _sample():
        mode = int(rng.choice(3, p=ps))
        if mode == 0 or len(pool_mgr.all()) == 0:
            return current_target
        if mode == 1:
            return pool_mgr.sample(rng)
        base = current_target if rng.random() < 0.5 else pool_mgr.sample(rng)
        return _jitter(base)

    return _sample

def find_similar_target_in_pool(pool_mgr: TargetPoolManager, target: "TargetProperties", rel_each: float = 0.05):
    best = None
    best_sum = 1e30
    for item in pool_mgr.all():
        err = target_error_vector(item, target)
        if (err["E"] <= rel_each) and (err["sigma_y"] <= rel_each) and (err["Kt"] <= rel_each):
            score = err["E"] + err["sigma_y"] + err["Kt"]
            if score < best_sum:
                best_sum = score
                best = item
    return best

def _safe_load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def _history_run_to_index_record(item: dict) -> dict:
    target = item["target"]
    run_dir = Path(item["run_dir"]).resolve()
    return {
        "run_dir": str(run_dir),
        "summary_path": str(Path(item["summary_path"]).resolve()),
        "target": {
            "E": float(target.E_target),
            "sigma_y": float(target.sigma_y_target),
            "Kt": float(target.Kt_target),
        },
        "best_reward": float(item.get("best_reward", -1e30)),
        "best_success": bool(item.get("best_success", False)),
        "best_error_sum": None if item.get("best_error_sum", None) is None else float(item.get("best_error_sum")),
        "best_params_path": str(Path(item["best_params_path"]).resolve()),
        "replay_path": str(Path(item["replay_path"]).resolve()),
        "checkpoint_path": str(Path(item["checkpoint_path"]).resolve()),
    }

def _index_record_to_history_run(record: dict) -> Optional[dict]:
    target_dict = record.get("target")
    if not isinstance(target_dict, dict):
        return None
    try:
        target = TargetProperties(
            E_target=float(target_dict["E"]),
            sigma_y_target=float(target_dict["sigma_y"]),
            Kt_target=float(target_dict["Kt"]),
        )
    except Exception:
        return None

    def _p(key: str) -> Path:
        return Path(record[key]).resolve()

    return {
        "run_dir": _p("run_dir"),
        "summary_path": _p("summary_path"),
        "target": target,
        "best_reward": float(record.get("best_reward", -1e30)),
        "best_success": bool(record.get("best_success", False)),
        "best_error_sum": None if record.get("best_error_sum", None) is None else float(record.get("best_error_sum")),
        "best_params_path": _p("best_params_path"),
        "replay_path": _p("replay_path"),
        "checkpoint_path": _p("checkpoint_path"),
    }

def load_experience_index(index_path: Path, *, exclude_dir: Optional[Path] = None) -> List[dict]:
    index_path = Path(index_path)
    exclude_dir = None if exclude_dir is None else Path(exclude_dir).resolve()
    payload = _safe_load_json(index_path)
    if not isinstance(payload, dict):
        return []

    runs = []
    for record in payload.get("runs", []):
        item = _index_record_to_history_run(record)
        if item is None:
            continue
        if exclude_dir is not None and item["run_dir"] == exclude_dir:
            continue
        if not item["summary_path"].exists():
            continue
        runs.append(item)
    return runs

def save_experience_index(index_path: Path, runs: List[dict]):
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "runs": [_history_run_to_index_record(item) for item in runs],
    }
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

def update_experience_index(index_path: Path, run_item: dict, *, max_runs: int = 500):
    runs = load_experience_index(index_path)
    run_dir = Path(run_item["run_dir"]).resolve()
    runs = [item for item in runs if Path(item["run_dir"]).resolve() != run_dir]
    runs.append(run_item)
    runs.sort(key=lambda item: str(item["run_dir"]))
    if len(runs) > int(max_runs):
        runs = runs[-int(max_runs):]
    save_experience_index(index_path, runs)

def is_high_quality_run(item: dict, *, reward_threshold: float = 4.0, max_error_sum: float = 0.12) -> bool:
    if not bool(item.get("best_success", False)):
        return False
    if float(item.get("best_reward", -1e30)) < float(reward_threshold):
        return False
    error_sum = item.get("best_error_sum", None)
    if error_sum is None:
        return True
    return float(error_sum) <= float(max_error_sum)

def update_two_level_experience_indexes(
    history_root: Path,
    run_item: dict,
    *,
    global_name: str = "experience_index.json",
    hq_name: str = "experience_index_hq.json",
    max_global_runs: int = 1000,
    max_hq_runs: int = 300,
    reward_threshold: float = 4.0,
    max_error_sum: float = 0.12,
):
    history_root = Path(history_root)
    global_index = history_root / global_name
    hq_index = history_root / hq_name

    update_experience_index(global_index, run_item, max_runs=max_global_runs)

    hq_runs = load_experience_index(hq_index)
    run_dir = Path(run_item["run_dir"]).resolve()
    hq_runs = [item for item in hq_runs if Path(item["run_dir"]).resolve() != run_dir]
    if is_high_quality_run(run_item, reward_threshold=reward_threshold, max_error_sum=max_error_sum):
        hq_runs.append(run_item)
    hq_runs.sort(key=lambda item: str(item["run_dir"]))
    if len(hq_runs) > int(max_hq_runs):
        hq_runs = hq_runs[-int(max_hq_runs):]
    save_experience_index(hq_index, hq_runs)

def discover_history_runs(
    history_root: Path,
    *,
    exclude_dir: Optional[Path] = None,
    max_runs: int = 200,
    prefer_high_quality: bool = True,
) -> List[dict]:
    history_root = Path(history_root)
    exclude_dir = None if exclude_dir is None else Path(exclude_dir).resolve()
    if not history_root.exists():
        return []

    index_candidates = []
    if prefer_high_quality:
        index_candidates.append(history_root / "experience_index_hq.json")
    index_candidates.append(history_root / "experience_index.json")

    for index_path in index_candidates:
        indexed_runs = load_experience_index(index_path, exclude_dir=exclude_dir)
        if len(indexed_runs) > 0:
            indexed_runs.sort(key=lambda item: str(item["run_dir"]))
            return indexed_runs[-int(max_runs):]

    runs = []
    for summary_path in history_root.rglob("run_summary.json"):
        run_dir = summary_path.parent.resolve()
        if exclude_dir is not None and run_dir == exclude_dir:
            continue
        payload = _safe_load_json(summary_path)
        if not isinstance(payload, dict):
            continue
        target_dict = payload.get("target")
        if not isinstance(target_dict, dict):
            continue
        try:
            target = TargetProperties(
                E_target=float(target_dict["E"]),
                sigma_y_target=float(target_dict["sigma_y"]),
                Kt_target=float(target_dict["Kt"]),
            )
        except Exception:
            continue

        runs.append(
            {
                "run_dir": run_dir,
                "summary_path": summary_path.resolve(),
                "target": target,
                "best_reward": float(payload.get("best_reward", -1e30)),
                "best_success": bool(payload.get("best_success", False)),
                "best_error_sum": (
                    None
                    if payload.get("best_E_error") is None or payload.get("best_sigma_y_error") is None or payload.get("best_Kt_error") is None
                    else float(payload.get("best_E_error", 0.0)) + float(payload.get("best_sigma_y_error", 0.0)) + float(payload.get("best_Kt_error", 0.0))
                ),
                "best_params_path": run_dir / "best_params.npy",
                "replay_path": run_dir / "diff_replay.npz",
                "checkpoint_path": run_dir / "checkpoint_final.pt",
            }
        )

    runs.sort(key=lambda item: str(item["run_dir"]))
    return runs[-int(max_runs):]

def rank_history_runs(
    runs: List[dict],
    target: "TargetProperties",
    *,
    prefer_success: bool = True,
) -> List[dict]:
    ranked = []
    for item in runs:
        distance = target_distance(item["target"], target)
        success_bonus = -0.25 if (prefer_success and item.get("best_success", False)) else 0.0
        reward_bonus = -0.02 * max(float(item.get("best_reward", -1e30)), -10.0)
        ranked.append({**item, "distance": float(distance), "rank_score": float(distance + success_bonus + reward_bonus)})
    ranked.sort(key=lambda item: (item["rank_score"], item["distance"], -item["best_reward"]))
    return ranked

def merge_replay_from_history(
    trainer,
    history_runs: List[dict],
    *,
    max_histories: int = 5,
    verbose: bool = True,
) -> int:
    if not hasattr(trainer, "diff_replay"):
        return 0

    merged = 0
    for item in history_runs[: max(0, int(max_histories))]:
        replay_path = Path(item["replay_path"])
        if not replay_path.exists():
            continue
        before = len(trainer.diff_replay)
        try:
            trainer.diff_replay.load_npz(str(replay_path), merge=True)
            delta = max(0, len(trainer.diff_replay) - before)
            merged += delta
            if verbose:
                print(f"[HistoryReplay] merged {delta} samples from {replay_path.parent}")
        except Exception as exc:
            if verbose:
                print(f"[HistoryReplay] failed to merge {replay_path}: {exc}")
    return merged

def auto_warmstart_from_history(
    trainer,
    target: "TargetProperties",
    *,
    enable: bool,
    history_runs: Optional[List[dict]] = None,
    topn: int = 5,
    verify_repeats: int = 2,
    verbose: bool = True,
) -> bool:
    env = trainer.env
    if not enable:
        if verbose:
            print("[WarmStart] skipped: no nearby historical target found, start from scratch.")
        trainer.best_reward = -float("inf")
        trainer.best_params = None
        return False

    if verbose:
        print("[WarmStart] nearby target found, searching reusable historical seeds...")

    chosen_params = None
    chosen_err_sum = 1e30
    chosen_reward = -float("inf")

    def _maybe_take(params, source: str):
        nonlocal chosen_params, chosen_err_sum, chosen_reward
        E, Sy, Kt = env.evaluate_params(params, metadata={"phase": "warmstart", "source": source})
        if not trainer._is_success(E, Sy, Kt):
            return
        err = target_error_vector(TargetProperties(E, Sy, Kt, tolerance=dict(target.tolerance)), target)
        err_sum = float(err["E"] + err["sigma_y"] + err["Kt"])
        rew = float(env.compute_reward(E, Sy, Kt))
        if verbose:
            print(
                f"[WarmStart] candidate from {source} | "
                f"err_sum={err_sum * 100:.2f}% | E={E:.1f} Sy={Sy:.1f} Kt={Kt:.3f} | reward={rew:.3f}"
            )
        if (err_sum < chosen_err_sum) or (err_sum <= chosen_err_sum + 1e-12 and rew > chosen_reward):
            chosen_params = np.asarray(params, dtype=np.float64).copy()
            chosen_err_sum = err_sum
            chosen_reward = rew

    if getattr(trainer, "best_params", None) is not None:
        if verbose:
            print("[WarmStart] checking current work_dir best_params...")
        _maybe_take(trainer.best_params.copy(), "checkpoint_best")

    if hasattr(trainer, "diff_replay") and len(trainer.diff_replay) > 0:
        buf = list(trainer.diff_replay.buffer)
        if len(buf) > 0:
            cond_cur = np.asarray(target.normalize(), dtype=np.float64).reshape(-1)

            def _cond_dist(item):
                cond = np.asarray(item["cond"], dtype=np.float64).reshape(-1)
                return float(np.linalg.norm(cond - cond_cur))

            buf.sort(key=lambda item: (_cond_dist(item), -float(item.get("reward", -1e30))))
            for idx, replay_item in enumerate(buf[: max(1, int(topn))]):
                z = np.asarray(replay_item["z"], dtype=np.float64)
                _maybe_take(trainer._denormalize_params(z), f"local_replay_top{idx + 1}")

    for idx, item in enumerate((history_runs or [])[: max(0, int(topn))]):
        best_params_path = Path(item["best_params_path"])
        if not best_params_path.exists():
            continue
        try:
            params = np.load(best_params_path, allow_pickle=False)
        except Exception as exc:
            if verbose:
                print(f"[WarmStart] failed to load {best_params_path}: {exc}")
            continue
        _maybe_take(params, f"history_best_{idx + 1}")

    if chosen_params is None:
        if verbose:
            print("[WarmStart] no reusable successful historical structure found.")
        trainer.best_reward = -float("inf")
        trainer.best_params = None
        return False

    trainer.best_params = chosen_params
    trainer.best_reward = chosen_reward
    trainer._no_improve = 0

    if verbose:
        print(f"[WarmStart] selected reusable seed with err_sum={chosen_err_sum * 100:.2f}%.")
    trainer.verify_best(repeats=max(1, int(verify_repeats)), verbose=verbose)
    return True
