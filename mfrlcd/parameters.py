from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BETA_CANDIDATES = [5, 10, 15]
MAX_N_PRIOR_BETA = max(BETA_CANDIDATES)


@dataclass
class MicrostructureParams:
    n_priorBeta: int = 5
    angles: np.ndarray = None
    seeds: np.ndarray = None
    lam_ratio: float = 1.0

    def __post_init__(self):
        if self.angles is None:
            self.angles = np.random.uniform(0, 360, self.n_priorBeta * 3)
        if self.seeds is None:
            self.seeds = np.random.randint(1, 30, self.n_priorBeta * 3)

    def to_vector(self):
        return np.concatenate([self.angles, self.seeds.astype(float), [self.lam_ratio]])

    @classmethod
    def from_vector(cls, vector, n_priorBeta=5, max_n_priorBeta=None):
        v = np.asarray(vector, dtype=np.float64)
        if max_n_priorBeta is None:
            max_n_priorBeta = n_priorBeta
        if len(v) < (max_n_priorBeta * 6 + 1):
            angles = v[:n_priorBeta * 3]
            seeds = v[n_priorBeta * 3:n_priorBeta * 6]
            lam_ratio = float(v[-1]) if len(v) > n_priorBeta * 6 else 1.0
            return cls(n_priorBeta=n_priorBeta, angles=angles, seeds=seeds, lam_ratio=lam_ratio)
        angles_full = v[:max_n_priorBeta * 3]
        seeds_full = v[max_n_priorBeta * 3:max_n_priorBeta * 6]
        lam_ratio = float(v[max_n_priorBeta * 6])
        phi1 = angles_full[0:n_priorBeta]
        Phi = angles_full[max_n_priorBeta:max_n_priorBeta + n_priorBeta]
        phi2 = angles_full[2 * max_n_priorBeta:2 * max_n_priorBeta + n_priorBeta]
        angles = np.concatenate([phi1, Phi, phi2]).astype(np.float64)
        sx = seeds_full[0:n_priorBeta]
        sy = seeds_full[max_n_priorBeta:max_n_priorBeta + n_priorBeta]
        sz = seeds_full[2 * max_n_priorBeta:2 * max_n_priorBeta + n_priorBeta]
        seeds = np.concatenate([sx, sy, sz]).astype(np.float64)
        return cls(n_priorBeta=n_priorBeta, angles=angles, seeds=seeds, lam_ratio=lam_ratio)


def decode_design_vector(vector):
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (MAX_N_PRIOR_BETA * 6 + 2,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite 92-element physical parameter vector")
    count = int(round(float(values[-1])))
    if count not in BETA_CANDIDATES or not np.isclose(values[-1], count):
        raise ValueError("Invalid beta-grain count")
    params = MicrostructureParams.from_vector(values, count, MAX_N_PRIOR_BETA)
    if np.any(params.seeds < 1) or np.any(params.seeds > 29) or not np.allclose(params.seeds, np.round(params.seeds)):
        raise ValueError("Seed coordinates must be integers in [1, 29]")
    if not 0.1 <= params.lam_ratio <= 4:
        raise ValueError("Lamellar ratio is outside the supported range")
    return params
