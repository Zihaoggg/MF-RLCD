import numpy as np


def yield_strength_cal(target):
    stresses = np.asarray(target, dtype=np.float64)
    if stresses.ndim != 2 or stresses.shape[1] != 6:
        raise ValueError("Expected one row of six stress values per sample")
    strain = np.linspace(0.0025, 0.015, 6)
    dense_strain = np.linspace(0.0025, 0.015, 60)
    moduli = []
    strengths = []
    for stress in stresses:
        modulus = stress[0] / strain[0]
        interpolated = np.interp(dense_strain, strain, stress)
        offset = modulus * dense_strain - modulus * 0.001
        crossings = np.flatnonzero(np.diff(np.sign(interpolated - offset)))
        if not crossings.size:
            raise ValueError("Stress curve does not cross the configured offset line")
        moduli.append(modulus)
        strengths.append(interpolated[crossings[0]])
    return moduli, strengths
