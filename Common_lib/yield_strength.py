import numpy as np
from scipy import interpolate


def yield_strength_cal(target):
    strain = np.linspace(0.0025, 0.015, 6)
    strain_new = np.linspace(0.0025, 0.015, 60)
    yield_strengths = []
    moduli = []

    for stress in target:
        stress = np.asarray(stress, dtype=np.float64)
        modulus = stress[0] / 0.0025
        offsetline = modulus * strain_new - modulus * 0.001
        stress_new = interpolate.interp1d(strain, stress, "linear")(strain_new)
        crossing = np.argwhere(np.diff(np.sign(stress_new - offsetline))).flatten()
        yield_strengths.append(stress_new[crossing][0])
        moduli.append(modulus)

    return moduli, yield_strengths
