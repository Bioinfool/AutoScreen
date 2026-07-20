"""Multi-objective metrics (maximize convention)."""
from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_efficient[i] = False
    return is_efficient


def hypervolume(Y: np.ndarray, ref_point: np.ndarray) -> float:
    if Y.shape[0] == 0:
        return 0.0
    ind = HV(ref_point=-np.asarray(ref_point, dtype=np.float64))
    return float(ind(-np.asarray(Y, dtype=np.float64)))


def make_ref_point(Y_all: np.ndarray, margin: float = 0.05) -> np.ndarray:
    lo = Y_all.min(axis=0)
    hi = Y_all.max(axis=0)
    return lo - margin * (hi - lo)
