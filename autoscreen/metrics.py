"""Multi-objective evaluation metrics (maximize convention).

All objective vectors here follow the *maximize* convention (see data.py).
Hypervolume is computed with pymoo's HV indicator, which minimizes, so we negate.
"""
from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (maximize all objectives)."""
    n = Y.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # a point j dominates i if j >= i on all and > on some
        dominates = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_efficient[i] = False
    return is_efficient


def hypervolume(Y: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume dominated by point set Y w.r.t. ref_point (all maximize).

    pymoo HV assumes minimization, so negate both Y and the reference point.
    ref_point should be a lower bound (worse than any point) in maximize space.
    """
    if Y.shape[0] == 0:
        return 0.0
    ind = HV(ref_point=-np.asarray(ref_point, dtype=np.float64))
    return float(ind(-np.asarray(Y, dtype=np.float64)))


def make_ref_point(Y_all: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """A fixed reference point slightly worse than the global minimum per objective."""
    lo = Y_all.min(axis=0)
    hi = Y_all.max(axis=0)
    return lo - margin * (hi - lo)
