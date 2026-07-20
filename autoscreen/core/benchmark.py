"""Offline benchmark metrics — the only Campaign-adjacent consumer of hidden labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from autoscreen.core.metrics import hypervolume, make_ref_point, pareto_mask
from autoscreen.core.oracle import ArrayLabelOracle


@dataclass
class BenchmarkReport:
    n_labeled: int
    hv_frac: float
    pareto_recall: float
    mean_activity: float
    global_hv: float


class BenchmarkEvaluator:
    """Evaluates labeled pool indices against a private label oracle.

    Must not be used inside acquisition selection. Reference points derived here
    stay inside this class.
    """

    def __init__(self, oracle: ArrayLabelOracle, *, use_expensive_only: bool = True):
        self.oracle = oracle
        Y = oracle.expensive_array() if use_expensive_only else oracle.as_array()
        self.Y = np.asarray(Y, dtype=np.float64)
        self.ref_point = make_ref_point(self.Y)
        gmask = pareto_mask(self.Y)
        self.global_front_idx = set(np.where(gmask)[0].tolist())
        self.global_hv = hypervolume(self.Y[gmask], self.ref_point) if gmask.any() else 1.0

    def evaluate(self, labeled_indices: Sequence[int]) -> BenchmarkReport:
        idxs = [int(i) for i in labeled_indices]
        if not idxs:
            return BenchmarkReport(0, 0.0, 0.0, 0.0, self.global_hv)
        Y = self.Y[idxs]
        hv = hypervolume(Y, self.ref_point)
        found = len(self.global_front_idx & set(idxs))
        return BenchmarkReport(
            n_labeled=len(idxs),
            hv_frac=hv / self.global_hv if self.global_hv > 0 else 0.0,
            pareto_recall=found / max(1, len(self.global_front_idx)),
            mean_activity=float(Y[:, 0].mean()) if Y.shape[1] else 0.0,
            global_hv=self.global_hv,
        )
