"""Batch feasibility and diversity constraints for plate-aware selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlateConfig:
    n_experimental: int = 80
    n_positive: int = 4
    n_negative: int = 4
    n_blank: int = 4
    n_replicate: int = 4
    diversity_lambda: float = 0.4
    sa_feasibility_quantile: float = 0.1

    @property
    def plate_size(self) -> int:
        return (
            self.n_experimental
            + self.n_positive
            + self.n_negative
            + self.n_blank
            + self.n_replicate
        )


def _tanimoto_to_set(fp: np.ndarray, fps_set: np.ndarray, pc_set: np.ndarray, pc_fp: float) -> np.ndarray:
    inter = (fps_set * fp).sum(axis=1)
    union = pc_set + pc_fp - inter
    union = np.where(union > 0, union, 1.0)
    return inter / union


def greedy_maxmin(
    scores: np.ndarray,
    fps: np.ndarray,
    k: int,
    diversity_lambda: float,
) -> list[int]:
    n = len(scores)
    k = min(k, n)
    popcount = fps.sum(axis=1).astype(float)
    s = scores.astype(float)
    srange = s.max() - s.min()
    s_norm = (s - s.min()) / srange if srange > 1e-12 else np.zeros_like(s)

    selected: list[int] = []
    max_sim = np.zeros(n)
    available = np.ones(n, dtype=bool)
    first = int(np.argmax(s_norm))
    selected.append(first)
    available[first] = False
    max_sim = np.maximum(max_sim, _tanimoto_to_set(fps[first], fps, popcount, popcount[first]))

    for _ in range(k - 1):
        obj = (1 - diversity_lambda) * s_norm - diversity_lambda * max_sim
        obj[~available] = -np.inf
        nxt = int(np.argmax(obj))
        if not available[nxt]:
            break
        selected.append(nxt)
        available[nxt] = False
        max_sim = np.maximum(max_sim, _tanimoto_to_set(fps[nxt], fps, popcount, popcount[nxt]))
    return selected


class ConstraintManager:
    """Filter infeasible compounds and optionally re-rank with diversity."""

    def __init__(
        self,
        plate: PlateConfig | None = None,
        stock_available: np.ndarray | None = None,
    ):
        self.plate = plate or PlateConfig()
        self.stock_available = stock_available

    def feasible_mask(
        self,
        pool_local_n: int,
        pred_sa_ease: np.ndarray | None = None,
        pool_global_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        feasible = np.ones(pool_local_n, dtype=bool)
        if pred_sa_ease is not None and self.plate.sa_feasibility_quantile > 0:
            thresh = np.quantile(pred_sa_ease, self.plate.sa_feasibility_quantile)
            feasible &= pred_sa_ease >= thresh
        if self.stock_available is not None and pool_global_idx is not None:
            feasible &= self.stock_available[pool_global_idx].astype(bool)
        if not feasible.any():
            return np.ones(pool_local_n, dtype=bool)
        return feasible

    def diversify(
        self,
        scores: np.ndarray,
        fps: np.ndarray,
        k: int,
    ) -> list[int]:
        return greedy_maxmin(scores, fps, k, self.plate.diversity_lambda)
