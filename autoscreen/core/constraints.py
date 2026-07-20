"""Batch feasibility and diversity constraints (static properties + stock)."""
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
    rows: int = 8
    cols: int = 12

    @property
    def plate_size(self) -> int:
        return (
            self.n_experimental
            + self.n_positive
            + self.n_negative
            + self.n_blank
            + self.n_replicate
        )

    @property
    def capacity(self) -> int:
        return self.rows * self.cols

    def validate(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"Invalid plate geometry rows={self.rows} cols={self.cols}")
        if self.plate_size > self.capacity:
            raise ValueError(
                f"plate_size={self.plate_size} exceeds capacity "
                f"{self.rows}x{self.cols}={self.capacity}"
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
    """Filter using static properties (not surrogate predictions of QED/SA)."""

    def __init__(
        self,
        plate: PlateConfig | None = None,
        stock_available: np.ndarray | None = None,
        static_sa_ease: np.ndarray | None = None,
        empty_policy: str = "fail_closed",  # fail_closed | relax | fail_open
    ):
        self.plate = plate or PlateConfig()
        self.stock_available = stock_available
        self.static_sa_ease = static_sa_ease
        if empty_policy not in ("fail_closed", "relax", "fail_open"):
            raise ValueError(f"Unknown empty_policy={empty_policy}")
        self.empty_policy = empty_policy
        self.last_relaxation: dict | None = None

    def feasible_mask(
        self,
        pool_local_n: int,
        pool_global_idx: np.ndarray | None = None,
        **_ignored,
    ) -> np.ndarray:
        from autoscreen.logging_utils import get_logger

        log = get_logger("constraints")
        self.last_relaxation = None
        feasible = np.ones(pool_local_n, dtype=bool)
        sa_thresh = None
        if (
            self.static_sa_ease is not None
            and pool_global_idx is not None
            and self.plate.sa_feasibility_quantile > 0
        ):
            vals = self.static_sa_ease[pool_global_idx]
            sa_thresh = float(np.quantile(self.static_sa_ease, self.plate.sa_feasibility_quantile))
            feasible &= vals >= sa_thresh
        if self.stock_available is not None and pool_global_idx is not None:
            feasible &= self.stock_available[pool_global_idx].astype(bool)
        if feasible.any():
            return feasible

        if self.empty_policy == "fail_closed":
            raise RuntimeError(
                "No feasible candidates under current constraints "
                "(empty_policy=fail_closed). Relax SA/stock settings or set "
                "constraints.empty_policy to 'relax' / 'fail_open'."
            )
        if self.empty_policy == "fail_open":
            log.warning(
                "All candidates infeasible; fail_open disables constraints for this step"
            )
            self.last_relaxation = {"policy": "fail_open", "n": pool_local_n}
            return np.ones(pool_local_n, dtype=bool)

        # relax: progressively widen SA quantile toward 0
        if (
            self.static_sa_ease is not None
            and pool_global_idx is not None
            and self.plate.sa_feasibility_quantile > 0
        ):
            vals = self.static_sa_ease[pool_global_idx]
            for q in (0.05, 0.02, 0.0):
                thresh = float(np.quantile(self.static_sa_ease, q))
                cand = vals >= thresh
                if self.stock_available is not None:
                    cand &= self.stock_available[pool_global_idx].astype(bool)
                if cand.any():
                    log.warning(
                        "Relaxed SA quantile %.3f -> %.3f to recover %d feasible",
                        self.plate.sa_feasibility_quantile,
                        q,
                        int(cand.sum()),
                    )
                    self.last_relaxation = {
                        "policy": "relax",
                        "sa_quantile": q,
                        "n_feasible": int(cand.sum()),
                    }
                    return cand
        log.warning("Relax failed; falling back to fail_open for this step")
        self.last_relaxation = {"policy": "relax_fallback_open", "n": pool_local_n}
        return np.ones(pool_local_n, dtype=bool)
    def diversify(
        self,
        scores: np.ndarray,
        fps: np.ndarray,
        k: int,
    ) -> list[int]:
        return greedy_maxmin(scores, fps, k, self.plate.diversity_lambda)
