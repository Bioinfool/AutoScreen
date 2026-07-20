"""Offline benchmark metrics — the only Campaign-adjacent consumer of hidden labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from autoscreen.core.enrichment import (
    bedroc_labeled_vs_truth,
    enrichment_factor,
    hit_recall,
    top_fraction_indices,
)
from autoscreen.core.metrics import hypervolume, make_ref_point, pareto_mask
from autoscreen.core.oracle import ArrayLabelOracle


@dataclass
class BenchmarkReport:
    n_labeled: int
    mean_activity: float
    # Single-objective primary metrics
    top01_recall: float  # top 0.1%
    top1_recall: float  # top 1%
    ef_top1: float  # enrichment vs top-1% hit definition
    bedroc: float
    n_hits_top1: int
    # Multi-objective (only meaningful when n_obj > 1)
    hv_frac: float
    pareto_recall: float
    global_hv: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkEvaluator:
    """Evaluates labeled pool indices against a private label oracle.

    Must not be used inside acquisition selection.
    """

    def __init__(
        self,
        oracle: ArrayLabelOracle,
        *,
        use_expensive_only: bool = True,
        top_fracs: tuple[float, float] = (0.001, 0.01),
        bedroc_alpha: float = 20.0,
    ):
        self.oracle = oracle
        Y = oracle.expensive_array() if use_expensive_only else oracle.as_array()
        self.Y = np.asarray(Y, dtype=np.float64)
        self.n_obj = self.Y.shape[1]
        self.activity = self.Y[:, 0]
        self.top01 = set(top_fraction_indices(self.activity, top_fracs[0]).tolist())
        self.top1 = set(top_fraction_indices(self.activity, top_fracs[1]).tolist())
        self.bedroc_alpha = bedroc_alpha

        self.ref_point = make_ref_point(self.Y)
        gmask = pareto_mask(self.Y)
        self.global_front_idx = set(np.where(gmask)[0].tolist())
        self.global_hv = hypervolume(self.Y[gmask], self.ref_point) if gmask.any() else 1.0

    def evaluate(self, labeled_indices: Sequence[int]) -> BenchmarkReport:
        idxs = [int(i) for i in labeled_indices]
        labeled = set(idxs)
        if not idxs:
            return BenchmarkReport(
                n_labeled=0,
                mean_activity=0.0,
                top01_recall=0.0,
                top1_recall=0.0,
                ef_top1=0.0,
                bedroc=0.0,
                n_hits_top1=0,
                hv_frac=0.0,
                pareto_recall=0.0,
                global_hv=self.global_hv,
            )

        mean_act = float(self.activity[idxs].mean())
        top01_r = hit_recall(labeled, self.top01)
        top1_r = hit_recall(labeled, self.top1)
        ef = enrichment_factor(labeled, self.top1, n_library=len(self.activity))
        bed = bedroc_labeled_vs_truth(
            self.activity, labeled, hit_frac=0.01, alpha=self.bedroc_alpha
        )
        n_hits = len(labeled & self.top1)

        if self.n_obj > 1:
            Y = self.Y[idxs]
            hv = hypervolume(Y, self.ref_point)
            hv_frac = hv / self.global_hv if self.global_hv > 0 else 0.0
            pareto_r = len(self.global_front_idx & labeled) / max(1, len(self.global_front_idx))
        else:
            hv_frac = float("nan")
            pareto_r = float("nan")

        return BenchmarkReport(
            n_labeled=len(idxs),
            mean_activity=mean_act,
            top01_recall=top01_r,
            top1_recall=top1_r,
            ef_top1=ef,
            bedroc=bed,
            n_hits_top1=n_hits,
            hv_frac=hv_frac,
            pareto_recall=pareto_r,
            global_hv=self.global_hv,
        )
