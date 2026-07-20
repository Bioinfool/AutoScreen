"""Acquisition strategies for multi-objective batch selection."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from autoscreen.core.metrics import hypervolume
from autoscreen.core.types import BatchSelection


def _normalize(Y: np.ndarray, range_: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi = range_
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    return (Y - lo) / span


def max_tanimoto_to_refs(fps: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    """For each row in ``fps``, max Tanimoto similarity to any row in ``ref_fps``.

    Uses elementwise products (no large ``@``) for Windows numerical stability.
    """
    if ref_fps is None or len(ref_fps) == 0 or len(fps) == 0:
        return np.zeros(len(fps), dtype=float)
    fps = np.asarray(fps, dtype=float)
    ref = np.asarray(ref_fps, dtype=float)
    pc = fps.sum(axis=1)
    max_sim = np.zeros(fps.shape[0], dtype=float)
    for j in range(ref.shape[0]):
        rj = ref[j]
        inter = (fps * rj).sum(axis=1)
        union = pc + float(rj.sum()) - inter
        union = np.where(union > 0, union, 1.0)
        max_sim = np.maximum(max_sim, inter / union)
    return max_sim


def apply_pending_penalty(
    scores: np.ndarray,
    cand_fps: np.ndarray | None,
    pending_fps: np.ndarray | None,
    lam: float = 0.5,
) -> np.ndarray:
    """Down-rank candidates similar to in-flight compounds: score - lam * max_sim."""
    if cand_fps is None or pending_fps is None or lam <= 0 or len(pending_fps) == 0:
        return scores
    return scores - float(lam) * max_tanimoto_to_refs(cand_fps, pending_fps)


class AcquisitionStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def select(
        self,
        pool_idx: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        k: int,
        *,
        labeled_Y: np.ndarray | None = None,
        ref_point: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        cand_fps: np.ndarray | None = None,
        pending_fps: np.ndarray | None = None,
        pending_penalty: float = 0.5,
    ) -> BatchSelection:
        ...


class RandomAcquisition(AcquisitionStrategy):
    name = "random"

    def select(self, pool_idx, means, stds, k, **kwargs) -> BatchSelection:
        rng = kwargs.get("rng") or np.random.default_rng(0)
        k = min(k, len(pool_idx))
        chosen = rng.choice(pool_idx, size=k, replace=False)
        return BatchSelection(pool_indices=chosen.tolist(), strategy=self.name)


class GreedyAcquisition(AcquisitionStrategy):
    name = "greedy"

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = weights

    def select(self, pool_idx, means, stds, k, **kwargs) -> BatchSelection:
        labeled_Y = kwargs.get("labeled_Y")
        if labeled_Y is None or labeled_Y.shape[0] == 0:
            norm_range = (means.min(0), means.max(0))
        else:
            norm_range = (labeled_Y.min(0), labeled_Y.max(0))
        w = self.weights if self.weights is not None else np.ones(means.shape[1]) / means.shape[1]
        Yn = _normalize(means, norm_range)
        scores = apply_pending_penalty(
            (Yn * w).sum(axis=1),
            kwargs.get("cand_fps"),
            kwargs.get("pending_fps"),
            float(kwargs.get("pending_penalty", 0.5)),
        )
        order = np.argsort(-scores)[:k]
        return BatchSelection(
            pool_indices=pool_idx[order].tolist(),
            scores=scores[order].tolist(),
            strategy=self.name,
        )


class WeightedAcquisition(AcquisitionStrategy):
    name = "weighted"

    def select(self, pool_idx, means, stds, k, **kwargs) -> BatchSelection:
        rng = kwargs.get("rng") or np.random.default_rng(0)
        w = rng.random(means.shape[1])
        w = w / w.sum()
        sel = GreedyAcquisition(weights=w).select(pool_idx, means, stds, k, **kwargs)
        sel.strategy = self.name
        sel.meta = {"weights": w.tolist()}
        return sel


class UCBAcquisition(AcquisitionStrategy):
    name = "ucb"

    def __init__(self, beta: float = 1.0, weights: np.ndarray | None = None):
        self.beta = beta
        self.weights = weights

    def select(self, pool_idx, means, stds, k, **kwargs) -> BatchSelection:
        labeled_Y = kwargs.get("labeled_Y")
        if labeled_Y is None or labeled_Y.shape[0] == 0:
            norm_range = (means.min(0), means.max(0))
        else:
            norm_range = (labeled_Y.min(0), labeled_Y.max(0))
        w = self.weights if self.weights is not None else np.ones(means.shape[1]) / means.shape[1]
        Yn = _normalize(means + self.beta * stds, norm_range)
        scores = apply_pending_penalty(
            (Yn * w).sum(axis=1),
            kwargs.get("cand_fps"),
            kwargs.get("pending_fps"),
            float(kwargs.get("pending_penalty", 0.5)),
        )
        order = np.argsort(-scores)[:k]
        return BatchSelection(
            pool_indices=pool_idx[order].tolist(),
            scores=scores[order].tolist(),
            strategy=self.name,
            meta={"beta": self.beta},
        )


class ParetoHVIAcquisition(AcquisitionStrategy):
    name = "pareto"

    def __init__(self, beta: float = 0.5, candidate_cap: int = 400):
        self.beta = beta
        self.candidate_cap = candidate_cap

    def select(self, pool_idx, means, stds, k, **kwargs) -> BatchSelection:
        ref_point = kwargs.get("ref_point")
        labeled_Y = kwargs.get("labeled_Y")
        if ref_point is None:
            raise ValueError("ParetoHVIAcquisition requires ref_point")
        opt = means + self.beta * stds
        scores = apply_pending_penalty(
            opt.sum(axis=1),
            kwargs.get("cand_fps"),
            kwargs.get("pending_fps"),
            float(kwargs.get("pending_penalty", 0.5)),
        )
        if len(pool_idx) > self.candidate_cap:
            keep = np.argsort(-scores)[: self.candidate_cap]
        else:
            keep = np.arange(len(pool_idx))
        cand = keep.tolist()
        front = list(labeled_Y) if labeled_Y is not None and labeled_Y.shape[0] else []
        chosen: list[int] = []
        base_hv = hypervolume(np.asarray(front), ref_point) if front else 0.0
        for _ in range(min(k, len(cand))):
            best_gain, best_j = -np.inf, None
            for j in cand:
                hv = hypervolume(np.asarray(front + [opt[j]]), ref_point)
                gain = hv - base_hv
                if gain > best_gain:
                    best_gain, best_j = gain, j
            chosen.append(best_j)
            front.append(opt[best_j])
            base_hv = hypervolume(np.asarray(front), ref_point)
            cand.remove(best_j)
        return BatchSelection(
            pool_indices=pool_idx[np.asarray(chosen, dtype=int)].tolist(),
            strategy=self.name,
        )


def build_acquisition(name: str, beta: float = 1.0) -> AcquisitionStrategy:
    key = name.lower()
    if key == "random":
        return RandomAcquisition()
    if key == "greedy":
        return GreedyAcquisition()
    if key == "weighted":
        return WeightedAcquisition()
    if key == "ucb":
        return UCBAcquisition(beta=beta)
    if key == "pareto":
        return ParetoHVIAcquisition(beta=beta)
    raise ValueError(f"Unknown acquisition: {name}")
