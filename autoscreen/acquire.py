"""Batch acquisition strategies for multi-objective active learning.

Every strategy scores the *unlabeled* pool and returns the indices (into the pool
array) of the next batch. All objective values follow the maximize convention.

Strategies
----------
random   : uniform random baseline
greedy   : scalarize predicted means with fixed weights, take top-k
weighted : greedy + per-batch random scalarization weights (ParEGO-style)
ucb      : scalarized upper-confidence bound (mean + beta*std), explores
pareto   : greedy Pareto-HVI-style; iteratively add the point that most
           improves hypervolume of the predicted batch front (mean + beta*std)
"""
from __future__ import annotations

import numpy as np

from .metrics import hypervolume


def _normalize(Y: np.ndarray, range_: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi = range_
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    return (Y - lo) / span


def select_random(rng: np.random.Generator, pool_idx: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(pool_idx))
    return rng.choice(pool_idx, size=k, replace=False)


def select_greedy(
    means: np.ndarray, pool_idx: np.ndarray, k: int, weights: np.ndarray, norm_range
) -> np.ndarray:
    Yn = _normalize(means, norm_range)
    scores = (Yn * weights).sum(axis=1)
    order = np.argsort(-scores)[:k]
    return pool_idx[order]


def select_weighted(
    rng: np.random.Generator, means: np.ndarray, pool_idx: np.ndarray, k: int, norm_range
) -> np.ndarray:
    """ParEGO-style: draw random weights each round, then greedy on them."""
    w = rng.random(means.shape[1])
    w = w / w.sum()
    return select_greedy(means, pool_idx, k, w, norm_range)


def select_ucb(
    means: np.ndarray,
    stds: np.ndarray,
    pool_idx: np.ndarray,
    k: int,
    weights: np.ndarray,
    norm_range,
    beta: float = 1.0,
) -> np.ndarray:
    Yn = _normalize(means + beta * stds, norm_range)
    scores = (Yn * weights).sum(axis=1)
    order = np.argsort(-scores)[:k]
    return pool_idx[order]


def select_pareto_hvi(
    means: np.ndarray,
    stds: np.ndarray,
    pool_idx: np.ndarray,
    k: int,
    ref_point: np.ndarray,
    labeled_Y: np.ndarray,
    beta: float = 0.5,
    candidate_cap: int = 2000,
) -> np.ndarray:
    """Greedy batch hypervolume improvement over an optimistic (mean+beta*std) score.

    Starts from the current labeled front and greedily adds the pool point whose
    optimistic objective vector yields the largest hypervolume gain. To stay cheap
    on CPU, restrict candidates to the top `candidate_cap` by scalarized score.
    """
    opt = means + beta * stds

    # pre-filter candidates by a cheap scalarization to bound the greedy cost
    if len(pool_idx) > candidate_cap:
        rough = opt.sum(axis=1)
        keep = np.argsort(-rough)[:candidate_cap]
    else:
        keep = np.arange(len(pool_idx))

    cand_local = keep.tolist()
    front = list(labeled_Y) if labeled_Y.shape[0] else []
    chosen_local: list[int] = []

    base_hv = hypervolume(np.asarray(front), ref_point) if front else 0.0

    for _ in range(min(k, len(cand_local))):
        best_gain = -np.inf
        best_j = None
        for j in cand_local:
            trial = front + [opt[j]]
            hv = hypervolume(np.asarray(trial), ref_point)
            gain = hv - base_hv
            if gain > best_gain:
                best_gain = gain
                best_j = j
        chosen_local.append(best_j)
        front.append(opt[best_j])
        base_hv = hypervolume(np.asarray(front), ref_point)
        cand_local.remove(best_j)

    return pool_idx[np.asarray(chosen_local, dtype=int)]
