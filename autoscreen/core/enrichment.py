"""Screening enrichment metrics (maximize convention for activity scores)."""
from __future__ import annotations

import numpy as np


def top_fraction_indices(scores: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the top ``frac`` fraction (maximize)."""
    n = len(scores)
    k = max(1, int(np.ceil(float(frac) * n)))
    return np.argsort(-np.asarray(scores, dtype=float))[:k]


def hit_recall(labeled: set[int], hit_set: set[int]) -> float:
    if not hit_set:
        return 0.0
    return len(labeled & hit_set) / len(hit_set)


def enrichment_factor(labeled: set[int], hit_set: set[int], n_library: int) -> float:
    """EF = (hits_found / n_labeled) / (n_hits / n_library)."""
    if not labeled or not hit_set or n_library <= 0:
        return 0.0
    hits_found = len(labeled & hit_set)
    base = len(hit_set) / n_library
    if base <= 0:
        return 0.0
    return (hits_found / len(labeled)) / base


def bedroc_from_ranks(active_ranks_0based: list[int], n: int, alpha: float = 20.0) -> float:
    """BEDROC given 0-based ranks of recovered actives in a list of length n."""
    if n <= 0 or not active_ranks_0based or alpha <= 0:
        return 0.0
    n_a = len(active_ranks_0based)
    ranks = np.asarray(active_ranks_0based, dtype=float) + 1.0  # 1-based
    s = float(np.exp(-alpha * ranks / n).sum())
    s_max = float(np.exp(-alpha * np.arange(1, n_a + 1) / n).sum())
    if s_max <= 0:
        return 0.0
    return float(np.clip(s / s_max, 0.0, 1.0))


def bedroc_labeled_vs_truth(
    scores_all: np.ndarray,
    labeled: set[int],
    hit_frac: float = 0.01,
    alpha: float = 20.0,
) -> float:
    """BEDROC for recovering the top-``hit_frac`` true actives among labeled compounds.

    Library is ranked by true activity. Actives = top hit_frac. BEDROC uses ranks of
    those actives that appear in ``labeled``.
    """
    scores_all = np.asarray(scores_all, dtype=float)
    n = len(scores_all)
    hits = set(top_fraction_indices(scores_all, hit_frac).tolist())
    order = np.argsort(-scores_all)
    rank = {int(idx): r for r, idx in enumerate(order)}
    found_ranks = [rank[i] for i in labeled if i in hits]
    if not found_ranks:
        return 0.0
    # Pad missing actives as if not found (omit from sum → lower score); normalize by all actives
    n_a = len(hits)
    s = float(np.exp(-alpha * (np.asarray(found_ranks, dtype=float) + 1.0) / n).sum())
    s_max = float(np.exp(-alpha * np.arange(1, n_a + 1) / n).sum())
    if s_max <= 0:
        return 0.0
    return float(np.clip(s / s_max, 0.0, 1.0))


def budget_to_hit_recall(
    labeled_order: list[int],
    hit_set: set[int],
    target_recall: float = 0.5,
) -> int | None:
    """Smallest prefix length of labeled_order achieving target_recall of hit_set."""
    if not hit_set or target_recall <= 0:
        return 0
    need = target_recall * len(hit_set)
    found = 0
    for i, idx in enumerate(labeled_order, start=1):
        if idx in hit_set:
            found += 1
            if found >= need:
                return i
    return None
