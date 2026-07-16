"""Plate-aware batch selection for wet-lab / robot experiments.

MolPAL's acquirers return "the top-k molecules". A robot runs 96-well plates, so
a batch is not a ranked list -- it is a *plate layout* that must also budget wells
for controls and replicates, avoid filling a plate with near-duplicate structures,
and drop compounds that are infeasible to run.

`select_plate` turns surrogate predictions + fingerprints into a `PlateLayout`:

    80 experimental wells   (score + diversity + feasibility)
     4 positive controls    (known actives, from a reference set)
     4 negative controls    (known inactives, from a reference set)
     4 blank controls       (no compound)
     4 replicate wells      (duplicates of chosen experimental wells for QC)
    -----------------------
    96 wells total

Diversity uses a greedy max-min Tanimoto selection on Morgan-style bit
fingerprints. Feasibility drops compounds whose predicted synthesizability is in
the worst tail and honours an external stock-availability mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlateConfig:
    n_experimental: int = 80
    n_positive: int = 4
    n_negative: int = 4
    n_blank: int = 4
    n_replicate: int = 4
    diversity_lambda: float = 0.4  # 0 = pure score, 1 = pure diversity
    sa_feasibility_quantile: float = 0.1  # drop worst 10% predicted synthesizability

    @property
    def plate_size(self) -> int:
        return (
            self.n_experimental
            + self.n_positive
            + self.n_negative
            + self.n_blank
            + self.n_replicate
        )


@dataclass
class Well:
    well_id: str
    kind: str  # experimental | positive | negative | blank | replicate
    pool_idx: int | None = None
    smiles: str | None = None
    replicate_of: str | None = None


@dataclass
class PlateLayout:
    round: int
    wells: list[Well] = field(default_factory=list)

    def experimental_idxs(self) -> list[int]:
        return [w.pool_idx for w in self.wells if w.kind == "experimental"]

    def count(self, kind: str) -> int:
        return sum(1 for w in self.wells if w.kind == kind)


def _tanimoto_to_set(fp: np.ndarray, fps_set: np.ndarray, pc_set: np.ndarray, pc_fp: float) -> np.ndarray:
    """Tanimoto of one fp against each row of fps_set (binary bit vectors)."""
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
    """Select k local indices maximizing score while penalizing similarity.

    At each step add argmax over remaining of
        (1 - lambda) * norm_score - lambda * max_similarity_to_selected
    """
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
    sim = _tanimoto_to_set(fps[first], fps, popcount, popcount[first])
    max_sim = np.maximum(max_sim, sim)

    for _ in range(k - 1):
        obj = (1 - diversity_lambda) * s_norm - diversity_lambda * max_sim
        obj[~available] = -np.inf
        nxt = int(np.argmax(obj))
        if not available[nxt]:
            break
        selected.append(nxt)
        available[nxt] = False
        sim = _tanimoto_to_set(fps[nxt], fps, popcount, popcount[nxt])
        max_sim = np.maximum(max_sim, sim)

    return selected


def select_plate(
    round: int,
    pool_idx: np.ndarray,
    scores: np.ndarray,
    pred_sa_ease: np.ndarray,
    fps_pool: np.ndarray,
    smis_pool: list[str],
    positive_idx: list[int],
    negative_idx: list[int],
    all_smis: list[str],
    config: PlateConfig,
    stock_available: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> PlateLayout:
    """Build a plate layout from surrogate predictions over the current pool.

    Parameters mirror what a campaign has on hand: `pool_idx` are global indices
    of unlabeled compounds; `scores` is a scalarized acquisition score aligned to
    `pool_idx`; `pred_sa_ease` is the surrogate's predicted synthesizability
    (higher = easier); `positive_idx`/`negative_idx` are global control indices.
    """
    rng = rng or np.random.default_rng(0)

    # feasibility: drop worst-tail predicted synthesizability + stock-outs
    feasible = np.ones(len(pool_idx), dtype=bool)
    if config.sa_feasibility_quantile > 0 and len(pred_sa_ease) > 0:
        thresh = np.quantile(pred_sa_ease, config.sa_feasibility_quantile)
        feasible &= pred_sa_ease >= thresh
    if stock_available is not None:
        feasible &= stock_available.astype(bool)

    feas_local = np.where(feasible)[0]
    if len(feas_local) == 0:
        feas_local = np.arange(len(pool_idx))

    chosen_local = greedy_maxmin(
        scores[feas_local], fps_pool[feas_local], config.n_experimental, config.diversity_lambda
    )
    exp_local = feas_local[chosen_local]

    wells: list[Well] = []
    row_letters = "ABCDEFGH"

    def next_well_id(i: int) -> str:
        return f"{row_letters[i // 12]}{i % 12 + 1:02d}"

    w = 0
    exp_well_ids: list[str] = []
    for loc in exp_local:
        gidx = int(pool_idx[loc])
        wid = next_well_id(w)
        wells.append(Well(wid, "experimental", pool_idx=gidx, smiles=smis_pool[loc]))
        exp_well_ids.append(wid)
        w += 1

    for gidx in positive_idx[: config.n_positive]:
        wells.append(Well(next_well_id(w), "positive", pool_idx=gidx, smiles=all_smis[gidx]))
        w += 1
    for gidx in negative_idx[: config.n_negative]:
        wells.append(Well(next_well_id(w), "negative", pool_idx=gidx, smiles=all_smis[gidx]))
        w += 1
    for _ in range(config.n_blank):
        wells.append(Well(next_well_id(w), "blank"))
        w += 1

    # replicates: duplicate a random subset of experimental wells for QC
    if exp_local.size and config.n_replicate > 0:
        rep_choice = rng.choice(len(exp_local), size=min(config.n_replicate, len(exp_local)), replace=False)
        for j in rep_choice:
            loc = exp_local[j]
            gidx = int(pool_idx[loc])
            wells.append(
                Well(
                    next_well_id(w),
                    "replicate",
                    pool_idx=gidx,
                    smiles=smis_pool[loc],
                    replicate_of=exp_well_ids[j],
                )
            )
            w += 1

    return PlateLayout(round=round, wells=wells)
