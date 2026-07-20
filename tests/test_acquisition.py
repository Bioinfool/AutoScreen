"""Acquisition and constraint unit tests."""
import numpy as np

from autoscreen.core.acquisition import build_acquisition
from autoscreen.core.constraints import ConstraintManager, PlateConfig
from autoscreen.core.metrics import make_ref_point


def test_acquisitions_return_k():
    rng = np.random.default_rng(0)
    pool = np.arange(100)
    means = rng.random((100, 3))
    stds = rng.random((100, 3)) * 0.1
    labeled = means[:10]
    ref = make_ref_point(means)
    for name in ("random", "greedy", "weighted", "ucb", "pareto"):
        acq = build_acquisition(name, beta=0.5)
        sel = acq.select(pool, means, stds, 10, labeled_Y=labeled, ref_point=ref, rng=rng)
        assert len(sel.pool_indices) == 10
        assert sel.strategy == name


def test_constraint_diversify():
    rng = np.random.default_rng(0)
    scores = rng.random(50)
    fps = (rng.random((50, 32)) > 0.7).astype(float)
    cm = ConstraintManager(PlateConfig(diversity_lambda=0.5))
    idxs = cm.diversify(scores, fps, 10)
    assert len(idxs) == 10
    assert len(set(idxs)) == 10
