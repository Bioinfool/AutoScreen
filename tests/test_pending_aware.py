"""Pending-aware acquisition penalty tests."""
import numpy as np

from autoscreen.core.acquisition import UCBAcquisition, apply_pending_penalty, max_tanimoto_to_refs


def test_pending_penalty_downranks_similar():
    # cand0 identical to pending, cand1 orthogonal
    cand = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    pending = np.array([[1.0, 1.0, 0.0, 0.0]])
    scores = np.array([10.0, 10.0])
    out = apply_pending_penalty(scores, cand, pending, lam=1.0)
    assert out[0] < out[1]
    assert max_tanimoto_to_refs(cand, pending)[0] == 1.0


def test_ucb_respects_pending_fps():
    rng = np.random.default_rng(0)
    pool = np.arange(20)
    means = rng.random((20, 1))
    stds = rng.random((20, 1)) * 0.1
    fps = rng.random((20, 16))
    # Make compound 0 fingerprint match a pending vector exactly and give it high mean
    means[0, 0] = 1.0
    fps[0] = 1.0
    pending = fps[0:1].copy()
    acq = UCBAcquisition(beta=0.0)
    sel = acq.select(
        pool,
        means,
        stds,
        5,
        labeled_Y=means[:3],
        cand_fps=fps,
        pending_fps=pending,
        pending_penalty=10.0,
        rng=rng,
    )
    # With huge pending penalty, identical-to-pending #0 should not be first
    assert sel.pool_indices[0] != 0
