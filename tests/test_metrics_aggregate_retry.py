"""Enrichment metrics, replicate aggregation, and retry policy."""
from __future__ import annotations

import numpy as np

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.candidate_state import CandidatePhase, CandidateStateStore
from autoscreen.core.enrichment import enrichment_factor, hit_recall, top_fraction_indices
from autoscreen.core.observations import AggregateConfig, ObservationStore
from autoscreen.core.oracle import ArrayLabelOracle
from autoscreen.core.types import ItemKind, Observation, WellState


def test_top1_recall_and_ef():
    scores = np.arange(1000, dtype=float)  # higher = better at the end
    # Make high indices best
    hits = set(top_fraction_indices(scores, 0.01).tolist())
    assert len(hits) == 10
    labeled = set(list(hits)[:5]) | {0, 1, 2}
    assert hit_recall(labeled, hits) == 0.5
    ef = enrichment_factor(labeled, hits, n_library=1000)
    assert ef > 1.0


def test_benchmark_single_obj_primary_metrics():
    rng = np.random.default_rng(0)
    Y = rng.random((500, 1))
    oracle = ArrayLabelOracle(Y=Y, column_names=("activity",), expensive_names=("activity",))
    ev = BenchmarkEvaluator(oracle)
    # Label the true top 1%
    top = list(ev.top1)
    rep = ev.evaluate(top)
    assert rep.top1_recall == 1.0
    assert rep.ef_top1 > 10
    assert np.isnan(rep.hv_frac)  # single-obj → HV not primary
    assert rep.bedroc > 0.5


def test_replicate_aggregation_mean_and_std_qc():
    store = ObservationStore(AggregateConfig(method="mean", max_std=0.2, min_replicates_for_qc=2))
    store.add(
        Observation(
            smiles="C",
            pool_idx=0,
            values=[1.0],
            state=WellState.COMPLETED,
            qc_passed=True,
            kind=ItemKind.EXPERIMENTAL,
            item_id="a",
        )
    )
    assert len(store) == 1
    assert store._by_pool[0].values == [1.0]

    store.add(
        Observation(
            smiles="C",
            pool_idx=0,
            values=[1.05],
            state=WellState.COMPLETED,
            qc_passed=True,
            kind=ItemKind.REPLICATE,
            item_id="b",
        )
    )
    assert len(store) == 1
    assert abs(store._by_pool[0].values[0] - 1.025) < 1e-9
    assert store._by_pool[0].raw["n_measurements"] == 2

    # Large disagreement → aggregate QC reject removes training label
    store2 = ObservationStore(AggregateConfig(method="mean", max_std=0.1, min_replicates_for_qc=2))
    store2.add(
        Observation(
            smiles="CC",
            pool_idx=1,
            values=[0.0],
            state=WellState.COMPLETED,
            qc_passed=True,
            kind=ItemKind.EXPERIMENTAL,
            item_id="c",
        )
    )
    store2.add(
        Observation(
            smiles="CC",
            pool_idx=1,
            values=[5.0],
            state=WellState.COMPLETED,
            qc_passed=True,
            kind=ItemKind.REPLICATE,
            item_id="d",
        )
    )
    assert 1 not in store2.labeled_indices
    assert store2.is_aggregate_qc_rejected(1)


def test_fail_and_qc_retry_then_permanent():
    cand = CandidateStateStore(5, max_fail_retries=1, max_qc_retries=1)
    cand.mark_selected([0, 1], "j")
    cand.mark_submitted([0, 1], "j")

    cand.apply_observation(
        Observation(
            smiles="C",
            pool_idx=0,
            values=None,
            state=WellState.FAILED,
            kind=ItemKind.EXPERIMENTAL,
            item_id="f1",
        )
    )
    assert cand.phase(0) is CandidatePhase.RETRYABLE
    assert 0 in cand.available_indices()

    cand.apply_observation(
        Observation(
            smiles="C",
            pool_idx=0,
            values=None,
            state=WellState.FAILED,
            kind=ItemKind.EXPERIMENTAL,
            item_id="f2",
        )
    )
    assert cand.phase(0) is CandidatePhase.FAILED
    assert 0 not in cand.available_indices()

    cand.apply_observation(
        Observation(
            smiles="CC",
            pool_idx=1,
            values=[1.0],
            state=WellState.QC_REJECTED,
            qc_passed=False,
            kind=ItemKind.EXPERIMENTAL,
            item_id="q1",
        )
    )
    assert cand.phase(1) is CandidatePhase.RETRYABLE
    cand.apply_observation(
        Observation(
            smiles="CC",
            pool_idx=1,
            values=[1.0],
            state=WellState.QC_REJECTED,
            qc_passed=False,
            kind=ItemKind.EXPERIMENTAL,
            item_id="q2",
        )
    )
    assert cand.phase(1) is CandidatePhase.QC_REJECTED
