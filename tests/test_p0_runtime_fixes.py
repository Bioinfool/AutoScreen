"""P0 fixes: local job_id binding, time-driven run, replicate state, schema, Vina campaign."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.candidate_state import CandidatePhase, CandidateStateStore
from autoscreen.core.jobs import JobLifecycle
from autoscreen.core.library import load_candidate_library
from autoscreen.core.objectives import parse_objective_schema
from autoscreen.core.observations import AggregateConfig, ObservationStore
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.core.types import ItemKind, Job, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor
from autoscreen.executors.replay import ReplayExecutor
from autoscreen.executors.vina import VinaConfig, VinaExecutor


def _lib_oracle():
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    return lib, oracle


class RemoteIdStub(Executor):
    """Returns a remote id different from the local job_id."""

    kind = "stub"

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self.submit_count = 0

    def submit(self, job: Job) -> str:
        self.submit_count += 1
        remote = f"remote-{self.submit_count}"
        self._jobs[remote] = job
        return remote

    def poll(self, job_id: str) -> JobStatus:
        job = self._jobs[job_id]
        obs = [
            Observation(
                smiles=it.smiles,
                pool_idx=it.pool_idx,
                values=[1.0] if it.pool_idx >= 0 else None,
                state=WellState.COMPLETED if it.pool_idx >= 0 else WellState.COMPLETED,
                qc_passed=it.pool_idx >= 0,
                kind=it.kind,
                item_id=it.item_id,
                source=self.kind,
            )
            for it in job.items
        ]
        return JobStatus(job_id=job_id, done=True, observations=obs, n_pending=0)

    def cancel(self, job_id: str) -> None:
        return None


def test_structured_objective_schema_dict_form():
    schema = parse_objective_schema(
        {
            "objectives": {
                "expensive": [{"name": "activity", "kind": "expensive", "source": "dock"}],
                "static": [{"name": "qed", "kind": "static"}],
            }
        }
    )
    assert schema.expensive_names == ("activity",)
    assert schema.static_names == ("qed",)
    assert schema.expensive[0].to_maximize(-8.0) == -8.0


def test_resume_uses_local_job_id_for_candidates(tmp_path: Path):
    lib, oracle = _lib_oracle()
    stub = RemoteIdStub()
    ckpt = tmp_path / "ckpt"
    camp = CampaignManager(
        library=lib,
        executor=stub,
        acquisition="greedy",
        campaign_id="idmix",
        seed=0,
        batch_size=5,
        init_frac=0.001,
        checkpoint_dir=ckpt,
        n_estimators=5,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=1,
        schema=lib.schema,
    )
    # Force a PREPARED record with mismatched remote, then fail release path
    rec = next(iter(camp.jobs.all()))
    local_id = rec.job.job_id
    assert rec.remote_job_id.startswith("remote-")
    # Candidate store must reference local id
    exp = [it.pool_idx for it in rec.job.items if it.pool_idx >= 0][:1]
    assert exp
    # Simulate crash: mark PREPARED again and resume with failing submit
    rec.status = JobLifecycle.PREPARED
    rec.remote_job_id = ""
    camp.cand.mark_selected(exp, local_id)
    camp.cand.mark_submitted(exp, local_id)
    camp._checkpoint()

    class FailSubmit(RemoteIdStub):
        def submit(self, job):
            raise RuntimeError("boom")

    camp2 = CampaignManager(
        library=lib,
        executor=FailSubmit(),
        acquisition="greedy",
        campaign_id="idmix",
        seed=0,
        batch_size=5,
        init_frac=0.001,
        checkpoint_dir=ckpt,
        n_estimators=5,
        resume=True,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=1,
        schema=lib.schema,
    )
    # After failed resume submit, candidates bound to local id must be releasable
    for i in exp:
        assert camp2.cand.phase(i) in (
            CandidatePhase.AVAILABLE,
            CandidatePhase.RETRYABLE,
            CandidatePhase.FAILED,
            CandidatePhase.LABELED,
        )
        # Critical: release path used local id — not stuck SUBMITTED with remote id
        assert camp2.cand.phase(i) is not CandidatePhase.SUBMITTED


def test_replicate_does_not_overwrite_labeled():
    cand = CandidateStateStore(3, max_fail_retries=0, max_qc_retries=0)
    cand.mark_labeled([0])
    cand.apply_observation(
        Observation(
            smiles="C",
            pool_idx=0,
            values=None,
            state=WellState.FAILED,
            kind=ItemKind.REPLICATE,
            item_id="rep-fail",
        )
    )
    assert cand.phase(0) is CandidatePhase.LABELED

    store = ObservationStore(AggregateConfig(max_std=None))
    store.add(
        Observation(
            smiles="C",
            pool_idx=0,
            values=[1.0],
            state=WellState.COMPLETED,
            qc_passed=True,
            kind=ItemKind.EXPERIMENTAL,
            item_id="e0",
        )
    )
    store.add(
        Observation(
            smiles="C",
            pool_idx=0,
            values=[1.2],
            state=WellState.COMPLETED,
            qc_passed=True,
            kind=ItemKind.REPLICATE,
            item_id="r0",
        )
    )
    assert 0 in store.labeled_indices
    assert store._by_pool[0].raw["n_measurements"] == 2


def test_run_sleeps_when_idle(tmp_path: Path, monkeypatch):
    lib, oracle = _lib_oracle()
    sleeps: list[float] = []

    def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    class SlowReplay(ReplayExecutor):
        def poll(self, job_id):
            st = super().poll(job_id)
            st.next_poll_after = 0.2
            return st

    camp = CampaignManager(
        library=lib,
        executor=SlowReplay(oracle, seed=0, min_latency=2, max_latency=3, stagger=True),
        acquisition="greedy",
        campaign_id="sleep",
        seed=0,
        batch_size=10,
        init_frac=0.002,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=5,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=1,
        schema=lib.schema,
        poll_interval_s=0.1,
    )
    camp.run(1, max_wall_time_s=30)
    assert any(s >= 0.1 for s in sleeps)


def test_vina_campaign_integration(tmp_path: Path):
    def dock(*, smiles, work, config):
        time.sleep(0.05)
        work.mkdir(parents=True, exist_ok=True)
        (work / "score.txt").write_text("-6.0\n", encoding="utf-8")
        return -6.0, "ok"

    lib, oracle = _lib_oracle()
    # Tiny slice via batching only — full library ok for few labels
    ex = VinaExecutor(
        VinaConfig(receptor="/fake.pdbqt", work_dir=str(tmp_path / "vina"), max_workers=2),
        dock_fn=dock,
    )
    camp = CampaignManager(
        library=lib,
        executor=ex,
        acquisition="greedy",
        campaign_id="vina_camp",
        seed=0,
        batch_size=4,
        init_frac=0.0005,  # ~5 molecules
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=8,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=2,
        schema=lib.schema,
        poll_interval_s=0.02,
        max_wall_time_s=60,
    )
    t0 = time.perf_counter()
    hist = camp.run(1)
    elapsed = time.perf_counter() - t0
    assert camp.state.init_done
    assert len(camp.store) >= 1
    assert hist
    # Should not busy-spin thousands of steps in << docking time
    assert elapsed > 0.05
    ex.close()


def test_constraint_fail_closed():
    from autoscreen.core.constraints import ConstraintManager, PlateConfig
    import numpy as np

    cm = ConstraintManager(
        PlateConfig(sa_feasibility_quantile=0.0),
        stock_available=np.zeros(3, dtype=bool),
        empty_policy="fail_closed",
    )
    with pytest.raises(RuntimeError, match="fail_closed"):
        cm.feasible_mask(3, pool_global_idx=np.array([0, 1, 2]))
