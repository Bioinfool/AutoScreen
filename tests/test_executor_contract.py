"""Executor contract tests + vina skip path."""
import shutil
from pathlib import Path

import pytest

from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.core.types import ItemKind, Job, JobItem
from autoscreen.executors.replay import ReplayExecutor
from autoscreen.executors.sim_dock import SimDockConfig, SimulatedDockExecutor
from autoscreen.executors.vina import VinaConfig, VinaExecutor, affinity_to_activity


def _lib_and_oracle():
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    return lib, oracle


def test_replay_executor_contract():
    lib, oracle = _lib_and_oracle()
    ex = ReplayExecutor(oracle, seed=1, fail_rate=0.0, qc_reject_rate=0.0)
    items = [
        JobItem(item_id="a", smiles=lib.smis[0], pool_idx=0, kind=ItemKind.EXPERIMENTAL),
        JobItem(item_id="b", smiles=lib.smis[1], pool_idx=1, kind=ItemKind.EXPERIMENTAL),
    ]
    job = Job(
        job_id="j1",
        campaign_id="c",
        round=1,
        items=items,
        executor_kind="replay",
        idempotency_key="k1",
    )
    jid = ex.submit(job)
    assert ex.submit(job) == jid
    status = ex.wait(jid, max_polls=5)
    assert status.done
    assert len(status.observations) == 2
    assert all(o.usable for o in status.observations)
    assert all(len(o.values) == 1 for o in status.observations)


def test_vina_missing_receptor_raises():
    ex = VinaExecutor(VinaConfig(receptor=None))
    items = [JobItem(item_id="a", smiles="CCO", pool_idx=0)]
    job = Job(job_id="v1", campaign_id="c", round=0, items=items, executor_kind="vina")
    with pytest.raises(RuntimeError, match="receptor"):
        ex.submit(job)


def test_vina_affinity_to_activity_maximize():
    assert affinity_to_activity(-8.5) == 8.5
    assert affinity_to_activity(-2.0) == 2.0


def test_vina_observation_uses_maximize_activity(tmp_path: Path):
    def dock(*, smiles, work, config):
        work.mkdir(parents=True, exist_ok=True)
        return -8.5, "ok"

    ex = VinaExecutor(
        VinaConfig(receptor="/fake.pdbqt", work_dir=str(tmp_path / "vina"), max_workers=1),
        dock_fn=dock,
    )
    try:
        items = [JobItem(item_id="a", smiles="CCO", pool_idx=0, kind=ItemKind.EXPERIMENTAL)]
        job = Job(job_id="v1", campaign_id="c", round=0, items=items, executor_kind="vina")
        jid = ex.submit(job)
        status = ex.wait(jid, max_polls=50)
        assert status.done
        assert len(status.observations) == 1
        obs = status.observations[0]
        assert obs.usable
        assert obs.values == [8.5]
        assert obs.raw.get("vina_affinity") == -8.5
        assert obs.raw.get("activity") == 8.5
    finally:
        ex.close()


@pytest.mark.skipif(shutil.which("vina") is None, reason="vina binary not installed")
def test_vina_binary_detected():
    ex = VinaExecutor(VinaConfig(receptor="/tmp/fake.pdbqt", vina_bin="vina"))
    assert ex.config.vina_bin == "vina"


def test_sim_dock_executor_partial_and_parallel():
    lib, oracle = _lib_and_oracle()
    ex = SimulatedDockExecutor(
        oracle, SimDockConfig(latency_s=0.02, max_workers=2, poll_hint_s=0.005)
    )
    try:
        items = [
            JobItem(item_id="a", smiles=lib.smis[0], pool_idx=0, kind=ItemKind.EXPERIMENTAL),
            JobItem(item_id="b", smiles=lib.smis[1], pool_idx=1, kind=ItemKind.EXPERIMENTAL),
        ]
        job = Job(
            job_id="s1",
            campaign_id="c",
            round=1,
            items=items,
            executor_kind="sim_dock",
            idempotency_key="sk1",
        )
        jid = ex.submit(job)
        assert ex.submit(job) == jid
        status = ex.poll(jid)
        # may be partial immediately after submit
        assert status.n_pending + len(status.observations) == 2
        status = ex.wait(jid, max_polls=200)
        assert status.done
        assert len(status.observations) == 2
        assert all(o.usable for o in status.observations)
    finally:
        ex.close()
