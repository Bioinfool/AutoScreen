"""P0/P1 fixes: unique IDs, durable submit, per-job stats, async vina."""
from __future__ import annotations

import json
import time
from pathlib import Path

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.jobs import JobLifecycle
from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.core.types import WellState
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


def test_concurrent_jobs_unique_item_ids_and_all_train(tmp_path: Path):
    lib, oracle = _lib_oracle()
    camp = CampaignManager(
        library=lib,
        executor=ReplayExecutor(oracle, seed=0, min_latency=1, max_latency=3, stagger=True),
        acquisition="greedy",
        campaign_id="uniq",
        seed=0,
        batch_size=20,
        init_frac=0.003,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=10,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=2,
        schema=lib.schema,
    )
    camp.run(2)

    all_item_ids = [o.item_id for o in camp.store.history if o.item_id]
    assert all_item_ids
    assert len(all_item_ids) == len(set(all_item_ids))

    job_ids = [r.job.job_id for r in camp.jobs.all()]
    assert len(job_ids) == len(set(job_ids))

    # Every usable experimental observation must be in the training store
    usable = [o for o in camp.store.history if o.contributes_measurement]
    assert usable
    assert set(camp.store.labeled_indices).issubset({o.pool_idx for o in usable})
    assert len(camp.store) == len(camp.store.labeled_indices)

    # Concurrent batches must use distinct batch_seq / job prefixes
    acq_jobs = [r for r in camp.jobs.all() if r.job.meta.get("acquisition") != "init"]
    if len(acq_jobs) >= 2:
        seqs = [r.job.meta["batch_seq"] for r in acq_jobs]
        assert len(seqs) == len(set(seqs))


def test_prepared_checkpoint_before_remote_submit(tmp_path: Path):
    """Crash window closed: PREPARED record exists before executor.submit returns."""
    lib, oracle = _lib_oracle()
    submitted = {"n": 0}

    class CountingReplay(ReplayExecutor):
        def submit(self, job):
            jobs_path = tmp_path / "ckpt" / "jobs.json"
            assert jobs_path.exists()
            data = json.loads(jobs_path.read_text(encoding="utf-8"))
            recs = {d["job"]["job_id"]: d for d in data["jobs"]}
            assert job.job_id in recs
            assert recs[job.job_id]["status"] == JobLifecycle.PREPARED.value
            submitted["n"] += 1
            return super().submit(job)

    camp = CampaignManager(
        library=lib,
        executor=CountingReplay(oracle, seed=0),
        acquisition="greedy",
        campaign_id="prep",
        seed=0,
        batch_size=15,
        init_frac=0.002,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=8,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=1,
        schema=lib.schema,
    )
    assert submitted["n"] >= 1
    camp.run(1)
    assert all(
        r.status is not JobLifecycle.PREPARED
        for r in camp.jobs.all()
    )


def test_per_job_history_not_cross_contaminated(tmp_path: Path):
    lib, oracle = _lib_oracle()
    camp = CampaignManager(
        library=lib,
        executor=ReplayExecutor(oracle, seed=0, min_latency=1, max_latency=2, stagger=True),
        acquisition="greedy",
        campaign_id="hist",
        seed=0,
        batch_size=25,
        init_frac=0.003,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=10,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=2,
        schema=lib.schema,
    )
    hist = camp.run(2)
    for row in hist:
        jid = row["job_id"]
        rec = camp.jobs.get(jid)
        item_ids = {it.item_id for it in rec.job.items}
        job_obs = [o for o in camp.store.history if o.item_id in item_ids]
        assert row["completed"] == sum(1 for o in job_obs if o.contributes_measurement)
        assert row["failed"] == sum(1 for o in job_obs if o.state is WellState.FAILED)
        assert row["qc_rejected"] == sum(1 for o in job_obs if o.state is WellState.QC_REJECTED)
        assert row["completed"] <= len(rec.job.items)


def test_vina_poll_is_nonblocking(tmp_path: Path):
    def slow_dock(*, smiles, work, config):
        time.sleep(0.35)
        work.mkdir(parents=True, exist_ok=True)
        (work / "score.txt").write_text(" -7.5\n", encoding="utf-8")
        return -7.5, "ok"

    ex = VinaExecutor(
        VinaConfig(receptor="/fake.pdbqt", work_dir=str(tmp_path / "vina"), max_workers=2),
        dock_fn=slow_dock,
    )
    from autoscreen.core.types import ItemKind, Job, JobItem

    items = [
        JobItem(item_id="j:i0", smiles="CCO", pool_idx=0, kind=ItemKind.EXPERIMENTAL),
        JobItem(item_id="j:i1", smiles="CCC", pool_idx=1, kind=ItemKind.EXPERIMENTAL),
    ]
    job = Job(job_id="vina-async", campaign_id="c", round=1, items=items, executor_kind="vina")
    jid = ex.submit(job)
    t0 = time.perf_counter()
    st = ex.poll(jid)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2  # must not wait for both ligands
    assert not st.done
    assert st.n_pending >= 1

    # Drain
    for _ in range(50):
        st = ex.poll(jid)
        if st.done:
            break
        time.sleep(0.05)
    assert st.done
    assert len([o for o in st.observations if o.usable]) == 2
    ex.close()


def test_controls_require_explicit_indices(tmp_path: Path):
    import pytest
    from autoscreen.core.constraints import PlateConfig

    lib, oracle = _lib_oracle()
    with pytest.raises(ValueError, match="explicit controls"):
        CampaignManager(
            library=lib,
            executor=ReplayExecutor(oracle, seed=0),
            checkpoint_dir=tmp_path / "bad",
            use_plate_layout=True,
            plate=PlateConfig(n_experimental=8, n_positive=2, n_negative=2, n_blank=0, n_replicate=0),
            schema=lib.schema,
        )
