"""Async step loop: partial results, pending exclusion, mid-job resume."""
from pathlib import Path

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.candidate_state import CandidatePhase
from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.executors.replay import ReplayExecutor


def _lib_oracle():
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    return lib, oracle


def test_partial_results_and_pending_not_reselected(tmp_path: Path):
    lib, oracle = _lib_oracle()
    ex = ReplayExecutor(oracle, seed=0, min_latency=1, max_latency=4, stagger=True)
    camp = CampaignManager(
        library=lib,
        executor=ex,
        acquisition="greedy",
        campaign_id="async1",
        seed=0,
        batch_size=30,
        init_frac=0.003,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=15,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=2,
        schema=lib.schema,
    )
    saw_partial = False
    for _ in range(100):
        info = camp.step()
        if info["open_jobs"] > 0:
            for rec in camp.jobs.open_jobs():
                for it in rec.job.items:
                    if it.pool_idx < 0:
                        continue
                    assert camp.cand.phase(it.pool_idx) != CandidatePhase.AVAILABLE
            if info["n_labeled"] > 0:
                saw_partial = True
                break
    assert saw_partial or camp.state.init_done

    hist = camp.run(2)
    assert camp.state.round >= 1
    assert len(camp.store) > 20
    assert any(h.get("round", -1) >= 1 for h in hist)


def test_resume_reattaches_open_job(tmp_path: Path):
    lib, oracle = _lib_oracle()
    ckpt = tmp_path / "resume_ckpt"
    ex = ReplayExecutor(oracle, seed=0, min_latency=2, max_latency=5, stagger=True)
    camp = CampaignManager(
        library=lib,
        executor=ex,
        acquisition="greedy",
        campaign_id="resume_job",
        seed=0,
        batch_size=25,
        init_frac=0.003,
        checkpoint_dir=ckpt,
        n_estimators=10,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=1,
        schema=lib.schema,
    )
    for _ in range(2):
        camp.step()
    assert camp.jobs.open_jobs()
    open_before = [r.job.job_id for r in camp.jobs.open_jobs()]
    labeled_before = len(camp.store)
    camp._checkpoint()

    ex2 = ReplayExecutor(oracle, seed=99, min_latency=1, max_latency=2, stagger=True)
    camp2 = CampaignManager(
        library=lib,
        executor=ex2,
        acquisition="greedy",
        campaign_id="resume_job",
        seed=0,
        batch_size=25,
        init_frac=0.003,
        checkpoint_dir=ckpt,
        n_estimators=10,
        resume=True,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=1,
        schema=lib.schema,
    )
    assert [r.job.job_id for r in camp2.jobs.open_jobs()] == open_before
    camp2.run(1)
    assert len(camp2.store) >= labeled_before
    assert camp2.state.init_done
