"""Replay campaign end-to-end (async step loop)."""
from pathlib import Path

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.executors.replay import ReplayExecutor


def _setup(tmp_path: Path, **kwargs):
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    ex = ReplayExecutor(oracle, seed=0, **{k: kwargs.pop(k) for k in list(kwargs) if k in (
        "fail_rate", "qc_reject_rate", "activity_noise", "min_latency", "max_latency", "stagger"
    )})
    camp = CampaignManager(
        library=lib,
        executor=ex,
        acquisition="greedy",
        campaign_id="test_replay",
        seed=0,
        batch_size=50,
        init_frac=0.005,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=20,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=kwargs.get("max_active_jobs", 1),
        schema=lib.schema,
    )
    return camp, lib, oracle


def test_replay_campaign_two_rounds(tmp_path: Path):
    camp, _, _ = _setup(tmp_path)
    hist = camp.run(2)
    assert len(hist) >= 2
    assert camp.state.round == 2
    assert len(camp.store) > 50
    camp2 = CampaignManager(
        library=camp.library,
        executor=ReplayExecutor(
            load_moo_oracle(
                Path(__file__).resolve().parents[1] / "data/Enamine10k_moo.csv.gz",
                camp.library.smis,
                schema=camp.library.schema,
            )[0],
            seed=0,
        ),
        acquisition="greedy",
        campaign_id="test_replay",
        seed=0,
        batch_size=50,
        init_frac=0.005,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=20,
        resume=True,
        max_active_jobs=1,
        schema=camp.library.schema,
    )
    assert camp2.state.round == 2
