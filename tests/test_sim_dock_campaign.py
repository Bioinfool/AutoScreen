"""Campaign + SimulatedDockExecutor integration."""
from pathlib import Path

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.executors.sim_dock import SimDockConfig, SimulatedDockExecutor


def test_sim_dock_campaign_one_round(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    ex = SimulatedDockExecutor(
        oracle, SimDockConfig(latency_s=0.0, max_workers=2, poll_hint_s=0.0)
    )
    camp = CampaignManager(
        library=lib,
        executor=ex,
        acquisition="random",
        campaign_id="sim_int",
        seed=0,
        batch_size=16,
        init_frac=0.002,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=10,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=2,
        schema=lib.schema,
        pending_penalty=0.0,
        poll_interval_s=0.0,
    )
    try:
        camp.run(1)
        assert camp.state.round == 1
        assert len(camp.store) > 10
        assert camp.executor.kind == "sim_dock"
    finally:
        camp.close()
