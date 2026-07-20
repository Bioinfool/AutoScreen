"""Replay campaign end-to-end (small rounds)."""
from pathlib import Path

from autoscreen.core.campaign import CampaignManager
from autoscreen.core.library import load_candidate_library
from autoscreen.executors.replay import ReplayExecutor


def test_replay_campaign_two_rounds(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "molpal/libraries/Enamine10k.csv.gz",
        root / "molpal/libraries/Enamine10k.h5",
        root / "molpal/data/Enamine10k_moo.csv.gz",
    )
    ex = ReplayExecutor(lib, seed=0)
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
        max_polls=10,
    )
    hist = camp.run(2)
    assert len(hist) >= 2
    assert camp.state.round == 2
    assert len(camp.store) > 50
    # resume
    camp2 = CampaignManager(
        library=lib,
        executor=ReplayExecutor(lib, seed=0),
        acquisition="greedy",
        campaign_id="test_replay",
        seed=0,
        batch_size=50,
        init_frac=0.005,
        checkpoint_dir=tmp_path / "ckpt",
        n_estimators=20,
        resume=True,
        max_polls=10,
    )
    assert camp2.state.round == 2
