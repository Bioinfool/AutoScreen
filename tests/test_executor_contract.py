"""Executor contract tests + vina skip path."""
import shutil
from pathlib import Path

import pytest

from autoscreen.core.library import load_candidate_library
from autoscreen.core.types import ItemKind, Job, JobItem
from autoscreen.executors.replay import ReplayExecutor
from autoscreen.executors.vina import VinaConfig, VinaExecutor


def _lib():
    root = Path(__file__).resolve().parents[1]
    return load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        root / "data/Enamine10k_moo.csv.gz",
    )


def test_replay_executor_contract():
    lib = _lib()
    ex = ReplayExecutor(lib, seed=1, fail_rate=0.0, qc_reject_rate=0.0)
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


def test_vina_missing_receptor_raises():
    ex = VinaExecutor(VinaConfig(receptor=None))
    items = [JobItem(item_id="a", smiles="CCO", pool_idx=0)]
    job = Job(job_id="v1", campaign_id="c", round=0, items=items, executor_kind="vina")
    with pytest.raises(RuntimeError, match="receptor"):
        ex.submit(job)


@pytest.mark.skipif(shutil.which("vina") is None, reason="vina binary not installed")
def test_vina_binary_detected():
    ex = VinaExecutor(VinaConfig(receptor="/tmp/fake.pdbqt", vina_bin="vina"))
    assert ex.config.vina_bin == "vina"
