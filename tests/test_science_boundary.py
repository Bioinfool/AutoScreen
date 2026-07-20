"""Scientific boundary: Campaign must not see hidden labels; schema is real."""
from pathlib import Path

import numpy as np
import pytest

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.library import load_candidate_library
from autoscreen.core.objectives import parse_objective_schema
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.core.types import ItemKind, Observation, WellState
from autoscreen.executors.replay import ReplayExecutor


def test_library_has_no_hidden_labels():
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    assert getattr(lib, "Y_hidden", None) is None
    assert "Y_hidden" not in lib.__dataclass_fields__
    assert lib.schema.n_expensive == 1
    assert "qed" in lib.static_names


def test_parse_objective_schema_legacy_list():
    schema = parse_objective_schema({"objectives": ["activity", "qed", "sa_ease"]})
    assert schema.expensive_names == ("activity",)
    assert "qed" in schema.static_names
    assert "sa_ease" in schema.static_names


def test_campaign_rejects_library_with_y_hidden(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    # Simulate a contaminated library object
    object.__setattr__(lib, "Y_hidden", np.zeros((lib.n, 1)))
    with pytest.raises(ValueError, match="Y_hidden"):
        CampaignManager(
            library=lib,
            executor=ReplayExecutor(oracle, seed=0),
            checkpoint_dir=tmp_path / "bad",
            evaluator=BenchmarkEvaluator(oracle),
        )


def test_qc_reject_and_fail_not_in_training(tmp_path: Path):
    from autoscreen.core.observations import ObservationStore

    store = ObservationStore()
    store.add(
        Observation(
            smiles="C",
            pool_idx=0,
            values=[1.0],
            state=WellState.FAILED,
            qc_passed=False,
            kind=ItemKind.EXPERIMENTAL,
            item_id="f0",
        )
    )
    store.add(
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
    store.add(
        Observation(
            smiles="CCC",
            pool_idx=2,
            values=[1.0],
            state=WellState.COMPLETED,
            qc_passed=True,
            kind=ItemKind.EXPERIMENTAL,
            item_id="ok2",
        )
    )
    assert len(store) == 1
    assert store.labeled_indices == [2]
    assert len(store.history) == 3


def test_duplicate_item_id_ingested_once():
    from autoscreen.core.observations import ObservationStore

    store = ObservationStore()
    obs = Observation(
        smiles="C",
        pool_idx=0,
        values=[1.0],
        state=WellState.COMPLETED,
        qc_passed=True,
        kind=ItemKind.EXPERIMENTAL,
        item_id="same",
    )
    assert store.add(obs) is True
    assert store.add(obs) is False
    assert len(store) == 1
