from pathlib import Path

import numpy as np

from autoscreen.core.library import load_candidate_library
from autoscreen.core.model import MultiOutputRFSurrogate
from autoscreen.core.observations import ObservationStore
from autoscreen.core.types import ItemKind, Observation, WellState


def test_library_observations_surrogate(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        root / "data/Enamine10k_moo.csv.gz",
    )
    assert lib.n > 1000
    assert lib.Y_hidden is not None

    store = ObservationStore()
    for i in range(20):
        store.add(
            Observation(
                smiles=lib.smis[i],
                pool_idx=i,
                values=lib.Y_hidden[i].tolist(),
                state=WellState.COMPLETED,
                qc_passed=True,
                source="test",
                kind=ItemKind.EXPERIMENTAL,
            )
        )
    X, Y = store.matrix(lib.X, lib.n_objectives)
    model = MultiOutputRFSurrogate(lib.n_objectives, n_estimators=10, seed=0)
    model.fit(X, Y)
    mu, sd = model.predict(lib.X[:5])
    assert mu.shape == (5, 3)
    assert sd.shape == (5, 3)

    p = tmp_path / "store.json"
    store.save(p)
    store2 = ObservationStore.load(p)
    assert len(store2) == 20
