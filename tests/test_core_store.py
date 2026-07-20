from pathlib import Path

from autoscreen.core.library import load_candidate_library
from autoscreen.core.model import MultiOutputRFSurrogate
from autoscreen.core.observations import ObservationStore
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.core.types import ItemKind, Observation, WellState


def test_library_observations_surrogate(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    assert lib.n > 1000
    assert not hasattr(lib, "Y_hidden") or getattr(lib, "Y_hidden", None) is None
    assert lib.static_Y is not None
    assert lib.n_objectives == 1

    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    store = ObservationStore()
    for i in range(20):
        store.add(
            Observation(
                smiles=lib.smis[i],
                pool_idx=i,
                values=oracle.lookup_expensive(i),
                state=WellState.COMPLETED,
                qc_passed=True,
                source="test",
                kind=ItemKind.EXPERIMENTAL,
                item_id=f"t{i}",
            )
        )
    X, Y = store.matrix(lib.X, lib.n_objectives)
    model = MultiOutputRFSurrogate(lib.n_objectives, n_estimators=10, seed=0)
    model.fit(X, Y)
    mu, sd = model.predict(lib.X[:5])
    assert mu.shape == (5, 1)
    assert sd.shape == (5, 1)

    p = tmp_path / "store.json"
    store.save(p)
    store2 = ObservationStore.load(p)
    assert len(store2) == 20
    assert len(store2.history) == 20
