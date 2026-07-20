"""Persistent store of usable observations used to train the surrogate."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .types import ItemKind, Observation, WellState


class ObservationStore:
    def __init__(self) -> None:
        self._by_pool: dict[int, Observation] = {}
        self._history: list[Observation] = []

    def __len__(self) -> int:
        return len(self._by_pool)

    @property
    def labeled_indices(self) -> list[int]:
        return sorted(self._by_pool.keys())

    def add(self, obs: Observation, *, replace: bool = False) -> bool:
        """Add an observation. Only usable experimental results enter training set."""
        self._history.append(obs)
        if not obs.usable:
            return False
        if obs.pool_idx in self._by_pool and not replace:
            return False
        self._by_pool[obs.pool_idx] = obs
        return True

    def add_many(self, observations: list[Observation]) -> int:
        return sum(1 for o in observations if self.add(o))

    def matrix(self, library_X: np.ndarray, n_objectives: int) -> tuple[np.ndarray, np.ndarray]:
        idxs = self.labeled_indices
        if not idxs:
            raise ValueError("ObservationStore is empty")
        X = library_X[idxs]
        Y = np.zeros((len(idxs), n_objectives), dtype=np.float64)
        for row, idx in enumerate(idxs):
            vals = self._by_pool[idx].values
            assert vals is not None
            Y[row] = np.asarray(vals, dtype=np.float64)
        return X, Y

    def mask(self, n: int) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        m[self.labeled_indices] = True
        return m

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "labeled": [
                {
                    **asdict(o),
                    "state": o.state.value,
                    "kind": o.kind.value,
                }
                for o in self._by_pool.values()
            ],
            "history_len": len(self._history),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ObservationStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        for d in data.get("labeled", []):
            obs = Observation(
                smiles=d["smiles"],
                pool_idx=int(d["pool_idx"]),
                values=d.get("values"),
                state=WellState(d["state"]),
                qc_passed=bool(d.get("qc_passed", False)),
                source=d.get("source", ""),
                compound_id=d.get("compound_id", ""),
                item_id=d.get("item_id", ""),
                kind=ItemKind(d.get("kind", "experimental")),
                raw=d.get("raw") or {},
                message=d.get("message", ""),
                timestamp=float(d.get("timestamp", 0.0)),
            )
            store.add(obs, replace=True)
        return store
