"""Persistent store of observations used to train the surrogate and for audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoscreen.core.jobs import observation_from_dict, observation_to_dict
from autoscreen.core.persist import atomic_write_json
from autoscreen.core.types import Observation


class ObservationStore:
    def __init__(self) -> None:
        self._by_pool: dict[int, Observation] = {}
        self._history: list[Observation] = []
        self._seen_keys: set[str] = set()

    def __len__(self) -> int:
        return len(self._by_pool)

    @property
    def labeled_indices(self) -> list[int]:
        return sorted(self._by_pool.keys())

    @property
    def history(self) -> list[Observation]:
        return list(self._history)

    def _key(self, obs: Observation) -> str:
        return obs.item_id or f"{obs.pool_idx}:{obs.state.value}:{obs.kind.value}"

    def add(self, obs: Observation, *, replace: bool = False) -> bool:
        """Add an observation. Deduplicate by item_id. Only usable → training set."""
        key = self._key(obs)
        if key in self._seen_keys and not replace:
            return False
        self._seen_keys.add(key)
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
            if len(vals) != n_objectives:
                raise ValueError(
                    f"Observation at {idx} has {len(vals)} values; expected {n_objectives}"
                )
            Y[row] = np.asarray(vals, dtype=np.float64)
        return X, Y

    def mask(self, n: int) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        m[self.labeled_indices] = True
        return m

    def save(self, path: str | Path) -> None:
        payload = {
            "labeled": [observation_to_dict(o) for o in self._by_pool.values()],
            "history": [observation_to_dict(o) for o in self._history],
            "seen_keys": sorted(self._seen_keys),
        }
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "ObservationStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        hist = data.get("history")
        if hist is not None:
            for d in hist:
                store.add(observation_from_dict(d), replace=True)
            store._seen_keys = set(data.get("seen_keys") or store._seen_keys)
        else:
            # Backward compatible: labeled-only checkpoints
            for d in data.get("labeled", []):
                store.add(observation_from_dict(d), replace=True)
        return store
