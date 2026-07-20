"""Persistent store with replicate aggregation into training labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoscreen.core.jobs import observation_from_dict, observation_to_dict
from autoscreen.core.persist import atomic_write_json
from autoscreen.core.types import ItemKind, Observation, WellState


@dataclass
class AggregateConfig:
    method: str = "mean"  # mean | median
    max_std: float | None = 1.0  # activity-std threshold; None disables
    min_replicates_for_qc: int = 2


class ObservationStore:
    def __init__(self, aggregate: AggregateConfig | None = None) -> None:
        self.aggregate = aggregate or AggregateConfig()
        self._by_pool: dict[int, Observation] = {}  # aggregated training labels
        self._events: dict[int, list[Observation]] = {}  # raw contributing measurements
        self._history: list[Observation] = []
        self._seen_keys: set[str] = set()
        self._aggregate_qc_rejected: set[int] = set()

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
        """Append to audit history; recompute aggregate training label when applicable.

        Returns False if ``item_id`` was already seen (duplicate). Otherwise returns
        whether ``pool_idx`` has a training label after this call.
        """
        key = self._key(obs)
        if key in self._seen_keys:
            if not replace:
                return False
            return obs.pool_idx in self._by_pool
        self._seen_keys.add(key)
        self._history.append(obs)

        if obs.contributes_measurement:
            self._events.setdefault(obs.pool_idx, []).append(obs)
            return self._recompute(obs.pool_idx)
        return False

    def add_many(self, observations: list[Observation]) -> int:
        """Add observations; return how many pool indices newly gained a training label."""
        before = set(self._by_pool.keys())
        for o in observations:
            self.add(o)
        return len(set(self._by_pool.keys()) - before)

    def _recompute(self, pool_idx: int) -> bool:
        events = [e for e in self._events.get(pool_idx, []) if e.contributes_measurement]
        if not events:
            self._by_pool.pop(pool_idx, None)
            return False

        mat = np.asarray([e.values for e in events], dtype=np.float64)
        if self.aggregate.method == "median":
            agg = np.median(mat, axis=0)
        else:
            agg = mat.mean(axis=0)
        std = mat.std(axis=0) if len(events) > 1 else np.zeros(mat.shape[1])

        if (
            self.aggregate.max_std is not None
            and len(events) >= self.aggregate.min_replicates_for_qc
            and float(std[0]) > float(self.aggregate.max_std)
        ):
            self._by_pool.pop(pool_idx, None)
            self._aggregate_qc_rejected.add(pool_idx)
            return False

        self._aggregate_qc_rejected.discard(pool_idx)
        primary = events[0]
        self._by_pool[pool_idx] = Observation(
            smiles=primary.smiles,
            pool_idx=pool_idx,
            values=agg.astype(float).tolist(),
            state=WellState.COMPLETED,
            qc_passed=True,
            source="aggregate",
            compound_id=primary.compound_id,
            item_id=f"agg:{pool_idx}:n{len(events)}",
            kind=ItemKind.EXPERIMENTAL,
            raw={
                "n_measurements": len(events),
                "std": std.astype(float).tolist(),
                "method": self.aggregate.method,
                "item_ids": [e.item_id for e in events],
            },
            message="aggregated",
            timestamp=max(e.timestamp for e in events),
        )
        return True

    def is_aggregate_qc_rejected(self, pool_idx: int) -> bool:
        return int(pool_idx) in self._aggregate_qc_rejected

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
            "aggregate": {
                "method": self.aggregate.method,
                "max_std": self.aggregate.max_std,
                "min_replicates_for_qc": self.aggregate.min_replicates_for_qc,
            },
            "labeled": [observation_to_dict(o) for o in self._by_pool.values()],
            "history": [observation_to_dict(o) for o in self._history],
            "seen_keys": sorted(self._seen_keys),
            "aggregate_qc_rejected": sorted(self._aggregate_qc_rejected),
        }
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "ObservationStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agg_cfg = data.get("aggregate") or {}
        store = cls(
            AggregateConfig(
                method=agg_cfg.get("method", "mean"),
                max_std=agg_cfg.get("max_std", 1.0),
                min_replicates_for_qc=int(agg_cfg.get("min_replicates_for_qc", 2)),
            )
        )
        hist = data.get("history")
        if hist is not None:
            for d in hist:
                store.add(observation_from_dict(d), replace=True)
            store._seen_keys = set(data.get("seen_keys") or store._seen_keys)
        else:
            for d in data.get("labeled", []):
                store.add(observation_from_dict(d), replace=True)
        store._aggregate_qc_rejected = set(data.get("aggregate_qc_rejected") or [])
        return store
