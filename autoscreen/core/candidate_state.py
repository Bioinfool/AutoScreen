"""Persistent per-candidate lifecycle for async campaign orchestration."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import numpy as np

from autoscreen.core.persist import atomic_write_json
from autoscreen.core.types import ItemKind, Observation, WellState


class CandidatePhase(str, Enum):
    AVAILABLE = "AVAILABLE"
    SELECTED = "SELECTED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    LABELED = "LABELED"
    FAILED = "FAILED"  # permanent assay failure
    QC_REJECTED = "QC_REJECTED"  # permanent QC
    RETRYABLE = "RETRYABLE"  # transient fail/QC — selectable again
    DEAD_LETTER = "DEAD_LETTER"
    CONTROL = "CONTROL"
    STOCKOUT = "STOCKOUT"
    BLANK = "BLANK"


_BLOCKING = {
    CandidatePhase.SELECTED,
    CandidatePhase.SUBMITTED,
    CandidatePhase.RUNNING,
    CandidatePhase.LABELED,
    CandidatePhase.FAILED,
    CandidatePhase.QC_REJECTED,
    CandidatePhase.DEAD_LETTER,
    CandidatePhase.CONTROL,
    CandidatePhase.STOCKOUT,
}


class CandidateStateStore:
    def __init__(
        self,
        n: int,
        *,
        max_fail_retries: int = 1,
        max_qc_retries: int = 1,
    ):
        self.n = int(n)
        self.max_fail_retries = int(max_fail_retries)
        self.max_qc_retries = int(max_qc_retries)
        self._phase = [CandidatePhase.AVAILABLE] * self.n
        self._job_id: list[str | None] = [None] * self.n
        self._fail_retries = [0] * self.n
        self._qc_retries = [0] * self.n

    def phase(self, idx: int) -> CandidatePhase:
        return self._phase[int(idx)]

    def unavailable_mask(self) -> np.ndarray:
        return np.array([p in _BLOCKING for p in self._phase], dtype=bool)

    def available_indices(self) -> np.ndarray:
        return np.where(~self.unavailable_mask())[0]

    def mark_control(self, idxs: list[int]) -> None:
        for i in idxs:
            if 0 <= i < self.n:
                self._phase[i] = CandidatePhase.CONTROL

    def mark_stockout(self, mask: np.ndarray) -> None:
        for i, out in enumerate(mask):
            if out and self._phase[i] is CandidatePhase.AVAILABLE:
                self._phase[i] = CandidatePhase.STOCKOUT

    def mark_selected(self, idxs: list[int], job_id: str = "") -> None:
        for i in idxs:
            i = int(i)
            if self._phase[i] in (CandidatePhase.AVAILABLE, CandidatePhase.RETRYABLE):
                self._phase[i] = CandidatePhase.SELECTED
                self._job_id[i] = job_id or self._job_id[i]

    def mark_submitted(self, idxs: list[int], job_id: str) -> None:
        for i in idxs:
            i = int(i)
            if i < 0:
                continue
            self._phase[i] = CandidatePhase.SUBMITTED
            self._job_id[i] = job_id

    def mark_running(self, idxs: list[int]) -> None:
        for i in idxs:
            i = int(i)
            if i >= 0 and self._phase[i] in (
                CandidatePhase.SELECTED,
                CandidatePhase.SUBMITTED,
            ):
                self._phase[i] = CandidatePhase.RUNNING

    def mark_labeled(self, idxs: list[int]) -> None:
        for i in idxs:
            i = int(i)
            if 0 <= i < self.n:
                self._phase[i] = CandidatePhase.LABELED

    def release(self, idxs: list[int], *, job_id: str | None = None) -> None:
        """Return in-flight candidates to AVAILABLE (transient job failure)."""
        for i in idxs:
            i = int(i)
            if i < 0 or i >= self.n:
                continue
            if self._phase[i] not in (
                CandidatePhase.SELECTED,
                CandidatePhase.SUBMITTED,
                CandidatePhase.RUNNING,
            ):
                continue
            if job_id is not None and self._job_id[i] not in (None, job_id):
                continue
            self._phase[i] = CandidatePhase.AVAILABLE
            self._job_id[i] = None

    def apply_observation(self, obs: Observation) -> None:
        """Update phase from a single observation (retry-aware)."""
        i = int(obs.pool_idx)
        if i < 0 or i >= self.n:
            return
        if obs.kind in (ItemKind.POSITIVE, ItemKind.NEGATIVE):
            self._phase[i] = CandidatePhase.CONTROL
            return
        if obs.kind is ItemKind.BLANK:
            return
        if obs.state is WellState.CANCELLED:
            self._phase[i] = CandidatePhase.AVAILABLE
            self._job_id[i] = None
            return
        if obs.state is WellState.FAILED:
            self._fail_retries[i] += 1
            if self._fail_retries[i] <= self.max_fail_retries:
                self._phase[i] = CandidatePhase.RETRYABLE
                self._job_id[i] = None
            else:
                self._phase[i] = CandidatePhase.FAILED
            return
        if obs.state is WellState.QC_REJECTED:
            self._qc_retries[i] += 1
            if self._qc_retries[i] <= self.max_qc_retries:
                self._phase[i] = CandidatePhase.RETRYABLE
                self._job_id[i] = None
            else:
                self._phase[i] = CandidatePhase.QC_REJECTED
            return
        if obs.contributes_measurement:
            # Tentative; campaign should call mark_labeled after aggregate confirms
            if self._phase[i] not in (CandidatePhase.LABELED, CandidatePhase.CONTROL):
                self._phase[i] = CandidatePhase.RUNNING
            return
        if obs.state is WellState.RUNNING:
            self._phase[i] = CandidatePhase.RUNNING
        elif obs.state is WellState.SUBMITTED:
            self._phase[i] = CandidatePhase.SUBMITTED

    def apply_observations(self, observations: list[Observation]) -> None:
        for o in observations:
            self.apply_observation(o)

    def sync_from_store(
        self,
        labeled_indices: list[int],
        aggregate_qc_rejected: set[int] | None = None,
    ) -> None:
        """Align LABELED / QC with ObservationStore aggregates (idempotent)."""
        for i in labeled_indices:
            if 0 <= i < self.n:
                self._phase[i] = CandidatePhase.LABELED
        if not aggregate_qc_rejected:
            return
        for i in aggregate_qc_rejected:
            if not (0 <= i < self.n):
                continue
            if self._phase[i] is CandidatePhase.LABELED:
                continue
            if self._phase[i] in (
                CandidatePhase.QC_REJECTED,
                CandidatePhase.DEAD_LETTER,
                CandidatePhase.RETRYABLE,
                CandidatePhase.FAILED,
                CandidatePhase.CONTROL,
            ):
                continue
            self._qc_retries[i] += 1
            if self._qc_retries[i] <= self.max_qc_retries:
                self._phase[i] = CandidatePhase.RETRYABLE
                self._job_id[i] = None
            else:
                self._phase[i] = CandidatePhase.QC_REJECTED

    def n_inflight(self) -> int:
        return sum(
            1
            for p in self._phase
            if p
            in (
                CandidatePhase.SELECTED,
                CandidatePhase.SUBMITTED,
                CandidatePhase.RUNNING,
            )
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "n": self.n,
            "phase": [p.value for p in self._phase],
            "job_id": self._job_id,
            "fail_retries": self._fail_retries,
            "qc_retries": self._qc_retries,
            "max_fail_retries": self.max_fail_retries,
            "max_qc_retries": self.max_qc_retries,
        }
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "CandidateStateStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(
            int(data["n"]),
            max_fail_retries=int(data.get("max_fail_retries", 1)),
            max_qc_retries=int(data.get("max_qc_retries", 1)),
        )
        store._phase = [CandidatePhase(p) for p in data["phase"]]
        store._job_id = list(data.get("job_id") or [None] * store.n)
        store._fail_retries = list(data.get("fail_retries") or [0] * store.n)
        store._qc_retries = list(data.get("qc_retries") or [0] * store.n)
        return store
