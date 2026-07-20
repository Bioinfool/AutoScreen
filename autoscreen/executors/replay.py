"""Offline oracle executor: reveals hidden multi-objective labels for submitted items."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from autoscreen.core.library import CandidateLibrary
from autoscreen.core.types import ItemKind, Job, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor


@dataclass
class _JobRec:
    job: Job
    tick: int = 0
    ready_at: dict[str, int] = field(default_factory=dict)
    results: dict[str, Observation] = field(default_factory=dict)


class ReplayExecutor(Executor):
    """Looks up hidden labels from CandidateLibrary.Y_hidden (maximize convention)."""

    kind = "replay"

    def __init__(
        self,
        library: CandidateLibrary,
        activity_noise: float = 0.0,
        fail_rate: float = 0.0,
        qc_reject_rate: float = 0.0,
        min_latency: int = 1,
        max_latency: int = 1,
        seed: int = 0,
    ):
        if library.Y_hidden is None:
            raise ValueError("ReplayExecutor requires CandidateLibrary.Y_hidden")
        self.library = library
        self.activity_noise = activity_noise
        self.fail_rate = fail_rate
        self.qc_reject_rate = qc_reject_rate
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.rng = np.random.default_rng(seed)
        self._jobs: dict[str, _JobRec] = {}
        self._idempotency: dict[str, str] = {}
        self._act_scale = float(np.std(library.Y_hidden[:, 0])) or 1.0

    def submit(self, job: Job) -> str:
        key = job.idempotency_key or job.job_id
        if key in self._idempotency:
            return self._idempotency[key]
        job_id = job.job_id or f"replay-{uuid.uuid4().hex[:12]}"
        ready = {
            it.item_id: int(self.rng.integers(self.min_latency, self.max_latency + 1))
            for it in job.items
        }
        self._jobs[job_id] = _JobRec(job=job, ready_at=ready)
        self._idempotency[key] = job_id
        return job_id

    def _finalize(self, job: Job, item) -> Observation:
        common = dict(
            smiles=item.smiles,
            pool_idx=item.pool_idx,
            source=self.kind,
            compound_id=item.compound_id,
            item_id=item.item_id,
            kind=item.kind,
            timestamp=time.time(),
        )
        if item.kind is ItemKind.BLANK or item.pool_idx < 0:
            return Observation(
                values=None, state=WellState.COMPLETED, qc_passed=False,
                message="blank/control", **common,
            )
        if self.rng.random() < self.fail_rate:
            return Observation(
                values=None, state=WellState.FAILED, qc_passed=False,
                message="simulated failure", **common,
            )
        base = self.library.Y_hidden[item.pool_idx].astype(float).copy()
        if self.activity_noise > 0:
            base[0] += self.rng.normal(0.0, self.activity_noise * self._act_scale)
        qc = self.rng.random() >= self.qc_reject_rate
        state = WellState.COMPLETED if qc else WellState.QC_REJECTED
        return Observation(
            values=base.tolist(), state=state, qc_passed=qc,
            message="ok" if qc else "qc rejected",
            raw={"dock_style": True}, **common,
        )

    def poll(self, job_id: str) -> JobStatus:
        rec = self._jobs[job_id]
        rec.tick += 1
        pending = 0
        for it in rec.job.items:
            if it.item_id in rec.results:
                continue
            if rec.tick >= rec.ready_at[it.item_id]:
                rec.results[it.item_id] = self._finalize(rec.job, it)
            else:
                pending += 1
        obs = list(rec.results.values())
        return JobStatus(
            job_id=job_id,
            done=pending == 0,
            observations=obs,
            n_pending=pending,
            round=rec.job.round,
        )

    def cancel(self, job_id: str) -> None:
        if job_id in self._jobs:
            rec = self._jobs[job_id]
            for it in rec.job.items:
                if it.item_id not in rec.results:
                    rec.results[it.item_id] = Observation(
                        smiles=it.smiles,
                        pool_idx=it.pool_idx,
                        values=None,
                        state=WellState.CANCELLED,
                        qc_passed=False,
                        source=self.kind,
                        item_id=it.item_id,
                        kind=it.kind,
                        message="cancelled",
                    )
