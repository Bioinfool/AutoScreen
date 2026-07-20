"""Offline oracle executor — owns hidden labels; Campaign never sees the oracle."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from autoscreen.core.oracle import ArrayLabelOracle
from autoscreen.core.types import ItemKind, Job, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor


@dataclass
class _JobRec:
    job: Job
    tick: int = 0
    ready_at: dict[str, int] = field(default_factory=dict)
    results: dict[str, Observation] = field(default_factory=dict)


class ReplayExecutor(Executor):
    """Reveals expensive objectives from a private LabelOracle."""

    kind = "replay"

    def __init__(
        self,
        oracle: ArrayLabelOracle,
        activity_noise: float = 0.0,
        fail_rate: float = 0.0,
        qc_reject_rate: float = 0.0,
        min_latency: int = 1,
        max_latency: int = 1,
        seed: int = 0,
        # Staggered completion for async / partial-result tests
        stagger: bool = False,
    ):
        self.oracle = oracle
        self.activity_noise = activity_noise
        self.fail_rate = fail_rate
        self.qc_reject_rate = qc_reject_rate
        self.min_latency = min_latency
        self.max_latency = max_latency if not stagger else max(max_latency, len(oracle.Y) and 3)
        self.rng = np.random.default_rng(seed)
        self._jobs: dict[str, _JobRec] = {}
        self._idempotency: dict[str, str] = {}
        act = oracle.expensive_array()[:, 0]
        self._act_scale = float(np.std(act)) or 1.0

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

    def _finalize(self, item) -> Observation:
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
        base = np.asarray(self.oracle.lookup_expensive(item.pool_idx), dtype=float)
        if self.activity_noise > 0 and len(base):
            base[0] += self.rng.normal(0.0, self.activity_noise * self._act_scale)
        qc = self.rng.random() >= self.qc_reject_rate
        state = WellState.COMPLETED if qc else WellState.QC_REJECTED
        return Observation(
            values=base.tolist(), state=state, qc_passed=qc,
            message="ok" if qc else "qc rejected",
            raw={}, **common,
        )

    def poll(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise KeyError(
                f"Unknown job_id={job_id}. In-process ReplayExecutor state is not "
                "shared across processes; resume must re-submit via JobStore."
            )
        rec = self._jobs[job_id]
        rec.tick += 1
        pending = 0
        for it in rec.job.items:
            if it.item_id in rec.results:
                continue
            if rec.tick >= rec.ready_at[it.item_id]:
                rec.results[it.item_id] = self._finalize(it)
            else:
                pending += 1
        return JobStatus(
            job_id=job_id,
            done=pending == 0,
            observations=list(rec.results.values()),
            n_pending=pending,
            round=rec.job.round,
        )

    def cancel(self, job_id: str) -> None:
        if job_id not in self._jobs:
            return
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
