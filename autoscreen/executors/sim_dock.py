"""Simulated docking executor: oracle labels + wall-clock latency + thread pool.

Measures Campaign async throughput without a real Vina binary.
Hidden labels stay inside this executor (same boundary as Replay/Vina).
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from autoscreen.core.oracle import ArrayLabelOracle
from autoscreen.core.types import ItemKind, Job, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor, JobNotFound


@dataclass
class SimDockConfig:
    latency_s: float = 0.05
    max_workers: int = 4
    poll_hint_s: float = 0.02


@dataclass
class _LigandTask:
    observation: Observation | None = None
    future: Future | None = None
    done: bool = False


@dataclass
class _JobRec:
    job: Job
    tasks: dict[str, _LigandTask] = field(default_factory=dict)


class SimulatedDockExecutor(Executor):
    """Vina-like async schedule with sleep + Replay oracle scores."""

    kind = "sim_dock"

    def __init__(self, oracle: ArrayLabelOracle, config: SimDockConfig | None = None):
        self.oracle = oracle
        self.config = config or SimDockConfig()
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(self.config.max_workers)))
        self._jobs: dict[str, _JobRec] = {}
        self._idempotency: dict[str, str] = {}

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _evaluate(self, item) -> Observation:
        common = dict(
            smiles=item.smiles,
            pool_idx=item.pool_idx,
            source=self.kind,
            compound_id=item.compound_id,
            item_id=item.item_id,
            kind=item.kind,
            timestamp=time.time(),
        )
        time.sleep(max(0.0, float(self.config.latency_s)))
        if item.kind is ItemKind.BLANK or item.pool_idx < 0:
            return Observation(
                values=None,
                state=WellState.COMPLETED,
                qc_passed=False,
                message="blank/control",
                **common,
            )
        vals = self.oracle.lookup_expensive(item.pool_idx)
        return Observation(
            values=list(vals),
            state=WellState.COMPLETED,
            qc_passed=True,
            message="ok",
            raw={"sim_latency_s": self.config.latency_s},
            **common,
        )

    def submit(self, job: Job) -> str:
        key = job.idempotency_key or job.job_id
        if key in self._idempotency:
            return self._idempotency[key]
        job_id = job.job_id or f"sim-{uuid.uuid4().hex[:12]}"
        rec = _JobRec(job=job)
        for it in job.items:
            task = _LigandTask()
            task.future = self._pool.submit(self._evaluate, it)
            rec.tasks[it.item_id] = task
        self._jobs[job_id] = rec
        self._idempotency[key] = job_id
        return job_id

    def poll(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise JobNotFound(f"Unknown job_id={job_id}")
        rec = self._jobs[job_id]
        pending = 0
        obs: list[Observation] = []
        for it in rec.job.items:
            task = rec.tasks[it.item_id]
            if task.done and task.observation is not None:
                obs.append(task.observation)
                continue
            fut = task.future
            if fut is not None and fut.done():
                task.observation = fut.result()
                task.done = True
                obs.append(task.observation)
            else:
                pending += 1
        return JobStatus(
            job_id=job_id,
            done=pending == 0,
            observations=obs,
            n_pending=pending,
            round=rec.job.round,
            next_poll_after=self.config.poll_hint_s if pending else 0.0,
        )

    def cancel(self, job_id: str) -> None:
        if job_id not in self._jobs:
            return
        rec = self._jobs[job_id]
        for it in rec.job.items:
            task = rec.tasks[it.item_id]
            if task.done:
                continue
            if task.future is not None:
                task.future.cancel()
            task.observation = Observation(
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
            task.done = True
