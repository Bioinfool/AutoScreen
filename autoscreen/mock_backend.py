"""A mock experiment backend that mimics a robot / assay platform.

It reads the *hidden* ground-truth labels (activity, qed, sa_ease) but only
reveals them for compounds that have been submitted, and only after a simulated
delay. It injects the realistic nuisances a real platform has:

  - staggered completion times (different wells finish on different polls)
  - hard failures (hardware/assay error -> FAILED, no usable value)
  - measurement noise added to the activity readout
  - QC rejection (a fraction of results fail QC and are discarded)
  - blank / control wells that carry no library compound

"Time" is modelled in discrete poll ticks rather than wall-clock seconds, so a
multi-hour async campaign can be exercised in milliseconds while still going
through every SUBMITTED -> RUNNING -> COMPLETED/FAILED transition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .experiment import (
    CompoundResult,
    ExperimentBackend,
    JobStatus,
    SubmittedCompound,
    WellState,
)


@dataclass
class _Tracked:
    comp: SubmittedCompound
    state: WellState
    ready_tick: int
    result: CompoundResult | None = None


@dataclass
class _Job:
    round: int
    tick: int = 0
    items: dict[str, _Tracked] = field(default_factory=dict)


class MockRobotBackend(ExperimentBackend):
    def __init__(
        self,
        truth: np.ndarray,
        activity_noise: float = 0.05,
        fail_rate: float = 0.05,
        qc_reject_rate: float = 0.05,
        min_latency: int = 1,
        max_latency: int = 3,
        seed: int = 0,
    ):
        """truth : (n_pool, 3) ground-truth objective values (maximize convention)."""
        self.truth = truth
        self.activity_noise = activity_noise
        self.fail_rate = fail_rate
        self.qc_reject_rate = qc_reject_rate
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.rng = np.random.default_rng(seed)
        self._jobs: dict[str, _Job] = {}
        # activity readout noise scale, in raw units of the activity objective
        self._act_scale = float(np.std(truth[:, 0])) if truth.shape[0] else 1.0

    def submit(self, job_id: str, round: int, compounds: list[SubmittedCompound]) -> None:
        job = _Job(round=round)
        for c in compounds:
            latency = int(self.rng.integers(self.min_latency, self.max_latency + 1))
            job.items[c.well_id] = _Tracked(comp=c, state=WellState.SUBMITTED, ready_tick=latency)
        self._jobs[job_id] = job

    def _finalize(self, tr: _Tracked) -> CompoundResult:
        c = tr.comp
        common = dict(
            well_id=c.well_id,
            compound_id=c.compound_id,
            smiles=c.smiles,
            pool_idx=c.pool_idx,
            kind=c.kind,
            replicate_of=c.replicate_of,
        )
        # blank wells never carry a compound: they always "complete" with no value
        if c.kind == "blank":
            return CompoundResult(state=WellState.COMPLETED, values=None, qc_passed=False,
                                  message="blank control", **common)

        if self.rng.random() < self.fail_rate:
            return CompoundResult(state=WellState.FAILED, values=None, qc_passed=False,
                                  message="assay/hardware failure", **common)

        base = self.truth[c.pool_idx].astype(float).copy()
        # noise only on the activity channel (qed/sa are deterministic descriptors)
        base[0] += self.rng.normal(0.0, self.activity_noise * self._act_scale)

        qc = self.rng.random() >= self.qc_reject_rate
        state = WellState.COMPLETED if qc else WellState.QC_REJECTED
        return CompoundResult(state=state, values=base.tolist(), qc_passed=qc,
                              message="ok" if qc else "qc rejected", **common)

    def poll(self, job_id: str) -> JobStatus:
        job = self._jobs[job_id]
        job.tick += 1
        results: list[CompoundResult] = []
        n_pending = 0
        for tr in job.items.values():
            if tr.state in (WellState.SUBMITTED, WellState.RUNNING):
                if tr.state is WellState.SUBMITTED:
                    tr.state = WellState.RUNNING
                if job.tick >= tr.ready_tick:
                    tr.result = self._finalize(tr)
                    tr.state = tr.result.state
                else:
                    n_pending += 1
            if tr.result is not None:
                results.append(tr.result)
        return JobStatus(job_id=job_id, round=job.round, done=(n_pending == 0), results=results, n_pending=n_pending)

    def is_done(self, job_id: str) -> bool:
        return all(
            tr.state in (WellState.COMPLETED, WellState.FAILED, WellState.QC_REJECTED)
            for tr in self._jobs[job_id].items.values()
        )
