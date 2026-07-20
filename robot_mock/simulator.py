"""In-process plate simulator used by the robot_mock HTTP service."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WellTrack:
    well: str
    item_id: str
    kind: str
    compound_id: str | None
    smiles: str | None
    pool_idx: int
    replicate_of: str | None
    ready_tick: int
    state: str = "SUBMITTED"
    values: list[float] | None = None
    qc_passed: bool = False
    message: str = ""


@dataclass
class JobTrack:
    job_id: str
    campaign_id: str
    round: int
    tick: int = 0
    wells: dict[str, WellTrack] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class PlateSimulator:
    def __init__(
        self,
        truth: dict[int, list[float]] | None = None,
        fail_rate: float = 0.05,
        qc_reject_rate: float = 0.05,
        activity_noise: float = 0.05,
        min_latency: int = 1,
        max_latency: int = 3,
        seed: int = 0,
    ):
        self.truth = truth or {}
        self.fail_rate = fail_rate
        self.qc_reject_rate = qc_reject_rate
        self.activity_noise = activity_noise
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.rng = np.random.default_rng(seed)
        self.jobs: dict[str, JobTrack] = {}
        self.idempotency: dict[str, str] = {}
        vals = list(self.truth.values())
        self._act_scale = float(np.std([v[0] for v in vals])) if vals else 1.0

    def submit(self, payload: dict, idempotency_key: str | None = None) -> str:
        if idempotency_key and idempotency_key in self.idempotency:
            return self.idempotency[idempotency_key]
        job_id = payload.get("job_id") or f"robot-{uuid.uuid4().hex[:12]}"
        wells: dict[str, WellTrack] = {}
        for w in payload["plate"]:
            latency = int(self.rng.integers(self.min_latency, self.max_latency + 1))
            well_id = w["well"]
            wells[well_id] = WellTrack(
                well=well_id,
                item_id=w["item_id"],
                kind=w["kind"],
                compound_id=w.get("compound_id"),
                smiles=w.get("smiles"),
                pool_idx=int(w.get("pool_idx", -1)),
                replicate_of=w.get("replicate_of"),
                ready_tick=latency,
            )
        self.jobs[job_id] = JobTrack(
            job_id=job_id,
            campaign_id=payload["campaign_id"],
            round=int(payload["round"]),
            wells=wells,
        )
        if idempotency_key:
            self.idempotency[idempotency_key] = job_id
        return job_id

    def _finalize(self, w: WellTrack) -> None:
        if w.kind == "blank" or w.pool_idx < 0:
            w.state = "COMPLETED"
            w.qc_passed = False
            w.message = "blank control"
            return
        if self.rng.random() < self.fail_rate:
            w.state = "FAILED"
            w.qc_passed = False
            w.message = "assay/hardware failure"
            return
        base = list(self.truth.get(w.pool_idx, [0.0, 0.5, -3.0]))
        base[0] = float(base[0]) + float(self.rng.normal(0.0, self.activity_noise * self._act_scale))
        qc = self.rng.random() >= self.qc_reject_rate
        w.values = base
        w.qc_passed = qc
        w.state = "COMPLETED" if qc else "QC_REJECTED"
        w.message = "ok" if qc else "qc rejected"

    def poll(self, job_id: str) -> dict:
        job = self.jobs[job_id]
        job.tick += 1
        pending = 0
        results = []
        for w in job.wells.values():
            if w.state in ("SUBMITTED", "RUNNING"):
                if w.state == "SUBMITTED":
                    w.state = "RUNNING"
                if job.tick >= w.ready_tick:
                    self._finalize(w)
                else:
                    pending += 1
            results.append(
                {
                    "well": w.well,
                    "item_id": w.item_id,
                    "kind": w.kind,
                    "compound_id": w.compound_id,
                    "smiles": w.smiles,
                    "pool_idx": w.pool_idx,
                    "replicate_of": w.replicate_of,
                    "state": w.state,
                    "values": w.values,
                    "qc_passed": w.qc_passed,
                    "message": w.message,
                }
            )
        return {
            "protocol_version": "v1",
            "job_id": job_id,
            "campaign_id": job.campaign_id,
            "round": job.round,
            "done": pending == 0,
            "n_pending": pending,
            "results": results,
        }

    def cancel(self, job_id: str) -> None:
        job = self.jobs[job_id]
        for w in job.wells.values():
            if w.state not in ("COMPLETED", "FAILED", "QC_REJECTED", "CANCELLED"):
                w.state = "CANCELLED"
                w.message = "cancelled"
