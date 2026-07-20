"""Persistent job ledger for async submit/poll and crash recovery."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoscreen.core.types import ItemKind, Job, JobItem, Observation, WellState


class JobLifecycle(str, Enum):
    QUEUED = "QUEUED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobRecord:
    job: Job
    remote_job_id: str
    status: JobLifecycle = JobLifecycle.SUBMITTED
    submitted_at: float = field(default_factory=time.time)
    last_poll_at: float = 0.0
    seen_item_ids: set[str] = field(default_factory=set)
    retry_count: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "remote_job_id": self.remote_job_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "last_poll_at": self.last_poll_at,
            "seen_item_ids": sorted(self.seen_item_ids),
            "retry_count": self.retry_count,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobRecord":
        jd = d["job"]
        items = [
            JobItem(
                item_id=it["item_id"],
                smiles=it.get("smiles", ""),
                pool_idx=int(it.get("pool_idx", -1)),
                kind=ItemKind(it.get("kind", "experimental")),
                compound_id=it.get("compound_id", ""),
                well_id=it.get("well_id"),
                replicate_of=it.get("replicate_of"),
                meta=it.get("meta") or {},
            )
            for it in jd.get("items", [])
        ]
        job = Job(
            job_id=jd["job_id"],
            campaign_id=jd["campaign_id"],
            round=int(jd["round"]),
            items=items,
            executor_kind=jd.get("executor_kind", ""),
            idempotency_key=jd.get("idempotency_key", ""),
            meta=jd.get("meta") or {},
        )
        return cls(
            job=job,
            remote_job_id=d["remote_job_id"],
            status=JobLifecycle(d.get("status", "SUBMITTED")),
            submitted_at=float(d.get("submitted_at", 0.0)),
            last_poll_at=float(d.get("last_poll_at", 0.0)),
            seen_item_ids=set(d.get("seen_item_ids") or []),
            retry_count=int(d.get("retry_count", 0)),
            message=d.get("message", ""),
        )


class JobStore:
    def __init__(self) -> None:
        self._by_id: dict[str, JobRecord] = {}

    def __len__(self) -> int:
        return len(self._by_id)

    def put(self, rec: JobRecord) -> None:
        self._by_id[rec.job.job_id] = rec

    def get(self, job_id: str) -> JobRecord:
        return self._by_id[job_id]

    def open_jobs(self) -> list[JobRecord]:
        return [
            r
            for r in self._by_id.values()
            if r.status in (JobLifecycle.QUEUED, JobLifecycle.SUBMITTED, JobLifecycle.RUNNING)
        ]

    def all(self) -> list[JobRecord]:
        return list(self._by_id.values())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": [r.to_dict() for r in self._by_id.values()]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "JobStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        for d in data.get("jobs", []):
            rec = JobRecord.from_dict(d)
            store.put(rec)
        return store


def observation_from_dict(d: dict[str, Any]) -> Observation:
    return Observation(
        smiles=d.get("smiles", ""),
        pool_idx=int(d.get("pool_idx", -1)),
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


def observation_to_dict(o: Observation) -> dict[str, Any]:
    d = asdict(o)
    d["state"] = o.state.value
    d["kind"] = o.kind.value
    return d
