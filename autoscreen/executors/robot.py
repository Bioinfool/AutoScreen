"""HTTP RobotExecutor — talks only over the network to a robot/LIMS endpoint."""
from __future__ import annotations

import time
import uuid

import httpx

from autoscreen.core.types import ItemKind, Job, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor
from autoscreen.protocol.v1 import plate_submit_payload


class RobotExecutor(Executor):
    kind = "robot"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def submit(self, job: Job) -> str:
        payload = plate_submit_payload(job.to_dict())
        headers = {}
        if job.idempotency_key:
            headers["Idempotency-Key"] = job.idempotency_key
        resp = self._client.post("/v1/plates", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["job_id"]

    def poll(self, job_id: str) -> JobStatus:
        # wall-clock pacing for async simulation
        time.sleep(self.poll_interval_s)
        resp = self._client.get(f"/v1/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        observations = []
        for r in data.get("results", []):
            state = WellState(r["state"])
            kind = ItemKind(r.get("kind", "experimental"))
            observations.append(
                Observation(
                    smiles=r.get("smiles") or "",
                    pool_idx=int(r.get("pool_idx", -1)),
                    values=r.get("values"),
                    state=state,
                    qc_passed=bool(r.get("qc_passed", False)),
                    source=self.kind,
                    compound_id=r.get("compound_id") or "",
                    item_id=r.get("item_id") or r.get("well") or "",
                    kind=kind,
                    message=r.get("message", ""),
                    raw={"well": r.get("well"), "replicate_of": r.get("replicate_of")},
                    timestamp=time.time(),
                )
            )
        return JobStatus(
            job_id=job_id,
            done=bool(data.get("done")),
            observations=observations,
            n_pending=int(data.get("n_pending", 0)),
            round=int(data.get("round", -1)),
        )

    def cancel(self, job_id: str) -> None:
        resp = self._client.post(f"/v1/jobs/{job_id}/cancel")
        resp.raise_for_status()

    def wait(self, job_id: str, max_polls: int = 100) -> JobStatus:
        status = self.poll(job_id)
        polls = 0
        while not status.done and polls < max_polls:
            polls += 1
            status = self.poll(job_id)
        if not status.done:
            raise TimeoutError(
                f"Job {job_id} not done after {max_polls} polls "
                f"(n_pending={status.n_pending})"
            )
        return status
