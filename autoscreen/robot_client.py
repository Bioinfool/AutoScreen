"""Robot / LIMS client interface.

The campaign submits a `PlateLayout` and later fetches results through a
`RobotClient`. Today the only implementation is `MockRobotClient`, which drives
the in-process `MockRobotBackend`. When the lab's platform is ready, implement
`RobotClient` against the real HTTP/queue API (see `HttpRobotClient` for the
payload contract) and the campaign code does not change at all.

Submission payload contract (JSON), matching the lab-facing spec:

    {
      "campaign_id": "screen_001",
      "round": 2,
      "plate": [
        {"well": "A01", "kind": "experimental",
         "compound_id": "CMP0007", "smiles": "..."},
        {"well": "H09", "kind": "blank", "compound_id": null, "smiles": null}
      ]
    }

Result payload contract (JSON):

    {
      "job_id": "screen_001-r2",
      "done": true,
      "results": [
        {"compound_id": "CMP0007", "state": "COMPLETED",
         "values": [activity, qed, sa_ease], "qc_passed": true,
         "replicate_of": null}
      ]
    }
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .experiment import (
    CompoundResult,
    ExperimentBackend,
    JobStatus,
    SubmittedCompound,
    WellState,
)
from .plate import PlateLayout


def layout_to_payload(campaign_id: str, layout: PlateLayout) -> dict:
    return {
        "campaign_id": campaign_id,
        "round": layout.round,
        "plate": [
            {
                "well": w.well_id,
                "kind": w.kind,
                "compound_id": None if w.pool_idx is None else f"CMP{w.pool_idx:07d}",
                "smiles": w.smiles,
                "replicate_of": w.replicate_of,
            }
            for w in layout.wells
        ],
    }


def _layout_to_submitted(layout: PlateLayout) -> list[SubmittedCompound]:
    out: list[SubmittedCompound] = []
    for w in layout.wells:
        if w.pool_idx is None:
            cid = f"BLANK-{w.well_id}"
        else:
            cid = f"CMP{w.pool_idx:07d}"
        out.append(
            SubmittedCompound(
                well_id=w.well_id,
                compound_id=cid,
                smiles=w.smiles or "",
                pool_idx=-1 if w.pool_idx is None else w.pool_idx,
                kind=w.kind,
                replicate_of=w.replicate_of,
            )
        )
    return out


class RobotClient(ABC):
    @abstractmethod
    def submit_plate(self, campaign_id: str, layout: PlateLayout) -> str:
        """Submit a plate; return a job id used to fetch results later."""

    @abstractmethod
    def fetch_results(self, job_id: str) -> JobStatus:
        """Poll the platform for the current job status/results."""

    @abstractmethod
    def is_done(self, job_id: str) -> bool:
        ...


class MockRobotClient(RobotClient):
    """In-process client backed by MockRobotBackend (no network)."""

    def __init__(self, backend: ExperimentBackend):
        self.backend = backend

    def submit_plate(self, campaign_id: str, layout: PlateLayout) -> str:
        job_id = f"{campaign_id}-r{layout.round}"
        self.backend.submit(job_id, layout.round, _layout_to_submitted(layout))
        return job_id

    def fetch_results(self, job_id: str) -> JobStatus:
        return self.backend.poll(job_id)

    def is_done(self, job_id: str) -> bool:
        return self.backend.is_done(job_id)


class HttpRobotClient(RobotClient):
    """Stub for a real lab platform reachable over HTTP.

    Not exercised in simulation. It documents exactly which endpoints and payload
    shapes a real integration needs, so wiring up the lab API is a drop-in.
    Uses only the standard library (urllib) to avoid adding dependencies.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit_plate(self, campaign_id: str, layout: PlateLayout) -> str:
        import json
        import urllib.request

        payload = layout_to_payload(campaign_id, layout)
        req = urllib.request.Request(
            f"{self.base_url}/plates",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["job_id"]

    def fetch_results(self, job_id: str) -> JobStatus:
        import json
        import urllib.request

        with urllib.request.urlopen(f"{self.base_url}/jobs/{job_id}", timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        results = [
            CompoundResult(
                well_id=r.get("well", r.get("well_id", "")),
                compound_id=r["compound_id"],
                smiles=r.get("smiles", ""),
                state=WellState(r["state"]),
                pool_idx=r.get("pool_idx", -1),
                kind=r.get("kind", "experimental"),
                values=r.get("values"),
                qc_passed=r.get("qc_passed", False),
                replicate_of=r.get("replicate_of"),
                message=r.get("message", ""),
            )
            for r in data.get("results", [])
        ]
        return JobStatus(
            job_id=job_id,
            round=data.get("round", -1),
            done=data.get("done", False),
            results=results,
            n_pending=data.get("n_pending", 0),
        )

    def is_done(self, job_id: str) -> bool:
        return self.fetch_results(job_id).done
