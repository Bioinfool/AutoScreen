"""Independent HTTP mock robot / HTS platform (protocol v1)."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from autoscreen.protocol.v1 import PROTOCOL_VERSION, validate_plate_payload
from robot_mock.simulator import PlateSimulator

app = FastAPI(title="AutoScreen robot_mock", version="0.2.0")


def _load_truth_moo_order() -> dict[int, list[float]]:
    """Load maximize-convention labels in Enamine10k_moo row order."""
    root = Path(__file__).resolve().parents[1]
    moo = root / "molpal" / "data" / "Enamine10k_moo.csv.gz"
    if not moo.exists():
        return {}
    truth: dict[int, list[float]] = {}
    with gzip.open(moo, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        for i, r in enumerate(reader):
            dock, qed, sa = float(r[1]), float(r[2]), float(r[3])
            truth[i] = [-dock, qed, -sa]
    return truth


SIM = PlateSimulator(truth=_load_truth_moo_order(), seed=0)


class PlateRequest(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    campaign_id: str
    round: int
    job_id: str | None = None
    plate: list[dict] = Field(default_factory=list)


@app.get("/v1/health")
def health():
    return {"status": "ok", "protocol_version": PROTOCOL_VERSION, "n_jobs": len(SIM.jobs)}


@app.post("/v1/plates")
def submit_plate(
    body: PlateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    payload = body.model_dump()
    try:
        validate_plate_payload(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    job_id = SIM.submit(payload, idempotency_key=idempotency_key)
    return {"job_id": job_id, "protocol_version": PROTOCOL_VERSION}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in SIM.jobs:
        raise HTTPException(status_code=404, detail="job not found")
    return SIM.poll(job_id)


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in SIM.jobs:
        raise HTTPException(status_code=404, detail="job not found")
    SIM.cancel(job_id)
    return {"job_id": job_id, "cancelled": True}
