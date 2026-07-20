"""Independent HTTP mock robot / HTS platform (protocol v1)."""
from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from autoscreen.protocol.v1 import PROTOCOL_VERSION, validate_plate_payload
from robot_mock.simulator import PlateSimulator

app = FastAPI(title="AutoScreen robot_mock", version="0.2.0")


def _default_truth_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "Enamine10k_moo.csv.gz"


def load_truth_moo(path: Path | None = None) -> dict[int, list[float]]:
    """Load maximize-convention labels in MOO CSV row order (= campaign pool_idx)."""
    moo = path or Path(os.environ.get("AUTOSCREEN_TRUTH_MOO", str(_default_truth_path())))
    if not moo.is_absolute():
        moo = Path(__file__).resolve().parents[1] / moo
    if not moo.exists():
        raise FileNotFoundError(
            f"robot_mock truth file not found: {moo}. "
            "Set AUTOSCREEN_TRUTH_MOO to the same moo_csv used by the campaign."
        )
    truth: dict[int, list[float]] = {}
    with gzip.open(moo, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        for i, r in enumerate(reader):
            dock = float(r[1])
            # Expensive objective only (activity = -dock). QED/SA stay on the library.
            truth[i] = [-dock]
    return truth


SIM = PlateSimulator(truth=load_truth_moo(), seed=0)


class PlateRequest(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    campaign_id: str
    round: int
    job_id: str | None = None
    plate: list[dict] = Field(default_factory=list)


@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "n_jobs": len(SIM.jobs),
        "n_truth": len(SIM.truth),
    }


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
