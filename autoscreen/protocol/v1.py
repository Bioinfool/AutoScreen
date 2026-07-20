"""Robot protocol v1 payload schemas and helpers."""
from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = "v1"


def plate_submit_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Build POST /v1/plates body from a Job.to_dict()-like mapping."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "campaign_id": job["campaign_id"],
        "round": job["round"],
        "job_id": job.get("job_id"),
        "plate": [
            {
                "well": it.get("well_id") or it["item_id"],
                "item_id": it["item_id"],
                "kind": it["kind"],
                "compound_id": it.get("compound_id") or None,
                "smiles": it.get("smiles") or None,
                "pool_idx": it.get("pool_idx", -1),
                "replicate_of": it.get("replicate_of"),
            }
            for it in job["items"]
        ],
    }


def validate_plate_payload(payload: dict[str, Any]) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol_version: {payload.get('protocol_version')}")
    for key in ("campaign_id", "round", "plate"):
        if key not in payload:
            raise ValueError(f"Missing field: {key}")
    if not isinstance(payload["plate"], list) or not payload["plate"]:
        raise ValueError("plate must be a non-empty list")
