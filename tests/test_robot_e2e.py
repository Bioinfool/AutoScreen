"""End-to-end AL against robot_mock over real HTTP (Starlette TestClient)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.constraints import ConstraintManager, PlateConfig
from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.core.types import ItemKind, Job, JobStatus, Observation, WellState
from autoscreen.executors.base import Executor
from autoscreen.protocol.v1 import PROTOCOL_VERSION, plate_submit_payload
from robot_mock.app import app


@pytest.fixture()
def robot_client():
    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["protocol_version"] == PROTOCOL_VERSION
    return client


def test_robot_protocol_submit_and_poll(robot_client):
    payload = {
        "protocol_version": "v1",
        "campaign_id": "c1",
        "round": 1,
        "plate": [
            {
                "well": "A01",
                "item_id": "i1",
                "kind": "experimental",
                "compound_id": "CMP0000000",
                "smiles": "CCO",
                "pool_idx": 0,
                "replicate_of": None,
            }
        ],
    }
    r = robot_client.post("/v1/plates", json=payload, headers={"Idempotency-Key": "abc"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    r2 = robot_client.post("/v1/plates", json=payload, headers={"Idempotency-Key": "abc"})
    assert r2.json()["job_id"] == jid

    done = False
    st = None
    for _ in range(10):
        st = robot_client.get(f"/v1/jobs/{jid}").json()
        if st["done"]:
            done = True
            break
    assert done
    assert st["results"][0]["state"] in ("COMPLETED", "FAILED", "QC_REJECTED")
    if st["results"][0]["values"] is not None:
        assert len(st["results"][0]["values"]) == 1


class ASGIRobotExecutor(Executor):
    kind = "robot"

    def __init__(self, client: TestClient):
        self.client = client

    def submit(self, job: Job) -> str:
        payload = plate_submit_payload(job.to_dict())
        headers = {}
        if job.idempotency_key:
            headers["Idempotency-Key"] = job.idempotency_key
        resp = self.client.post("/v1/plates", json=payload, headers=headers)
        assert resp.status_code == 200, resp.text
        return resp.json()["job_id"]

    def poll(self, job_id: str) -> JobStatus:
        resp = self.client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        observations = []
        for r in data.get("results", []):
            observations.append(
                Observation(
                    smiles=r.get("smiles") or "",
                    pool_idx=int(r.get("pool_idx", -1)),
                    values=r.get("values"),
                    state=WellState(r["state"]),
                    qc_passed=bool(r.get("qc_passed", False)),
                    source=self.kind,
                    compound_id=r.get("compound_id") or "",
                    item_id=r.get("item_id") or r.get("well") or "",
                    kind=ItemKind(r.get("kind", "experimental")),
                    message=r.get("message", ""),
                    raw={"well": r.get("well")},
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
        self.client.post(f"/v1/jobs/{job_id}/cancel")


def test_robot_campaign_http(tmp_path, robot_client):
    root = Path(__file__).resolve().parents[1]
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    plate = PlateConfig(n_experimental=20, n_positive=2, n_negative=2, n_blank=2, n_replicate=2)
    ex = ASGIRobotExecutor(robot_client)
    camp = CampaignManager(
        library=lib,
        executor=ex,
        acquisition="ucb",
        campaign_id="robot_e2e",
        seed=0,
        batch_size=20,
        init_frac=0.005,
        checkpoint_dir=tmp_path / "robot_ckpt",
        n_estimators=15,
        plate=plate,
        constraints=ConstraintManager(plate=plate, static_sa_ease=lib.static_col("sa_ease")),
        use_plate_layout=True,
        beta=0.5,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=1,
        schema=lib.schema,
        positive_idx=[0, 1],
        negative_idx=[2, 3],
    )
    hist = camp.run(1)
    assert camp.state.round == 1
    assert len(camp.store) > 10
    assert hist[-1]["submitted"] == plate.plate_size
