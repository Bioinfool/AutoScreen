"""robot_mock must not invent labels for unknown pool indices."""
from robot_mock.simulator import PlateSimulator


def test_missing_truth_marks_failed():
    sim = PlateSimulator(truth={0: [1.0, 0.5, -2.0]}, fail_rate=0.0, qc_reject_rate=0.0, seed=0)
    jid = sim.submit(
        {
            "job_id": "j1",
            "campaign_id": "c",
            "round": 1,
            "plate": [
                {
                    "well": "A01",
                    "item_id": "i0",
                    "kind": "experimental",
                    "pool_idx": 0,
                    "smiles": "CCO",
                },
                {
                    "well": "A02",
                    "item_id": "i1",
                    "kind": "experimental",
                    "pool_idx": 999,
                    "smiles": "CCC",
                },
            ],
        }
    )
    st = None
    for _ in range(5):
        st = sim.poll(jid)
        if st["done"]:
            break
    assert st["done"]
    by_well = {r["well"]: r for r in st["results"]}
    assert by_well["A01"]["state"] == "COMPLETED"
    assert by_well["A02"]["state"] == "FAILED"
    assert "missing truth" in by_well["A02"]["message"]


def test_executor_wait_timeout():
    from autoscreen.core.types import JobStatus
    from autoscreen.executors.base import Executor

    class NeverDone(Executor):
        kind = "stub"

        def submit(self, job):
            return "x"

        def poll(self, job_id):
            return JobStatus(job_id=job_id, done=False, n_pending=1)

        def cancel(self, job_id):
            return None

    try:
        NeverDone().wait("x", max_polls=2)
        assert False, "expected TimeoutError"
    except TimeoutError as e:
        assert "not done" in str(e)
