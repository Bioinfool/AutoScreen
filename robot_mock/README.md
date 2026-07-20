# robot_mock: independent HTTP service simulating a robotic HTS / LIMS endpoint.

Run locally:

```bash
uvicorn robot_mock.app:app --host 0.0.0.0 --port 8080
```

Endpoints (protocol v1):

- `GET  /v1/health`
- `POST /v1/plates`  (header `Idempotency-Key` optional)
- `GET  /v1/jobs/{job_id}`
- `POST /v1/jobs/{job_id}/cancel`

This service is a **simulator**. Connecting AutoScreen here proves the orchestration
and protocol layer; it does not mean a real robot assay was performed.
