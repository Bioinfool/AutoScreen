"""Executor abstract base: submit / poll / cancel."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from autoscreen.core.types import Job, JobStatus


class JobNotFound(KeyError):
    """Remote / in-process executor has no record of this job_id."""


class Executor(ABC):
    """Evaluation backend interface used by CampaignManager."""

    kind: str = "base"

    @abstractmethod
    def submit(self, job: Job) -> str:
        """Accept a job; return job_id (may honor job.idempotency_key)."""

    @abstractmethod
    def poll(self, job_id: str) -> JobStatus:
        """Return current status including any available observations."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Best-effort cancellation."""

    def close(self) -> None:
        """Release pools / HTTP clients. Default no-op."""

    def wait(self, job_id: str, max_polls: int = 100) -> JobStatus:
        status = self.poll(job_id)
        polls = 0
        while not status.done and polls < max_polls:
            sleep_s = float(getattr(status, "next_poll_after", 0.0) or 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)
            polls += 1
            status = self.poll(job_id)
        if not status.done:
            raise TimeoutError(
                f"Job {job_id} not done after {max_polls} polls "
                f"(n_pending={status.n_pending})"
            )
        return status
