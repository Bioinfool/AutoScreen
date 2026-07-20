"""Executor abstract base: submit / poll / cancel."""
from __future__ import annotations

from abc import ABC, abstractmethod

from autoscreen.core.types import Job, JobStatus


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
