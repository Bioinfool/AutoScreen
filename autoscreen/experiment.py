"""Experiment state machine and the abstract experiment backend interface.

This is the seam that lets AutoScreen submit a *plate* of compounds to an
experimental resource (docking, a robot, or a lab LIMS) and collect results
asynchronously. The active-learning loop never talks to a backend directly; it
talks to this interface, so a mock backend today can be swapped for a real
robot/LIMS client later without touching the campaign logic.

Lifecycle of a single well/compound:

    SELECTED -> SUBMITTED -> RUNNING -> COMPLETED
                                     -> FAILED        (hardware / assay error)
                          -> QC_REJECTED             (result failed quality control)

Only COMPLETED + qc_passed results are allowed into the surrogate training set.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class WellState(str, Enum):
    SELECTED = "SELECTED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    QC_REJECTED = "QC_REJECTED"


TERMINAL_STATES = {WellState.COMPLETED, WellState.FAILED, WellState.QC_REJECTED}


@dataclass
class CompoundResult:
    """One measured well coming back from the backend.

    `values` holds the measured objectives in AutoScreen's maximize convention
    (activity, qed, sa_ease). `qc_passed` gates entry into training. `well_id` is
    unique per plate; `pool_idx` maps back to the library row (-1 for blanks).
    """

    well_id: str
    compound_id: str
    smiles: str
    state: WellState
    pool_idx: int = -1
    kind: str = "experimental"
    values: Optional[list[float]] = None
    qc_passed: bool = False
    replicate_of: Optional[str] = None
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.state is WellState.COMPLETED and self.qc_passed and self.values is not None


@dataclass
class SubmittedCompound:
    """A well handed to the backend as part of a submission.

    `well_id` is the unique plate coordinate (e.g. "A01"); `compound_id` is the
    chemical identifier and may repeat across wells (replicates share it).
    """

    well_id: str
    compound_id: str
    smiles: str
    pool_idx: int
    kind: str = "experimental"  # experimental | positive | negative | blank | replicate
    replicate_of: Optional[str] = None


@dataclass
class JobStatus:
    job_id: str
    round: int
    done: bool
    results: list[CompoundResult] = field(default_factory=list)
    n_pending: int = 0


class ExperimentBackend(ABC):
    """Abstract asynchronous experiment resource.

    Implementations must be safe to poll repeatedly and must eventually drive
    every submitted compound to a terminal state.
    """

    @abstractmethod
    def submit(self, job_id: str, round: int, compounds: list[SubmittedCompound]) -> None:
        """Accept a plate of compounds for (asynchronous) evaluation."""

    @abstractmethod
    def poll(self, job_id: str) -> JobStatus:
        """Return current status; advances any internal simulated clock."""

    @abstractmethod
    def is_done(self, job_id: str) -> bool:
        """True once every compound in the job has reached a terminal state."""
