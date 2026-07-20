"""Shared domain types for the decision layer and executors."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class WellState(str, Enum):
    SELECTED = "SELECTED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    QC_REJECTED = "QC_REJECTED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = {
    WellState.COMPLETED,
    WellState.FAILED,
    WellState.QC_REJECTED,
    WellState.CANCELLED,
}


class ItemKind(str, Enum):
    EXPERIMENTAL = "experimental"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BLANK = "blank"
    REPLICATE = "replicate"


@dataclass
class Molecule:
    """A library compound identified by pool index and SMILES."""

    pool_idx: int
    smiles: str
    compound_id: str = ""

    def __post_init__(self) -> None:
        if not self.compound_id:
            self.compound_id = f"CMP{self.pool_idx:07d}"


@dataclass
class JobItem:
    """One item inside a submitted evaluation job (docking or plate well)."""

    item_id: str
    smiles: str
    pool_idx: int = -1
    kind: ItemKind = ItemKind.EXPERIMENTAL
    compound_id: str = ""
    well_id: Optional[str] = None
    replicate_of: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """A batch evaluation request handed to an Executor."""

    job_id: str
    campaign_id: str
    round: int
    items: list[JobItem]
    executor_kind: str
    idempotency_key: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [
            {**asdict(it), "kind": it.kind.value if isinstance(it.kind, ItemKind) else it.kind}
            for it in self.items
        ]
        return d


@dataclass
class Observation:
    """One measured / computed evaluation result (maximize convention for values)."""

    smiles: str
    pool_idx: int
    values: Optional[list[float]]
    state: WellState
    qc_passed: bool = False
    source: str = ""
    compound_id: str = ""
    item_id: str = ""
    kind: ItemKind = ItemKind.EXPERIMENTAL
    raw: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: float = 0.0

    @property
    def usable(self) -> bool:
        return (
            self.state is WellState.COMPLETED
            and self.qc_passed
            and self.values is not None
            and self.kind is ItemKind.EXPERIMENTAL
            and self.pool_idx >= 0
        )


@dataclass
class JobStatus:
    job_id: str
    done: bool
    observations: list[Observation] = field(default_factory=list)
    n_pending: int = 0
    round: int = -1
    message: str = ""


@dataclass
class BatchSelection:
    """Indices (into the unlabeled pool / library) chosen for the next round."""

    pool_indices: list[int]
    scores: Optional[list[float]] = None
    strategy: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
