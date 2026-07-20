"""Candidate molecule library: SMILES, fingerprints, and static properties only."""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from autoscreen.core.objectives import ObjectiveSchema, default_schema
from autoscreen.core.oracle import filter_library_to_moo_order, load_moo_oracle
from autoscreen.logging_utils import get_logger

log = get_logger("library")


@dataclass
class CandidateLibrary:
    """Decision-layer library. Must never carry hidden expensive labels."""

    smis: list[str]
    X: np.ndarray  # (n, n_bits)
    static_Y: np.ndarray | None = None  # (n, n_static)
    static_names: tuple[str, ...] = ()
    schema: ObjectiveSchema = field(default_factory=default_schema)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.smis)

    @property
    def n_bits(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_objectives(self) -> int:
        """Number of expensive objectives optimized by the surrogate."""
        return self.schema.n_expensive

    def static_col(self, name: str) -> np.ndarray:
        if self.static_Y is None or name not in self.static_names:
            raise KeyError(f"Static property {name!r} not on library")
        return self.static_Y[:, self.static_names.index(name)]

    def smiles_at(self, idxs: list[int] | np.ndarray) -> list[str]:
        return [self.smis[int(i)] for i in idxs]

    def fingerprint_at(self, idxs: list[int] | np.ndarray) -> np.ndarray:
        return self.X[np.asarray(idxs, dtype=int)]


def load_candidate_library(
    library_csv: str | Path,
    fps_h5: str | Path,
    *,
    moo_csv: str | Path | None = None,
    schema: ObjectiveSchema | None = None,
) -> CandidateLibrary:
    """Load fingerprints (+ optional static props from MOO CSV).

    When ``moo_csv`` is provided, the library is filtered/reordered to the MOO
    overlap order and QED/SA (per schema.static) are attached as ``static_Y``.
    Docking / activity labels are **not** stored on the library.
    """
    schema = schema or default_schema()
    library_csv = Path(library_csv)
    fps_h5 = Path(fps_h5)

    with gzip.open(library_csv, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        lib_smis = [row[0] for row in reader]

    with h5py.File(fps_h5, "r") as f:
        fps = f["fps"][:]

    if len(lib_smis) != fps.shape[0]:
        raise ValueError(
            f"library ({len(lib_smis)}) and fps ({fps.shape[0]}) length mismatch"
        )

    if moo_csv is None:
        return CandidateLibrary(
            smis=lib_smis,
            X=fps.astype(np.float32),
            schema=schema,
        )

    moo_csv = Path(moo_csv)
    smis, X = filter_library_to_moo_order(lib_smis, fps, moo_csv)
    # Build oracle only to extract static columns; discard oracle handle here.
    _oracle, static_Y = load_moo_oracle(moo_csv, lib_smis, schema=schema)
    if len(smis) != static_Y.shape[0]:
        raise RuntimeError("Internal error: static_Y / library length mismatch")

    n_dropped = len(lib_smis) - len(smis)
    if n_dropped:
        log.warning(
            "Aligned library to MOO for static props: lib=%d kept=%d dropped=%d. "
            "pool_idx is kept-row order. Expensive labels are NOT stored on the library.",
            len(lib_smis),
            len(smis),
            n_dropped,
        )

    return CandidateLibrary(
        smis=smis,
        X=X,
        static_Y=static_Y,
        static_names=schema.static_names,
        schema=schema,
        meta={"moo_csv": str(moo_csv), "source_library": str(library_csv)},
    )
