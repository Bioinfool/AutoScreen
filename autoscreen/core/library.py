"""Candidate molecule library aligned with fingerprints and optional MOO labels."""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from autoscreen.logging_utils import get_logger

OBJECTIVE_NAMES = ("activity", "qed", "sa_ease")
log = get_logger("library")


@dataclass
class CandidateLibrary:
    smis: list[str]
    X: np.ndarray  # (n, n_bits)
    Y_hidden: np.ndarray | None = None  # maximize convention; None if unknown
    raw: np.ndarray | None = None  # original dock/qed/sa if available

    @property
    def n(self) -> int:
        return len(self.smis)

    @property
    def n_bits(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_objectives(self) -> int:
        if self.Y_hidden is None:
            return 0
        return int(self.Y_hidden.shape[1])

    def smiles_at(self, idxs: list[int] | np.ndarray) -> list[str]:
        return [self.smis[int(i)] for i in idxs]

    def fingerprint_at(self, idxs: list[int] | np.ndarray) -> np.ndarray:
        return self.X[np.asarray(idxs, dtype=int)]


def load_candidate_library(
    library_csv: str | Path,
    fps_h5: str | Path,
    moo_csv: str | Path | None = None,
) -> CandidateLibrary:
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
        return CandidateLibrary(smis=lib_smis, X=fps.astype(np.float32))

    smi_to_row = {smi: i for i, smi in enumerate(lib_smis)}
    smis: list[str] = []
    rows: list[int] = []
    raw: list[tuple[float, float, float]] = []
    n_moo = 0
    n_moo_missing = 0
    with gzip.open(moo_csv, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        for r in reader:
            n_moo += 1
            smi = r[0]
            if smi not in smi_to_row:
                n_moo_missing += 1
                continue
            smis.append(smi)
            rows.append(smi_to_row[smi])
            raw.append((float(r[1]), float(r[2]), float(r[3])))

    n_lib = len(lib_smis)
    n_kept = len(smis)
    if n_kept < n_lib or n_moo_missing:
        log.warning(
            "Aligned library to MOO labels: lib=%d moo=%d kept=%d "
            "(dropped_from_lib=%d, moo_not_in_lib=%d). "
            "pool_idx is the kept-row order, not original CSV line numbers.",
            n_lib,
            n_moo,
            n_kept,
            n_lib - n_kept,
            n_moo_missing,
        )
    if n_kept == 0:
        raise ValueError(f"No overlapping SMILES between {library_csv} and {moo_csv}")

    X = fps[rows].astype(np.float32)
    raw_arr = np.asarray(raw, dtype=np.float64)
    Y = np.column_stack([-raw_arr[:, 0], raw_arr[:, 1], -raw_arr[:, 2]])
    return CandidateLibrary(smis=smis, X=X, Y_hidden=Y, raw=raw_arr)
