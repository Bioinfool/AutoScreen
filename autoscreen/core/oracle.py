"""Label oracles — the only place hidden experimental labels may live."""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoscreen.core.objectives import ObjectiveSchema, default_schema


@dataclass
class ArrayLabelOracle:
    """In-memory maximize-convention labels aligned to library pool_idx order.

    Columns follow ``schema.expensive + schema.static`` when built from MOO CSV
    (dock→activity, qed, sa→sa_ease). Campaign code must never hold this object
    for acquisition; only ReplayExecutor and BenchmarkEvaluator may.
    """

    Y: np.ndarray  # (n, n_cols) maximize
    column_names: tuple[str, ...]
    expensive_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.Y.ndim != 2:
            raise ValueError("Y must be 2-D")
        if self.Y.shape[1] != len(self.column_names):
            raise ValueError("column_names length must match Y.shape[1]")

    @property
    def n(self) -> int:
        return int(self.Y.shape[0])

    @property
    def expensive_indices(self) -> list[int]:
        return [self.column_names.index(n) for n in self.expensive_names]

    def lookup_expensive(self, pool_idx: int) -> list[float]:
        row = self.Y[int(pool_idx)]
        return [float(row[i]) for i in self.expensive_indices]

    def lookup_full(self, pool_idx: int) -> list[float]:
        return self.Y[int(pool_idx)].astype(float).tolist()

    def as_array(self) -> np.ndarray:
        return self.Y

    def expensive_array(self) -> np.ndarray:
        return self.Y[:, self.expensive_indices]


def load_moo_oracle(
    moo_csv: str | Path,
    smis: list[str],
    schema: ObjectiveSchema | None = None,
) -> tuple[ArrayLabelOracle, np.ndarray]:
    """Align MOO CSV to library SMILES order.

    Returns
    -------
    oracle
        Hidden labels (activity/qed/sa_ease maximize) for aligned rows.
    static_Y
        Columns matching ``schema.static`` for the same row order.
    """
    schema = schema or default_schema()
    moo_csv = Path(moo_csv)
    smi_to_i = {s: i for i, s in enumerate(smis)}

    # Read MOO in file order, keep only SMILES present in library, preserve moo order
    # so pool_idx matches filtered library order used by load_candidate_library.
    kept_smis: list[str] = []
    raw_rows: list[tuple[float, float, float]] = []
    with gzip.open(moo_csv, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        for r in reader:
            smi = r[0]
            if smi not in smi_to_i:
                continue
            kept_smis.append(smi)
            raw_rows.append((float(r[1]), float(r[2]), float(r[3])))

    if not kept_smis:
        raise ValueError(f"No overlapping SMILES between library and {moo_csv}")

    raw = np.asarray(raw_rows, dtype=np.float64)
    # maximize: activity=-dock, qed, sa_ease=-sa
    full = np.column_stack([-raw[:, 0], raw[:, 1], -raw[:, 2]])
    col_names = ("activity", "qed", "sa_ease")
    expensive_names = schema.expensive_names
    for name in expensive_names:
        if name not in col_names:
            raise ValueError(f"MOO oracle missing expensive objective {name!r}")

    oracle = ArrayLabelOracle(Y=full, column_names=col_names, expensive_names=expensive_names)

    static_cols = []
    name_to_col = {n: i for i, n in enumerate(col_names)}
    for spec in schema.static:
        if spec.name not in name_to_col:
            raise ValueError(f"MOO oracle missing static property {spec.name!r}")
        static_cols.append(full[:, name_to_col[spec.name]])
    static_Y = (
        np.column_stack(static_cols).astype(np.float32)
        if static_cols
        else np.zeros((len(kept_smis), 0), dtype=np.float32)
    )
    return oracle, static_Y


def filter_library_to_moo_order(
    smis: list[str],
    X: np.ndarray,
    moo_csv: str | Path,
) -> tuple[list[str], np.ndarray]:
    """Reorder/filter (smis, X) to MOO-overlap order (same as oracle rows)."""
    moo_csv = Path(moo_csv)
    smi_to_row = {s: i for i, s in enumerate(smis)}
    out_smis: list[str] = []
    rows: list[int] = []
    with gzip.open(moo_csv, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        for r in reader:
            smi = r[0]
            if smi not in smi_to_row:
                continue
            out_smis.append(smi)
            rows.append(smi_to_row[smi])
    if not out_smis:
        raise ValueError(f"No overlapping SMILES for {moo_csv}")
    return out_smis, X[rows].astype(np.float32)
