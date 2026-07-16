"""Load fingerprints and the multi-objective label table, aligned by SMILES.

All objectives are converted to a *maximize* convention internally:
  - activity:  -dock   (docking is "lower is better")
  - qed:        qed    (already "higher is better")
  - sa_ease:   -sa     (synthetic accessibility is "lower is easier")
"""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

OBJECTIVE_NAMES = ("activity", "qed", "sa_ease")


@dataclass
class MOODataset:
    smis: list[str]
    X: np.ndarray  # (n, n_bits) fingerprints
    Y: np.ndarray  # (n, 3) raw-but-maximize objective values
    raw: np.ndarray  # (n, 3) original dock/qed/sa for reporting

    @property
    def n(self) -> int:
        return len(self.smis)

    @property
    def n_objectives(self) -> int:
        return self.Y.shape[1]


def load_moo_dataset(
    molpal_root: str | Path = None,
    library_csv: str | Path = None,
    fps_h5: str | Path = None,
    moo_csv: str | Path = None,
) -> MOODataset:
    root = Path(molpal_root) if molpal_root else Path(__file__).resolve().parents[1] / "molpal"
    library_csv = Path(library_csv) if library_csv else root / "libraries" / "Enamine10k.csv.gz"
    fps_h5 = Path(fps_h5) if fps_h5 else root / "libraries" / "Enamine10k.h5"
    moo_csv = Path(moo_csv) if moo_csv else root / "data" / "Enamine10k_moo.csv.gz"

    # library order == fps row order
    with gzip.open(library_csv, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        lib_smis = [row[0] for row in reader]

    with h5py.File(fps_h5, "r") as f:
        fps = f["fps"][:]

    if len(lib_smis) != fps.shape[0]:
        raise ValueError(f"library ({len(lib_smis)}) and fps ({fps.shape[0]}) length mismatch")

    smi_to_row = {smi: i for i, smi in enumerate(lib_smis)}

    smis: list[str] = []
    rows: list[int] = []
    raw: list[tuple[float, float, float]] = []
    with gzip.open(moo_csv, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        for r in reader:
            smi = r[0]
            if smi not in smi_to_row:
                continue
            smis.append(smi)
            rows.append(smi_to_row[smi])
            raw.append((float(r[1]), float(r[2]), float(r[3])))

    X = fps[rows].astype(np.float32)
    raw_arr = np.asarray(raw, dtype=np.float64)  # [dock, qed, sa]

    Y = np.column_stack([-raw_arr[:, 0], raw_arr[:, 1], -raw_arr[:, 2]])  # maximize convention

    return MOODataset(smis=smis, X=X, Y=Y, raw=raw_arr)
