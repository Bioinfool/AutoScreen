"""Precompute atom-pair fingerprint HDF5 for a library under data/.

Usage:
  python scripts/prep_fps_serial.py --name Enamine10k
  python scripts/prep_fps_serial.py --name Enamine50k
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import h5py
import numpy as np
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

RADIUS = 2
LENGTH = 2048


def featurize(smi: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = rdmd.GetHashedAtomPairFingerprintAsBitVect(
        mol, minLength=1, maxLength=1 + RADIUS, nBits=LENGTH
    )
    x = np.empty(len(fp))
    ConvertToNumpyArray(fp, x)
    return x


def load_smis(library: Path) -> list[str]:
    with gzip.open(library, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        return [row[0] for row in reader]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Enamine10k", help="Library stem under data/")
    p.add_argument("--library", default=None, help="Override library csv.gz path")
    p.add_argument("--output", default=None, help="Override output .h5 path")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    library = Path(args.library) if args.library else root / "data" / f"{args.name}.csv.gz"
    output = Path(args.output) if args.output else root / "data" / f"{args.name}.h5"

    smis = load_smis(library)
    size = len(smis)

    invalid_idxs: set[int] = set()
    fps_rows: list[np.ndarray] = []

    for idx, smi in enumerate(tqdm(smis, desc=f"Featurizing {args.name}", unit="smi")):
        fp = featurize(smi)
        if fp is None:
            invalid_idxs.add(idx)
        else:
            fps_rows.append(fp.astype(np.int8))

    valid_size = size - len(invalid_idxs)
    with h5py.File(output, "w") as h5f:
        dset = h5f.create_dataset(
            "fps", (valid_size, LENGTH), chunks=(512, LENGTH), dtype="int8"
        )
        dset[:] = np.stack(fps_rows)

    print(f"Wrote {output} ({valid_size} fps, {len(invalid_idxs)} invalid)")
    if invalid_idxs:
        print(f"invalid_idxs = {sorted(invalid_idxs)[:20]}{'...' if len(invalid_idxs) > 20 else ''}")


if __name__ == "__main__":
    main()
