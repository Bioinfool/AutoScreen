"""Precompute Enamine10k fingerprint HDF5 (serial, no Ray).

Uses hashed atom-pair fingerprints (radius 2, 2048 bits).
"""
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
    root = Path(__file__).resolve().parents[1]
    library = root / "data" / "Enamine10k.csv.gz"
    output = root / "data" / "Enamine10k.h5"

    smis = load_smis(library)
    size = len(smis)

    invalid_idxs: set[int] = set()
    fps_rows: list[np.ndarray] = []

    for idx, smi in enumerate(tqdm(smis, desc="Featurizing", unit="smi")):
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
        print(f"invalid_idxs = {sorted(invalid_idxs)}")


if __name__ == "__main__":
    main()
