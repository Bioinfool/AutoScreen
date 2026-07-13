"""Precompute MolPAL fingerprint HDF5 without Ray (Windows-friendly)."""
import csv
import gzip
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from molpal.featurizer import Featurizer, featurize


def load_smis(library: Path) -> list[str]:
    with gzip.open(library, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        return [row[0] for row in reader]


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "molpal"
    library = root / "libraries" / "Enamine10k.csv.gz"
    output = root / "libraries" / "Enamine10k.h5"

    featurizer = Featurizer(fingerprint="pair", radius=2, length=2048)
    smis = load_smis(library)
    size = len(smis)

    invalid_idxs: set[int] = set()
    fps_rows: list[np.ndarray] = []

    for idx, smi in enumerate(tqdm(smis, desc="Featurizing", unit="smi")):
        fp = featurize(smi, featurizer.fingerprint, featurizer.radius, len(featurizer))
        if fp is None:
            invalid_idxs.add(idx)
        else:
            fps_rows.append(fp.astype(np.int8))

    valid_size = size - len(invalid_idxs)
    with h5py.File(output, "w") as h5f:
        dset = h5f.create_dataset(
            "fps",
            (valid_size, len(featurizer)),
            chunks=(512, len(featurizer)),
            dtype="int8",
        )
        dset[:] = np.stack(fps_rows)

    print(f"Wrote {output} ({valid_size} fps, {len(invalid_idxs)} invalid)")
    if invalid_idxs:
        print(f"invalid_idxs = {sorted(invalid_idxs)}")


if __name__ == "__main__":
    main()
