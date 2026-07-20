"""Build a multi-objective label table for Enamine10k.

Objectives (raw values; direction handled downstream):
  - dock : docking score from the precomputed score table (lower = better)
  - qed  : QED drug-likeness (higher = better)
  - sa   : synthetic accessibility (lower = easier to make)

Output: data/Enamine10k_moo.csv.gz  with columns [smiles, dock, qed, sa]
"""
import csv
import gzip
import os
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import QED, RDConfig
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402


def load_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with gzip.open(path, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        for row in reader:
            try:
                scores[row[0]] = float(row[1])
            except (ValueError, IndexError):
                pass
    return scores


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    library = root / "data" / "Enamine10k.csv.gz"
    scores_path = root / "data" / "Enamine10k_scores.csv.gz"
    out_path = root / "data" / "Enamine10k_moo.csv.gz"

    dock = load_scores(scores_path)

    with gzip.open(library, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        smis = [row[0] for row in reader]

    rows: list[tuple[str, float, float, float]] = []
    n_bad = 0
    for smi in tqdm(smis, desc="QED/SA", unit="smi"):
        if smi not in dock:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_bad += 1
            continue
        try:
            qed = QED.qed(mol)
            sa = sascorer.calculateScore(mol)
        except Exception:
            n_bad += 1
            continue
        rows.append((smi, dock[smi], qed, sa))

    with gzip.open(out_path, "wt", newline="") as fid:
        writer = csv.writer(fid)
        writer.writerow(["smiles", "dock", "qed", "sa"])
        writer.writerows(rows)

    print(f"Wrote {out_path} with {len(rows)} molecules ({n_bad} skipped)")
    dvals = [r[1] for r in rows]
    qvals = [r[2] for r in rows]
    svals = [r[3] for r in rows]
    print(f"dock: [{min(dvals):.2f}, {max(dvals):.2f}]")
    print(f"qed : [{min(qvals):.3f}, {max(qvals):.3f}]")
    print(f"sa  : [{min(svals):.2f}, {max(svals):.2f}]")


if __name__ == "__main__":
    main()
