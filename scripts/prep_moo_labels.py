"""Build multi-objective label tables (dock + QED + SA).

Usage:
  python scripts/prep_moo_labels.py --name Enamine10k
  python scripts/prep_moo_labels.py --name Enamine50k
"""
from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Enamine10k", help="Dataset stem under data/")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    library = root / "data" / f"{args.name}.csv.gz"
    scores_path = root / "data" / f"{args.name}_scores.csv.gz"
    out_path = root / "data" / f"{args.name}_moo.csv.gz"

    dock = load_scores(scores_path)

    with gzip.open(library, "rt") as fid:
        reader = csv.reader(fid)
        next(reader)
        smis = [row[0] for row in reader]

    rows: list[tuple[str, float, float, float]] = []
    n_bad = 0
    for smi in tqdm(smis, desc=f"QED/SA {args.name}", unit="smi"):
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
    if rows:
        dvals = [r[1] for r in rows]
        qvals = [r[2] for r in rows]
        svals = [r[3] for r in rows]
        print(f"dock: [{min(dvals):.2f}, {max(dvals):.2f}]")
        print(f"qed : [{min(qvals):.3f}, {max(qvals):.3f}]")
        print(f"sa  : [{min(svals):.2f}, {max(svals):.2f}]")


if __name__ == "__main__":
    main()
