#!/usr/bin/env python
"""Run a minimal real-Vina closed loop: sync vs async wall-clock.

Requires: vina binary, OpenBabel (obabel), RDKit, demo receptor.

Example:
  python scripts/run_vina_closed_loop.py
  python scripts/run_vina_closed_loop.py --out docs/bench/vina_mini_results.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _have_obabel() -> bool:
    return bool(shutil.which("obabel") or shutil.which("obabel.exe"))


def _resolve_vina(root: Path) -> Path | None:
    tools = root / "tools" / "bin" / "vina.exe"
    if tools.is_file():
        return tools
    found = shutil.which("vina") or shutil.which("vina.exe")
    return Path(found) if found else None


def _preflight(root: Path) -> list[str]:
    errs: list[str] = []
    if _resolve_vina(root) is None:
        errs.append("Vina binary missing (run scripts/install_vina_windows.ps1 or put vina on PATH)")
    if not _have_obabel():
        errs.append("OpenBabel missing (pip install openbabel-wheel)")
    receptor = root / "data" / "receptors" / "1iep_receptor.pdbqt"
    if not receptor.is_file():
        errs.append(f"Receptor not found: {receptor}")
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        errs.append("RDKit missing (pip install -e \".[prep]\")")
    return errs


def _run_mode(
    *,
    root: Path,
    cfg: dict,
    mode: str,
    workers: int,
    max_jobs: int,
) -> dict:
    from autoscreen.cli import build_from_config

    c = deepcopy(cfg)
    c["campaign_id"] = f"vina_mini_{mode}"
    c["checkpoint_dir"] = str(root / "runs" / f"vina_mini_{mode}")
    c.setdefault("vina", {})
    c["vina"]["max_workers"] = workers
    c["vina"]["work_dir"] = str(root / "runs" / f"vina_mini_work_{mode}")
    vina_bin = _resolve_vina(root)
    assert vina_bin is not None
    c["vina"]["vina_bin"] = str(vina_bin)
    c.setdefault("async", {})
    c["async"]["max_active_jobs"] = max_jobs
    c["async"]["pending_penalty"] = 0.0

    ckpt = Path(c["checkpoint_dir"])
    if ckpt.exists():
        shutil.rmtree(ckpt)
    work = Path(c["vina"]["work_dir"])
    if work.exists():
        shutil.rmtree(work)

    camp = build_from_config(c)
    t0 = time.perf_counter()
    try:
        n_rounds = int(c.get("n_rounds", 2))
        hist = camp.run(n_rounds)
        wall = time.perf_counter() - t0
        mean_act = None
        if len(camp.store) > 0:
            _, Y = camp.store.matrix(camp.library.X, camp.n_obj)
            if Y.size:
                mean_act = round(float(Y[:, 0].mean()), 4)
        return {
            "mode": mode,
            "max_workers": workers,
            "max_active_jobs": max_jobs,
            "n_rounds": n_rounds,
            "batch_size": int(c.get("batch_size", 0)),
            "init_frac": float(c.get("init_frac", 0)),
            "n_labeled": len(camp.store),
            "wall_clock_s": round(wall, 3),
            "mean_activity": mean_act,
            "history_len": len(hist),
            "checkpoint": str(camp.checkpoint_dir),
        }
    finally:
        camp.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/vina_mini.yaml")
    p.add_argument("--out", default="docs/bench/vina_mini_results.json")
    args = p.parse_args()

    root = ROOT
    errs = _preflight(root)
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Install steps: docs/vina_setup.md  |  scripts/install_vina_windows.ps1",
            file=sys.stderr,
        )
        return 2

    from autoscreen.config import load_config

    cfg = load_config(root / args.config)
    rows = []
    print("Running sync (workers=1, max_jobs=1) ...", flush=True)
    sync = _run_mode(root=root, cfg=cfg, mode="sync", workers=1, max_jobs=1)
    rows.append(sync)
    print(
        f"  labeled={sync['n_labeled']} wall={sync['wall_clock_s']:.1f}s "
        f"mean_activity={sync['mean_activity']}",
        flush=True,
    )

    print("Running async (workers=2, max_jobs=2) ...", flush=True)
    async_row = _run_mode(root=root, cfg=cfg, mode="async", workers=2, max_jobs=2)
    if sync["wall_clock_s"] > 0:
        async_row["speedup_vs_sync"] = round(sync["wall_clock_s"] / max(async_row["wall_clock_s"], 1e-9), 3)
    rows.append(async_row)
    print(
        f"  labeled={async_row['n_labeled']} wall={async_row['wall_clock_s']:.1f}s "
        f"mean_activity={async_row['mean_activity']} "
        f"speedup={async_row.get('speedup_vs_sync')}",
        flush=True,
    )

    payload = {
        "version": "vina_mini_v1",
        "receptor": "data/receptors/1iep_receptor.pdbqt",
        "notes": [
            "Observation.activity = -vina_affinity (maximize convention).",
            "Wall-clock compares VinaExecutor thread-pool scheduling, not Replay.",
            "MOO/Replay labels are not used as docking ground truth here.",
        ],
        "rows": rows,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
