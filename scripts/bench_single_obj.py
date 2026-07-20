#!/usr/bin/env python
"""Single-objective VS benchmark: random / greedy / UCB × sync vs async.

Writes JSON summary under results/ (gitignored) and prints a compact table.

Example:
  python scripts/bench_single_obj.py --rounds 3 --batch-size 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.executors.replay import ReplayExecutor


def _run_one(
    *,
    root: Path,
    out_dir: Path,
    acquisition: str,
    mode: str,
    rounds: int,
    batch_size: int,
    init_frac: float,
    seed: int,
    n_estimators: int,
    pending_penalty: float,
) -> dict:
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    async_mode = mode == "async"
    ex = ReplayExecutor(
        oracle,
        seed=seed,
        min_latency=1,
        max_latency=3 if async_mode else 1,
        stagger=async_mode,
    )
    ckpt = out_dir / f"{acquisition}_{mode}_s{seed}"
    if ckpt.exists():
        # allow re-runs in fresh dirs only
        import shutil

        shutil.rmtree(ckpt)
    t0 = time.perf_counter()
    camp = CampaignManager(
        library=lib,
        executor=ex,
        acquisition=acquisition,
        campaign_id=f"bench_{acquisition}_{mode}",
        seed=seed,
        batch_size=batch_size,
        init_frac=init_frac,
        checkpoint_dir=ckpt,
        n_estimators=n_estimators,
        evaluator=BenchmarkEvaluator(oracle),
        max_active_jobs=2 if async_mode else 1,
        schema=lib.schema,
        pending_penalty=pending_penalty if async_mode else 0.0,
        poll_interval_s=0.0,
        beta=0.5,
    )
    hist = camp.run(rounds)
    wall = time.perf_counter() - t0
    m = camp._metrics_dict()
    return {
        "acquisition": acquisition,
        "mode": mode,
        "seed": seed,
        "rounds": rounds,
        "batch_size": batch_size,
        "n_labeled": len(camp.store),
        "wall_clock_s": round(wall, 3),
        "top01_recall": m.get("top01_recall"),
        "top1_recall": m.get("top1_recall"),
        "ef_top1": m.get("ef_top1"),
        "bedroc": m.get("bedroc"),
        "n_hits_top1": m.get("n_hits_top1"),
        "mean_activity": m.get("mean_activity"),
        "history_len": len(hist),
        "max_active_jobs": 2 if async_mode else 1,
        "pending_penalty": pending_penalty if async_mode else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--init-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-estimators", type=int, default=40)
    p.add_argument("--pending-penalty", type=float, default=0.5)
    p.add_argument(
        "--acquisitions",
        default="random,greedy,ucb",
        help="Comma-separated acquisition names",
    )
    p.add_argument("--modes", default="sync,async")
    p.add_argument("--out", default="results/bench_single_obj.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for acq in [a.strip() for a in args.acquisitions.split(",") if a.strip()]:
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            print(f"Running {acq} / {mode} ...", flush=True)
            row = _run_one(
                root=root,
                out_dir=out.parent / "bench_runs",
                acquisition=acq,
                mode=mode,
                rounds=args.rounds,
                batch_size=args.batch_size,
                init_frac=args.init_frac,
                seed=args.seed,
                n_estimators=args.n_estimators,
                pending_penalty=args.pending_penalty,
            )
            rows.append(row)
            print(
                f"  labeled={row['n_labeled']} wall={row['wall_clock_s']:.2f}s "
                f"top1={row['top1_recall']:.3f} EF={row['ef_top1']:.2f}",
                flush=True,
            )

    payload = {"version": "single_obj_v1", "rows": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
