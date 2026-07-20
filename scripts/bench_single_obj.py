#!/usr/bin/env python
"""Single-objective VS benchmark: enrichment + async throughput.

Two suites (use ``--suite both`` by default):

1. **enrichment** — ``ReplayExecutor`` (tick latency): random / greedy / UCB
   enrichment metrics on Enamine10k.
2. **throughput** — ``SimulatedDockExecutor`` (wall-clock sleep + thread pool):
   sync vs async Campaign scheduling. Labels still come from the Replay oracle;
   this does **not** require a Vina binary.

Writes JSON under ``docs/bench/`` (committed summaries) and optionally
``results/`` for local scratch.

Example:
  python scripts/bench_single_obj.py --suite both --rounds 5 --batch-size 50
  python scripts/bench_single_obj.py --suite throughput --ligand-latency 0.05
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from autoscreen.core.benchmark import BenchmarkEvaluator
from autoscreen.core.campaign import CampaignManager
from autoscreen.core.library import load_candidate_library
from autoscreen.core.oracle import load_moo_oracle
from autoscreen.executors.replay import ReplayExecutor
from autoscreen.executors.sim_dock import SimDockConfig, SimulatedDockExecutor


def _load(root: Path):
    lib = load_candidate_library(
        root / "data/Enamine10k.csv.gz",
        root / "data/Enamine10k.h5",
        moo_csv=root / "data/Enamine10k_moo.csv.gz",
    )
    oracle, _ = load_moo_oracle(root / "data/Enamine10k_moo.csv.gz", lib.smis, schema=lib.schema)
    return lib, oracle


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metrics_row(camp: CampaignManager, *, wall: float, **extra) -> dict:
    m = camp._metrics_dict()
    return {
        **extra,
        "n_labeled": len(camp.store),
        "wall_clock_s": round(wall, 3),
        "top01_recall": m.get("top01_recall"),
        "top1_recall": m.get("top1_recall"),
        "ef_top1": m.get("ef_top1"),
        "bedroc": m.get("bedroc"),
        "n_hits_top1": m.get("n_hits_top1"),
        "mean_activity": m.get("mean_activity"),
        "history_len": len(camp.state.history),
    }


def run_enrichment(
    *,
    root: Path,
    run_root: Path,
    acquisitions: list[str],
    rounds: int,
    batch_size: int,
    init_frac: float,
    seed: int,
    n_estimators: int,
) -> list[dict]:
    rows = []
    for acq in acquisitions:
        print(f"[enrichment] {acq} ...", flush=True)
        lib, oracle = _load(root)
        ex = ReplayExecutor(oracle, seed=seed, min_latency=1, max_latency=1, stagger=False)
        ckpt = _fresh(run_root / f"enrich_{acq}_s{seed}")
        t0 = time.perf_counter()
        camp = CampaignManager(
            library=lib,
            executor=ex,
            acquisition=acq,
            campaign_id=f"enrich_{acq}",
            seed=seed,
            batch_size=batch_size,
            init_frac=init_frac,
            checkpoint_dir=ckpt,
            n_estimators=n_estimators,
            evaluator=BenchmarkEvaluator(oracle),
            max_active_jobs=1,
            schema=lib.schema,
            pending_penalty=0.0,
            poll_interval_s=0.0,
            beta=0.5,
        )
        camp.run(rounds)
        wall = time.perf_counter() - t0
        row = _metrics_row(
            camp,
            wall=wall,
            suite="enrichment",
            acquisition=acq,
            mode="sync",
            seed=seed,
            rounds=rounds,
            batch_size=batch_size,
            executor="replay",
        )
        rows.append(row)
        print(
            f"  labeled={row['n_labeled']} wall={row['wall_clock_s']:.2f}s "
            f"top1={row['top1_recall']:.3f} EF={row['ef_top1']:.2f} "
            f"BEDROC={row['bedroc']:.3f}",
            flush=True,
        )
    return rows


def run_throughput(
    *,
    root: Path,
    run_root: Path,
    acquisition: str,
    rounds: int,
    batch_size: int,
    init_frac: float,
    seed: int,
    n_estimators: int,
    ligand_latency: float,
    pending_penalty: float,
) -> list[dict]:
    configs = [
        ("sync", 1, 1, 0.0),
        ("async", 4, 2, pending_penalty),
    ]
    rows = []
    for mode, workers, max_jobs, pen in configs:
        print(f"[throughput] {mode} workers={workers} max_jobs={max_jobs} ...", flush=True)
        lib, oracle = _load(root)
        ex = SimulatedDockExecutor(
            oracle,
            SimDockConfig(latency_s=ligand_latency, max_workers=workers, poll_hint_s=0.01),
        )
        ckpt = _fresh(run_root / f"thru_{mode}_s{seed}")
        t0 = time.perf_counter()
        try:
            camp = CampaignManager(
                library=lib,
                executor=ex,
                acquisition=acquisition,
                campaign_id=f"thru_{mode}",
                seed=seed,
                batch_size=batch_size,
                init_frac=init_frac,
                checkpoint_dir=ckpt,
                n_estimators=n_estimators,
                evaluator=BenchmarkEvaluator(oracle),
                max_active_jobs=max_jobs,
                schema=lib.schema,
                pending_penalty=pen,
                poll_interval_s=0.0,
                beta=0.5,
            )
            camp.run(rounds)
            wall = time.perf_counter() - t0
            row = _metrics_row(
                camp,
                wall=wall,
                suite="throughput",
                acquisition=acquisition,
                mode=mode,
                seed=seed,
                rounds=rounds,
                batch_size=batch_size,
                executor="sim_dock",
                ligand_latency_s=ligand_latency,
                max_workers=workers,
                max_active_jobs=max_jobs,
                pending_penalty=pen,
            )
        finally:
            ex.close()
        rows.append(row)
        print(
            f"  labeled={row['n_labeled']} wall={row['wall_clock_s']:.2f}s "
            f"top1={row['top1_recall']:.3f}",
            flush=True,
        )
    if len(rows) == 2 and rows[0]["wall_clock_s"] > 0:
        speedup = rows[0]["wall_clock_s"] / max(rows[1]["wall_clock_s"], 1e-9)
        rows[1]["speedup_vs_sync"] = round(speedup, 3)
        print(f"  async speedup vs sync: {speedup:.2f}x", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("enrichment", "throughput", "both"), default="both")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--init-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-estimators", type=int, default=40)
    p.add_argument("--pending-penalty", type=float, default=0.5)
    p.add_argument("--ligand-latency", type=float, default=0.05)
    p.add_argument(
        "--throughput-acquisition",
        default="random",
        help="Prefer a cheap acquisition so wall-clock reflects executor parallelism",
    )
    p.add_argument("--acquisitions", default="random,greedy,ucb")
    p.add_argument("--out", default="docs/bench/single_obj_results.json")
    p.add_argument("--run-dir", default="results/bench_runs")
    p.add_argument(
        "--throughput-pending-penalty",
        type=float,
        default=0.0,
        help="Pending penalty for throughput suite (0 isolates scheduling)",
    )
    p.add_argument("--throughput-n-estimators", type=int, default=20)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    run_root = root / args.run_dir
    run_root.mkdir(parents=True, exist_ok=True)

    acqs = [a.strip() for a in args.acquisitions.split(",") if a.strip()]
    rows: list[dict] = []
    if args.suite in ("enrichment", "both"):
        rows.extend(
            run_enrichment(
                root=root,
                run_root=run_root,
                acquisitions=acqs,
                rounds=args.rounds,
                batch_size=args.batch_size,
                init_frac=args.init_frac,
                seed=args.seed,
                n_estimators=args.n_estimators,
            )
        )
    if args.suite in ("throughput", "both"):
        # Docking-dominated settings: larger latency, modest AL overhead
        t_batch = args.batch_size if args.suite == "throughput" else min(args.batch_size, 40)
        t_rounds = args.rounds if args.suite == "throughput" else min(args.rounds, 4)
        t_latency = args.ligand_latency
        if args.suite == "both" and t_latency < 0.1:
            t_latency = 0.12  # ensure sleep dominates RF when running combined suite
        rows.extend(
            run_throughput(
                root=root,
                run_root=run_root,
                acquisition=args.throughput_acquisition,
                rounds=t_rounds,
                batch_size=t_batch,
                init_frac=args.init_frac,
                seed=args.seed,
                n_estimators=args.throughput_n_estimators,
                ligand_latency=t_latency,
                pending_penalty=args.throughput_pending_penalty,
            )
        )

    payload = {
        "version": "single_obj_v2",
        "library": "Enamine10k",
        "notes": [
            "enrichment uses ReplayExecutor (oracle scores, tick latency).",
            "throughput uses SimulatedDockExecutor (oracle scores + wall-clock sleep + thread pool).",
            "throughput is a scheduling proxy for Vina async; not real docking.",
        ],
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
