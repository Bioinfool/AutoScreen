"""Multi-objective active-learning virtual screening loop.

Compares acquisition strategies on the hidden-label Enamine10k benchmark:
each round selects a batch, "reveals" its true (dock, qed, sa) labels, retrains
the per-objective surrogates, and records hypervolume / Pareto recovery.

Usage:
    python -m autoscreen.run --strategy pareto --seeds 3 --out results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .acquire import (
    select_greedy,
    select_pareto_hvi,
    select_random,
    select_ucb,
    select_weighted,
)
from .data import OBJECTIVE_NAMES, load_moo_dataset
from .metrics import hypervolume, make_ref_point, pareto_mask
from .model import MultiOutputRFSurrogate

STRATEGIES = ("random", "greedy", "weighted", "ucb", "pareto")


def run_campaign(
    strategy: str,
    seed: int,
    init_frac: float,
    batch_frac: float,
    n_iters: int,
    beta: float,
    candidate_cap: int,
    verbose: bool = True,
) -> dict:
    ds = load_moo_dataset()
    rng = np.random.default_rng(seed)

    n = ds.n
    init_k = max(1, int(init_frac * n))
    batch_k = max(1, int(batch_frac * n))

    ref_point = make_ref_point(ds.Y)
    global_front = pareto_mask(ds.Y)
    global_front_idx = set(np.where(global_front)[0].tolist())
    global_hv = hypervolume(ds.Y[global_front], ref_point)

    weights = np.ones(ds.n_objectives) / ds.n_objectives

    labeled = np.zeros(n, dtype=bool)
    init_sel = rng.choice(n, size=init_k, replace=False)
    labeled[init_sel] = True

    history = []

    def log_round(rnd: int):
        Y_lab = ds.Y[labeled]
        hv = hypervolume(Y_lab, ref_point)
        found_front = len(global_front_idx & set(np.where(labeled)[0].tolist()))
        rec = {
            "round": rnd,
            "n_labeled": int(labeled.sum()),
            "hv": hv,
            "hv_frac": hv / global_hv if global_hv > 0 else 0.0,
            "pareto_recovered": found_front,
            "pareto_recall": found_front / max(1, len(global_front_idx)),
        }
        history.append(rec)
        if verbose:
            print(
                f"[{strategy} s{seed}] r{rnd:02d} "
                f"n={rec['n_labeled']:5d} hv_frac={rec['hv_frac']:.3f} "
                f"pareto={found_front}/{len(global_front_idx)}"
            )

    log_round(0)

    for rnd in range(1, n_iters + 1):
        pool_idx = np.where(~labeled)[0]
        if len(pool_idx) == 0:
            break

        if strategy == "random":
            chosen = select_random(rng, pool_idx, batch_k)
        else:
            model = MultiOutputRFSurrogate(ds.n_objectives, seed=seed)
            model.fit(ds.X[labeled], ds.Y[labeled])
            means, stds = model.predict(ds.X[pool_idx])
            norm_range = (ds.Y[labeled].min(axis=0), ds.Y[labeled].max(axis=0))

            if strategy == "greedy":
                chosen = select_greedy(means, pool_idx, batch_k, weights, norm_range)
            elif strategy == "weighted":
                chosen = select_weighted(rng, means, pool_idx, batch_k, norm_range)
            elif strategy == "ucb":
                chosen = select_ucb(means, stds, pool_idx, batch_k, weights, norm_range, beta)
            elif strategy == "pareto":
                chosen = select_pareto_hvi(
                    means, stds, pool_idx, batch_k, ref_point,
                    ds.Y[labeled], beta=beta, candidate_cap=candidate_cap,
                )
            else:
                raise ValueError(f"unknown strategy {strategy}")

        labeled[chosen] = True
        log_round(rnd)

    return {
        "strategy": strategy,
        "seed": seed,
        "objectives": list(OBJECTIVE_NAMES),
        "global_hv": global_hv,
        "n_global_pareto": len(global_front_idx),
        "history": history,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-objective AL virtual screening")
    p.add_argument("--strategy", choices=STRATEGIES + ("all",), default="all")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--init-frac", type=float, default=0.01)
    p.add_argument("--batch-frac", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--candidate-cap", type=int, default=400)
    p.add_argument("--out", type=str, default="results")
    args = p.parse_args()

    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for strat in strategies:
        for seed in range(args.seeds):
            res = run_campaign(
                strat, seed, args.init_frac, args.batch_frac,
                args.iters, args.beta, args.candidate_cap,
            )
            all_results.append(res)

    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_file} ({len(all_results)} campaigns)")


if __name__ == "__main__":
    main()
