"""Aggregate results.json across seeds and plot strategy comparison curves.

Produces:
  results/summary.csv       : mean +/- std of hv_frac and pareto_recall per (strategy, round)
  results/hv_curve.svg      : hypervolume fraction vs molecules evaluated
  results/pareto_curve.svg  : Pareto recall vs molecules evaluated

SVG output is written by a small in-repo renderer (autoscreen.svgplot) because
matplotlib's Agg backend segfaults on some Windows/conda setups.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .svgplot import PALETTE, Series, line_chart


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def aggregate(results: list[dict]):
    # (strategy, round) -> {metric: [values across seeds]}
    by_key = defaultdict(lambda: defaultdict(list))
    n_labeled = {}
    for res in results:
        strat = res["strategy"]
        for rec in res["history"]:
            key = (strat, rec["round"])
            by_key[key]["hv_frac"].append(rec["hv_frac"])
            by_key[key]["pareto_recall"].append(rec["pareto_recall"])
            n_labeled[key] = rec["n_labeled"]
    return by_key, n_labeled


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.json")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out_dir = Path(args.out)
    results = load(Path(args.results))
    by_key, n_labeled = aggregate(results)

    strategies = sorted({s for s, _ in by_key})
    rounds = sorted({r for _, r in by_key})

    # summary csv
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy", "round", "n_labeled",
            "hv_frac_mean", "hv_frac_std",
            "pareto_recall_mean", "pareto_recall_std",
        ])
        for strat in strategies:
            for rnd in rounds:
                key = (strat, rnd)
                if key not in by_key:
                    continue
                hv = np.array(by_key[key]["hv_frac"])
                pr = np.array(by_key[key]["pareto_recall"])
                w.writerow([
                    strat, rnd, n_labeled[key],
                    f"{hv.mean():.4f}", f"{hv.std():.4f}",
                    f"{pr.mean():.4f}", f"{pr.std():.4f}",
                ])

    # plots (SVG)
    for metric, fname, ylabel in [
        ("hv_frac", "hv_curve.svg", "Hypervolume fraction of global front"),
        ("pareto_recall", "pareto_curve.svg", "Pareto-front recall"),
    ]:
        series = []
        for i, strat in enumerate(strategies):
            xs, means, los, his = [], [], [], []
            for rnd in rounds:
                key = (strat, rnd)
                if key not in by_key:
                    continue
                vals = np.array(by_key[key][metric])
                xs.append(float(n_labeled[key]))
                m = float(vals.mean())
                s = float(vals.std())
                means.append(m)
                los.append(m - s)
                his.append(m + s)
            series.append(
                Series(label=strat, xs=xs, ys=means, color=PALETTE[i % len(PALETTE)], lo=los, hi=his)
            )
        svg = line_chart(
            series,
            title=ylabel,
            xlabel="Molecules evaluated (labeling budget)",
            ylabel=ylabel,
        )
        (out_dir / fname).write_text(svg, encoding="utf-8")

    print(f"Wrote {out_dir/'summary.csv'}, {out_dir/'hv_curve.svg'}, {out_dir/'pareto_curve.svg'}")

    # final-round leaderboard
    last = max(rounds)
    print(f"\nFinal round (r{last}) leaderboard:")
    rows = []
    for strat in strategies:
        key = (strat, last)
        if key not in by_key:
            continue
        hv = np.array(by_key[key]["hv_frac"])
        pr = np.array(by_key[key]["pareto_recall"])
        rows.append((strat, hv.mean(), pr.mean()))
    for strat, hv, pr in sorted(rows, key=lambda t: -t[1]):
        print(f"  {strat:9s}  hv_frac={hv:.3f}  pareto_recall={pr:.3f}")


if __name__ == "__main__":
    main()
