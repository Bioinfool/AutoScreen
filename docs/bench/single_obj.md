# Single-objective benchmark (Enamine10k)

Frozen snapshot from `scripts/bench_single_obj.py` (seed=0). Raw JSON: [`single_obj_results.json`](single_obj_results.json).

## 1. Enrichment (Replay oracle)

Settings: 5 rounds × batch 50, `init_frac=0.005`, ~302 labels, RF 40 trees.

| Acquisition | Top-1% recall | EF@1% | BEDROC | Top-0.1% recall |
|-------------|---------------|-------|--------|-----------------|
| random      | 0.029         | 0.99  | 0.028  | 0.000           |
| greedy      | 0.200         | 6.92  | 0.200  | 0.273           |
| ucb         | 0.219         | 7.58  | 0.223  | 0.273           |

UCB ≥ greedy ≫ random under the same labeling budget.

## 2. Throughput (SimulatedDock)

Proxy for Vina-style async scheduling: oracle scores + fixed per-ligand sleep (`0.12s`) + thread pool. **Not real docking.**

Settings: 4 rounds × batch 40, random acquisition, RF 20 trees, `pending_penalty=0`.

| Mode  | workers | max_active_jobs | wall-clock (s) | speedup |
|-------|---------|-----------------|----------------|---------|
| sync  | 1       | 1               | 28.17          | 1.0×    |
| async | 4       | 2               | 7.19           | **3.92×** |

When per-ligand latency dominates surrogate cost, Campaign async + worker pool recovers near-linear speedup vs serial.

## Reproduce

```bash
pip install -e ".[dev]"
python scripts/bench_single_obj.py --suite both --rounds 5 --batch-size 50
```

Suites: `--suite enrichment|throughput|both`.

## Claims / non-claims

- Claim: AL policies beat random on this Replay oracle; async orchestration speeds wall-clock under simulated docking latency.
- Non-claim: real AutoDock Vina wall-clock (binary not required for this bench); production robotic screening.
