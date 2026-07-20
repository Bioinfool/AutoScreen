# Real Vina mini closed loop (Abl / 1iep)

Frozen snapshot from `scripts/run_vina_closed_loop.py`. Raw JSON: [`vina_mini_results.json`](vina_mini_results.json).

## Setup

- Receptor: `data/receptors/1iep_receptor.pdbqt`
- Library: Enamine10k (`init_frac=0.001`, batch 4, 2 rounds) → **18** docks
- `activity = -vina_affinity` (maximize)

## Wall-clock (this machine)

| Mode | workers | max_jobs | labeled | wall-clock (s) | mean activity | speedup |
|------|---------|----------|---------|----------------|---------------|---------|
| sync | 1 | 1 | 18 | 97.6 | 8.56 | 1.0× |
| async | 2 | 2 | 18 | 76.5 | 8.86 | **1.28×** |

Small budgets are overhead-limited (surrogate + ligand prep); larger batches / more workers increase the async gap. Numbers are machine-dependent.

## Reproduce

```bash
python scripts/run_vina_closed_loop.py
```

Requires Vina + OpenBabel + RDKit (see [`../vina_setup.md`](../vina_setup.md)).
