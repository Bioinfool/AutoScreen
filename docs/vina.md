"""docs: how to run AutoScreen with AutoDock Vina.

## Prerequisites

1. Install [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) and ensure `vina` is on PATH.
2. Install OpenBabel (`obabel`) for ligand PDB → PDBQT conversion.
3. Prepare a receptor `.pdbqt` file and set the docking box in `configs/vina_demo.yaml`.

## Config

```yaml
executor: vina
vina:
  receptor: /path/to/receptor.pdbqt
  box_center: [x, y, z]
  box_size: [20, 20, 20]
  vina_bin: vina
```

## Run

```bash
autoscreen run --config configs/vina_demo.yaml
```

If `receptor` is null or `vina` is missing, the executor raises a clear `RuntimeError`.
Offline Replay campaigns do not need Vina.
"""
