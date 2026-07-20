# AutoDock Vina setup — real docking closed loop

## Dependencies

| Component | Role | How to get it |
|-----------|------|---------------|
| `vina` / `vina.exe` | Docking engine | `scripts/install_vina_windows.ps1` → `tools/bin/vina.exe`, or put `vina` on PATH |
| OpenBabel `obabel` | SMILES→PDB→PDBQT | `pip install openbabel-wheel` |
| RDKit | Embed / optimize ligand | `pip install -e ".[prep]"` |
| Receptor `.pdbqt` | Protein target | Demo: `data/receptors/1iep_receptor.pdbqt` |

## Score convention (critical)

AutoScreen AL uses **maximize** `activity`. Vina reports affinity in kcal/mol (more negative = stronger).

```text
Observation.values[0] = activity = -vina_affinity
Observation.raw["vina_affinity"] = <raw Vina score>
```

So greedy/UCB prefer stronger binders. Replay MOO `dock` labels are a different scale — do not treat them as Vina ground truth.

## Install (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_vina_windows.ps1
pip install -e ".[prep,dev]" openbabel-wheel
```

## Punch-through closed loop (recommended)

Mini budget (~≤30 docks on Enamine10k, Abl/1iep box):

```powershell
# Ensure vina + obabel are findable
$env:Path = "$(Resolve-Path tools\bin);$env:Path"

python scripts/run_vina_closed_loop.py
# writes docs/bench/vina_mini_results.json (sync vs async wall-clock)
```

Or a single campaign:

```bash
python -m autoscreen.cli run --config configs/vina_mini.yaml
```

## Swap in your own receptor

1. Prepare receptor PDBQT (Meeko / ADFR / OpenBabel — your choice).
2. Edit `configs/vina_mini.yaml` (or copy it):

```yaml
vina:
  receptor: path/to/your_receptor.pdbqt
  box_center: [x, y, z]
  box_size: [sx, sy, sz]
  vina_bin: tools/bin/vina.exe   # or vina on PATH
```

3. Keep `batch_size` / `n_rounds` small until the pipeline is stable.

## CI / tests

- Unit tests mock `dock_fn` (no Vina required).
- Real docking tests are `@pytest.mark.skipif` when `vina` is absent.
- CI does **not** run the closed-loop script.

## Notes

- Demo path only — not a production VS cluster.
- Offline Replay / `sim_dock` campaigns do not need Vina.
