# Changelog

## 0.3.0 — 2026-07-20

Real docking closed loop + engineering hardening (AL VS engine, not production robotic screening).

### Added
- `SimulatedDockExecutor` and CLI `sim_dock` (`configs/sim_dock_demo.yaml`)
- Real Vina mini closed loop: `configs/vina_mini.yaml`, `scripts/run_vina_closed_loop.py`
- Windows Vina install script + `docs/vina_setup.md` + demo receptor `data/receptors/1iep_receptor.pdbqt`
- Bench snapshots: `docs/bench/` (Replay/SimDock enrichment + Vina mini wall-clock)
- Pending-aware acquisition (Tanimoto local penalty vs in-flight compounds)

### Fixed
- Vina Observation maximize convention: `activity = -vina_affinity`
- Campaign/CLI always `close()` executors
- Vina receptor/binary early validation; OpenBabel errors include stderr
- Removed dead Campaign `max_polls` and Vina `qed_sa_lookup`
- Time-driven campaign loop; local `job_id` binding; fair Vina scheduling

### Changed
- Legacy multi-objective HV/Pareto plots under `docs/legacy_multiobjective/`
- Dropped lightweight smoke-only tests (`test_smoke`, `test_vina_real_smoke`); keep contract/integration tests

## Unreleased

(none)
