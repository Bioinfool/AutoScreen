# Changelog

## Unreleased

### Added
- Pending-aware acquisition: Tanimoto local penalty vs in-flight compounds
- `scripts/bench_single_obj.py` for sync vs async single-objective comparison
- Ruff (critical) + coverage in CI; LICENSE / CHANGELOG / CITATION already present

### Fixed
- Candidate state always binds **local** `job_id` (never remote id) on resume
- Campaign `run()` is time-driven (`poll_interval_s` / `next_poll_after` / wall & idle timeouts) to avoid Vina busy-spin
- Replicate observations no longer demote an already `LABELED` compound
- Structured `objectives: {expensive, static}` YAML parsing
- Constraint empty-set policy is explicit (`fail_closed` default; `relax` / `fail_open`)
- Vina ligand scheduling is fair across concurrent jobs (round-robin)
- Windows-safe pending Tanimoto (no large `@` matmul)

### Changed
- Legacy multi-objective HV/Pareto plots archived under `docs/legacy_multiobjective/`
