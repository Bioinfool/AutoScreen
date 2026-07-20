# Changelog

## Unreleased

### Fixed
- Candidate state always binds **local** `job_id` (never remote id) on resume
- Campaign `run()` is time-driven (`poll_interval_s` / `next_poll_after` / wall & idle timeouts) to avoid Vina busy-spin
- Replicate observations no longer demote an already `LABELED` compound
- Structured `objectives: {expensive, static}` YAML parsing
- Constraint empty-set policy is explicit (`fail_closed` default; `relax` / `fail_open`)
- Vina ligand scheduling is fair across concurrent jobs (round-robin)

### Added
- Single-objective enrichment metrics (Top-k recall, EF, BEDROC)
- Replicate aggregation into training labels with std QC
- Fail/QC retryable candidate phases
- Vina campaign integration test

### Changed
- Legacy multi-objective HV/Pareto plots moved to `results/legacy_multiobjective/`
