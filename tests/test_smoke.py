"""Smoke test for package importability (phase 0)."""


def test_version_and_types_import():
    import autoscreen
    from autoscreen.core.types import Job, Molecule, Observation, WellState

    assert autoscreen.__version__
    m = Molecule(pool_idx=1, smiles="CCO")
    assert m.compound_id.startswith("CMP")
    assert WellState.COMPLETED.value == "COMPLETED"
    assert Job
    assert Observation
