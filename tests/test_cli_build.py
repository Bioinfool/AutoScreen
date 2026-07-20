"""CLI build_from_config: vina early checks + sim_dock wiring."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from autoscreen.cli import build_from_config, _resolve_vina_bin
from autoscreen.config import project_root


def _base_data(root: Path) -> dict:
    return {
        "library_csv": str(root / "data/Enamine10k.csv.gz"),
        "fps_h5": str(root / "data/Enamine10k.h5"),
        "moo_csv": str(root / "data/Enamine10k_moo.csv.gz"),
    }


def test_resolve_vina_bin_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="Vina binary not found"):
        _resolve_vina_bin(str(tmp_path / "no_such_vina.exe"), project_root())


def test_vina_missing_receptor_file_raises():
    root = project_root()
    cfg = {
        "executor": "vina",
        "campaign_id": "vina_bad",
        "data": _base_data(root),
        "checkpoint_dir": str(root / "runs" / "_test_vina_missing_rec"),
        "vina": {
            "receptor": "data/receptors/does_not_exist.pdbqt",
            "vina_bin": "vina",
        },
    }
    with pytest.raises(ValueError, match="receptor not found"):
        build_from_config(cfg)


def test_vina_missing_receptor_key_raises():
    root = project_root()
    cfg = {
        "executor": "vina",
        "data": _base_data(root),
        "vina": {},
    }
    with pytest.raises(ValueError, match="receptor is required"):
        build_from_config(cfg)


def test_sim_dock_build_from_config(tmp_path: Path):
    root = project_root()
    cfg = {
        "executor": "sim_dock",
        "campaign_id": "sim_cli",
        "seed": 0,
        "batch_size": 8,
        "init_frac": 0.001,
        "n_rounds": 1,
        "acquisition": "random",
        "data": _base_data(root),
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "sim_dock": {"latency_s": 0.0, "max_workers": 2, "poll_hint_s": 0.0},
        "async": {"max_active_jobs": 1, "pending_penalty": 0.0},
        "surrogate": {"n_estimators": 10},
    }
    camp = build_from_config(cfg)
    try:
        assert camp.executor.kind == "sim_dock"
        hist = camp.run(1)
        assert camp.state.round == 1
        assert len(camp.store) > 0
        assert isinstance(hist, list)
    finally:
        camp.close()


def test_sim_dock_demo_yaml_loads():
    root = project_root()
    path = root / "configs/sim_dock_demo.yaml"
    assert path.is_file()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["executor"] == "sim_dock"


def test_demo_receptor_checked_in():
    root = project_root()
    assert (root / "data/receptors/1iep_receptor.pdbqt").is_file()
    assert (root / "tools/bin/.gitkeep").is_file()
