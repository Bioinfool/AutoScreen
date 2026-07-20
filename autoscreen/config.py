"""YAML config loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    data["_config_path"] = str(path.resolve())
    return data


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
