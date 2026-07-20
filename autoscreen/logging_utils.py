"""Logging setup for AutoScreen."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger("autoscreen")
    if root.handlers:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"autoscreen.{name}")
