"""Filesystem locations used by JARVIS.

Every persistent file lives under ``data_dir()`` so nothing depends on the
current working directory. Override the location with ``JARVIS_DATA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    path = Path(os.environ.get("JARVIS_DATA_DIR") or (BASE_DIR / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts: str) -> Path:
    """Return a path inside the data directory, creating parent folders."""
    path = data_dir().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = Path(os.environ.get("JARVIS_LOG_DIR") or (BASE_DIR / "logs"))
    path.mkdir(parents=True, exist_ok=True)
    return path
