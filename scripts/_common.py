"""Shared helpers for the entry-point scripts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_split(name_or_path: str) -> list[Path]:
    """Resolve a split file to a list of absolute HDF5 paths.

    Accepts a bare split name (looked up under ``data/splits/``) or a path.
    Each non-empty line is a path relative to the release root (or absolute).
    """
    p = Path(name_or_path)
    if not p.exists():
        p = ROOT / "data" / "splits" / name_or_path
    sessions = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sp = Path(line)
        sessions.append(sp if sp.is_absolute() else ROOT / sp)
    return sessions
