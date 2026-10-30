"""Public, side-effect-free path helpers shared by tools and file operations."""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]|^\\")  # "C:\\…", "\\Windows\\…" (rooted) and UNC "\\\\server\\…"


def resolve(path: str) -> Path:
    """Natural-language or literal path → absolute Path ('downloads', '~/x.txt', 'notes.txt on my desktop')."""
    from automation.files_read import resolve_user_system_path

    raw = str(path or "").strip().strip("'\"")
    if _WIN_ABS.match(raw):  # "C:\\…" is absolute even when tests run on Linux — never guess
        return Path(PureWindowsPath(raw).as_posix()) if os.name != "nt" else Path(os.path.abspath(raw))
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    if p.is_absolute():
        return p
    if p.root:  # "/etc/x" or "\\Windows\\x": rooted but driveless on Windows → the current drive, never a guess
        return Path(os.path.abspath(p))
    return resolve_user_system_path(raw)
