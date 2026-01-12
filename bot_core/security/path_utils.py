"""Path utilities for security modules."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_tilde_path(path: str | Path) -> Path:
    path_str = str(path)
    if path_str.startswith("~"):
        if len(path_str) > 1 and path_str[1] not in ("/", "\\"):
            raise ValueError("Nieobsługiwany prefiks ~user w ścieżce.")
        home = os.environ.get("HOME") or os.path.expanduser("~")
        suffix = path_str[1:]
        if suffix in ("", "/", "\\"):
            return Path(home)
        return Path(home) / suffix.lstrip("/\\")
    return Path(path)
