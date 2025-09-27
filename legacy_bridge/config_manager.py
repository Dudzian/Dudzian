"""Wejściowy moduł zgodności dla ``config_manager``. (READ ONLY: przekierowuje do pakietu `KryptoLowca`)."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        package_init = candidate / "KryptoLowca" / "__init__.py"
        if package_init.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break


if __package__ in (None, ""):
    _ensure_repo_root()


from KryptoLowca.config_manager import *  # noqa: F401,F403
