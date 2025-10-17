"""Umożliwia import lokalnych modułów bez instalacji pakietu."""
from __future__ import annotations

from pathlib import Path

from pathbootstrap import ensure_repo_root_on_sys_path


ensure_repo_root_on_sys_path(Path(__file__).resolve().parent)
