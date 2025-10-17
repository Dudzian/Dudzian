"""Wspólny helper do bootstrapu ścieżki importu w testach."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

__all__ = [
    "PATHBOOTSTRAP_MODULE",
    "REPO_ROOT",
    "ensure_repo_root_on_sys_path",
    "get_repo_info",
    "get_repo_root",
    "clear_cache",
    "repo_on_sys_path",
    "chdir_repo_root",
    "load_pathbootstrap",
]


def load_pathbootstrap() -> ModuleType:
    """Załaduj moduł :mod:`pathbootstrap`, niezależnie od bieżącego sys.path."""

    repo_root = Path(__file__).resolve().parents[1]
    try:
        return importlib.import_module("pathbootstrap")
    except ModuleNotFoundError:
        module_path = repo_root / "pathbootstrap.py"
        spec = importlib.util.spec_from_file_location("_pathbootstrap_for_tests", module_path)
        if spec is None or spec.loader is None:  # pragma: no cover - ścieżka awaryjna
            raise ImportError("Nie można załadować modułu pathbootstrap")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


PATHBOOTSTRAP_MODULE = load_pathbootstrap()
REPO_ROOT = PATHBOOTSTRAP_MODULE.get_repo_root(Path(__file__).resolve().parents[1])
PATHBOOTSTRAP_MODULE.ensure_repo_root_on_sys_path(REPO_ROOT)

ensure_repo_root_on_sys_path = PATHBOOTSTRAP_MODULE.ensure_repo_root_on_sys_path
get_repo_info = PATHBOOTSTRAP_MODULE.get_repo_info
get_repo_root = PATHBOOTSTRAP_MODULE.get_repo_root
clear_cache = PATHBOOTSTRAP_MODULE.clear_cache
repo_on_sys_path = PATHBOOTSTRAP_MODULE.repo_on_sys_path
chdir_repo_root = PATHBOOTSTRAP_MODULE.chdir_repo_root
