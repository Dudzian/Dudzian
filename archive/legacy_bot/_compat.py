"""Utilities for wiring legacy modules to their canonical replacements."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["proxy_globals"]


def _exported_names(module: Any) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return list(exported)
    return [name for name in dir(module) if not name.startswith("_")]


def proxy_globals(module_globals: dict[str, Any], target: str, legacy_proxy: str) -> None:
    """Populate ``module_globals`` with a live view of ``target`` module."""

    impl = import_module(target)
    exported = _exported_names(impl)
    module_globals["__legacy_proxy__"] = legacy_proxy
    module_globals["__all__"] = exported
    module_globals["_impl"] = impl
    module_globals.update({name: getattr(impl, name) for name in exported})
    module_globals["__getattr__"] = lambda name: getattr(impl, name)
    module_globals["__dir__"] = lambda: sorted(set(exported) | set(dir(impl)))
