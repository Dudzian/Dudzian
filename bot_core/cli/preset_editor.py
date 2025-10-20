"""Kompatybilne wejście dla migratora Stage6 presetów."""

from __future__ import annotations

import sys
from importlib import import_module

_legacy_cli = import_module("KryptoLowca.scripts.preset_editor_cli")

__all__ = getattr(_legacy_cli, "__all__", [])  # type: ignore[assignment]

sys.modules[__name__] = _legacy_cli

