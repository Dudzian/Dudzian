"""Compatibility wrapper exposing the legacy AIManager under the bot_core namespace."""
from __future__ import annotations

from importlib import import_module
import sys

_LEGACY_MODULE = import_module("KryptoLowca.ai_manager")

sys.modules[__name__] = _LEGACY_MODULE
