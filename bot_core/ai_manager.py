"""Compatibility module kept for legacy imports.

The new implementation lives in :mod:`bot_core.ai.manager`.  This module simply
re-exports the public API for callers still importing ``bot_core.ai_manager``.
"""
from __future__ import annotations

from bot_core.ai.manager import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
