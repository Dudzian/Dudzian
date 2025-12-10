"""Warstwa zgodności – implementacja przeniesiona do bot_core.data.sources."""
from __future__ import annotations

from bot_core.data.sources.sqlite_storage import SQLiteCacheStorage

__all__ = ["SQLiteCacheStorage"]
