"""Dodatkowe implementacje CacheStorage."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Mapping, Sequence

from bot_core.data.base import CacheStorage


class DualCacheStorage(CacheStorage):
    """Łączy magazyn danych (np. Parquet) oraz manifest metadanych (SQLite)."""

    def __init__(self, primary: CacheStorage, manifest: CacheStorage) -> None:
        self._primary = primary
        self._manifest = manifest

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        return self._primary.read(key)

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._primary.write(key, payload)
        self._manifest.write(key, payload)

    def metadata(self) -> MutableMapping[str, str]:
        return self._manifest.metadata()

    def latest_timestamp(self, key: str) -> float | None:
        timestamp = self._manifest.latest_timestamp(key)
        if timestamp is not None:
            return timestamp
        return self._primary.latest_timestamp(key)


__all__ = ["DualCacheStorage"]
