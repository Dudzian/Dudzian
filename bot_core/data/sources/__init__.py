"""Konkretne implementacje źródeł danych oraz magazynów cache."""

from __future__ import annotations

from bot_core.data.sources.ohlcv_cache import (
    CachedOHLCVSource,
    OfflineOnlyDataSource,
    PublicAPIDataSource,
)
from bot_core.data.sources.sqlite_storage import SQLiteCacheStorage
from bot_core.data.sources.storage import DualCacheStorage

try:
    from bot_core.data.sources.parquet_storage import ParquetCacheStorage
except ModuleNotFoundError as exc:
    if exc.name not in {"pyarrow", "pyarrow.parquet"}:
        raise

    _PARQUET_IMPORT_ERROR = exc

    class ParquetCacheStorage:  # type: ignore[no-redef]
        """Placeholder informujący o brakującej zależności optionalnej."""

        def __init__(self, *args, **kwargs) -> None:
            dep = getattr(_PARQUET_IMPORT_ERROR, "name", None) or "pyarrow"
            raise RuntimeError(
                f"ParquetCacheStorage wymaga opcjonalnej zależności '{dep}'."
            ) from _PARQUET_IMPORT_ERROR


__all__ = [
    "CachedOHLCVSource",
    "OfflineOnlyDataSource",
    "PublicAPIDataSource",
    "ParquetCacheStorage",
    "SQLiteCacheStorage",
    "DualCacheStorage",
]
