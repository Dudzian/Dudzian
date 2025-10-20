"""Warstwa danych rynkowych oraz biblioteka znormalizowanych zestawów backtestowych."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - tylko dla wskazówek typów
    from bot_core.config.models import EnvironmentConfig
else:  # pragma: no cover - środowiska bootstrapowe mogą mieć niepełne modele
    EnvironmentConfig = object  # type: ignore[assignment]
from bot_core.data.backtest_library import (
    BacktestDatasetLibrary,
    DataQualityReport,
    DataQualityValidator,
    DatasetDescriptor,
)
from bot_core.data.base import CacheStorage, DataSource, OHLCVRequest, OHLCVResponse
from bot_core.data.intervals import interval_to_milliseconds
from bot_core.data.migrations import ExchangeDataToolkit, prepare_exchange_data_toolkit
from bot_core.data.ohlcv.cache import CachedOHLCVSource, OfflineOnlyDataSource, PublicAPIDataSource
from bot_core.data.ohlcv.parquet_storage import ParquetCacheStorage
from bot_core.data.ohlcv.sqlite_storage import SQLiteCacheStorage
from bot_core.data.ohlcv.storage import DualCacheStorage
from bot_core.exchanges.base import ExchangeAdapter


def resolve_cache_namespace(environment: EnvironmentConfig) -> str:
    """Zwraca nazwę przestrzeni cache zgodną z konfiguracją środowiska."""

    data_source_cfg = getattr(environment, "data_source", None)
    namespace = environment.exchange
    if data_source_cfg is not None:
        custom = getattr(data_source_cfg, "cache_namespace", None)
        if custom:
            namespace = custom
    return namespace


def create_cached_ohlcv_source(
    exchange_adapter: ExchangeAdapter,
    *,
    cache_directory: str | Path,
    manifest_path: str | Path,
    enable_snapshots: bool = True,
    allow_network_upstream: bool = True,
    namespace: str | None = None,
) -> CachedOHLCVSource:
    """Buduje domyślne źródło OHLCV z Parquet + manifestem SQLite."""

    storage = DualCacheStorage(
        ParquetCacheStorage(cache_directory, namespace=namespace or exchange_adapter.name),
        SQLiteCacheStorage(manifest_path, store_rows=False),
    )
    if allow_network_upstream:
        upstream: DataSource = PublicAPIDataSource(exchange_adapter=exchange_adapter)
    else:
        upstream = OfflineOnlyDataSource(exchange_name=exchange_adapter.name)

    snapshot_fetcher = None
    if enable_snapshots and allow_network_upstream:

        def _snapshot(request: OHLCVRequest) -> Sequence[Sequence[float]]:
            interval_ms = interval_to_milliseconds(request.interval)
            window_start = max(request.start, request.end - max(interval_ms * 2, interval_ms))
            limit = request.limit if request.limit and request.limit > 0 else 1
            return exchange_adapter.fetch_ohlcv(
                request.symbol,
                request.interval,
                start=window_start,
                end=request.end,
                limit=limit,
            )

        snapshot_fetcher = _snapshot

    return CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=snapshot_fetcher,
    )

__all__ = [
    "CacheStorage",
    "CachedOHLCVSource",
    "create_cached_ohlcv_source",
    "DataSource",
    "OHLCVRequest",
    "OHLCVResponse",
    "OfflineOnlyDataSource",
    "PublicAPIDataSource",
    "ExchangeDataToolkit",
    "prepare_exchange_data_toolkit",
    "resolve_cache_namespace",
    "BacktestDatasetLibrary",
    "DatasetDescriptor",
    "DataQualityValidator",
    "DataQualityReport",
]
