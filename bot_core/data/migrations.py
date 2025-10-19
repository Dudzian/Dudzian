"""Narzędzia wspomagające migracje danych OHLCV dla nowych giełd."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bot_core.config.models import EnvironmentConfig
from bot_core.data.ohlcv import CachedOHLCVSource, OHLCVBackfillService
from bot_core.exchanges.base import ExchangeAdapter


@dataclass(slots=True)
class ExchangeDataToolkit:
    """Zestaw komponentów do migracji danych pojedynczej giełdy."""

    environment: EnvironmentConfig
    adapter: ExchangeAdapter
    namespace: str
    cache_directory: Path
    manifest_path: Path
    data_source: CachedOHLCVSource
    backfill_service: OHLCVBackfillService

    def warm_cache(self, *, symbols: Iterable[str], intervals: Iterable[str]) -> None:
        """Pomocnicza metoda upraszczająca rozgrzanie cache'u."""

        self.data_source.warm_cache(symbols, intervals)

    def latest_timestamp(self, symbol: str, interval: str) -> float | None:
        """Zwraca znacznik czasu ostatniej świecy z manifestu."""

        cache_key = f"{symbol}::{interval}"
        return self.data_source.storage.latest_timestamp(cache_key)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_exchange_data_toolkit(
    environment: EnvironmentConfig,
    adapter: ExchangeAdapter,
    *,
    base_directory: str | Path | None = None,
    enable_snapshots: bool | None = None,
    allow_network_upstream: bool | None = None,
) -> ExchangeDataToolkit:
    """Buduje zestaw komponentów cache/backfill dla wskazanego środowiska."""

    from bot_core.data import create_cached_ohlcv_source, resolve_cache_namespace

    cache_root = Path(base_directory or environment.data_cache_path)
    namespace = resolve_cache_namespace(environment)
    cache_directory = _ensure_directory(cache_root / "ohlcv_parquet")
    manifest_path = _ensure_directory(cache_root / "manifests") / "ohlcv_manifest.sqlite"

    snapshots_enabled = enable_snapshots
    if snapshots_enabled is None:
        snapshots_enabled = not environment.offline_mode

    upstream_enabled = allow_network_upstream
    if upstream_enabled is None:
        upstream_enabled = not environment.offline_mode

    data_source = create_cached_ohlcv_source(
        adapter,
        cache_directory=cache_directory,
        manifest_path=manifest_path,
        enable_snapshots=snapshots_enabled,
        allow_network_upstream=upstream_enabled,
        namespace=namespace,
    )
    backfill_service = OHLCVBackfillService(data_source)

    return ExchangeDataToolkit(
        environment=environment,
        adapter=adapter,
        namespace=namespace,
        cache_directory=cache_directory,
        manifest_path=manifest_path,
        data_source=data_source,
        backfill_service=backfill_service,
    )


__all__ = ["ExchangeDataToolkit", "prepare_exchange_data_toolkit"]
