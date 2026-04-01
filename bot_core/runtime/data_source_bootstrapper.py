"""Bootstrap warstwy źródeł danych i streamingu dla runtime pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot_core.config.models import EnvironmentConfig

try:  # pragma: no cover - fallback dla gałęzi bez pełnego modułu data
    from bot_core.data import CachedOHLCVSource, create_cached_ohlcv_source, resolve_cache_namespace
    from bot_core.data.backfill_scheduler import BackfillScheduler
    from bot_core.data.ohlcv import OHLCVBackfillService
except Exception:  # pragma: no cover
    CachedOHLCVSource = Any  # type: ignore
    create_cached_ohlcv_source = None  # type: ignore
    resolve_cache_namespace = None  # type: ignore
    BackfillScheduler = None  # type: ignore
    OHLCVBackfillService = Any  # type: ignore

from bot_core.exchanges.base import ExchangeAdapter
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.observability.metrics import MetricsRegistry
from bot_core.runtime.multi_strategy_scheduler import StrategyDataFeed

_LOGGER = logging.getLogger(__name__)
_DEFAULT_OHLCV_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class DataSourceBootstrapper:
    """Wydziela tworzenie źródeł OHLCV i wiring feedów streaming/polling."""

    def create_cached_source(
        self,
        *,
        adapter: ExchangeAdapter,
        environment: EnvironmentConfig,
    ) -> CachedOHLCVSource:
        if create_cached_ohlcv_source is None or resolve_cache_namespace is None:
            raise RuntimeError("Brakuje modułów data wymaganych do bootstrapu źródeł OHLCV")
        try:
            cache_root = Path(environment.data_cache_path)
            data_source_cfg = getattr(environment, "data_source", None)
            enable_snapshots = True
            namespace = resolve_cache_namespace(environment)
            offline_mode = bool(getattr(environment, "offline_mode", False))
            allow_network_upstream = not offline_mode
            if data_source_cfg is not None:
                enable_snapshots = bool(getattr(data_source_cfg, "enable_snapshots", True))
            if offline_mode:
                enable_snapshots = False

            return create_cached_ohlcv_source(
                adapter,
                cache_directory=cache_root / "ohlcv_parquet",
                manifest_path=cache_root / "ohlcv_manifest.sqlite",
                enable_snapshots=enable_snapshots,
                allow_network_upstream=allow_network_upstream,
                namespace=namespace,
            )
        except Exception:
            _LOGGER.exception("Data source bootstrap failed during cached source creation")
            raise

    def ensure_local_market_data_availability(
        self,
        *,
        environment: EnvironmentConfig,
        data_source: CachedOHLCVSource,
        markets: Mapping[str, object],
        interval: str,
        backfill_service: OHLCVBackfillService | None = None,
        adapter: ExchangeAdapter | None = None,
    ) -> None:
        if BackfillScheduler is None:
            raise RuntimeError("Brakuje BackfillScheduler wymaganego do bootstrapu źródeł danych")
        try:
            scheduler = BackfillScheduler(
                data_source,
                backfill_service=backfill_service,
                adapter=adapter,
                default_columns=_DEFAULT_OHLCV_COLUMNS,
            )
            scheduler.ensure_ohlcv_availability(
                symbols=markets.keys(),
                interval=interval,
                environment=environment,
            )
        except Exception:
            _LOGGER.exception("Data source bootstrap failed during OHLCV availability check")
            raise

    @staticmethod
    def resolve_adapter_metrics_registry(
        adapter: ExchangeAdapter | object | None,
    ) -> MetricsRegistry | None:
        if adapter is None:
            return None

        candidate = getattr(adapter, "metrics_registry", None)
        if isinstance(candidate, MetricsRegistry):
            return candidate

        private_candidate = getattr(adapter, "_metrics", None)
        if isinstance(private_candidate, MetricsRegistry):
            return private_candidate

        return None

    def build_streaming_feed(
        self,
        *,
        stream_factory: Any,
        stream_config: object | None,
        stream_settings: Mapping[str, object] | None,
        adapter_metrics: MetricsRegistry | None,
        base_feed: StrategyDataFeed | None,
        symbols_map: Mapping[str, Sequence[str]] | None,
        exchange: str | None,
        environment_name: str | None,
    ):
        try:
            return stream_factory(
                stream_config=stream_config,
                stream_settings=stream_settings,
                adapter_metrics=adapter_metrics,
                base_feed=base_feed,
                symbols_map=symbols_map,
                exchange=exchange,
                environment_name=environment_name,
            )
        except Exception:
            _LOGGER.exception("Data source bootstrap failed during streaming feed wiring")
            raise


__all__ = ["DataSourceBootstrapper", "LocalLongPollStream"]
