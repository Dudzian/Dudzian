"""Automatyczny backfill oraz odświeżanie inkrementalne danych OHLCV."""
from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig, InstrumentUniverseConfig
from bot_core.data.ohlcv import (
    CachedOHLCVSource,
    DualCacheStorage,
    OHLCVBackfillService,
    OHLCVRefreshScheduler,
    ParquetCacheStorage,
    PublicAPIDataSource,
    SQLiteCacheStorage,
)
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter

_LOGGER = logging.getLogger(__name__)

_MILLISECONDS_IN_DAY = 86_400_000


@dataclass(slots=True)
class _IntervalPlan:
    symbols: set[str]
    backfill_start_ms: int
    incremental_lookback_ms: int


def _build_public_source(exchange: str, environment: Environment) -> PublicAPIDataSource:
    builders: Mapping[str, Callable[[Environment], PublicAPIDataSource]] = {
        "binance_spot": lambda env: PublicAPIDataSource(
            exchange_adapter=BinanceSpotAdapter(ExchangeCredentials(key_id="public", environment=env))
        ),
        "binance_futures": lambda env: PublicAPIDataSource(
            exchange_adapter=BinanceFuturesAdapter(ExchangeCredentials(key_id="public", environment=env), environment=env)
        ),
        "kraken_spot": lambda env: PublicAPIDataSource(
            exchange_adapter=KrakenSpotAdapter(ExchangeCredentials(key_id="public", environment=env), environment=env)
        ),
        "kraken_futures": lambda env: PublicAPIDataSource(
            exchange_adapter=KrakenFuturesAdapter(
                ExchangeCredentials(key_id="public", environment=env),
                environment=env,
            )
        ),
        "zonda_spot": lambda env: PublicAPIDataSource(
            exchange_adapter=ZondaSpotAdapter(
                ExchangeCredentials(key_id="public", environment=env),
                environment=env,
            )
        ),
    }
    try:
        builder = builders[exchange]
    except KeyError as exc:  # pragma: no cover - zabezpieczenie przed przyszłymi giełdami
        raise ValueError(f"Brak obsługi exchange={exchange} dla backfillu") from exc
    return builder(environment)


def _resolve_universe(core_config: CoreConfig, environment: EnvironmentConfig) -> InstrumentUniverseConfig:
    if not environment.instrument_universe:
        raise SystemExit(
            "Środowisko nie posiada przypisanego uniwersum instrumentów – zdefiniuj instrument_universe w config/core.yaml."
        )
    try:
        return core_config.instrument_universes[environment.instrument_universe]
    except KeyError as exc:
        raise SystemExit(
            f"Środowisko {environment.name} wskazuje nieistniejące uniwersum {environment.instrument_universe}."
        ) from exc


def _utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _build_interval_plans(
    *,
    universe: InstrumentUniverseConfig,
    exchange_name: str,
    incremental_lookback_days: int,
) -> tuple[dict[str, _IntervalPlan], set[str]]:
    plans: dict[str, _IntervalPlan] = {}
    symbols: set[str] = set()
    now_ms = _utc_now_ms()

    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange_name)
        if not symbol:
            continue
        symbols.add(symbol)

        for window in instrument.backfill_windows:
            start = now_ms - window.lookback_days * _MILLISECONDS_IN_DAY
            plan = plans.get(window.interval)
            if plan is None:
                plan = _IntervalPlan(symbols=set(), backfill_start_ms=start, incremental_lookback_ms=0)
                plans[window.interval] = plan
            plan.symbols.add(symbol)
            plan.backfill_start_ms = min(plan.backfill_start_ms, start)
            effective_days = max(1, min(window.lookback_days, incremental_lookback_days))
            plan.incremental_lookback_ms = max(plan.incremental_lookback_ms, effective_days * _MILLISECONDS_IN_DAY)

    return plans, symbols


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill danych OHLCV zgodnie z config/core.yaml")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku konfiguracyjnego CoreConfig")
    parser.add_argument("--environment", default="binance_paper", help="Nazwa środowiska do backfillu")
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=900,
        help="Częstotliwość odświeżania inkrementalnego (w sekundach)",
    )
    parser.add_argument(
        "--incremental-lookback-days",
        type=int,
        default=3,
        help="Ile dni historii pobierać przy odświeżaniu inkrementalnym",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Wykonaj tylko pełny backfill i zakończ (bez harmonogramu)",
    )
    return parser.parse_args(argv)


def _perform_backfill(
    *,
    service: OHLCVBackfillService,
    plans: Mapping[str, _IntervalPlan],
    end_timestamp: int,
) -> None:
    for interval, plan in plans.items():
        start = max(0, plan.backfill_start_ms)
        _LOGGER.info(
            "Backfill interval=%s, start=%s, end=%s, symbole=%s",
            interval,
            start,
            end_timestamp,
            ",".join(sorted(plan.symbols)),
        )
        summaries = service.synchronize(
            symbols=tuple(sorted(plan.symbols)),
            interval=interval,
            start=start,
            end=end_timestamp,
        )
        total = sum(summary.fetched_candles for summary in summaries)
        _LOGGER.info(
            "Zakończono backfill dla interval=%s – pobrano %s nowych świec",
            interval,
            total,
        )


async def _run_scheduler(
    *,
    scheduler: OHLCVRefreshScheduler,
    plans: Mapping[str, _IntervalPlan],
    refresh_seconds: int,
) -> None:
    for interval, plan in plans.items():
        scheduler.add_job(
            symbols=tuple(sorted(plan.symbols)),
            interval=interval,
            lookback_ms=plan.incremental_lookback_ms or (_MILLISECONDS_IN_DAY * 1),
            frequency_seconds=refresh_seconds,
            name=f"{interval}:{len(plan.symbols)}",
        )

    _LOGGER.info("Uruchamiam harmonogram odświeżania (co %s sekund)", refresh_seconds)
    try:
        await scheduler.run_forever()
    finally:
        scheduler.stop()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    config = load_core_config(args.config)
    try:
        environment = config.environments[args.environment]
    except KeyError as exc:
        raise SystemExit(f"Nie znaleziono środowiska {args.environment} w konfiguracji") from exc

    universe = _resolve_universe(config, environment)

    plans, symbols = _build_interval_plans(
        universe=universe,
        exchange_name=environment.exchange,
        incremental_lookback_days=max(1, args.incremental_lookback_days),
    )
    if not plans:
        _LOGGER.warning("Brak instrumentów z zakresem backfill dla giełdy %s", environment.exchange)
        return 0

    cache_root = Path(environment.data_cache_path)
    parquet_storage = ParquetCacheStorage(cache_root / "ohlcv_parquet", namespace=environment.exchange)
    manifest_storage = SQLiteCacheStorage(cache_root / "ohlcv_manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(primary=parquet_storage, manifest=manifest_storage)

    upstream_source = _build_public_source(environment.exchange, environment.environment)
    upstream_source.exchange_adapter.configure_network(ip_allowlist=environment.ip_allowlist)

    cached_source = CachedOHLCVSource(storage=storage, upstream=upstream_source)
    cached_source.warm_cache(symbols, plans.keys())

    backfill_service = OHLCVBackfillService(cached_source)
    now_ts = _utc_now_ms()
    _perform_backfill(service=backfill_service, plans=plans, end_timestamp=now_ts)

    if args.run_once:
        _LOGGER.info("Tryb run-once – kończę po pełnym backfillu")
        return 0

    scheduler = OHLCVRefreshScheduler(backfill_service)
    try:
        asyncio.run(
            _run_scheduler(
                scheduler=scheduler,
                plans=plans,
                refresh_seconds=args.refresh_seconds,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - obsługa CLI
        _LOGGER.info("Przerwano przez użytkownika – zamykam harmonogram")

    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())

