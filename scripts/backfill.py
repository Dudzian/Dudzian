"""Automatyczny backfill oraz odświeżanie inkrementalne danych OHLCV."""
from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from bot_core.alerts import DefaultAlertRouter
from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig, InstrumentUniverseConfig
from bot_core.data.ohlcv import (
    BackfillSummary,
    CachedOHLCVSource,
    DataGapIncidentTracker,
    DualCacheStorage,
    GapAlertPolicy,
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
from bot_core.runtime.bootstrap import build_alert_channels
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage

_LOGGER = logging.getLogger(__name__)

_MILLISECONDS_IN_DAY = 86_400_000

# Domyślne częstotliwości odświeżania per interwał (sekundy)
_DEFAULT_REFRESH_SECONDS: Mapping[str, int] = {
    "1d": 24 * 60 * 60,
    "1h": 15 * 60,
    "15m": 5 * 60,
}


@dataclass(slots=True)
class _IntervalPlan:
    symbols: set[str]
    backfill_start_ms: int
    incremental_lookback_ms: int
    # 0 => użyj wartości przekazanej z CLI (--refresh-seconds)
    refresh_seconds: int


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
    except KeyError as exc:  # pragma: no cover
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
    interval_refresh_overrides: Mapping[str, int] | None = None,
) -> tuple[dict[str, _IntervalPlan], set[str]]:
    # przefiltruj override’y do dodatnich intów
    overrides: dict[str, int] = {}
    if interval_refresh_overrides:
        for interval, value in interval_refresh_overrides.items():
            try:
                seconds = int(value)
            except (TypeError, ValueError):
                continue
            if seconds > 0:
                overrides[interval] = seconds

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
                refresh_seconds = overrides.get(window.interval, _DEFAULT_REFRESH_SECONDS.get(window.interval, 0))
                plan = _IntervalPlan(
                    symbols=set(),
                    backfill_start_ms=start,
                    incremental_lookback_ms=0,
                    refresh_seconds=refresh_seconds,
                )
                plans[window.interval] = plan
            plan.symbols.add(symbol)
            plan.backfill_start_ms = min(plan.backfill_start_ms, start)
            effective_days = max(1, min(window.lookback_days, incremental_lookback_days))
            plan.incremental_lookback_ms = max(plan.incremental_lookback_ms, effective_days * _MILLISECONDS_IN_DAY)

    return plans, symbols


def _extract_gap_policy(environment: EnvironmentConfig) -> GapAlertPolicy:
    settings: Mapping[str, object] = {}
    if isinstance(environment.adapter_settings, Mapping):
        candidate = environment.adapter_settings.get("ohlcv_gap_alerts")
        if isinstance(candidate, Mapping):
            settings = candidate

    warnings_cfg = {}
    raw_warnings = settings.get("warning_gap_minutes") if settings else None
    if isinstance(raw_warnings, Mapping):
        warnings_cfg = {
            str(interval): max(1, int(value))
            for interval, value in raw_warnings.items()
            if value is not None and int(value) > 0
        }

    def _safe_int(key: str, default: int) -> int:
        value = settings.get(key) if settings else None
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    return GapAlertPolicy(
        warning_gap_minutes=warnings_cfg,
        incident_threshold_count=_safe_int("incident_threshold_count", 5),
        incident_window_minutes=_safe_int("incident_window_minutes", 10),
        sms_escalation_minutes=_safe_int("sms_escalation_minutes", 15),
    )


def _build_gap_callback(
    gap_tracker: DataGapIncidentTracker | None,
) -> Callable[[str, Sequence[BackfillSummary], int], None] | None:
    if gap_tracker is None:
        return None

    def _callback(interval: str, summaries: Sequence[BackfillSummary], as_of_ms: int) -> None:
        gap_tracker.handle_summaries(interval=interval, summaries=summaries, as_of_ms=as_of_ms)

    return _callback


def _initialize_alerting(
    *,
    args: argparse.Namespace,
    config: CoreConfig,
    environment: EnvironmentConfig,
) -> tuple[DefaultAlertRouter | None, GapAlertPolicy | None, str]:
    if not args.enable_alerts:
        return None, None, "Alerty wyłączone flagą CLI"

    try:
        storage = create_default_secret_storage(
            namespace=args.secret_namespace,
            headless_passphrase=args.headless_passphrase,
            headless_path=args.headless_secrets_path,
        )
    except SecretStorageError as exc:
        return None, None, f"Nie udało się przygotować magazynu sekretów: {exc}"

    secret_manager = SecretManager(storage, namespace=args.secret_namespace)

    try:
        _, router, _ = build_alert_channels(
            core_config=config,
            environment=environment,
            secret_manager=secret_manager,
        )
    except SecretStorageError as exc:
        return None, None, f"Nie udało się zbudować kanałów alertów: {exc}"

    policy = _extract_gap_policy(environment)
    return router, policy, "Kanały alertowe zainicjalizowane"


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
    parser.add_argument(
        "--enable-alerts",
        action="store_true",
        help="Aktywuj wysyłkę alertów o lukach danych (wymaga skonfigurowanych sekretów)",
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Namespace używany przy odczycie sekretów (keychain / plik szyfrowany)",
    )
    parser.add_argument(
        "--headless-passphrase",
        default=None,
        help="Hasło do magazynu sekretów w środowiskach headless (Linux).",
    )
    parser.add_argument(
        "--headless-secrets-path",
        default=None,
        help="Ścieżka do zaszyfrowanego magazynu sekretów w trybie headless.",
    )
    return parser.parse_args(argv)


def _perform_backfill(
    *,
    service: OHLCVBackfillService,
    plans: Mapping[str, _IntervalPlan],
    end_timestamp: int,
    gap_tracker: DataGapIncidentTracker | None = None,
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
        if gap_tracker:
            gap_tracker.handle_summaries(interval=interval, summaries=summaries, as_of_ms=end_timestamp)
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
        frequency = plan.refresh_seconds or refresh_seconds
        scheduler.add_job(
            symbols=tuple(sorted(plan.symbols)),
            interval=interval,
            lookback_ms=plan.incremental_lookback_ms or (_MILLISECONDS_IN_DAY * 1),
            frequency_seconds=frequency,
            name=f"{interval}:{len(plan.symbols)}",
        )
        _LOGGER.debug(
            "Zarejestrowano zadanie interval=%s, refresh_seconds=%s, lookback_ms=%s",
            interval,
            frequency,
            plan.incremental_lookback_ms,
        )

    _LOGGER.info(
        "Uruchamiam harmonogram odświeżania (%s zadań, domyślna częstotliwość %s sekund)",
        len(plans),
        refresh_seconds,
    )
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

    # odczyt override’ów częstotliwości z adapter_settings (opcjonalnie)
    adapter_settings = getattr(environment, "adapter_settings", {}) or {}
    raw_interval_overrides = adapter_settings.get("ohlcv_refresh_seconds", {})
    interval_refresh_overrides: Mapping[str, int] | None = None
    if isinstance(raw_interval_overrides, Mapping):
        interval_refresh_overrides = raw_interval_overrides

    plans, symbols = _build_interval_plans(
        universe=universe,
        exchange_name=environment.exchange,
        incremental_lookback_days=max(1, args.incremental_lookback_days),
        interval_refresh_overrides=interval_refresh_overrides,
    )
    if not plans:
        _LOGGER.warning("Brak instrumentów z zakresem backfill dla giełdy %s", environment.exchange)
        return 0

    cache_root = Path(environment.data_cache_path)
    parquet_storage = ParquetCacheStorage(cache_root / "ohlcv_parquet", namespace=environment.exchange)
    manifest_storage = SQLiteCacheStorage(cache_root / "ohlcv_manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(primary=parquet_storage, manifest=manifest_storage)

    alert_router, gap_policy, alert_message = _initialize_alerting(
        args=args,
        config=config,
        environment=environment,
    )
    gap_tracker: DataGapIncidentTracker | None = None
    if alert_router and gap_policy:
        gap_tracker = DataGapIncidentTracker(
            router=alert_router,
            metadata_provider=storage.metadata,
            policy=gap_policy,
            environment_name=environment.name,
            exchange=environment.exchange,
        )
        if alert_message:
            _LOGGER.info(alert_message)
    elif alert_message:
        level = logging.INFO if not args.enable_alerts else logging.ERROR
        _LOGGER.log(level, alert_message)

    upstream_source = _build_public_source(environment.exchange, environment.environment)
    upstream_source.exchange_adapter.configure_network(ip_allowlist=environment.ip_allowlist)

    cached_source = CachedOHLCVSource(storage=storage, upstream=upstream_source)
    cached_source.warm_cache(symbols, plans.keys())

    backfill_service = OHLCVBackfillService(cached_source)
    now_ts = _utc_now_ms()
    _perform_backfill(
        service=backfill_service,
        plans=plans,
        end_timestamp=now_ts,
        gap_tracker=gap_tracker,
    )

    if args.run_once:
        _LOGGER.info("Tryb run-once – kończę po pełnym backfillu")
        return 0

    scheduler = OHLCVRefreshScheduler(
        backfill_service,
        on_job_complete=_build_gap_callback(gap_tracker),
    )
    try:
        asyncio.run(
            _run_scheduler(
                scheduler=scheduler,
                plans=plans,
                refresh_seconds=args.refresh_seconds,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover
        _LOGGER.info("Przerwano przez użytkownika – zamykam harmonogram")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
