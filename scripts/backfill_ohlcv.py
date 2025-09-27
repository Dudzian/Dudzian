"""Uruchamia proces backfillu OHLCV zgodny z architekturą etapu 1."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

from bot_core.config.loader import load_core_config
from bot_core.config.models import InstrumentUniverseConfig
from bot_core.data.ohlcv import (
    OHLCVBackfillService,
    CachedOHLCVSource,
    PublicAPIDataSource,
    SQLiteCacheStorage,
)
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.binance.spot import BinanceSpotAdapter

_LOGGER = logging.getLogger(__name__)


def _parse_timestamp(value: str) -> int:
    """Zamienia wejście użytkownika na timestamp w milisekundach (UTC)."""

    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - walidacja wejścia
        raise argparse.ArgumentTypeError(
            "Użyj formatu ISO (YYYY-MM-DD) lub liczby milisekund od epochy."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_adapter(exchange: str, environment: Environment) -> PublicAPIDataSource:
    """Zwraca adapter publicznego API dla wskazanej giełdy."""

    builders: dict[str, Callable[[Environment], PublicAPIDataSource]] = {
        "binance_spot": lambda env: PublicAPIDataSource(
            exchange_adapter=BinanceSpotAdapter(
                ExchangeCredentials(key_id="public", environment=env)
            )
        )
    }

    try:
        builder = builders[exchange]
    except KeyError as exc:  # pragma: no cover - zabezpieczenie przyszłych rozszerzeń
        raise ValueError(f"Brak obsługi exchange={exchange} w procesie backfillu") from exc

    return builder(environment)


def _resolve_symbols(
    universe: InstrumentUniverseConfig | None,
    exchange_name: str,
    explicit_symbols: Sequence[str] | None,
) -> tuple[str, ...]:
    if explicit_symbols:
        return tuple(explicit_symbols)

    if universe is None:
        raise SystemExit(
            "Środowisko nie posiada przypisanego uniwersum instrumentów – podaj symbole ręcznie."
        )

    resolved: list[str] = []
    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange_name)
        if symbol:
            resolved.append(symbol)

    if not resolved:
        raise SystemExit(
            "Żaden instrument z uniwersum nie jest dostępny dla wskazanej giełdy. Zdefiniuj aliasy lub"
            " przekaż symbole przez --symbols."
        )
    return tuple(resolved)


def _determine_start_timestamp(
    interval: str,
    requested_start: int | None,
    universe: InstrumentUniverseConfig | None,
) -> int:
    if requested_start is not None:
        return requested_start

    if universe is None:
        raise SystemExit(
            "Nie określono daty początkowej (--start), a środowisko nie wskazuje uniwersum z parametrami"
            " backfillu."
        )

    max_lookback_days = 0
    for instrument in universe.instruments:
        for window in instrument.backfill_windows:
            if window.interval == interval:
                if window.lookback_days > max_lookback_days:
                    max_lookback_days = window.lookback_days

    if max_lookback_days == 0:
        raise SystemExit(
            "Uniwersum nie zawiera zakresu backfill dla wskazanego interwału – podaj --start ręcznie."
        )

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max_lookback_days)
    return int(cutoff.timestamp() * 1000)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill danych OHLCV z publicznych API")
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku konfiguracji",
    )
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Nazwa środowiska z konfiguracji",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Lista symboli do backfillu (domyślnie pobierane z uniwersum środowiska)",
    )
    parser.add_argument("--interval", default="1d", help="Interwał (np. 1d, 1h)")
    parser.add_argument(
        "--start",
        type=_parse_timestamp,
        help="Początek zakresu w ms od epochy lub dacie ISO (domyślnie wg uniwersum)",
    )
    parser.add_argument(
        "--end",
        type=_parse_timestamp,
        help="Koniec zakresu (domyślnie teraz)",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=1000,
        help="Liczba świec pobieranych jednorazowo (maks. limit API)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    config = load_core_config(args.config)
    try:
        env_cfg = config.environments[args.environment]
    except KeyError as exc:
        raise SystemExit(f"Nie znaleziono środowiska {args.environment} w konfiguracji") from exc

    universe: InstrumentUniverseConfig | None = None
    if env_cfg.instrument_universe:
        universe = config.instrument_universes.get(env_cfg.instrument_universe)
        if universe is None:
            raise SystemExit(
                f"Środowisko {args.environment} wskazuje nieistniejące uniwersum "
                f"{env_cfg.instrument_universe}."
            )

    symbols = _resolve_symbols(universe, env_cfg.exchange, args.symbols)

    storage_path = Path(env_cfg.data_cache_path) / "ohlcv.sqlite"
    storage = SQLiteCacheStorage(storage_path)

    upstream_source = _build_adapter(env_cfg.exchange, env_cfg.environment)
    upstream_source.exchange_adapter.configure_network(ip_allowlist=env_cfg.ip_allowlist)

    cached_source = CachedOHLCVSource(storage=storage, upstream=upstream_source)
    backfill_service = OHLCVBackfillService(cached_source, chunk_limit=args.chunk_limit)

    start_ts = _determine_start_timestamp(args.interval, args.start, universe)
    end_ts = args.end or int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    _LOGGER.info(
        "Rozpoczynam backfill: env=%s, interval=%s, start=%s, end=%s, symbole=%s",
        args.environment,
        args.interval,
        start_ts,
        end_ts,
        ",".join(symbols),
    )

    summaries = backfill_service.synchronize(
        symbols=symbols,
        interval=args.interval,
        start=start_ts,
        end=end_ts,
    )

    for summary in summaries:
        _LOGGER.info(
            "Symbol %s (%s): pobrano %s nowych świec (pominięto %s).",
            summary.symbol,
            summary.interval,
            summary.fetched_candles,
            summary.skipped_candles,
        )

    _LOGGER.info("Backfill zakończony – łączna liczba symboli: %s", len(summaries))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
