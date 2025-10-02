"""Narzędzie do przygotowania lokalnego cache'u OHLCV dla smoke testów paper tradingu."""
from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig, InstrumentConfig
from bot_core.data.ohlcv import ParquetCacheStorage, SQLiteCacheStorage
from bot_core.exchanges.base import Environment as ExchangeEnvironment

_LOGGER = logging.getLogger(__name__)

_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")

_BASE_PRICE_BY_ASSET = {
    "BTC": 45_000.0,
    "ETH": 2_500.0,
    "SOL": 95.0,
    "BNB": 320.0,
    "XRP": 0.6,
    "ADA": 0.45,
    "LTC": 78.0,
    "MATIC": 0.85,
}

_BASE_VOLUME_BY_ASSET = {
    "BTC": 1_500.0,
    "ETH": 3_000.0,
    "SOL": 45_000.0,
    "BNB": 25_000.0,
    "XRP": 8_000_000.0,
    "ADA": 9_500_000.0,
    "LTC": 120_000.0,
    "MATIC": 6_000_000.0,
}


@dataclass(slots=True)
class GeneratedSeries:
    symbol: str
    interval: str
    candles: int
    start_timestamp: int
    end_timestamp: int


def _parse_start_date(value: str | None, *, days: int) -> datetime:
    if value is None:
        return datetime.now(timezone.utc) - timedelta(days=days)
    text = value.strip()
    if not text:
        raise ValueError("start-date nie może być pusty")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _base_price(symbol: str, base_asset: str) -> float:
    asset = base_asset.upper()
    if asset in _BASE_PRICE_BY_ASSET:
        return _BASE_PRICE_BY_ASSET[asset]
    digest = hashlib.sha256(f"{symbol}:{asset}".encode("utf-8")).digest()
    anchor = int.from_bytes(digest[:4], "big")
    return 10.0 + (anchor % 10_000) / 100.0


def _base_volume(symbol: str, base_asset: str) -> float:
    asset = base_asset.upper()
    if asset in _BASE_VOLUME_BY_ASSET:
        return _BASE_VOLUME_BY_ASSET[asset]
    digest = hashlib.sha256(f"vol:{symbol}:{asset}".encode("utf-8")).digest()
    anchor = int.from_bytes(digest[4:8], "big")
    return 1_000.0 + (anchor % 200_000)


def _generate_rows(
    *,
    symbol: str,
    base_asset: str,
    days: int,
    start: datetime,
    interval: str,
    seed: int | None,
) -> list[list[float]]:
    if interval != "1d":
        raise ValueError("Skrypt obsługuje wyłącznie interwał 1d na potrzeby smoke testu")
    step = timedelta(days=1)
    price = _base_price(symbol, base_asset)
    volume_anchor = _base_volume(symbol, base_asset)
    if seed is not None:
        digest = hashlib.sha256(f"{symbol}:{seed}".encode("utf-8")).digest()
        rng_state = int.from_bytes(digest[:8], "big")
    else:
        rng_state = hash((symbol, days, start.toordinal())) & 0xFFFFFFFF

    def _rand() -> float:
        nonlocal rng_state
        rng_state = (1103515245 * rng_state + 12345) % (2 ** 31)
        return rng_state / float(2 ** 31)

    rows: list[list[float]] = []
    current = start
    for _ in range(days):
        open_price = price
        drift = 0.0025 + (_rand() - 0.5) * 0.004
        shock = (_rand() - 0.5) * 0.015
        close_price = max(0.0001, open_price * (1.0 + drift + shock))
        high_price = max(open_price, close_price) * (1.0 + abs((_rand() - 0.5) * 0.01))
        low_price = min(open_price, close_price) * (1.0 - abs((_rand() - 0.5) * 0.01))
        volume = volume_anchor * (0.75 + _rand() * 0.5)
        timestamp = int(current.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        rows.append([
            float(timestamp),
            float(round(open_price, 6)),
            float(round(high_price, 6)),
            float(round(low_price, 6)),
            float(round(close_price, 6)),
            float(round(volume, 6)),
        ])
        price = close_price
        current += step
    return rows


def _resolve_symbols(
    *,
    config: CoreConfig,
    environment: EnvironmentConfig,
) -> list[tuple[str, InstrumentConfig]]:
    if not environment.instrument_universe:
        raise ValueError(
            f"Środowisko {environment.name} nie ma przypisanego instrument_universe w konfiguracji"
        )
    try:
        universe = config.instrument_universes[environment.instrument_universe]
    except KeyError as exc:
        raise KeyError(
            f"Brak uniwersum instrumentów '{environment.instrument_universe}' w konfiguracji"
        ) from exc

    raw_settings = getattr(environment, "adapter_settings", {}) or {}
    paper_settings = raw_settings.get("paper_trading", {}) or {}
    quote_assets = paper_settings.get("quote_assets")
    if quote_assets:
        allowed_quotes = {str(asset).upper() for asset in quote_assets}
    else:
        valuation = str(paper_settings.get("valuation_asset", "USDT")).upper()
        allowed_quotes = {valuation}

    symbols: list[tuple[str, InstrumentConfig]] = []
    for instrument in universe.instruments:
        exchange_symbol = instrument.exchange_symbols.get(environment.exchange)
        if not exchange_symbol:
            continue
        if instrument.quote_asset.upper() not in allowed_quotes:
            continue
        symbols.append((exchange_symbol, instrument))
    return symbols


def generate_smoke_cache(
    *,
    config_path: Path,
    environment_name: str,
    interval: str,
    days: int,
    start_date: datetime,
    seed: int | None = None,
) -> list[GeneratedSeries]:
    if days <= 0:
        raise ValueError("Liczba dni musi być dodatnia")

    config = load_core_config(config_path)
    try:
        environment = config.environments[environment_name]
    except KeyError as exc:
        raise KeyError(f"Brak środowiska '{environment_name}' w konfiguracji") from exc

    if environment.environment not in {ExchangeEnvironment.PAPER, ExchangeEnvironment.TESTNET}:
        raise ValueError("Cache smoke obsługuje wyłącznie środowiska paper/testnet")

    symbols = _resolve_symbols(config=config, environment=environment)
    if not symbols:
        raise ValueError(
            f"Uniwersum {environment.instrument_universe} nie posiada instrumentów dla giełdy {environment.exchange}"
        )

    cache_root = Path(environment.data_cache_path)
    parquet_storage = ParquetCacheStorage(cache_root / "ohlcv_parquet", namespace=environment.exchange)
    manifest_storage = SQLiteCacheStorage(cache_root / "ohlcv_manifest.sqlite", store_rows=False)
    metadata = parquet_storage.metadata()

    generated: list[GeneratedSeries] = []
    for symbol, instrument in symbols:
        rows = _generate_rows(
            symbol=symbol,
            base_asset=instrument.base_asset,
            days=days,
            start=start_date,
            interval=interval,
            seed=seed,
        )
        payload = {"columns": _COLUMNS, "rows": rows}
        key = f"{symbol}::{interval}"
        parquet_storage.write(key, payload)
        manifest_storage.write(key, payload)
        metadata[f"row_count::{symbol}::{interval}"] = str(len(rows))
        metadata[f"last_timestamp::{symbol}::{interval}"] = str(int(rows[-1][0]))
        generated.append(
            GeneratedSeries(
                symbol=symbol,
                interval=interval,
                candles=len(rows),
                start_timestamp=int(rows[0][0]),
                end_timestamp=int(rows[-1][0]),
            )
        )
        _LOGGER.info(
            "Zapisano %s świec dla %s (%s) w %s",
            len(rows),
            symbol,
            interval,
            cache_root,
        )

    return generated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generuje deterministyczne dane OHLCV 1d dla środowiska paper/testnet, "
            "aby umożliwić offline smoke test strategii Daily Trend."
        )
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku konfiguracyjnego core")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Nazwa środowiska paper/testnet, dla którego generujemy cache",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Interwał OHLCV (obecnie obsługiwany 1d)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Liczba kolejnych dni do wygenerowania",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Data początkowa (ISO 8601, UTC) pierwszej świecy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Opcjonalne ziarno generatora szumu (dla powtarzalności)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Ogranicz logowanie do ostrzeżeń/błędów",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    try:
        start_date = _parse_start_date(args.start_date, days=args.days)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    try:
        results = generate_smoke_cache(
            config_path=Path(args.config),
            environment_name=args.environment,
            interval=args.interval,
            days=args.days,
            start_date=start_date,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Nie udało się zbudować cache'u smoke: %s", exc)
        return 1

    total = sum(entry.candles for entry in results)
    earliest = min(entry.start_timestamp for entry in results)
    latest = max(entry.end_timestamp for entry in results)
    _LOGGER.info(
        "Cache smoke gotowy: %s świec, zakres %s – %s (UTC)",
        total,
        datetime.fromtimestamp(earliest / 1000, tz=timezone.utc).isoformat(),
        datetime.fromtimestamp(latest / 1000, tz=timezone.utc).isoformat(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
