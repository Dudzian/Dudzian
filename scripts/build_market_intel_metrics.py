#!/usr/bin/env python3
"""Buduje metryki Market Intelligence dla Stage6 (zunifikowany CLI).

Obsługiwane tryby:
- ohlcv  : wylicza metryki z lokalnego cache OHLCV (Parquet) dla aktywów governora
           i zapisuje jeden plik JSON (lub CSV) ze snapshotami.
- sqlite : eksportuje bazowe metryki z SQLite do osobnych plików per-symbol + manifest.

Tryb wybierany jest automatycznie na podstawie dostępnych klas w bot_core.market_intel,
ale można go wymusić przez --mode ohlcv|sqlite.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

# --- konfiguracja / modele
from bot_core.config import load_core_config
from bot_core.data import resolve_cache_namespace

# Opcjonalne importy – różne gałęzie mają różne interfejsy
try:
    # API dla trybu OHLCV
    from bot_core.market_intel import MarketIntelAggregator as OHLCVAggregator  # type: ignore
    from bot_core.market_intel import MarketIntelQuery  # type: ignore
    _HAS_OHLCV = True
except Exception:
    _HAS_OHLCV = False
    OHLCVAggregator = None  # type: ignore
    MarketIntelQuery = None  # type: ignore

try:
    # API dla trybu SQLite
    from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig  # type: ignore
    from bot_core.market_intel import MarketIntelAggregator as SqliteAggregator  # type: ignore
    _HAS_SQLITE_TYPES = True
except Exception:
    _HAS_SQLITE_TYPES = False
    MarketIntelConfig = None  # type: ignore
    MarketIntelSqliteConfig = None  # type: ignore
    SqliteAggregator = None  # type: ignore

_LOGGER = logging.getLogger("stage6.market_intel.cli")


# ---------------------------- wspólne utility ----------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_output(governor: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("var/market_intel") / f"market_intel_{governor}_{timestamp}.json"


def _write_json(path: Path, payload: Mapping[str, object], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, snapshots: Iterable[tuple[str, dict[str, object]]]) -> None:
    fieldnames = [
        "symbol",
        "interval",
        "bar_count",
        "price_change_pct",
        "volatility_pct",
        "max_drawdown_pct",
        "average_volume",
        "liquidity_usd",
        "momentum_score",
        "start",
        "end",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for symbol, snapshot in snapshots:
            row = {key: snapshot.get(key) for key in fieldnames}
            row["symbol"] = symbol
            writer.writerow(row)


# ---------------------------- parser argumentów ----------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    # Wspólne
    p.add_argument("--config", default="config/core.yaml", help="Ścieżka do core.yaml (domyślnie config/core.yaml)")
    p.add_argument("--log-level", default="INFO", help="Poziom logowania (domyślnie INFO)")
    p.add_argument("--mode", choices=("auto", "ohlcv", "sqlite"), default="auto",
                   help="Wymuś tryb działania: ohlcv/sqlite (domyślnie auto)")

    # --- Tryb OHLCV (Parquet)
    p.add_argument("--environment", help="[OHLCV] Środowisko z core.yaml (np. binance_live)")
    p.add_argument("--governor", help="[OHLCV] Nazwa PortfolioGovernora (sekcja portfolio_governors)")
    p.add_argument("--interval", default="1h", help="[OHLCV] Interwał OHLCV (domyślnie 1h)")
    p.add_argument("--lookback-bars", type=int, default=168, help="[OHLCV] Liczba świec do agregacji (domyślnie 168)")
    p.add_argument("--cache-base", help="[OHLCV] Katalog bazowy cache OHLCV (domyślnie z env.data_cache_path)")
    p.add_argument("--namespace", help="[OHLCV] Namespace cache Parquet (domyślnie environment z core.yaml)")
    p.add_argument("--output", help="[OHLCV] Ścieżka pliku wynikowego (domyślnie var/market_intel/...)")
    p.add_argument("--format", choices=("json", "csv"), default="json", help="[OHLCV] Format eksportu")
    p.add_argument("--symbols", nargs="*", help="[OHLCV] Lista symboli (domyślnie aktywa governora)")
    p.add_argument("--pretty", action="store_true", help="[OHLCV] Format JSON z wcięciami")

    # --- Tryb SQLite (baseline + manifest)
    p.add_argument("--output-dir", help="[SQLite] Katalog wyjściowy metryk (domyślnie wg konfiguracji)")
    p.add_argument("--manifest", help="[SQLite] Ścieżka pliku manifestu (domyślnie wg konfiguracji)")
    p.add_argument("--sqlite-path", help="[SQLite] Nadpisz ścieżkę do bazy SQLite z metrykami")
    p.add_argument("--sqlite-table", help="[SQLite] Nadpisz nazwę tabeli w bazie SQLite")
    p.add_argument("--required-symbol", action="append", dest="required_symbols",
                   help="[SQLite] Wymagany symbol (można podać wielokrotnie)")
    p.add_argument("--default-weight", type=float, help="[SQLite] Domyślna waga gdy brak kolumny weight")

    return p


# ---------------------------- Tryb OHLCV ----------------------------
def _run_ohlcv(args: argparse.Namespace) -> int:
    if not _HAS_OHLCV:
        raise SystemExit("Tryb OHLCV nieobsługiwany w tej gałęzi (brak MarketIntelQuery/aggregatora OHLCV). Użyj --mode sqlite.")

    if not args.environment or not args.governor:
        raise SystemExit("Dla trybu OHLCV wymagane są argumenty: --environment oraz --governor.")

    core_config = load_core_config(args.config)
    try:
        environment_cfg = core_config.environments[args.environment]
    except KeyError as exc:
        raise SystemExit(f"Nie znaleziono środowiska {args.environment} w {args.config}") from exc

    if args.governor not in core_config.portfolio_governors:
        raise SystemExit(f"PortfolioGovernor {args.governor} nie istnieje w configu")
    governor_cfg = core_config.portfolio_governors[args.governor]

    cache_base = args.cache_base or environment_cfg.data_cache_path
    namespace = args.namespace or resolve_cache_namespace(environment_cfg)
    # environment może być Enum'em, więc wyciągamy value gdy ma atrybut
    if hasattr(namespace, "value"):
        namespace = namespace.value
    if not cache_base:
        raise SystemExit("Brak ścieżki cache dla środowiska – uzupełnij data_cache_path lub podaj --cache-base")

    # Lazy import backendu Parquet (nie każda gałąź go ma)
    try:
        from bot_core.data.ohlcv.parquet_storage import ParquetCacheStorage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Brak backendu ParquetCacheStorage: {exc}") from exc

    storage = ParquetCacheStorage(cache_base, namespace=str(namespace))
    aggregator = OHLCVAggregator(storage)  # type: ignore

    symbols = args.symbols or [asset.symbol for asset in governor_cfg.assets]
    if not symbols:
        raise SystemExit("PortfolioGovernor nie ma zdefiniowanych aktywów do agregacji")

    queries = [
        MarketIntelQuery(symbol=symbol, interval=args.interval, lookback_bars=args.lookback_bars)  # type: ignore
        for symbol in symbols
    ]
    snapshots = aggregator.build_many(queries)  # {symbol: MarketIntelSnapshot}

    output_path = Path(args.output) if args.output else _default_output(args.governor)

    if args.format == "json":
        payload: dict[str, object] = {
            "generated_at": _now_iso(),
            "environment": args.environment,
            "governor": args.governor,
            "interval": args.interval,
            "lookback_bars": args.lookback_bars,
            "symbols": symbols,
            "snapshots": {symbol: snapshot.to_dict() for symbol, snapshot in snapshots.items()},
        }
        _write_json(output_path, payload, args.pretty)
        print(f"Zapisano raport Market Intelligence (OHLCV) do {output_path}")
    else:
        rows = [(symbol, snapshot.to_dict()) for symbol, snapshot in snapshots.items()]
        _write_csv(output_path, rows)
        print(f"Zapisano raport Market Intelligence CSV (OHLCV) do {output_path}")

    return 0


# ---------------------------- Tryb SQLite ----------------------------
def _override_sqlite_config(
    base: MarketIntelSqliteConfig, *, path: str | None, table: str | None
) -> MarketIntelSqliteConfig:
    kwargs = {
        "path": path or base.path,
        "table": table or base.table,
        "symbol_column": base.symbol_column,
        "mid_price_column": base.mid_price_column,
        "depth_column": base.depth_column,
        "spread_column": base.spread_column,
        "funding_column": base.funding_column,
        "sentiment_column": base.sentiment_column,
        "volatility_column": base.volatility_column,
        "weight_column": base.weight_column,
    }
    return MarketIntelSqliteConfig(**kwargs)


def _override_required_symbols(
    override: Iterable[str] | None, base: Iterable[str]
) -> tuple[str, ...]:
    if override:
        symbols = [str(value).strip() for value in override if str(value).strip()]
        if symbols:
            return tuple(symbols)
    return tuple(base)


def _apply_overrides_sqlite(config: MarketIntelConfig, args: argparse.Namespace) -> MarketIntelConfig:
    sqlite_cfg = config.sqlite
    if sqlite_cfg is None:
        raise ValueError("Konfiguracja market_intel nie posiada sekcji sqlite")
    if args.sqlite_path or args.sqlite_table:
        sqlite_cfg = _override_sqlite_config(
            sqlite_cfg, path=args.sqlite_path, table=args.sqlite_table
        )
    manifest_path = args.manifest if args.manifest else config.manifest_path
    output_dir = args.output_dir if args.output_dir else config.output_directory
    required_symbols = _override_required_symbols(args.required_symbols, config.required_symbols)
    default_weight = config.default_weight if args.default_weight is None else float(args.default_weight)

    return MarketIntelConfig(
        enabled=True,
        output_directory=str(output_dir),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        sqlite=sqlite_cfg,
        required_symbols=required_symbols,
        default_weight=default_weight,
    )


def _run_sqlite(args: argparse.Namespace) -> int:
    if not (_HAS_SQLITE_TYPES and SqliteAggregator is not None):
        raise SystemExit("Tryb SQLite nieobsługiwany w tej gałęzi (brak MarketIntelConfig/aggregatora SQLite). Użyj --mode ohlcv.")

    config = load_core_config(args.config)
    market_config = getattr(config, "market_intel", None)
    if market_config is None or not market_config.enabled:
        _LOGGER.warning("Sekcja market_intel jest niedostępna lub wyłączona w konfiguracji")
        return 0

    effective_config = _apply_overrides_sqlite(market_config, args)
    # Sprawdzamy, czy importowany aggregator ma API SQLite (write_outputs)
    if not hasattr(SqliteAggregator, "write_outputs"):
        raise SystemExit("W tej gałęzi MarketIntelAggregator nie wspiera write_outputs (SQLite). Użyj --mode ohlcv.")

    aggregator = SqliteAggregator(effective_config)  # type: ignore[arg-type]
    written = aggregator.write_outputs(
        output_directory=Path(effective_config.output_directory),
        manifest_path=Path(effective_config.manifest_path) if effective_config.manifest_path else None,
    )
    for path in written:
        _LOGGER.info("Zapisano %s", path)
    if written:
        print(f"Zapisano {len(written)} plików Market Intelligence (SQLite).")
    return 0


# ---------------------------- main ----------------------------
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    mode = args.mode
    if mode == "auto":
        # preferencja: jeśli mamy SQLite typy i aggregator ma write_outputs -> sqlite,
        # w przeciwnym razie jeśli mamy OHLCV -> ohlcv
        picked = None
        if _HAS_SQLITE_TYPES and SqliteAggregator is not None and hasattr(SqliteAggregator, "write_outputs"):
            picked = "sqlite"
        elif _HAS_OHLCV:
            picked = "ohlcv"
        else:
            raise SystemExit("Nie wykryto obsługi żadnego trybu Market Intelligence (brak kompatybilnych klas).")
        mode = picked
        _LOGGER.info("Auto-detected mode: %s", mode)

    if mode == "sqlite":
        return _run_sqlite(args)
    else:
        return _run_ohlcv(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
