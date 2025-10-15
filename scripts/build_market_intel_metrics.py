"""Buduje podpisane metryki Market Intelligence dla Stage6."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from bot_core.config import load_core_config
from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig
from bot_core.market_intel import MarketIntelAggregator

_LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Plik konfiguracji core (domyślnie config/core.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        help="Katalog wyjściowy metryk (domyślnie według konfiguracji)",
    )
    parser.add_argument(
        "--manifest",
        help="Ścieżka pliku manifestu (domyślnie według konfiguracji)",
    )
    parser.add_argument(
        "--sqlite-path",
        help="Nadpisuje ścieżkę do bazy SQLite z metrykami",
    )
    parser.add_argument(
        "--sqlite-table",
        help="Nadpisuje nazwę tabeli w bazie SQLite",
    )
    parser.add_argument(
        "--required-symbol",
        action="append",
        dest="required_symbols",
        help="Wymagany symbol (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--default-weight",
        type=float,
        help="Domyślna waga rynku gdy brak kolumny weight",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser


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


def _apply_overrides(config: MarketIntelConfig, args: argparse.Namespace) -> MarketIntelConfig:
    sqlite_cfg = config.sqlite
    if sqlite_cfg is None:
        raise ValueError("Konfiguracja market_intel nie posiada sekcji sqlite")
    if args.sqlite_path or args.sqlite_table:
        sqlite_cfg = _override_sqlite_config(
            sqlite_cfg,
            path=args.sqlite_path,
            table=args.sqlite_table,
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    config = load_core_config(args.config)
    market_config = getattr(config, "market_intel", None)
    if market_config is None or not market_config.enabled:
        _LOGGER.warning("Sekcja market_intel jest niedostępna lub wyłączona w konfiguracji")
        return 0

    effective_config = _apply_overrides(market_config, args)
    aggregator = MarketIntelAggregator(effective_config)
    written = aggregator.write_outputs(
        output_directory=Path(effective_config.output_directory),
        manifest_path=Path(effective_config.manifest_path) if effective_config.manifest_path else None,
    )
    for path in written:
        _LOGGER.info("Zapisano %s", path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
