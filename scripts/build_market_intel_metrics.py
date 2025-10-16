#!/usr/bin/env python3
"""Buduje metryki Market Intelligence dla PortfolioGovernora Stage6."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bot_core.config import load_core_config
from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery


def _default_output(governor: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("var/market_intel") / f"market_intel_{governor}_{timestamp}.json"


def _write_json(path: Path, payload: dict[str, object], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generuje raport Market Intelligence Stage6")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument("--environment", required=True, help="Nazwa środowiska z core.yaml")
    parser.add_argument("--governor", required=True, help="Nazwa PortfolioGovernora")
    parser.add_argument("--interval", default="1h", help="Interwał OHLCV (domyślnie 1h)")
    parser.add_argument(
        "--lookback-bars",
        type=int,
        default=168,
        help="Liczba świec użyta do agregacji (domyślnie 168)",
    )
    parser.add_argument(
        "--cache-base",
        help="Nadrzędny katalog cache OHLCV (domyślnie z configu środowiska)",
    )
    parser.add_argument(
        "--namespace",
        help="Namespace cache Parquet (domyślnie wartość environment z core.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Ścieżka pliku wynikowego (domyślnie var/market_intel/...)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Format eksportu (json lub csv)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Lista symboli do przetworzenia (domyślnie aktywa governora)",
    )
    parser.add_argument("--pretty", action="store_true", help="Format JSON z wcięciami")
    args = parser.parse_args()

    core_config = load_core_config(args.config)
    try:
        environment_cfg = core_config.environments[args.environment]
    except KeyError as exc:  # pragma: no cover - defensywna walidacja CLI
        raise SystemExit(f"Nie znaleziono środowiska {args.environment} w {args.config}") from exc

    if args.governor not in core_config.portfolio_governors:
        raise SystemExit(f"PortfolioGovernor {args.governor} nie istnieje w configu")
    governor_cfg = core_config.portfolio_governors[args.governor]

    cache_base = args.cache_base or environment_cfg.data_cache_path
    namespace = args.namespace or environment_cfg.environment.value
    if not cache_base:
        raise SystemExit("Brak ścieżki cache dla środowiska – uzupełnij data_cache_path")

    from bot_core.data.ohlcv.parquet_storage import ParquetCacheStorage

    storage = ParquetCacheStorage(cache_base, namespace=namespace)
    aggregator = MarketIntelAggregator(storage)

    symbols = args.symbols or [asset.symbol for asset in governor_cfg.assets]
    if not symbols:
        raise SystemExit("PortfolioGovernor nie ma zdefiniowanych aktywów do agregacji")

    queries = [
        MarketIntelQuery(symbol=symbol, interval=args.interval, lookback_bars=args.lookback_bars)
        for symbol in symbols
    ]
    snapshots = aggregator.build_many(queries)
    generated_at = datetime.now(timezone.utc).isoformat()

    output_path = Path(args.output) if args.output else _default_output(args.governor)

    if args.format == "json":
        payload = {
            "generated_at": generated_at,
            "environment": args.environment,
            "governor": args.governor,
            "interval": args.interval,
            "lookback_bars": args.lookback_bars,
            "symbols": symbols,
            "snapshots": {symbol: snapshot.to_dict() for symbol, snapshot in snapshots.items()},
        }
        _write_json(output_path, payload, args.pretty)
    else:
        rows = [
            (symbol, snapshot.to_dict())
            for symbol, snapshot in snapshots.items()
        ]
        _write_csv(output_path, rows)

    print(f"Zapisano raport Market Intelligence do {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
