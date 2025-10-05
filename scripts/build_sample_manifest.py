"""Generator zanonimizowanego manifestu OHLCV do testów i CI."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_SYMBOLS = {
    "BTCUSDT": {"interval": "1d"},
    "ETHUSDT": {"interval": "1d"},
}


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buduje zanonimizowany manifest OHLCV na potrzeby smoketestów coverage",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/assets/coverage_sample",
        help="Katalog docelowy, gdzie zostanie zapisany manifest SQLite",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=40,
        help="Liczba obserwacji na każdy symbol/interwał",
    )
    parser.add_argument(
        "--last-date",
        default="2024-01-30",
        help="Data (UTC) ostatniej świecy w formacie YYYY-MM-DD",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=sorted(DEFAULT_SYMBOLS.keys()),
        help="Lista symboli do wygenerowania (domyślnie BTCUSDT i ETHUSDT)",
    )
    return parser.parse_args(argv)


def _ensure_metadata_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _write_entry(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    interval: str,
    last_dt: datetime,
    rows: int,
) -> None:
    timestamp_ms = int(last_dt.timestamp() * 1000)
    entries: Mapping[str, str] = {
        f"last_timestamp::{symbol}::{interval}": str(timestamp_ms),
        f"row_count::{symbol}::{interval}": str(rows),
    }
    for key, value in entries.items():
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def build_sample_manifest(
    *,
    output_dir: Path,
    symbols: Iterable[str],
    rows: int,
    last_date: datetime,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ohlcv_manifest.sqlite"
    with sqlite3.connect(manifest_path) as connection:
        _ensure_metadata_table(connection)
        for symbol in symbols:
            spec = DEFAULT_SYMBOLS.get(symbol) or {"interval": "1d"}
            interval = spec["interval"]
            last_dt = last_date.astimezone(timezone.utc)
            # Zakładamy równomierny rozkład świec, więc wyznaczamy początek z odstępem 1d
            start_dt = last_dt - timedelta(days=max(rows - 1, 0))
            _write_entry(
                connection,
                symbol=symbol,
                interval=interval,
                last_dt=last_dt,
                rows=rows,
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (f"first_timestamp::{symbol}::{interval}", str(int(start_dt.timestamp() * 1000))),
            )
    return manifest_path


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        last_date = datetime.fromisoformat(args.last_date)
    except ValueError:  # pragma: no cover - walidacja wejścia CLI
        print(f"Niepoprawny format daty: {args.last_date}", file=sys.stderr)
        return 2
    if last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=timezone.utc)
    manifest_path = build_sample_manifest(
        output_dir=Path(args.output_dir),
        symbols=args.symbols,
        rows=args.rows,
        last_date=last_date,
    )
    print(str(manifest_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())
