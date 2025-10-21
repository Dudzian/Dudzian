"""Raport stanu manifestu danych OHLCV na podstawie SQLite."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import tylko dla typów
    from bot_core.config.models import InstrumentUniverseConfig
else:  # pragma: no cover - fallback podczas wczesnego bootstrapu
    InstrumentUniverseConfig = object  # type: ignore[assignment]

from bot_core.data.ohlcv.utils import interval_to_minutes

@dataclass(slots=True)
class ManifestEntry:
    """Pojedynczy wpis raportu z manifestu danych OHLCV."""

    symbol: str
    interval: str
    row_count: int | None
    last_timestamp_ms: int | None
    last_timestamp_iso: str | None
    gap_minutes: float | None
    threshold_minutes: int | None
    status: str


def _default_threshold_minutes(interval: str) -> int:
    return max(1, interval_to_minutes(interval) * 2)


def _parse_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _load_metadata(path: Path) -> Mapping[str, str]:
    if not path.exists():
        return {}
    connection = sqlite3.connect(path)
    try:
        cursor = connection.execute("SELECT key, value FROM metadata")
        return {str(row[0]): str(row[1]) for row in cursor.fetchall()}
    finally:
        connection.close()


def _expected_pairs(
    universe: InstrumentUniverseConfig,
    exchange_name: str,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange_name)
        if not symbol:
            continue
        for window in instrument.backfill_windows:
            pairs.add((symbol, window.interval))
    return pairs


def generate_manifest_report(
    *,
    manifest_path: str | Path,
    universe: InstrumentUniverseConfig,
    exchange_name: str,
    as_of: datetime | None = None,
    warning_thresholds: Mapping[str, int] | None = None,
) -> list[ManifestEntry]:
    """Buduje raport stanu manifestu SQLite dla wskazanego uniwersum."""

    snapshot = _load_metadata(Path(manifest_path))
    pairs = _expected_pairs(universe, exchange_name)
    if not pairs:
        return []

    as_of_dt = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows: list[ManifestEntry] = []

    for symbol, interval in sorted(pairs):
        last_key = f"last_timestamp::{symbol}::{interval}"
        row_key = f"row_count::{symbol}::{interval}"
        raw_last = snapshot.get(last_key)
        raw_rows = snapshot.get(row_key)

        row_count = _parse_int(raw_rows)
        threshold = None
        if warning_thresholds and interval in warning_thresholds:
            threshold = max(1, int(warning_thresholds[interval]))
        else:
            try:
                threshold = _default_threshold_minutes(interval)
            except ValueError:
                threshold = None

        if raw_last is None:
            rows.append(
                ManifestEntry(
                    symbol=symbol,
                    interval=interval,
                    row_count=row_count,
                    last_timestamp_ms=None,
                    last_timestamp_iso=None,
                    gap_minutes=None,
                    threshold_minutes=threshold,
                    status="missing_metadata",
                )
            )
            continue

        last_ts = _parse_int(raw_last)
        if last_ts is None:
            rows.append(
                ManifestEntry(
                    symbol=symbol,
                    interval=interval,
                    row_count=row_count,
                    last_timestamp_ms=None,
                    last_timestamp_iso=str(raw_last),
                    gap_minutes=None,
                    threshold_minutes=threshold,
                    status="invalid_metadata",
                )
            )
            continue

        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        gap_minutes = max(0.0, (as_of_dt - last_dt).total_seconds() / 60)

        status = "ok"
        if threshold is None:
            status = "unknown"
        elif gap_minutes >= threshold:
            status = "warning"

        rows.append(
            ManifestEntry(
                symbol=symbol,
                interval=interval,
                row_count=row_count,
                last_timestamp_ms=last_ts,
                last_timestamp_iso=last_dt.isoformat(),
                gap_minutes=gap_minutes,
                threshold_minutes=threshold,
                status=status,
            )
        )

    return rows


def summarize_status(entries: Sequence[ManifestEntry]) -> Mapping[str, int]:
    summary: dict[str, int] = {}
    for entry in entries:
        summary[entry.status] = summary.get(entry.status, 0) + 1
    return summary


__all__ = ["ManifestEntry", "generate_manifest_report", "summarize_status"]
