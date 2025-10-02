"""Kontrola pokrycia danych OHLCV względem wymagań backfillu."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bot_core.config.models import InstrumentBackfillWindow, InstrumentUniverseConfig
from bot_core.data.ohlcv.manifest_report import ManifestEntry, generate_manifest_report


@dataclass(slots=True)
class CoverageStatus:
    """Rezultat walidacji pojedynczego symbolu/interwału."""

    symbol: str
    interval: str
    manifest_entry: ManifestEntry
    required_rows: int | None
    issues: Sequence[str]

    @property
    def status(self) -> str:
        return "ok" if not self.issues else "error"


def _interval_minutes(interval: str) -> int:
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
        "3d": 4320,
        "1w": 10_080,
        "1M": 43_200,
    }
    try:
        return mapping[interval]
    except KeyError as exc:  # pragma: no cover - format wejściowy waliduje konfiguracja
        raise ValueError(f"Nieobsługiwany interwał: {interval}") from exc


def _build_requirements(
    universe: InstrumentUniverseConfig,
    exchange_name: str,
) -> Mapping[tuple[str, str], InstrumentBackfillWindow]:
    requirements: dict[tuple[str, str], InstrumentBackfillWindow] = {}
    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange_name)
        if not symbol:
            continue
        for window in instrument.backfill_windows:
            key = (symbol, window.interval)
            existing = requirements.get(key)
            if existing is None or window.lookback_days > existing.lookback_days:
                requirements[key] = window
    return requirements


def evaluate_coverage(
    *,
    manifest_path: str | Path,
    universe: InstrumentUniverseConfig,
    exchange_name: str,
    as_of: datetime | None = None,
) -> Sequence[CoverageStatus]:
    """Zwraca status pokrycia danych względem manifestu."""

    as_of_dt = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
    manifest_entries = generate_manifest_report(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=exchange_name,
        as_of=as_of_dt,
    )
    requirements = _build_requirements(universe, exchange_name)

    statuses: list[CoverageStatus] = []
    for entry in manifest_entries:
        window = requirements.get((entry.symbol, entry.interval))
        required_rows: int | None = None
        issues: list[str] = []

        if entry.status != "ok":
            issues.append(f"manifest_status:{entry.status}")

        if window is not None:
            try:
                interval_minutes = _interval_minutes(entry.interval)
                required_rows = ceil(window.lookback_days * 24 * 60 / interval_minutes)
            except ValueError as exc:
                issues.append(str(exc))
            else:
                row_count = entry.row_count
                if row_count is None:
                    issues.append("missing_row_count")
                elif row_count < required_rows:
                    issues.append(
                        f"insufficient_rows:{row_count}<{required_rows}"
                    )
        statuses.append(
            CoverageStatus(
                symbol=entry.symbol,
                interval=entry.interval,
                manifest_entry=entry,
                required_rows=required_rows,
                issues=tuple(issues),
            )
        )
    return statuses


def summarize_issues(statuses: Iterable[CoverageStatus]) -> list[str]:
    """Agreguje komunikaty o problemach w listę opisową."""

    issues: list[str] = []
    for status in statuses:
        if not status.issues:
            continue
        for issue in status.issues:
            issues.append(f"{status.symbol}/{status.interval}: {issue}")
    return issues


__all__ = ["CoverageStatus", "evaluate_coverage", "summarize_issues"]
