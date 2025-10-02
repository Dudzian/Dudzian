"""Kontrola pokrycia danych OHLCV względem wymagań backfillu."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bot_core.config.models import InstrumentBackfillWindow, InstrumentUniverseConfig
from bot_core.data.intervals import interval_to_milliseconds, normalize_interval_token
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
    try:
        return interval_to_milliseconds(interval) // 60_000
    except ValueError as exc:  # pragma: no cover - format wejściowy waliduje konfiguracja
        raise ValueError(f"Nieobsługiwany interwał: {interval}") from exc


def _build_requirements(
    universe: InstrumentUniverseConfig,
    exchange_name: str,
    allowed_intervals: set[str] | None = None,
) -> Mapping[tuple[str, str], InstrumentBackfillWindow]:
    requirements: dict[tuple[str, str], InstrumentBackfillWindow] = {}
    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange_name)
        if not symbol:
            continue
        for window in instrument.backfill_windows:
            normalized = normalize_interval_token(window.interval)
            if not normalized:
                continue
            if allowed_intervals is not None and normalized not in allowed_intervals:
                continue
            key = (symbol, normalized)
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
    intervals: Sequence[str] | None = None,
) -> Sequence[CoverageStatus]:
    """Zwraca status pokrycia danych względem manifestu."""

    as_of_dt = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
    allowed_intervals: set[str] | None = None
    if intervals:
        allowed_intervals = {
            token
            for token in (
                normalize_interval_token(value) for value in intervals if value is not None
            )
            if token
        }
        if not allowed_intervals:
            allowed_intervals = set()

    manifest_entries = generate_manifest_report(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=exchange_name,
        as_of=as_of_dt,
    )
    requirements = _build_requirements(universe, exchange_name, allowed_intervals)

    statuses: list[CoverageStatus] = []
    for entry in manifest_entries:
        normalized_interval = normalize_interval_token(entry.interval)
        if allowed_intervals is not None and (
            not normalized_interval or normalized_interval not in allowed_intervals
        ):
            continue

        window = requirements.get((entry.symbol, normalized_interval or entry.interval))
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


def summarize_coverage(statuses: Iterable[CoverageStatus]) -> Mapping[str, object]:
    """Zwraca zagregowane informacje o pokryciu manifestu."""

    total = 0
    ok = 0
    errors = 0
    manifest_counts: dict[str, int] = {}
    worst_gap_minutes: float | int | None = None
    worst_gap_symbol: tuple[str, str] | None = None

    for status in statuses:
        total += 1
        if status.issues:
            errors += 1
        else:
            ok += 1

        manifest_status = status.manifest_entry.status or "unknown"
        manifest_counts[manifest_status] = manifest_counts.get(manifest_status, 0) + 1

        gap = status.manifest_entry.gap_minutes
        if gap is not None:
            try:
                numeric_gap = float(gap)
            except (TypeError, ValueError):  # pragma: no cover - defensywnie
                continue
            if worst_gap_minutes is None or numeric_gap > float(worst_gap_minutes):
                worst_gap_minutes = numeric_gap
                worst_gap_symbol = (status.symbol, status.interval)

    payload: dict[str, object] = {
        "total": total,
        "ok": ok,
        "error": errors,
        "manifest_status_counts": manifest_counts,
    }
    if worst_gap_minutes is not None and worst_gap_symbol is not None:
        payload["worst_gap"] = {
            "symbol": worst_gap_symbol[0],
            "interval": worst_gap_symbol[1],
            "gap_minutes": worst_gap_minutes,
        }

    return payload


__all__ = ["CoverageStatus", "evaluate_coverage", "summarize_issues", "summarize_coverage"]
