"""Kontrola pokrycia danych OHLCV względem wymagań backfillu."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from math import ceil, isnan
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


@dataclass(slots=True)
class CoverageSummary:
    """Zbiorcze metryki jakości manifestu danych OHLCV."""

    total: int
    ok: int
    error: int
    warning: int
    manifest_status_counts: Mapping[str, int]
    issue_counts: Mapping[str, int]
    issue_examples: Mapping[str, str]
    stale_entries: int
    worst_gap: Mapping[str, object] | None

    @property
    def ok_ratio(self) -> float | None:
        if self.total <= 0:
            return None
        return self.ok / self.total

    @property
    def status(self) -> str:
        if self.error > 0:
            return "error"
        if self.warning > 0 or self.stale_entries > 0:
            return "warning"
        return "ok"

    def to_mapping(self) -> dict[str, object]:
        """Reprezentacja słownikowa używana przez raporty i CLI."""

        payload: dict[str, object] = {
            "status": self.status,
            "total": self.total,
            "ok": self.ok,
            "warning": self.warning,
            "error": self.error,
            "ok_ratio": self.ok_ratio,
            "manifest_status_counts": dict(self.manifest_status_counts),
            "issue_counts": dict(self.issue_counts),
            "issue_examples": dict(self.issue_examples),
            "stale_entries": self.stale_entries,
            "worst_gap": self.worst_gap,
        }
        return payload


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


def _issue_code(issue: object) -> str:
    raw = str(issue)
    prefix, _, _ = raw.partition(":")
    return prefix or raw


def _build_worst_gap(statuses: Sequence[CoverageStatus]) -> Mapping[str, object] | None:
    worst: tuple[float, CoverageStatus] | None = None
    for status in statuses:
        entry = status.manifest_entry
        gap = entry.gap_minutes
        if gap is None:
            continue
        try:
            gap_value = float(gap)
        except (TypeError, ValueError):
            continue
        if isnan(gap_value):
            continue
        if worst is None or gap_value > worst[0]:
            worst = (gap_value, status)

    if worst is None:
        return None

    gap_value, status = worst
    entry = status.manifest_entry
    payload: dict[str, object] = {
        "symbol": status.symbol,
        "interval": status.interval,
        "gap_minutes": gap_value,
        "manifest_status": entry.status,
    }
    if entry.threshold_minutes is not None:
        payload["threshold_minutes"] = int(entry.threshold_minutes)
    if entry.last_timestamp_iso is not None:
        payload["last_timestamp_iso"] = entry.last_timestamp_iso
    return payload


def summarize_coverage(statuses: Sequence[CoverageStatus]) -> CoverageSummary:
    """Buduje zagregowane metryki przydatne w automatycznych pre-checkach."""

    normalized = list(statuses)
    total = len(normalized)

    status_counts = Counter(status.status for status in normalized)
    manifest_counts = Counter(status.manifest_entry.status for status in normalized)

    issue_counts = Counter()
    issue_examples: dict[str, str] = {}
    stale_entries = 0

    for status in normalized:
        entry = status.manifest_entry
        for issue in status.issues:
            code = _issue_code(issue)
            issue_counts[code] += 1
            issue_examples.setdefault(code, str(issue))

        gap = entry.gap_minutes
        threshold = entry.threshold_minutes
        if gap is not None and threshold is not None:
            try:
                gap_value = float(gap)
                threshold_value = float(threshold)
            except (TypeError, ValueError):
                continue
            if not isnan(gap_value) and gap_value >= threshold_value:
                stale_entries += 1

    ok = int(status_counts.get("ok", 0))
    error = int(total - ok)
    warning = int(manifest_counts.get("warning", 0))

    worst_gap = _build_worst_gap(normalized)

    summary = CoverageSummary(
        total=total,
        ok=ok,
        error=error,
        warning=warning,
        manifest_status_counts=dict(manifest_counts),
        issue_counts=dict(issue_counts),
        issue_examples=issue_examples,
        stale_entries=stale_entries,
        worst_gap=worst_gap,
    )
    return summary


def coerce_summary_mapping(
    summary: CoverageSummary | Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalizuje wynik `summarize_coverage` do słownika z kompletem pól."""

    defaults: dict[str, object] = {
        "status": "unknown",
        "total": 0,
        "ok": 0,
        "warning": 0,
        "error": 0,
        "ok_ratio": None,
        "manifest_status_counts": {},
        "issue_counts": {},
        "issue_examples": {},
        "stale_entries": 0,
        "worst_gap": None,
    }

    if summary is None:
        return dict(defaults)

    payload: Mapping[str, object] | dict[str, object] | None = None

    if isinstance(summary, CoverageSummary):
        payload = summary.to_mapping()
    elif hasattr(summary, "to_mapping"):
        try:
            candidate = summary.to_mapping()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensywne dla nietypowych implementacji
            candidate = None
        if isinstance(candidate, Mapping):
            payload = candidate

    if payload is None and is_dataclass(summary):
        payload = asdict(summary)

    if payload is None and isinstance(summary, Mapping):
        payload = summary

    if payload is None:
        payload = {"status": getattr(summary, "status", defaults["status"])}

    normalized = dict(payload)
    for key, default in defaults.items():
        normalized.setdefault(key, default)
    return normalized


__all__ = [
    "CoverageStatus",
    "CoverageSummary",
    "coerce_summary_mapping",
    "evaluate_coverage",
    "summarize_coverage",
    "summarize_issues",
]
