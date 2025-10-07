"""Kontrola pokrycia danych OHLCV względem wymagań backfillu."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from math import ceil, floor, isnan
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.config.models import (
    EnvironmentDataQualityConfig,
    InstrumentBackfillWindow,
    InstrumentUniverseConfig,
)
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


@dataclass(slots=True)
class GapStatistics:
    """Podstawowe statystyki dotyczące luk czasowych w manifeście."""

    total_entries: int
    with_gap_measurement: int
    min_gap_minutes: float | None
    max_gap_minutes: float | None
    median_gap_minutes: float | None
    percentile_95_gap_minutes: float | None
    percentile_99_gap_minutes: float | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "total_entries": self.total_entries,
            "with_gap_measurement": self.with_gap_measurement,
            "min_gap_minutes": self.min_gap_minutes,
            "max_gap_minutes": self.max_gap_minutes,
            "median_gap_minutes": self.median_gap_minutes,
            "percentile_95_gap_minutes": self.percentile_95_gap_minutes,
            "percentile_99_gap_minutes": self.percentile_99_gap_minutes,
        }


@dataclass(slots=True)
class SummaryThresholdResult:
    """Wynik walidacji progów jakości danych."""

    issues: tuple[str, ...] = ()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    observed: Mapping[str, float | None] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return {
            "issues": list(self.issues),
            "thresholds": dict(self.thresholds),
            "observed": dict(self.observed),
        }


@dataclass(slots=True)
class CoverageReportPayload:
    """Pakiet danych wykorzystywany przez runnera alertów i raporty."""

    payload: Mapping[str, object]
    statuses: tuple[CoverageStatus, ...]
    summary: Mapping[str, object]
    threshold_result: SummaryThresholdResult | None
    threshold_issues: tuple[str, ...]
    issues: tuple[str, ...]
    gap_statistics: GapStatistics | None = None
    gap_statistics_by_interval: Mapping[str, GapStatistics] | None = None

    def to_mapping(self) -> dict[str, object]:
        data: MutableMapping[str, object] = dict(self.payload)
        if self.gap_statistics is not None:
            data["gap_statistics"] = self.gap_statistics.to_mapping()
        if self.gap_statistics_by_interval:
            data["gap_statistics_by_interval"] = {
                interval: stats.to_mapping()
                for interval, stats in self.gap_statistics_by_interval.items()
            }
        return dict(data)


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


def status_to_mapping(status: CoverageStatus) -> dict[str, object]:
    """Serializuje wynik walidacji do formatu JSON-owego."""

    entry = status.manifest_entry
    payload: dict[str, object] = {
        "symbol": status.symbol,
        "interval": status.interval,
        "status": status.status,
        "issues": list(status.issues),
        "required_rows": status.required_rows,
        "row_count": entry.row_count,
        "gap_minutes": entry.gap_minutes,
        "threshold_minutes": entry.threshold_minutes,
        "last_timestamp_iso": entry.last_timestamp_iso,
    }
    return payload


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    q = max(0.0, min(1.0, quantile))
    position = (len(values) - 1) * q
    lower = floor(position)
    upper = min(len(values) - 1, lower + 1)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] + (values[upper] - values[lower]) * weight)


def compute_gap_statistics(statuses: Sequence[CoverageStatus]) -> GapStatistics:
    """Oblicza statystyki luk czasowych dla całego zestawu statusów."""

    total = len(statuses)
    if total == 0:
        return GapStatistics(
            total_entries=0,
            with_gap_measurement=0,
            min_gap_minutes=None,
            max_gap_minutes=None,
            median_gap_minutes=None,
            percentile_95_gap_minutes=None,
            percentile_99_gap_minutes=None,
        )

    gaps: list[float] = []
    for status in statuses:
        gap = status.manifest_entry.gap_minutes
        if gap is None:
            continue
        try:
            value = float(gap)
        except (TypeError, ValueError):
            continue
        if isnan(value):
            continue
        gaps.append(value)

    if not gaps:
        return GapStatistics(
            total_entries=total,
            with_gap_measurement=0,
            min_gap_minutes=None,
            max_gap_minutes=None,
            median_gap_minutes=None,
            percentile_95_gap_minutes=None,
            percentile_99_gap_minutes=None,
        )

    ordered = sorted(gaps)
    stats = GapStatistics(
        total_entries=total,
        with_gap_measurement=len(ordered),
        min_gap_minutes=float(ordered[0]),
        max_gap_minutes=float(ordered[-1]),
        median_gap_minutes=float(median(ordered)),
        percentile_95_gap_minutes=_percentile(ordered, 0.95),
        percentile_99_gap_minutes=_percentile(ordered, 0.99),
    )
    return stats


def compute_gap_statistics_by_interval(
    statuses: Sequence[CoverageStatus],
) -> Mapping[str, GapStatistics]:
    """Buduje statystyki luk z rozbiciem na interwały."""

    per_interval: dict[str, list[CoverageStatus]] = {}
    for status in statuses:
        per_interval.setdefault(status.interval, []).append(status)

    result: dict[str, GapStatistics] = {}
    for interval, items in per_interval.items():
        result[interval] = compute_gap_statistics(items)
    return result


def evaluate_summary_thresholds(
    summary: CoverageSummary | Mapping[str, object] | None,
    *,
    max_gap_minutes: float | None = None,
    min_ok_ratio: float | None = None,
) -> SummaryThresholdResult:
    """Porównuje zagregowane metryki z zadanymi progami."""

    thresholds: dict[str, float] = {}
    observed: dict[str, float | None] = {}
    issues: list[str] = []

    if summary is None:
        summary_payload: Mapping[str, object] = {}
    elif isinstance(summary, CoverageSummary):
        summary_payload = summary.to_mapping()
    else:
        summary_payload = dict(summary)

    worst_gap = summary_payload.get("worst_gap") if summary_payload else None
    worst_gap_minutes: float | None = None
    if isinstance(worst_gap, Mapping):
        raw_gap = worst_gap.get("gap_minutes")
        try:
            worst_gap_minutes = float(raw_gap) if raw_gap is not None else None
        except (TypeError, ValueError):
            worst_gap_minutes = None
    elif worst_gap is not None:
        try:
            worst_gap_minutes = float(worst_gap)
        except (TypeError, ValueError):
            worst_gap_minutes = None

    if max_gap_minutes is not None:
        thresholds["max_gap_minutes"] = float(max_gap_minutes)
        observed["worst_gap_minutes"] = worst_gap_minutes
        if worst_gap_minutes is not None and worst_gap_minutes > float(max_gap_minutes):
            issues.append(f"max_gap_exceeded:{worst_gap_minutes}>{float(max_gap_minutes)}")

    ok_ratio_value: float | None = None
    if summary_payload:
        raw_ok_ratio = summary_payload.get("ok_ratio")
        try:
            ok_ratio_value = float(raw_ok_ratio) if raw_ok_ratio is not None else None
        except (TypeError, ValueError):
            ok_ratio_value = None

    if min_ok_ratio is not None:
        thresholds["min_ok_ratio"] = float(min_ok_ratio)
        observed["ok_ratio"] = ok_ratio_value
        if ok_ratio_value is None:
            issues.append("ok_ratio_missing")
        elif ok_ratio_value < float(min_ok_ratio):
            issues.append(f"ok_ratio_below_threshold:{ok_ratio_value}<{float(min_ok_ratio)}")

    if "ok_ratio" not in observed and min_ok_ratio is None:
        observed["ok_ratio"] = ok_ratio_value
    if "worst_gap_minutes" not in observed and max_gap_minutes is None:
        observed["worst_gap_minutes"] = worst_gap_minutes

    return SummaryThresholdResult(
        issues=tuple(issues),
        thresholds=thresholds,
        observed=observed,
    )


__all__ = [
    "CoverageStatus",
    "CoverageSummary",
    "CoverageReportPayload",
    "GapStatistics",
    "SummaryThresholdResult",
    "coerce_summary_mapping",
    "compute_gap_statistics",
    "compute_gap_statistics_by_interval",
    "evaluate_summary_thresholds",
    "evaluate_coverage",
    "status_to_mapping",
    "summarize_coverage",
    "summarize_issues",
]
