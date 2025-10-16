"""Kalibracja progów Stress Lab Stage6 na podstawie metryk rynkowych."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import math

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.risk.simulation import RiskSimulationReport
from bot_core.security.signing import build_hmac_signature

_REPORT_SCHEMA = "stage6.risk.stress_lab.calibration"
_REPORT_SCHEMA_VERSION = 1
_SIGNATURE_SCHEMA = "stage6.risk.stress_lab.calibration.signature"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Nie można policzyć percentyla pustej kolekcji")
    if quantile <= 0:
        return min(values)
    if quantile >= 1:
        return max(values)
    sorted_values = sorted(values)
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


@dataclass(slots=True)
class StressLabCalibrationSettings:
    """Parametry sterujące obliczeniami progów."""

    liquidity_warning_percentile: float = 0.35
    liquidity_critical_percentile: float = 0.2
    latency_warning_percentile: float = 0.65
    latency_critical_percentile: float = 0.85
    min_liquidity_threshold: float = 25_000.0
    min_latency_threshold_ms: float = 120.0


@dataclass(slots=True)
class StressLabCalibrationSegment:
    """Deklaracja segmentu kalibracyjnego (np. core vs DeFi)."""

    name: str
    symbols: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    risk_budgets: Sequence[str] = field(default_factory=tuple)


def build_volume_segments(
    market_snapshots: Mapping[str, MarketIntelSnapshot],
    *,
    buckets: int,
    min_symbols_per_bucket: int = 3,
    name_prefix: str = "volume",
    risk_budget_prefix: str | None = None,
) -> tuple[StressLabCalibrationSegment, ...]:
    """Buduje segmenty kalibracyjne na podstawie płynności/volumenu.

    Segmenty powstają poprzez posortowanie aktywów malejąco po wartości
    ``liquidity_usd`` i podział listy na zadane kubełki. Pozwala to automatycznie
    kalibrować progi dla grup o wysokiej i niskiej płynności bez przygotowywania
    ręcznych definicji.

    Raises:
        ValueError: Gdy ``buckets`` jest mniejsze niż 1 lub brak jest
            wystarczających metryk płynności do wyznaczenia segmentów.
    """

    if buckets < 1:
        raise ValueError("Liczba kubełków musi być dodatnia")

    ranked: list[tuple[str, float]] = [
        (symbol, float(snapshot.liquidity_usd))
        for symbol, snapshot in market_snapshots.items()
        if snapshot.liquidity_usd is not None
    ]
    if not ranked:
        raise ValueError("Brak metryk płynności w danych Market Intel")

    ranked.sort(key=lambda item: item[1], reverse=True)
    bucket_count = min(buckets, len(ranked))

    min_per_bucket = (
        min_symbols_per_bucket
        if len(ranked) >= min_symbols_per_bucket * bucket_count
        else 1
    )

    segments: list[StressLabCalibrationSegment] = []
    cursor = 0
    for index in range(bucket_count):
        remaining = len(ranked) - cursor
        if remaining <= 0:
            break
        buckets_left = bucket_count - index

        average_size = math.ceil(remaining / buckets_left)
        max_allowed = max(1, remaining - min_per_bucket * (buckets_left - 1))

        size = max(min_per_bucket, min(average_size, max_allowed, remaining))

        bucket_symbols = [symbol for symbol, _ in ranked[cursor : cursor + size]]
        risk_budgets: Sequence[str]
        if risk_budget_prefix is not None:
            risk_budgets = (f"{risk_budget_prefix}{index + 1}",)
        else:
            risk_budgets = ()
        segments.append(
            StressLabCalibrationSegment(
                name=f"{name_prefix}_{index + 1}",
                symbols=tuple(bucket_symbols),
                risk_budgets=risk_budgets,
            )
        )
        cursor += size

    return tuple(segments)


@dataclass(slots=True)
class StressLabSegmentThresholds:
    """Wynik kalibracji dla pojedynczego segmentu."""

    segment: str
    liquidity_warning_threshold_usd: float | None
    liquidity_critical_threshold_usd: float | None
    symbol_count: int
    reference_symbols: Sequence[str] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "segment": self.segment,
            "symbol_count": self.symbol_count,
            "liquidity_warning_threshold_usd": self.liquidity_warning_threshold_usd,
            "liquidity_critical_threshold_usd": self.liquidity_critical_threshold_usd,
        }
        if self.reference_symbols:
            payload["symbols"] = list(self.reference_symbols)
        return payload


@dataclass(slots=True)
class StressLabCalibrationReport:
    """Zbiorczy raport kalibracji progów Stress Lab."""

    generated_at: datetime
    settings: StressLabCalibrationSettings
    liquidity_segments: Sequence[StressLabSegmentThresholds]
    latency_warning_threshold_ms: float | None
    latency_critical_threshold_ms: float | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": _REPORT_SCHEMA,
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "settings": {
                "liquidity_warning_percentile": self.settings.liquidity_warning_percentile,
                "liquidity_critical_percentile": self.settings.liquidity_critical_percentile,
                "latency_warning_percentile": self.settings.latency_warning_percentile,
                "latency_critical_percentile": self.settings.latency_critical_percentile,
                "min_liquidity_threshold": self.settings.min_liquidity_threshold,
                "min_latency_threshold_ms": self.settings.min_latency_threshold_ms,
            },
            "segments": [segment.to_payload() for segment in self.liquidity_segments],
            "latency_warning_threshold_ms": self.latency_warning_threshold_ms,
            "latency_critical_threshold_ms": self.latency_critical_threshold_ms,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class StressLabCalibrator:
    """Agreguje dane rynku i raporty ryzyka w celu kalibracji progów."""

    def __init__(
        self,
        *,
        settings: StressLabCalibrationSettings | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings or StressLabCalibrationSettings()
        self._clock = clock or _now_utc

    def calibrate(
        self,
        *,
        market_snapshots: Mapping[str, MarketIntelSnapshot],
        segments: Iterable[StressLabCalibrationSegment],
        risk_report: RiskSimulationReport | None = None,
    ) -> StressLabCalibrationReport:
        liquidity_results: list[StressLabSegmentThresholds] = []
        total_symbols = 0
        for segment in segments:
            thresholds = self._calibrate_segment(segment, market_snapshots)
            liquidity_results.append(thresholds)
            total_symbols += thresholds.symbol_count

        latency_warning, latency_critical = self._calibrate_latency(risk_report)

        metadata: MutableMapping[str, object] = {
            "segments": len(liquidity_results),
            "symbols_considered": total_symbols,
        }
        if risk_report is not None:
            metadata["profiles_evaluated"] = len(risk_report.profiles)

        return StressLabCalibrationReport(
            generated_at=self._clock(),
            settings=self._settings,
            liquidity_segments=tuple(liquidity_results),
            latency_warning_threshold_ms=latency_warning,
            latency_critical_threshold_ms=latency_critical,
            metadata=metadata,
        )

    def _calibrate_segment(
        self,
        segment: StressLabCalibrationSegment,
        market_snapshots: Mapping[str, MarketIntelSnapshot],
    ) -> StressLabSegmentThresholds:
        symbols = set(segment.symbols)
        if segment.tags or segment.risk_budgets:
            # Jeżeli segment zdefiniowano po tagach/budżetach, pozostawiamy filtrację
            symbols.update(
                symbol
                for symbol, snapshot in market_snapshots.items()
                if _matches_segment(snapshot, segment)
            )

        liquidity_values = [
            float(snapshot.liquidity_usd)
            for symbol, snapshot in market_snapshots.items()
            if symbol in symbols and snapshot.liquidity_usd is not None
        ]

        if not liquidity_values:
            return StressLabSegmentThresholds(
                segment=segment.name,
                liquidity_warning_threshold_usd=None,
                liquidity_critical_threshold_usd=None,
                symbol_count=len(symbols),
                reference_symbols=tuple(sorted(symbols)),
            )

        warning = max(
            self._settings.min_liquidity_threshold,
            _percentile(liquidity_values, self._settings.liquidity_warning_percentile),
        )
        critical = max(
            self._settings.min_liquidity_threshold,
            _percentile(liquidity_values, self._settings.liquidity_critical_percentile),
        )

        return StressLabSegmentThresholds(
            segment=segment.name,
            liquidity_warning_threshold_usd=warning,
            liquidity_critical_threshold_usd=critical,
            symbol_count=len(symbols),
            reference_symbols=tuple(sorted(symbols)),
        )

    def _calibrate_latency(
        self, risk_report: RiskSimulationReport | None
    ) -> tuple[float | None, float | None]:
        if risk_report is None:
            return None, None

        latency_values: list[float] = []
        for profile in risk_report.profiles:
            for stress in profile.stress_tests:
                candidates = [
                    stress.metrics.get("avg_order_latency_ms"),
                    stress.metrics.get("p95_order_latency_ms"),
                    stress.metrics.get("max_order_latency_ms"),
                    stress.metrics.get("latency_ms"),
                ]
                latency_values.extend(
                    float(value)
                    for value in candidates
                    if value is not None and isinstance(value, (int, float))
                )

        if not latency_values:
            return None, None

        warning = max(
            self._settings.min_latency_threshold_ms,
            _percentile(latency_values, self._settings.latency_warning_percentile),
        )
        critical = max(
            self._settings.min_latency_threshold_ms,
            _percentile(latency_values, self._settings.latency_critical_percentile),
        )
        return warning, critical


def _matches_segment(snapshot: MarketIntelSnapshot, segment: StressLabCalibrationSegment) -> bool:
    metadata_tags = set()
    metadata_budget = None
    if snapshot.metadata:
        tags = snapshot.metadata.get("tags")
        if isinstance(tags, (list, tuple, set)):
            metadata_tags = {str(tag) for tag in tags}
        budget = snapshot.metadata.get("risk_budget")
        if budget is not None:
            metadata_budget = str(budget)

    if segment.tags and metadata_tags.intersection(segment.tags):
        return True
    if segment.risk_budgets and metadata_budget in segment.risk_budgets:
        return True
    return False


def write_calibration_json(report: StressLabCalibrationReport, path: Path) -> dict[str, object]:
    payload = report.to_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def write_calibration_csv(report: StressLabCalibrationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "segment",
                "symbol_count",
                "liquidity_warning_threshold_usd",
                "liquidity_critical_threshold_usd",
                "symbols",
            ),
        )
        writer.writeheader()
        for segment in report.liquidity_segments:
            row = segment.to_payload()
            row["symbols"] = ",".join(segment.reference_symbols)
            writer.writerow(row)


def write_calibration_signature(
    payload: Mapping[str, object],
    path: Path,
    *,
    key: bytes,
    key_id: str | None,
    target: str | None = None,
) -> Mapping[str, object]:
    document = {
        "schema": _SIGNATURE_SCHEMA,
        "schema_version": _REPORT_SCHEMA_VERSION,
        "signed_at": _now_utc().isoformat(),
        "target": target or path.name,
        "signature": build_hmac_signature(payload, key=key, key_id=key_id),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return document


__all__ = [
    "StressLabCalibrator",
    "StressLabCalibrationReport",
    "StressLabCalibrationSegment",
    "StressLabCalibrationSettings",
    "StressLabSegmentThresholds",
    "build_volume_segments",
    "write_calibration_csv",
    "write_calibration_json",
    "write_calibration_signature",
]
