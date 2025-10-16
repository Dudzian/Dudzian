"""Monitorowanie SLO dla Observability Stage6."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

_SEVERITY_ORDER = {
    "debug": -1,
    "info": 0,
    "notice": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}


def _normalize_severity(value: str) -> str:
    normalized = (value or "info").lower()
    if normalized not in _SEVERITY_ORDER:
        return "warning"
    return normalized


@dataclass(slots=True)
class SLODefinition:
    """Definicja SLO oparta na pojedynczym wskaźniku."""

    name: str
    indicator: str
    target: float
    comparison: str = ">="
    warning_threshold: float | None = None
    severity: str = "critical"
    description: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.comparison not in {">=", "<="}:
            raise ValueError("Dozwolone porównania to '>=' oraz '<='")
        self.severity = _normalize_severity(self.severity)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "indicator": self.indicator,
            "target": self.target,
            "comparison": self.comparison,
            "severity": self.severity,
        }
        if self.warning_threshold is not None:
            payload["warning_threshold"] = self.warning_threshold
        if self.description:
            payload["description"] = self.description
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(slots=True)
class SLOCompositeDefinition:
    """Definicja kompozytowego (SLO2) celu składającego się z kilku SLO."""

    name: str
    objectives: Sequence[str]
    max_breaches: int = 0
    max_warnings: int | None = None
    min_ok_ratio: float | None = None
    severity: str = "critical"
    description: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("Kompozyt SLO musi zawierać co najmniej jeden cel")
        if self.max_breaches < 0:
            raise ValueError("max_breaches nie może być ujemne")
        if self.max_warnings is not None and self.max_warnings < 0:
            raise ValueError("max_warnings nie może być ujemne")
        if self.min_ok_ratio is not None and not (0 <= self.min_ok_ratio <= 1):
            raise ValueError("min_ok_ratio musi zawierać się w przedziale <0,1>")
        self.severity = _normalize_severity(self.severity)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "objectives": list(self.objectives),
            "max_breaches": self.max_breaches,
            "severity": self.severity,
        }
        if self.max_warnings is not None:
            payload["max_warnings"] = self.max_warnings
        if self.min_ok_ratio is not None:
            payload["min_ok_ratio"] = self.min_ok_ratio
        if self.description:
            payload["description"] = self.description
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(slots=True)
class SLOMeasurement:
    """Pomiar wskaźnika używanego w SLO."""

    indicator: str
    value: float | None
    window_start: datetime | None = None
    window_end: datetime | None = None
    sample_size: int = 0
    metadata: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "indicator": self.indicator,
            "value": self.value,
            "sample_size": self.sample_size,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class SLOStatus:
    """Wynik ewaluacji pojedynczego SLO."""

    name: str
    indicator: str
    value: float | None
    target: float
    comparison: str
    status: str
    severity: str
    warning_threshold: float | None = None
    error_budget_pct: float | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    sample_size: int = 0
    reason: str | None = None
    metadata: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "indicator": self.indicator,
            "value": self.value,
            "target": self.target,
            "comparison": self.comparison,
            "status": self.status,
            "severity": self.severity,
            "warning_threshold": self.warning_threshold,
            "error_budget_pct": self.error_budget_pct,
            "sample_size": self.sample_size,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @property
    def is_breach(self) -> bool:
        return self.status == "breach"


@dataclass(slots=True)
class SLOCompositeStatus:
    """Podsumowanie kompozytowego celu SLO2."""

    name: str
    status: str
    severity: str
    counts: Mapping[str, int]
    objectives: Sequence[str]
    missing_objectives: Sequence[str] = field(default_factory=tuple)
    reason: str | None = None
    metadata: Mapping[str, float] = field(default_factory=dict)
    tags: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "counts": dict(self.counts),
            "objectives": list(self.objectives),
            "missing_objectives": list(self.missing_objectives),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload

    @property
    def is_breach(self) -> bool:
        return self.status == "breach"


class SLOMonitor:
    """Silnik ewaluujący definicje SLO na podstawie pomiarów."""

    def __init__(
        self,
        definitions: Sequence[SLODefinition],
        *,
        composites: Sequence[SLOCompositeDefinition] | None = None,
    ):
        self._definitions: dict[str, SLODefinition] = {
            definition.name: definition for definition in definitions
        }
        self._definitions_by_indicator: dict[str, list[SLODefinition]] = {}
        for definition in definitions:
            self._definitions_by_indicator.setdefault(definition.indicator, []).append(
                definition
            )
        self._composites: dict[str, SLOCompositeDefinition] = {
            composite.name: composite for composite in composites or []
        }

    @property
    def definitions(self) -> Sequence[SLODefinition]:
        return tuple(self._definitions.values())

    @property
    def composites(self) -> Sequence[SLOCompositeDefinition]:
        return tuple(self._composites.values())

    def evaluate(
        self,
        measurements: Mapping[str, SLOMeasurement] | Iterable[SLOMeasurement],
    ) -> dict[str, SLOStatus]:
        if isinstance(measurements, Mapping):
            measurement_map = dict(measurements)
        else:
            measurement_map = {measurement.indicator: measurement for measurement in measurements}

        statuses: dict[str, SLOStatus] = {}
        for name, definition in self._definitions.items():
            measurement = measurement_map.get(definition.indicator)
            status = self._evaluate_single(definition, measurement)
            statuses[name] = status
        return statuses

    def evaluate_composites(
        self, statuses: Mapping[str, SLOStatus]
    ) -> dict[str, SLOCompositeStatus]:
        results: dict[str, SLOCompositeStatus] = {}
        for name, composite in self._composites.items():
            results[name] = self._evaluate_composite(composite, statuses)
        return results

    def summary(
        self,
        statuses: Mapping[str, SLOStatus],
        composite_statuses: Mapping[str, SLOCompositeStatus] | None = None,
    ) -> dict[str, object]:
        counts: MutableMapping[str, int] = {"ok": 0, "warning": 0, "breach": 0, "unknown": 0}
        breached: list[str] = []
        for name, status in statuses.items():
            counts[status.status] = counts.get(status.status, 0) + 1
            if status.is_breach:
                breached.append(name)
        summary: dict[str, object] = {
            "evaluated": len(statuses),
            "status_counts": dict(counts),
            "breached": breached,
        }
        if composite_statuses:
            composite_counts: MutableMapping[str, int] = {
                "ok": 0,
                "warning": 0,
                "breach": 0,
                "unknown": 0,
            }
            composite_breached: list[str] = []
            for name, status in composite_statuses.items():
                composite_counts[status.status] = (
                    composite_counts.get(status.status, 0) + 1
                )
                if status.is_breach:
                    composite_breached.append(name)
            summary["composites"] = {
                "evaluated": len(composite_statuses),
                "status_counts": dict(composite_counts),
                "breached": composite_breached,
            }
        return summary

    def _evaluate_single(
        self, definition: SLODefinition, measurement: SLOMeasurement | None
    ) -> SLOStatus:
        if measurement is None or measurement.value is None:
            return SLOStatus(
                name=definition.name,
                indicator=definition.indicator,
                value=None,
                target=definition.target,
                comparison=definition.comparison,
                status="unknown",
                severity="warning",
                warning_threshold=definition.warning_threshold,
                error_budget_pct=None,
                window_start=measurement.window_start if measurement else None,
                window_end=measurement.window_end if measurement else None,
                sample_size=measurement.sample_size if measurement else 0,
                reason="brak danych",
            )

        value = float(measurement.value)
        warning_threshold = definition.warning_threshold
        status = "ok"
        reason: str | None = None
        severity = "info"
        error_budget_pct: float | None = None

        if definition.comparison == ">=":
            if warning_threshold is None:
                warning_threshold = definition.target
            if value < definition.target:
                status = "breach"
                severity = definition.severity
                reason = f"{value:.6g} < target {definition.target:.6g}"
                if definition.target != 0:
                    error_budget_pct = max(0.0, (definition.target - value) / abs(definition.target))
            elif value < warning_threshold:
                status = "warning"
                severity = "warning"
                reason = f"{value:.6g} < warning {warning_threshold:.6g}"
        else:  # "<="
            if warning_threshold is None:
                warning_threshold = definition.target
            if value > definition.target:
                status = "breach"
                severity = definition.severity
                reason = f"{value:.6g} > target {definition.target:.6g}"
                if definition.target != 0:
                    error_budget_pct = max(0.0, (value - definition.target) / abs(definition.target))
            elif value > warning_threshold:
                status = "warning"
                severity = "warning"
                reason = f"{value:.6g} > warning {warning_threshold:.6g}"

        metadata = dict(measurement.metadata)
        if error_budget_pct is not None:
            metadata.setdefault("error_budget_pct", error_budget_pct)

        return SLOStatus(
            name=definition.name,
            indicator=definition.indicator,
            value=value,
            target=definition.target,
            comparison=definition.comparison,
            status=status,
            severity=severity,
            warning_threshold=warning_threshold,
            error_budget_pct=error_budget_pct,
            window_start=measurement.window_start,
            window_end=measurement.window_end,
            sample_size=measurement.sample_size,
            reason=reason,
            metadata=metadata,
        )

    def _evaluate_composite(
        self,
        definition: SLOCompositeDefinition,
        statuses: Mapping[str, SLOStatus],
    ) -> SLOCompositeStatus:
        counts: MutableMapping[str, int] = {"ok": 0, "warning": 0, "breach": 0, "unknown": 0}
        missing: list[str] = []
        total = len(definition.objectives)
        status_map: dict[str, SLOStatus] = {}
        for name in definition.objectives:
            status = statuses.get(name)
            if status is None:
                counts["unknown"] = counts.get("unknown", 0) + 1
                missing.append(name)
            else:
                counts[status.status] = counts.get(status.status, 0) + 1
                status_map[name] = status

        metadata: MutableMapping[str, float] = {
            "total_objectives": float(total),
            "max_breaches": float(definition.max_breaches),
        }
        if definition.max_warnings is not None:
            metadata["max_warnings"] = float(definition.max_warnings)
        if definition.min_ok_ratio is not None:
            metadata["min_ok_ratio"] = float(definition.min_ok_ratio)

        status_value = "ok"
        severity = "info"
        reason: str | None = None
        evaluated = total - counts.get("unknown", 0)
        required_ok_ratio = definition.min_ok_ratio if definition.min_ok_ratio is not None else 1.0
        required_ok = int(ceil(required_ok_ratio * total)) if total else 0

        ok_like = counts.get("ok", 0)
        warning_count = counts.get("warning", 0)
        breach_count = counts.get("breach", 0)

        if not status_map and missing:
            status_value = "unknown"
            severity = "warning"
            reason = "brak danych dla celów składowych"
        else:
            if breach_count > definition.max_breaches:
                status_value = "breach"
                severity = definition.severity
                reason = (
                    f"{breach_count}/{total} celów w stanie breach (limit {definition.max_breaches})"
                )
            else:
                tolerated_breach = breach_count > 0
                effective_ok = ok_like + warning_count
                if definition.min_ok_ratio is not None and effective_ok < required_ok:
                    status_value = "warning"
                    severity = "warning"
                    reason = (
                        f"tylko {effective_ok}/{total} celów spełnia wymagany udział"
                    )
                elif (
                    definition.max_warnings is not None
                    and warning_count > definition.max_warnings
                ):
                    status_value = "warning"
                    severity = "warning"
                    reason = (
                        f"{warning_count}/{total} celów w stanie warning (limit {definition.max_warnings})"
                    )
                elif warning_count > 0 or tolerated_breach:
                    status_value = "warning"
                    severity = "warning"
                    if tolerated_breach:
                        reason = (
                            f"tolerowane breach: {breach_count}/{definition.max_breaches}"
                        )
                    else:
                        reason = f"{warning_count}/{total} celów w stanie warning"
                elif missing:
                    status_value = "warning"
                    severity = "warning"
                    reason = "brak pełnych danych dla wszystkich celów"

        metadata.update({
            "counts_ok": float(counts.get("ok", 0)),
            "counts_warning": float(warning_count),
            "counts_breach": float(breach_count),
            "counts_unknown": float(counts.get("unknown", 0)),
            "evaluated": float(evaluated),
        })

        return SLOCompositeStatus(
            name=definition.name,
            status=status_value,
            severity=severity if status_value != "breach" else definition.severity,
            counts=dict(counts),
            objectives=tuple(definition.objectives),
            missing_objectives=tuple(missing),
            reason=reason,
            metadata=dict(metadata),
            tags=tuple(definition.tags),
        )


@dataclass(slots=True)
class SLOReport:
    """Pełny raport z ewaluacji SLO wraz z kompozytami."""

    generated_at: datetime
    definitions: Sequence[SLODefinition]
    measurements: Mapping[str, SLOMeasurement]
    statuses: Mapping[str, SLOStatus]
    summary: Mapping[str, Any]
    composites: Sequence[SLOCompositeDefinition] = field(default_factory=tuple)
    composite_statuses: Mapping[str, SLOCompositeStatus] = field(default_factory=dict)

    def _timestamp(self) -> str:
        value = self.generated_at
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self._timestamp(),
            "definitions": [definition.to_dict() for definition in self.definitions],
            "measurements": {
                key: measurement.to_dict() for key, measurement in self.measurements.items()
            },
            "results": {name: status.to_dict() for name, status in self.statuses.items()},
            "summary": dict(self.summary),
        }
        if self.composites:
            payload["composites"] = {
                "definitions": [composite.to_dict() for composite in self.composites],
                "results": {
                    name: status.to_dict()
                    for name, status in self.composite_statuses.items()
                },
            }
        return payload

    def write_json(self, path: Path, *, pretty: bool = False) -> Path:
        payload = self.to_payload()
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            if pretty:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            else:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        return path

    def write_csv(self, path: Path) -> Path:
        return write_slo_results_csv(self.statuses, path, composites=self.composite_statuses)


def evaluate_slo(
    definitions: Sequence[SLODefinition],
    measurements: Mapping[str, SLOMeasurement],
    *,
    composites: Sequence[SLOCompositeDefinition] | None = None,
    generated_at: datetime | None = None,
) -> SLOReport:
    """Buduje raport z ewaluacji SLO i opcjonalnych kompozytów."""

    monitor = SLOMonitor(definitions, composites=composites)
    statuses = monitor.evaluate(measurements)
    composite_statuses = monitor.evaluate_composites(statuses)
    summary = monitor.summary(statuses, composite_statuses)
    timestamp = generated_at or datetime.now(timezone.utc)
    return SLOReport(
        generated_at=timestamp,
        definitions=tuple(definitions),
        measurements=dict(measurements),
        statuses=dict(statuses),
        summary=dict(summary),
        composites=tuple(composites or ()),
        composite_statuses=dict(composite_statuses),
    )
    def _evaluate_single(
        self, definition: SLODefinition, measurement: SLOMeasurement | None
    ) -> SLOStatus:
        if measurement is None or measurement.value is None:
            return SLOStatus(
                name=definition.name,
                indicator=definition.indicator,
                value=None,
                target=definition.target,
                comparison=definition.comparison,
                status="unknown",
                severity="warning",
                warning_threshold=definition.warning_threshold,
                error_budget_pct=None,
                window_start=measurement.window_start if measurement else None,
                window_end=measurement.window_end if measurement else None,
                sample_size=measurement.sample_size if measurement else 0,
                reason="brak danych",
            )

        value = float(measurement.value)
        warning_threshold = definition.warning_threshold
        status = "ok"
        reason: str | None = None
        severity = "info"
        error_budget_pct: float | None = None

        if definition.comparison == ">=":
            if warning_threshold is None:
                warning_threshold = definition.target
            if value < definition.target:
                status = "breach"
                severity = definition.severity
                reason = f"{value:.6g} < target {definition.target:.6g}"
                if definition.target != 0:
                    error_budget_pct = max(0.0, (definition.target - value) / abs(definition.target))
            elif value < warning_threshold:
                status = "warning"
                severity = "warning"
                reason = f"{value:.6g} < warning {warning_threshold:.6g}"
        else:  # "<="
            if warning_threshold is None:
                warning_threshold = definition.target
            if value > definition.target:
                status = "breach"
                severity = definition.severity
                reason = f"{value:.6g} > target {definition.target:.6g}"
                if definition.target != 0:
                    error_budget_pct = max(0.0, (value - definition.target) / abs(definition.target))
            elif value > warning_threshold:
                status = "warning"
                severity = "warning"
                reason = f"{value:.6g} > warning {warning_threshold:.6g}"

        metadata = dict(measurement.metadata)
        if error_budget_pct is not None:
            metadata.setdefault("error_budget_pct", error_budget_pct)

        return SLOStatus(
            name=definition.name,
            indicator=definition.indicator,
            value=value,
            target=definition.target,
            comparison=definition.comparison,
            status=status,
            severity=severity,
            warning_threshold=warning_threshold,
            error_budget_pct=error_budget_pct,
            window_start=measurement.window_start,
            window_end=measurement.window_end,
            sample_size=measurement.sample_size,
            reason=reason,
            metadata=metadata,
        )

    def _evaluate_composite(
        self,
        definition: SLOCompositeDefinition,
        statuses: Mapping[str, SLOStatus],
    ) -> SLOCompositeStatus:
        counts: MutableMapping[str, int] = {"ok": 0, "warning": 0, "breach": 0, "unknown": 0}
        missing: list[str] = []
        total = len(definition.objectives)
        status_map: dict[str, SLOStatus] = {}
        for name in definition.objectives:
            status = statuses.get(name)
            if status is None:
                counts["unknown"] = counts.get("unknown", 0) + 1
                missing.append(name)
            else:
                counts[status.status] = counts.get(status.status, 0) + 1
                status_map[name] = status

        metadata: MutableMapping[str, float] = {
            "total_objectives": float(total),
            "max_breaches": float(definition.max_breaches),
        }
        if definition.max_warnings is not None:
            metadata["max_warnings"] = float(definition.max_warnings)
        if definition.min_ok_ratio is not None:
            metadata["min_ok_ratio"] = float(definition.min_ok_ratio)

        status_value = "ok"
        severity = "info"
        reason: str | None = None
        evaluated = total - counts.get("unknown", 0)
        required_ok_ratio = definition.min_ok_ratio if definition.min_ok_ratio is not None else 1.0
        required_ok = int(ceil(required_ok_ratio * total)) if total else 0

        ok_like = counts.get("ok", 0)
        warning_count = counts.get("warning", 0)
        breach_count = counts.get("breach", 0)

        if not status_map and missing:
            status_value = "unknown"
            severity = "warning"
            reason = "brak danych dla celów składowych"
        else:
            if breach_count > definition.max_breaches:
                status_value = "breach"
                severity = definition.severity
                reason = (
                    f"{breach_count}/{total} celów w stanie breach (limit {definition.max_breaches})"
                )
            else:
                tolerated_breach = breach_count > 0
                effective_ok = ok_like + warning_count
                if definition.min_ok_ratio is not None and effective_ok < required_ok:
                    status_value = "warning"
                    severity = "warning"
                    reason = (
                        f"tylko {effective_ok}/{total} celów spełnia wymagany udział"
                    )
                elif (
                    definition.max_warnings is not None
                    and warning_count > definition.max_warnings
                ):
                    status_value = "warning"
                    severity = "warning"
                    reason = (
                        f"{warning_count}/{total} celów w stanie warning (limit {definition.max_warnings})"
                    )
                elif warning_count > 0 or tolerated_breach:
                    status_value = "warning"
                    severity = "warning"
                    if tolerated_breach:
                        reason = (
                            f"tolerowane breach: {breach_count}/{definition.max_breaches}"
                        )
                    else:
                        reason = f"{warning_count}/{total} celów w stanie warning"
                elif missing:
                    status_value = "warning"
                    severity = "warning"
                    reason = "brak pełnych danych dla wszystkich celów"

        metadata.update({
            "counts_ok": float(counts.get("ok", 0)),
            "counts_warning": float(warning_count),
            "counts_breach": float(breach_count),
            "counts_unknown": float(counts.get("unknown", 0)),
            "evaluated": float(evaluated),
        })

        return SLOCompositeStatus(
            name=definition.name,
            status=status_value,
            severity=severity if status_value != "breach" else definition.severity,
            counts=dict(counts),
            objectives=tuple(definition.objectives),
            missing_objectives=tuple(missing),
            reason=reason,
            metadata=dict(metadata),
            tags=tuple(definition.tags),
        )


def _format_optional_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _format_metadata(metadata: Mapping[str, float]) -> str:
    if not metadata:
        return ""
    return " | ".join(f"{key}={value:.6g}" for key, value in sorted(metadata.items()))


def write_slo_results_csv(
    statuses: Mapping[str, SLOStatus],
    output_path: Path,
    *,
    composites: Mapping[str, SLOCompositeStatus] | None = None,
) -> Path:
    """Zapisuje wyniki ewaluacji SLO do pliku CSV."""

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "type",
                "name",
                "status",
                "severity",
                "indicator",
                "value",
                "target",
                "comparison",
                "warning_threshold",
                "error_budget_pct",
                "sample_size",
                "window_start",
                "window_end",
                "reason",
                "metadata",
                "objectives",
                "missing_objectives",
                "tags",
            ]
        )
        for name in sorted(statuses):
            status = statuses[name]
            writer.writerow(
                [
                    "slo",
                    status.name,
                    status.status,
                    status.severity,
                    status.indicator,
                    _format_optional_number(status.value),
                    f"{status.target:.6g}",
                    status.comparison,
                    _format_optional_number(status.warning_threshold),
                    _format_optional_number(status.error_budget_pct),
                    status.sample_size,
                    status.window_start.isoformat() if status.window_start else "",
                    status.window_end.isoformat() if status.window_end else "",
                    status.reason or "",
                    _format_metadata(status.metadata),
                    "",
                    "",
                    "",
                ]
            )
        if composites:
            for name in sorted(composites):
                composite = composites[name]
                writer.writerow(
                    [
                        "composite",
                        composite.name,
                        composite.status,
                        composite.severity,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        composite.reason or "",
                        _format_metadata(composite.metadata),
                        " | ".join(composite.objectives),
                        " | ".join(composite.missing_objectives),
                        " | ".join(composite.tags),
                    ]
                )
    return output_path


__all__ = [
    "SLODefinition",
    "SLOCompositeDefinition",
    "SLOCompositeStatus",
    "SLOMeasurement",
    "SLOMonitor",
    "SLOStatus",
    "SLOReport",
    "evaluate_slo",
    "write_slo_results_csv",
]
