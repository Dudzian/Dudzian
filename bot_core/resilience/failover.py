"""Obsługa drillów failover i raportów resilience Stage6."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.config.models import (
    ResilienceConfig,
    ResilienceDrillConfig,
    ResilienceDrillThresholdsConfig,
)
from bot_core.security.signing import HmacSignedReportMixin

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FailoverDrillMetrics:
    """Metryki z pojedynczego drill'u failover."""

    max_latency_ms: float
    error_rate: float
    failover_duration_seconds: float
    orders_redirected: int
    orders_failed: int
    notes: Sequence[str] = field(default_factory=tuple)

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "max_latency_ms": float(self.max_latency_ms),
            "error_rate": float(self.error_rate),
            "failover_duration_seconds": float(self.failover_duration_seconds),
            "orders_redirected": int(self.orders_redirected),
            "orders_failed": int(self.orders_failed),
        }
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


@dataclass(slots=True)
class FailoverDrillResult:
    """Wynik pojedynczego drill'u failover."""

    name: str
    primary: str
    fallbacks: Sequence[str]
    status: str
    metrics: FailoverDrillMetrics
    thresholds: ResilienceDrillThresholdsConfig
    failures: Sequence[str] = field(default_factory=tuple)
    description: str | None = None
    dataset_path: str | None = None

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "name": self.name,
            "primary": self.primary,
            "fallbacks": list(self.fallbacks),
            "status": self.status,
            "metrics": self.metrics.to_mapping(),
            "thresholds": asdict(self.thresholds),
        }
        if self.failures:
            payload["failures"] = list(self.failures)
        if self.description:
            payload["description"] = self.description
        if self.dataset_path:
            payload["dataset_path"] = self.dataset_path
        return payload

    def has_failures(self) -> bool:
        return self.status.lower() not in {"passed", "ok", "success"}


@dataclass(slots=True)
class FailoverDrillReport(HmacSignedReportMixin):
    """Raport zbiorczy z drillów failover."""

    generated_at: str
    drills: Sequence[FailoverDrillResult]

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at,
            "failure_count": sum(1 for drill in self.drills if drill.has_failures()),
            "drills": [drill.to_mapping() for drill in self.drills],
        }

    def has_failures(self) -> bool:
        return any(drill.has_failures() for drill in self.drills)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path

class ResilienceFailoverDrill:
    """Wykonuje drille failover na podstawie konfiguracji resilience Stage6."""

    def __init__(self, config: ResilienceConfig) -> None:
        self._config = config

    def run(self) -> FailoverDrillReport:
        generated_at = datetime.now(timezone.utc).isoformat()
        results: list[FailoverDrillResult] = []
        for drill_config in self._config.drills:
            results.append(self._run_single_drill(drill_config))
        return FailoverDrillReport(generated_at=generated_at, drills=tuple(results))

    def _run_single_drill(self, drill_config: ResilienceDrillConfig) -> FailoverDrillResult:
        dataset = self._load_dataset(Path(drill_config.dataset_path))
        metrics = self._extract_metrics(dataset)
        thresholds = drill_config.thresholds

        failures: list[str] = []
        if metrics.max_latency_ms > thresholds.max_latency_ms:
            failures.append(
                f"max_latency_ms={metrics.max_latency_ms:.2f}>{thresholds.max_latency_ms:.2f}"
            )
        if metrics.error_rate > thresholds.max_error_rate:
            failures.append(
                f"error_rate={metrics.error_rate:.4f}>{thresholds.max_error_rate:.4f}"
            )
        if metrics.failover_duration_seconds > thresholds.max_failover_duration_seconds:
            failures.append(
                "failover_duration_seconds="
                f"{metrics.failover_duration_seconds:.2f}>{thresholds.max_failover_duration_seconds:.2f}"
            )
        if metrics.orders_failed > thresholds.max_orders_failed:
            failures.append(
                f"orders_failed={metrics.orders_failed}>{thresholds.max_orders_failed}"
            )

        status = "failed" if failures else "passed"
        return FailoverDrillResult(
            name=drill_config.name,
            primary=drill_config.primary,
            fallbacks=tuple(drill_config.fallbacks),
            status=status,
            metrics=metrics,
            thresholds=thresholds,
            failures=tuple(failures),
            description=drill_config.description,
            dataset_path=drill_config.dataset_path,
        )

    def _load_dataset(self, path: Path) -> Mapping[str, object]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError(f"Dataset resilience '{path}' musi być mapą")
        return data

    def _extract_metrics(self, dataset: Mapping[str, object]) -> FailoverDrillMetrics:
        def _extract_float(key: str, default: float = 0.0) -> float:
            value = dataset.get(key, default)
            if isinstance(value, Mapping):
                numeric_values = [
                    float(item)
                    for item in value.values()
                    if isinstance(item, (int, float))
                ]
                if numeric_values:
                    value = max(numeric_values)
                else:
                    value = default
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _extract_int(key: str, default: int = 0) -> int:
            value = dataset.get(key, default)
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return int(default)

        notes_raw = dataset.get("notes")
        if isinstance(notes_raw, str):
            notes = (notes_raw.strip(),) if notes_raw.strip() else ()
        elif isinstance(notes_raw, Sequence):
            notes = tuple(
                str(entry).strip()
                for entry in notes_raw
                if isinstance(entry, (str, int, float)) and str(entry).strip()
            )
        else:
            notes = ()

        metrics = FailoverDrillMetrics(
            max_latency_ms=max(0.0, _extract_float("max_latency_ms")),
            error_rate=max(0.0, _extract_float("error_rate")),
            failover_duration_seconds=max(0.0, _extract_float("failover_duration_seconds")),
            orders_redirected=max(0, _extract_int("orders_redirected")),
            orders_failed=max(0, _extract_int("orders_failed")),
            notes=notes,
        )
        _LOGGER.debug("Wczytane metryki drill'u resilience: %s", metrics)
        return metrics


__all__ = [
    "FailoverDrillMetrics",
    "FailoverDrillResult",
    "FailoverDrillReport",
    "ResilienceFailoverDrill",
]
