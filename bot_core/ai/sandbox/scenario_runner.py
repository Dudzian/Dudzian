"""Scenariusze sandboxowe dla DecisionModelInference."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

import math

import yaml

from bot_core.ai.inference import DecisionModelInference
from bot_core.alerts import AlertSeverity
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
from .cost_guard import (
    SandboxAlertConfig,
    SandboxBudgetConfig,
    SandboxCostGuard,
    SandboxMetricsConfig,
)
from .stream_ingest import SandboxStreamEvent, TradingStubStreamIngestor

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _ROOT / "config" / "ai" / "sandbox.yaml"
_DEFAULT_DASHBOARD_PATH = _ROOT / "var" / "observability" / "sandbox_annotations.json"


@dataclass(slots=True)
class SandboxScenarioConfig:
    """Konfiguracja sandboxa wczytywana z pliku YAML."""

    budgets: SandboxBudgetConfig
    metrics: SandboxMetricsConfig
    alerts: SandboxAlertConfig
    dashboard_output: Path = field(default=_DEFAULT_DASHBOARD_PATH)
    dashboard_pretty: bool = False
    default_dataset: str = "multi_asset_performance"
    metric_labels: Mapping[str, str] = field(default_factory=dict)

    def copy_with(
        self,
        *,
        dashboard_output: Path | None = None,
        dashboard_pretty: bool | None = None,
        metric_labels: Mapping[str, str] | None = None,
    ) -> "SandboxScenarioConfig":
        return SandboxScenarioConfig(
            budgets=self.budgets,
            metrics=self.metrics,
            alerts=self.alerts,
            dashboard_output=dashboard_output or self.dashboard_output,
            dashboard_pretty=self.dashboard_pretty if dashboard_pretty is None else dashboard_pretty,
            default_dataset=self.default_dataset,
            metric_labels=dict(metric_labels or self.metric_labels),
        )


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        return _CONFIG_PATH
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_ROOT / candidate).resolve()


def load_sandbox_config(path: str | Path | None = None) -> SandboxScenarioConfig:
    """Ładuje konfigurację sandboxa AI z pliku YAML."""

    config_path = _resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Konfiguracja sandboxa musi być mapowaniem")
    budgets_section = raw.get("budgets", {})
    if not isinstance(budgets_section, Mapping):
        raise ValueError("Sekcja 'budgets' musi być mapowaniem")
    metrics_section = raw.get("metrics", {})
    if not isinstance(metrics_section, Mapping):
        raise ValueError("Sekcja 'metrics' musi być mapowaniem")
    alerts_section = raw.get("alerts", {})
    if not isinstance(alerts_section, Mapping):
        raise ValueError("Sekcja 'alerts' musi być mapowaniem")
    output_section = raw.get("output", {})
    if output_section and not isinstance(output_section, Mapping):
        raise ValueError("Sekcja 'output' musi być mapowaniem")
    defaults_section = raw.get("defaults", {})
    if defaults_section and not isinstance(defaults_section, Mapping):
        raise ValueError("Sekcja 'defaults' musi być mapowaniem")

    budgets = SandboxBudgetConfig(
        wall_time_seconds=float(budgets_section.get("wall_time_seconds", 30.0))
        if budgets_section.get("wall_time_seconds") is not None
        else None,
        cpu_utilization_percent=float(budgets_section.get("cpu_utilization_percent", 95.0))
        if budgets_section.get("cpu_utilization_percent") is not None
        else None,
        gpu_utilization_percent=float(budgets_section.get("gpu_utilization_percent", 95.0))
        if budgets_section.get("gpu_utilization_percent") is not None
        else None,
    )

    def _resolve_counter(
        section: Mapping[str, object] | object,
        *,
        default_name: str,
        default_description: str,
        allow_disable: bool = False,
    ) -> tuple[str, str] | None:
        if isinstance(section, Mapping):
            enabled = section.get("enabled")
            if allow_disable and enabled is not None and not bool(enabled):
                return None
            name = str(section.get("name", default_name))
            description = str(section.get("description", default_description))
            return name, description
        if allow_disable and section is None:
            return None
        return default_name, default_description

    metrics = SandboxMetricsConfig(
        cpu_counter=_resolve_counter(
            metrics_section.get("cpu_counter", {}),
            default_name="bot_ai_sandbox_cpu_percent_total",
            default_description="Suma procentowego wykorzystania CPU w sandboxie AI",
        ),
        gpu_counter=_resolve_counter(
            metrics_section.get("gpu_counter", {}),
            default_name="bot_ai_sandbox_gpu_percent_total",
            default_description="Suma procentowego wykorzystania GPU w sandboxie AI",
        ),
        wall_time_histogram=_resolve_counter(
            metrics_section.get("wall_time_histogram", {}),
            default_name="bot_ai_sandbox_wall_time_seconds",
            default_description="Czas wykonania scenariusza sandboxowego",
        ),
        wall_time_buckets=tuple(
            float(value)
            for value in metrics_section.get("wall_time_histogram", {}).get(
                "buckets", (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
            )
        ),
        events_counter=_resolve_counter(
            metrics_section.get("events_counter", {}),
            default_name="bot_ai_sandbox_events_total",
            default_description="Liczba zdarzeń przetworzonych w sandboxie AI",
            allow_disable=True,
        ),
        decisions_counter=_resolve_counter(
            metrics_section.get("decisions_counter", {}),
            default_name="bot_ai_sandbox_decisions_total",
            default_description="Liczba decyzji wygenerowanych w sandboxie AI",
            allow_disable=True,
        ),
        risk_limit_utilization_gauge=_resolve_counter(
            metrics_section.get("risk_limit_utilization_gauge", {}),
            default_name="bot_ai_sandbox_risk_limit_utilization",
            default_description="Bieżące wartości limitów ryzyka obserwowane w sandboxie AI",
            allow_disable=True,
        ),
        risk_limit_breach_counter=_resolve_counter(
            metrics_section.get("risk_limit_breach_counter", {}),
            default_name="bot_ai_sandbox_risk_limit_breaches_total",
            default_description="Liczba naruszeń limitów ryzyka obserwowanych w sandboxie AI",
            allow_disable=True,
        ),
    )

    severity_raw = alerts_section.get("severity", AlertSeverity.WARNING.value)
    if isinstance(severity_raw, AlertSeverity):
        severity = severity_raw
    else:
        severity = AlertSeverity(str(severity_raw).lower())
    alerts = SandboxAlertConfig(
        source=str(alerts_section.get("source", "ai.sandbox")),
        severity=severity,
        enabled=bool(alerts_section.get("enabled", True)),
    )

    dashboard_output = output_section.get("dashboard_annotations") if isinstance(output_section, Mapping) else None
    dashboard_pretty = bool(output_section.get("pretty", False)) if isinstance(output_section, Mapping) else False
    dashboard_path = Path(dashboard_output) if isinstance(dashboard_output, (str, Path)) else _DEFAULT_DASHBOARD_PATH
    if not dashboard_path.is_absolute():
        dashboard_path = (_ROOT / dashboard_path).resolve()

    default_dataset = str(defaults_section.get("dataset", "multi_asset_performance"))
    metric_labels = defaults_section.get("metric_labels", {})
    if metric_labels and not isinstance(metric_labels, Mapping):
        raise ValueError("Sekcja defaults.metric_labels musi być mapowaniem")

    return SandboxScenarioConfig(
        budgets=budgets,
        metrics=metrics,
        alerts=alerts,
        dashboard_output=dashboard_path,
        dashboard_pretty=dashboard_pretty,
        default_dataset=default_dataset,
        metric_labels=dict({str(key): str(value) for key, value in (metric_labels or {}).items()}),
    )


@dataclass(slots=True)
class SandboxDecisionRecord:
    event: SandboxStreamEvent
    features: Mapping[str, float]
    score: Mapping[str, float]


@dataclass(slots=True)
class SandboxScenarioResult:
    scenario: str
    dataset: Path
    started_at: datetime
    completed_at: datetime
    processed_events: int
    decisions: Sequence[SandboxDecisionRecord]
    resource_statistics: Mapping[str, float]
    event_type_counts: Mapping[str, int]
    risk_limit_summary: Mapping[str, Sequence["RiskLimitSummary"]] = field(default_factory=dict)

    def to_dashboard_payload(self) -> Mapping[str, object]:
        return {
            "schema": "stage6.ai.sandbox",
            "schema_version": "1.0",
            "generated_at": self.completed_at.astimezone(timezone.utc).isoformat(),
            "scenario": self.scenario,
            "dataset": str(self.dataset),
            "processed_events": int(self.processed_events),
            "event_type_counts": dict(self.event_type_counts),
            "decisions": [
                {
                    "timestamp": record.event.timestamp.astimezone(timezone.utc).isoformat(),
                    "instrument": record.event.instrument.symbol,
                    "exchange": record.event.instrument.exchange,
                    "score": dict(record.score),
                }
                for record in self.decisions
            ],
            "resource_statistics": dict(self.resource_statistics),
            "risk_limit_summary": {
                instrument: [summary.to_payload() for summary in summaries]
                for instrument, summaries in self.risk_limit_summary.items()
            },
        }


FeatureBuilder = Callable[[SandboxStreamEvent], Mapping[str, float]]


def _normalize_feature_key(value: str) -> str:
    normalized = [
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
        if character
    ]
    key = "".join(normalized).strip("_")
    return key or "value"


def _extract_risk_state_features(payload: Mapping[str, object]) -> Mapping[str, float]:
    features: MutableMapping[str, float] = {}
    for field in ("portfolio_value", "current_drawdown", "max_daily_loss", "used_leverage"):
        value = payload.get(field)
        if isinstance(value, (int, float)):
            features[field] = float(value)
    limits = payload.get("limits")
    if isinstance(limits, Sequence):
        for index, limit in enumerate(limits):
            if not isinstance(limit, Mapping):
                continue
            code = str(limit.get("code", f"limit_{index}"))
            code_key = _normalize_feature_key(code)
            for field in ("max_value", "current_value", "threshold_value"):
                value = limit.get(field)
                if isinstance(value, (int, float)):
                    features[f"limit_{code_key}_{field}"] = float(value)
            max_value = limit.get("max_value")
            current_value = limit.get("current_value")
            if isinstance(max_value, (int, float)) and isinstance(current_value, (int, float)) and max_value:
                features[f"limit_{code_key}_utilization"] = float(current_value) / float(max_value)
            threshold_value = limit.get("threshold_value")
            if (
                isinstance(threshold_value, (int, float))
                and isinstance(current_value, (int, float))
                and threshold_value
            ):
                features[f"limit_{code_key}_threshold_utilization"] = float(current_value) / float(threshold_value)
    return features


def default_feature_builder(event: SandboxStreamEvent) -> Mapping[str, float]:
    payload = event.payload
    features: MutableMapping[str, float] = {}
    for name in ("open", "high", "low", "close", "volume"):
        value = payload.get(name)
        if isinstance(value, (int, float)):
            features[name] = float(value)
    if "high" in features and "low" in features:
        features["range"] = features["high"] - features["low"]
    if "close" in features and "open" in features:
        features["delta_close_open"] = features["close"] - features["open"]
    if event.event_type == "risk_state":
        features.update(_extract_risk_state_features(payload))
    return features


@dataclass(slots=True)
class RiskLimitSummary:
    code: str
    instrument: str
    max_utilization: float = 0.0
    max_threshold_utilization: float = 0.0
    hard_limit_breaches: int = 0
    threshold_breaches: int = 0
    observations: int = 0
    last_current_value: float | None = None
    last_max_value: float | None = None
    last_threshold_value: float | None = None
    last_observed_at: datetime | None = None

    def record_observation(
        self,
        *,
        current_value: float | None,
        max_value: float | None,
        threshold_value: float | None,
        timestamp: datetime,
    ) -> None:
        self.observations += 1
        self.last_observed_at = timestamp
        if current_value is not None:
            self.last_current_value = current_value
        if max_value is not None:
            self.last_max_value = max_value
        if threshold_value is not None:
            self.last_threshold_value = threshold_value
        if (
            current_value is not None
            and max_value is not None
            and max_value not in {0.0, -0.0}
            and math.isfinite(max_value)
        ):
            utilization = current_value / max_value
            if math.isfinite(utilization):
                self.max_utilization = max(self.max_utilization, utilization)
            if current_value > max_value:
                self.hard_limit_breaches += 1
        if (
            current_value is not None
            and threshold_value is not None
            and threshold_value not in {0.0, -0.0}
            and math.isfinite(threshold_value)
        ):
            threshold_utilization = current_value / threshold_value
            if math.isfinite(threshold_utilization):
                self.max_threshold_utilization = max(
                    self.max_threshold_utilization,
                    threshold_utilization,
                )
            if current_value > threshold_value:
                self.threshold_breaches += 1

    def to_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "code": self.code,
            "instrument": self.instrument,
            "observations": self.observations,
            "max_utilization": self.max_utilization,
            "max_threshold_utilization": self.max_threshold_utilization,
            "hard_limit_breaches": self.hard_limit_breaches,
            "threshold_breaches": self.threshold_breaches,
        }
        if self.last_current_value is not None:
            payload["last_current_value"] = self.last_current_value
        if self.last_max_value is not None:
            payload["last_max_value"] = self.last_max_value
        if self.last_threshold_value is not None:
            payload["last_threshold_value"] = self.last_threshold_value
        if self.last_observed_at is not None:
            payload["last_observed_at"] = self.last_observed_at.astimezone(timezone.utc).isoformat()
        return payload


class SandboxScenarioRunner:
    """Wykonuje scenariusz sandboxowy używając DecisionModelInference."""

    def __init__(
        self,
        *,
        config: SandboxScenarioConfig | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        self._config = config or load_sandbox_config()
        self._metrics_registry = metrics_registry or get_global_metrics_registry()

    def run(
        self,
        scenario: str,
        *,
        inference: DecisionModelInference,
        dataset: str | Path | None = None,
        feature_builder: FeatureBuilder | None = None,
        cost_guard: SandboxCostGuard | None = None,
        instruments: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        on_event: Callable[[SandboxStreamEvent], None] | None = None,
    ) -> SandboxScenarioResult:
        ingestor = TradingStubStreamIngestor(dataset or self._config.default_dataset)
        builder = feature_builder or default_feature_builder
        guard_labels = {**self._config.metric_labels, "scenario": scenario}
        dataset_label = ingestor.dataset_path.stem
        guard_labels.setdefault("dataset", dataset_label)
        guard = cost_guard or SandboxCostGuard(
            budgets=self._config.budgets,
            metrics=self._config.metrics,
            alerts=self._config.alerts,
            metrics_registry=self._metrics_registry,
            metric_labels=guard_labels,
        )
        if cost_guard is not None and cost_guard.metric_labels is None:
            cost_guard.metric_labels = guard_labels
        events_counter_metric = None
        decisions_counter_metric = None
        risk_utilization_metric = None
        risk_breach_metric = None
        if self._config.metrics.events_counter is not None:
            events_counter_metric = self._metrics_registry.counter(
                self._config.metrics.events_counter[0],
                self._config.metrics.events_counter[1],
            )
        if self._config.metrics.decisions_counter is not None:
            decisions_counter_metric = self._metrics_registry.counter(
                self._config.metrics.decisions_counter[0],
                self._config.metrics.decisions_counter[1],
            )
        if self._config.metrics.risk_limit_utilization_gauge is not None:
            risk_utilization_metric = self._metrics_registry.gauge(
                self._config.metrics.risk_limit_utilization_gauge[0],
                self._config.metrics.risk_limit_utilization_gauge[1],
            )
        if self._config.metrics.risk_limit_breach_counter is not None:
            risk_breach_metric = self._metrics_registry.counter(
                self._config.metrics.risk_limit_breach_counter[0],
                self._config.metrics.risk_limit_breach_counter[1],
            )
        decisions: list[SandboxDecisionRecord] = []
        event_type_counts: MutableMapping[str, int] = {}
        started_at = datetime.now(timezone.utc)
        processed_events = 0
        risk_limit_track: dict[str, dict[str, RiskLimitSummary]] = {}
        with guard:
            for event in ingestor.iter_events(
                instruments=instruments,
                event_types=event_types,
            ):
                processed_events += 1
                event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
                if events_counter_metric is not None:
                    event_labels = dict(guard_labels)
                    event_labels["event_type"] = event.event_type
                    events_counter_metric.inc(labels=event_labels)
                if on_event is not None:
                    on_event(event)
                guard.update(
                    context={
                        "event_type": event.event_type,
                        "instrument": event.instrument.symbol,
                        "dataset": dataset_label,
                        "sequence": event.sequence,
                    }
                )
                if event.event_type == "risk_state":
                    limits = event.payload.get("limits")
                    if isinstance(limits, Sequence):
                        instrument_limits = risk_limit_track.setdefault(
                            event.instrument.symbol,
                            {},
                        )
                        for index, raw_limit in enumerate(limits):
                            if not isinstance(raw_limit, Mapping):
                                continue
                            code = str(raw_limit.get("code", f"limit_{index}"))
                            code_key = _normalize_feature_key(code) or f"limit_{index}"
                            summary = instrument_limits.get(code_key)
                            if summary is None:
                                summary = RiskLimitSummary(
                                    code=code,
                                    instrument=event.instrument.symbol,
                                )
                                instrument_limits[code_key] = summary
                            current_value = raw_limit.get("current_value")
                            max_value = raw_limit.get("max_value")
                            threshold_value = raw_limit.get("threshold_value")
                            current_float = (
                                float(current_value)
                                if isinstance(current_value, (int, float))
                                else None
                            )
                            max_float = (
                                float(max_value) if isinstance(max_value, (int, float)) else None
                            )
                            threshold_float = (
                                float(threshold_value)
                                if isinstance(threshold_value, (int, float))
                                else None
                            )
                            summary.record_observation(
                                current_value=current_float,
                                max_value=max_float,
                                threshold_value=threshold_float,
                                timestamp=event.timestamp,
                            )
                            metric_labels: Mapping[str, str] | None = None
                            if (risk_utilization_metric is not None) or (
                                risk_breach_metric is not None
                            ):
                                labels = dict(guard_labels)
                                labels["instrument"] = event.instrument.symbol
                                labels["limit_code"] = code
                                metric_labels = labels
                            if risk_utilization_metric is not None and metric_labels is not None:
                                if current_float is not None:
                                    labels = dict(metric_labels)
                                    labels["dimension"] = "current_value"
                                    risk_utilization_metric.set(current_float, labels=labels)
                                if max_float is not None:
                                    labels = dict(metric_labels)
                                    labels["dimension"] = "max_value"
                                    risk_utilization_metric.set(max_float, labels=labels)
                                if threshold_float is not None:
                                    labels = dict(metric_labels)
                                    labels["dimension"] = "threshold_value"
                                    risk_utilization_metric.set(threshold_float, labels=labels)
                                if (
                                    current_float is not None
                                    and max_float not in {None, 0.0, -0.0}
                                    and math.isfinite(max_float)
                                ):
                                    labels = dict(metric_labels)
                                    labels["dimension"] = "hard_utilization"
                                    risk_utilization_metric.set(
                                        current_float / max_float,
                                        labels=labels,
                                    )
                                if (
                                    current_float is not None
                                    and threshold_float not in {None, 0.0, -0.0}
                                    and math.isfinite(threshold_float)
                                ):
                                    labels = dict(metric_labels)
                                    labels["dimension"] = "threshold_utilization"
                                    risk_utilization_metric.set(
                                        current_float / threshold_float,
                                        labels=labels,
                                    )
                            if risk_breach_metric is not None and metric_labels is not None:
                                if (
                                    current_float is not None
                                    and max_float is not None
                                    and math.isfinite(max_float)
                                    and current_float > max_float
                                ):
                                    labels = dict(metric_labels)
                                    labels["breach_type"] = "hard"
                                    risk_breach_metric.inc(labels=labels)
                                if (
                                    current_float is not None
                                    and threshold_float is not None
                                    and math.isfinite(threshold_float)
                                    and current_float > threshold_float
                                ):
                                    labels = dict(metric_labels)
                                    labels["breach_type"] = "threshold"
                                    risk_breach_metric.inc(labels=labels)
                features = builder(event)
                if not features:
                    continue
                score = inference.score(features, context={"instrument": event.instrument.symbol})
                decisions.append(
                    SandboxDecisionRecord(
                        event=event,
                        features=dict(features),
                        score={
                            "expected_return_bps": float(getattr(score, "expected_return_bps", 0.0)),
                            "success_probability": float(getattr(score, "success_probability", 0.0)),
                        },
                    )
                )
                if decisions_counter_metric is not None:
                    decision_labels = dict(guard_labels)
                    decision_labels["event_type"] = event.event_type
                    decisions_counter_metric.inc(labels=decision_labels)
        completed_at = datetime.now(timezone.utc)
        return SandboxScenarioResult(
            scenario=scenario,
            dataset=ingestor.dataset_path,
            started_at=started_at,
            completed_at=completed_at,
            processed_events=processed_events,
            decisions=tuple(decisions),
            resource_statistics=guard.statistics(),
            event_type_counts=dict(event_type_counts),
            risk_limit_summary={
                instrument: tuple(summaries.values())
                for instrument, summaries in risk_limit_track.items()
            },
        )


__all__ = [
    "SandboxScenarioConfig",
    "SandboxScenarioResult",
    "SandboxScenarioRunner",
    "RiskLimitSummary",
    "default_feature_builder",
    "load_sandbox_config",
]
