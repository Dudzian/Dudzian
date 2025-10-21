"""Narzędzia do agregacji i raportowania jakości decyzji AI."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.decision.models import DecisionEngineSummary


def _coerce_float(value: object) -> float | None:
    """Próbuje rzutować dowolną wartość na float."""

from .utils import coerce_float


def _normalize_thresholds(snapshot: Mapping[str, object] | None) -> Mapping[str, float | None] | None:
    if not snapshot or not isinstance(snapshot, Mapping):
        return None
    normalized: MutableMapping[str, float | None] = {}
    for key, value in snapshot.items():
        normalized[str(key)] = coerce_float(value)
    return normalized


def _extract_candidate_metadata(candidate: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if not candidate or not isinstance(candidate, Mapping):
        return None
    metadata = candidate.get("metadata")
    if isinstance(metadata, Mapping):
        return {str(key): metadata[key] for key in metadata}
    return None


def _extract_generated_at(payload: Mapping[str, object]) -> str | None:
    candidate = payload.get("candidate")
    if isinstance(candidate, Mapping):
        metadata = _extract_candidate_metadata(candidate)
        if metadata:
            generated_at = metadata.get("generated_at") or metadata.get("timestamp")
            if generated_at is not None:
                return str(generated_at)
        candidate_generated = candidate.get("generated_at")
        if candidate_generated is not None:
            return str(candidate_generated)

from .schema import DecisionEngineSummary
from .utils import coerce_float

QuantileSpec = tuple[float, str]


def _compute_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Brak wartości do policzenia kwantyla")
    if quantile <= 0:
        return sorted_values[0]
    if quantile >= 1:
        return sorted_values[-1]
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + weight * (upper_value - lower_value)


def _compute_std(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Brak wartości do policzenia odchylenia standardowego")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


@dataclass
class StatsAccumulator:
    values: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

def _iter_strings(values: object) -> Iterable[str]:
    if values is None:
        return ()
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else ()
    if isinstance(values, Mapping):
        return (str(key) for key in values.keys())
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        return (str(item) for item in values if item not in (None, ""))
    return (str(values),)


class _SegmentMetricAccumulator:
    """Akumulator wartości metryk dla pojedynczego klucza breakdownu."""
    def add(self, value: float) -> None:
        self.values.append(value)

    def inject(
        self,
        summary: MutableMapping[str, object],
        prefix: str,
        *,
        segment_prefix: str = "",
        quantiles: Sequence[QuantileSpec] = (),
        include_minmax: bool = False,
        include_std: bool = False,
        include_sum: bool = True,
        include_avg: bool = True,
        include_count: bool = False,
    ) -> None:
        if include_count:
            summary[f"{segment_prefix}{prefix}_count"] = self.count
        if not self.values:
            return
        total = sum(self.values)
        if include_avg:
            summary[f"{segment_prefix}avg_{prefix}"] = total / self.count
        if include_sum:
            summary[f"{segment_prefix}sum_{prefix}"] = total
        if include_minmax:
            summary[f"{segment_prefix}min_{prefix}"] = min(self.values)
            summary[f"{segment_prefix}max_{prefix}"] = max(self.values)
        if quantiles:
            sorted_values = sorted(self.values)
            for quantile, label in quantiles:
                summary[f"{segment_prefix}{label}_{prefix}"] = _compute_quantile(
                    sorted_values, quantile
                )
        if include_std:
            summary[f"{segment_prefix}std_{prefix}"] = _compute_std(self.values)


@dataclass(frozen=True)
class MetricSpec:
    quantiles_total: Sequence[QuantileSpec] = ()
    quantiles_segment: Sequence[QuantileSpec] = ()
    include_minmax_total: bool = False
    include_minmax_segment: bool = False
    include_std_total: bool = False
    include_std_segment: bool = False
    include_sum_total: bool = True
    include_sum_segment: bool = True
    include_count_total: bool = False
    include_count_segment: bool = True


@dataclass
class SegmentedMetricCollector:
    spec: MetricSpec
    total: StatsAccumulator = field(default_factory=StatsAccumulator)
    accepted: StatsAccumulator = field(default_factory=StatsAccumulator)
    rejected: StatsAccumulator = field(default_factory=StatsAccumulator)

    def add(self, value: float, *, accepted: bool) -> None:
        self.total.add(value)
        if accepted:
            self.accepted.add(value)
        else:
            self.rejected.add(value)

    def inject(self, summary: MutableMapping[str, object], prefix: str) -> None:
        self.total.inject(
            summary,
            prefix,
            quantiles=self.spec.quantiles_total,
            include_minmax=self.spec.include_minmax_total,
            include_std=self.spec.include_std_total,
            include_sum=self.spec.include_sum_total,
            include_count=self.spec.include_count_total,
        )
        self.accepted.inject(
            summary,
            prefix,
            segment_prefix="accepted_",
            quantiles=self.spec.quantiles_segment,
            include_minmax=self.spec.include_minmax_segment,
            include_std=self.spec.include_std_segment,
            include_sum=self.spec.include_sum_segment,
            include_count=self.spec.include_count_segment,
        )
        self.rejected.inject(
            summary,
            prefix,
            segment_prefix="rejected_",
            quantiles=self.spec.quantiles_segment,
            include_minmax=self.spec.include_minmax_segment,
            include_std=self.spec.include_std_segment,
            include_sum=self.spec.include_sum_segment,
            include_count=self.spec.include_count_segment,
        )


@dataclass
class SegmentMetricAccumulator:
    total_sum: float = 0.0
    total_count: int = 0
    accepted_sum: float = 0.0
    accepted_count: int = 0
    rejected_sum: float = 0.0
    rejected_count: int = 0

    def update(self, value: float, *, accepted: bool) -> None:
        self.total_sum += value
        self.total_count += 1
        if accepted:
            self.accepted_sum += value
            self.accepted_count += 1
        else:
            self.rejected_sum += value
            self.rejected_count += 1

    def build_summary(self) -> Mapping[str, float | int]:
        total_avg = self.total_sum / self.total_count if self.total_count else 0.0
        accepted_avg = (
            self.accepted_sum / self.accepted_count if self.accepted_count else 0.0
        )
        rejected_avg = (
            self.rejected_sum / self.rejected_count if self.rejected_count else 0.0
        )
        return {
            "total_sum": self.total_sum,
            "total_avg": total_avg,
            "total_count": self.total_count,
            "accepted_sum": self.accepted_sum,
            "accepted_avg": accepted_avg,
            "accepted_count": self.accepted_count,
            "rejected_sum": self.rejected_sum,
            "rejected_avg": rejected_avg,
            "rejected_count": self.rejected_count,
        }


@dataclass
class ThresholdCollector:
    prefix: str
    base_key: str
    quantiles: Sequence[QuantileSpec]
    segment_quantiles: Sequence[QuantileSpec]
    values: StatsAccumulator = field(default_factory=StatsAccumulator)
    accepted_values: StatsAccumulator = field(default_factory=StatsAccumulator)
    rejected_values: StatsAccumulator = field(default_factory=StatsAccumulator)
    breaches: int = 0
    accepted_breaches: int = 0
    rejected_breaches: int = 0

    def add(self, margin: float, *, accepted: bool) -> None:
        self.values.add(margin)
        if margin < 0:
            self.breaches += 1
            if accepted:
                self.accepted_breaches += 1
            else:
                self.rejected_breaches += 1
        if accepted:
            self.accepted_values.add(margin)
        else:
            self.rejected_values.add(margin)

    def inject(self, summary: MutableMapping[str, object]) -> None:
        self.values.inject(
            summary,
            self.prefix,
            quantiles=self.quantiles,
            include_minmax=True,
            include_std=True,
            include_count=True,
        )
        total_count = self.values.count
        summary[f"{self.base_key}_breaches"] = self.breaches
        summary[f"{self.base_key}_breach_rate"] = (
            self.breaches / total_count if total_count else 0.0
        )
        self.accepted_values.inject(
            summary,
            self.prefix,
            segment_prefix="accepted_",
            quantiles=self.segment_quantiles,
            include_minmax=True,
            include_std=True,
            include_count=True,
        )
        accepted_count = self.accepted_values.count
        summary[f"accepted_{self.base_key}_breaches"] = self.accepted_breaches
        summary[f"accepted_{self.base_key}_breach_rate"] = (
            self.accepted_breaches / accepted_count if accepted_count else 0.0
        )
        self.rejected_values.inject(
            summary,
            self.prefix,
            segment_prefix="rejected_",
            quantiles=self.segment_quantiles,
            include_minmax=True,
            include_std=True,
            include_count=True,
        )
        rejected_count = self.rejected_values.count
        summary[f"rejected_{self.base_key}_breaches"] = self.rejected_breaches
        summary[f"rejected_{self.base_key}_breach_rate"] = (
            self.rejected_breaches / rejected_count if rejected_count else 0.0
        )


class ThresholdAggregator:
    _quantiles = ((0.1, "p10"), (0.5, "median"), (0.9, "p90"))

    def __init__(self) -> None:
        self._collectors: dict[str, ThresholdCollector] = {
            "probability_threshold_margin": ThresholdCollector(
                "probability_threshold_margin",
                "probability_threshold",
                self._quantiles,
                self._quantiles,
            ),
            "cost_threshold_margin": ThresholdCollector(
                "cost_threshold_margin",
                "cost_threshold",
                self._quantiles,
                self._quantiles,
            ),
            "net_edge_threshold_margin": ThresholdCollector(
                "net_edge_threshold_margin",
                "net_edge_threshold",
                self._quantiles,
                self._quantiles,
            ),
            "latency_threshold_margin": ThresholdCollector(
                "latency_threshold_margin",
                "latency_threshold",
                self._quantiles,
                self._quantiles,
            ),
            "notional_threshold_margin": ThresholdCollector(
                "notional_threshold_margin",
                "notional_threshold",
                self._quantiles,
                self._quantiles,
            ),
        }

    def add(
        self,
        thresholds: Mapping[str, float | None] | None,
        observed: Mapping[str, float],
        *,
        accepted: bool,
    ) -> None:
        if not thresholds:
            return
        for threshold_key, (metric_key, direction, prefix) in _THRESHOLD_DEFINITIONS.items():
            collector = self._collectors.get(prefix)
            if collector is None:
                continue
            threshold_value = thresholds.get(threshold_key)
            threshold_float = coerce_float(threshold_value)
            if threshold_float is None:
                continue
            observed_value = observed.get(metric_key)
            if observed_value is None:
                continue
            if direction == "min":
                margin = observed_value - threshold_float
            else:
                margin = threshold_float - observed_value
            collector.add(margin, accepted=accepted)

    def inject(self, summary: MutableMapping[str, object]) -> None:
        for collector in self._collectors.values():
            collector.inject(summary)


_THRESHOLD_DEFINITIONS: Mapping[str, tuple[str, str, str]] = {
    "min_probability": ("expected_probability", "min", "probability_threshold_margin"),
    "min_net_edge_bps": ("net_edge_bps", "min", "net_edge_threshold_margin"),
    "max_cost_bps": ("cost_bps", "max", "cost_threshold_margin"),
    "max_latency_ms": ("latency_ms", "max", "latency_threshold_margin"),
    "max_trade_notional": ("notional", "max", "notional_threshold_margin"),
}


_BREAKDOWN_METRIC_KEYS = {
    "net_edge_bps",
    "cost_bps",
    "expected_value_bps",
    "expected_value_minus_cost_bps",
    "notional",
    "latency_ms",
}


_THRESHOLD_DEFINITIONS: Mapping[str, tuple[str, str, str]] = {
    "min_probability": ("expected_probability", "min", "probability_threshold_margin"),
    "min_net_edge_bps": ("net_edge_bps", "min", "net_edge_threshold_margin"),
    "max_cost_bps": ("cost_bps", "max", "cost_threshold_margin"),
    "max_latency_ms": ("latency_ms", "max", "latency_threshold_margin"),
    "max_trade_notional": ("notional", "max", "notional_threshold_margin"),
}


@dataclass(frozen=True)
class AggregationSpec:
    quantiles: Sequence[tuple[float, str]] = ()
    include_minmax: bool = True
    include_sum: bool = True
    include_std: bool = True


@dataclass(frozen=True)
class MetricSettings:
    overall: AggregationSpec
    segment: AggregationSpec


class SegmentedMetricCollector:
    def __init__(self) -> None:
        self._values: list[float] = []
        self._accepted: list[float] = []
        self._rejected: list[float] = []

    def add(self, value: float, *, accepted: bool) -> None:
        self._values.append(value)
        if accepted:
            self._accepted.append(value)
        else:
            self._rejected.append(value)

    def inject(self, summary: MutableMapping[str, object], prefix: str, settings: MetricSettings) -> None:
        self._inject_overall(summary, prefix, settings.overall)
        self._inject_segment(summary, self._accepted, prefix, "accepted", settings.segment)
        self._inject_segment(summary, self._rejected, prefix, "rejected", settings.segment)

    def _inject_overall(
        self,
        summary: MutableMapping[str, object],
        prefix: str,
        spec: AggregationSpec,
    ) -> None:
        if not self._values:
            return
        total = sum(self._values)
        count = len(self._values)
        summary[f"avg_{prefix}"] = total / count
        if spec.include_sum:
            summary[f"sum_{prefix}"] = total
        if spec.include_minmax or spec.quantiles:
            _inject_distribution_metrics(
                summary,
                self._values,
                prefix=prefix,
                quantiles=spec.quantiles,
                include_minmax=spec.include_minmax,
            )
        if spec.include_std:
            _inject_std_metric(summary, self._values, prefix=prefix)

    def _inject_segment(
        self,
        summary: MutableMapping[str, object],
        values: Sequence[float],
        prefix: str,
        segment: str,
        spec: AggregationSpec,
    ) -> None:
        if not values:
            return
        total = sum(values)
        count = len(values)
        summary[f"{segment}_avg_{prefix}"] = total / count
        summary[f"{segment}_{prefix}_count"] = count
        if spec.include_sum:
            summary[f"{segment}_sum_{prefix}"] = total
        sorted_values = sorted(values)
        if spec.include_minmax:
            summary[f"{segment}_min_{prefix}"] = sorted_values[0]
            summary[f"{segment}_max_{prefix}"] = sorted_values[-1]
        for quantile, label in spec.quantiles:
            summary[f"{segment}_{label}_{prefix}"] = _compute_quantile(
                sorted_values, quantile
            )
        if spec.include_std:
            summary[f"{segment}_std_{prefix}"] = _compute_std(values)


class ThresholdMarginCollector:
    def __init__(self) -> None:
        self._values: list[float] = []
        self._accepted: list[float] = []
        self._rejected: list[float] = []
        self._total_breaches = 0
        self._accepted_breaches = 0
        self._rejected_breaches = 0

    def add(self, value: float, *, accepted: bool, breached: bool) -> None:
        self._values.append(value)
        if accepted:
            self._accepted.append(value)
        else:
            self._rejected.append(value)
        if breached:
            self._total_breaches += 1
            if accepted:
                self._accepted_breaches += 1
            else:
                self._rejected_breaches += 1

    def inject(self, summary: MutableMapping[str, object], prefix: str) -> None:
        values = self._values
        if values:
            count = len(values)
            total = sum(values)
            summary[f"{prefix}_count"] = count
            summary[f"avg_{prefix}"] = total / count
            summary[f"sum_{prefix}"] = total
            _inject_distribution_metrics(
                summary,
                values,
                prefix=prefix,
                quantiles=((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
            )
            _inject_std_metric(summary, values, prefix=prefix)
        base_key = prefix.rsplit("_margin", 1)[0] if prefix.endswith("_margin") else prefix
        total_count = len(values)
        summary[f"{base_key}_breaches"] = self._total_breaches
        summary[f"{base_key}_breach_rate"] = (
            self._total_breaches / total_count if total_count else 0.0
        )

        accepted_count = len(self._accepted)
        if self._accepted:
            accepted_total = sum(self._accepted)
            summary[f"accepted_avg_{prefix}"] = accepted_total / accepted_count
            summary[f"accepted_sum_{prefix}"] = accepted_total
            summary[f"accepted_{prefix}_count"] = accepted_count
            _inject_segmented_threshold_metrics(
                summary,
                self._accepted,
                prefix=prefix,
                segment="accepted",
            )
        summary[f"accepted_{base_key}_breaches"] = self._accepted_breaches
        summary[f"accepted_{base_key}_breach_rate"] = (
            self._accepted_breaches / accepted_count if accepted_count else 0.0
def _iter_strings(values: object) -> Iterable[str]:
    if values is None:
        return ()
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else ()
    if isinstance(values, Mapping):
        return (str(key) for key in values.keys())
    if isinstance(values, Sequence):
        return (str(item) for item in values if item not in (None, ""))
    return (str(values),)


def _normalize_thresholds(
    snapshot: Mapping[str, object] | None,
) -> Mapping[str, float | None] | None:
    if not snapshot or not isinstance(snapshot, Mapping):
        return None
    normalized: MutableMapping[str, float | None] = {}
    for key, value in snapshot.items():
        normalized[str(key)] = coerce_float(value)
    return normalized


def _extract_candidate_metadata(candidate: Mapping[str, object]) -> Mapping[str, object] | None:
    metadata = candidate.get("metadata")
    if isinstance(metadata, Mapping):
        return {str(key): metadata[key] for key in metadata}
    return None

def summarize_evaluation_payloads(
    evaluations: Sequence[Mapping[str, object]],
    *,
    history_limit: int | None = None,
) -> DecisionEngineSummary:
    """Buduje zagregowane podsumowanie Decision Engine na podstawie ewaluacji."""

    items = list(evaluations)
    full_total = len(items)
    effective_limit = _resolve_history_limit(history_limit, full_total)
    if full_total and effective_limit and effective_limit < full_total:
        window_start = full_total - effective_limit
        windowed = items[window_start:]
    else:
        windowed = items
    total = len(windowed)
    summary: MutableMapping[str, object] = {
        "total": total,
        "accepted": 0,
        "rejected": 0,
        "acceptance_rate": 0.0,
        "history_limit": effective_limit if effective_limit else full_total,
        "history_window": total,
        "rejection_reasons": {},
        "unique_rejection_reasons": 0,
        "unique_risk_flags": 0,
        "risk_flags_with_accepts": 0,
        "unique_stress_failures": 0,
        "stress_failures_with_accepts": 0,
        "unique_models": 0,
        "models_with_accepts": 0,
        "unique_actions": 0,
        "actions_with_accepts": 0,
        "unique_strategies": 0,
        "strategies_with_accepts": 0,
        "unique_symbols": 0,
        "symbols_with_accepts": 0,
        "full_total": full_total,
        "current_acceptance_streak": 0,
        "current_rejection_streak": 0,
        "longest_acceptance_streak": 0,
        "longest_rejection_streak": 0,
    }
    if full_total and full_total != total:
        full_accepted = sum(
            1 for payload in items if isinstance(payload, Mapping) and bool(payload.get("accepted"))
        )

        rejected_count = len(self._rejected)
        if self._rejected:
            rejected_total = sum(self._rejected)
            summary[f"rejected_avg_{prefix}"] = rejected_total / rejected_count
            summary[f"rejected_sum_{prefix}"] = rejected_total
            summary[f"rejected_{prefix}_count"] = rejected_count
            _inject_segmented_threshold_metrics(
                summary,
                self._rejected,
                prefix=prefix,
                segment="rejected",
            )
        summary[f"rejected_{base_key}_breaches"] = self._rejected_breaches
        summary[f"rejected_{base_key}_breach_rate"] = (
            self._rejected_breaches / rejected_count if rejected_count else 0.0
        )


METRIC_SETTINGS: Mapping[str, MetricSettings] = {
    "net_edge_bps": MetricSettings(
        overall=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
        ),
        segment=AggregationSpec(
            quantiles=((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
        ),
    ),
    "cost_bps": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"), (0.9, "p90"))),
        segment=AggregationSpec(
            quantiles=((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
        ),
    ),
    "expected_probability": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),), include_minmax=False, include_sum=False),
        segment=AggregationSpec(
            quantiles=((0.5, "median"),),
            include_minmax=False,
            include_sum=False,
        ),
    ),
    "expected_return_bps": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),)),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90")),
        ),
    ),
    "expected_value_bps": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),)),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90")),
        ),
    ),
    "expected_value_minus_cost_bps": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),)),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90")),
        ),
    ),
    "notional": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),)),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90")),
        ),
    ),
    "latency_ms": MetricSettings(
        overall=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
        ),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
        ),
    ),
    "model_success_probability": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),), include_minmax=False, include_sum=False),
        segment=AggregationSpec(
            quantiles=((0.5, "median"),),
            include_minmax=False,
            include_sum=False,
        ),
    ),
    "model_expected_return_bps": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),)),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90")),
        ),
    ),
    "model_expected_value_bps": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),)),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90")),
        ),
    ),
    "model_expected_value_minus_cost_bps": MetricSettings(
        overall=AggregationSpec(quantiles=((0.5, "median"),)),
        segment=AggregationSpec(
            quantiles=((0.5, "median"), (0.9, "p90")),
        ),
    ),
}


class DecisionSummaryAggregator:
    """Agreguje metryki Decision Engine dla kolekcji ewaluacji."""

    breakdown_metric_keys = (
        "net_edge_bps",
        "cost_bps",
        "expected_value_bps",
        "expected_value_minus_cost_bps",
        "notional",
        "latency_ms",
    )

    def __init__(
        self,
        evaluations: Sequence[Mapping[str, object]],
        *,
        history_limit: int | None = None,
    ) -> None:
        self._items = list(evaluations)
        self.full_total = len(self._items)
        self.effective_limit = _resolve_history_limit(history_limit, self.full_total)
        if self.full_total and self.effective_limit and self.effective_limit < self.full_total:
            start = self.full_total - self.effective_limit
            self.windowed = [self._items[index] for index in range(start, self.full_total)]
        else:
            self.windowed = list(self._items)
        self.total = len(self.windowed)

        full_accepted = sum(
            1
            for payload in self._items
            if isinstance(payload, Mapping) and bool(payload.get("accepted"))
        )
        full_rejected = self.full_total - full_accepted
        full_acceptance_rate = full_accepted / self.full_total if self.full_total else 0.0

        self.summary: MutableMapping[str, object] = {
            "total": self.total,
            "accepted": 0,
            "rejected": 0,
            "acceptance_rate": 0.0,
            "history_limit": self.effective_limit if self.effective_limit else self.full_total,
            "history_window": self.total,
            "rejection_reasons": {},
            "unique_rejection_reasons": 0,
            "unique_risk_flags": 0,
            "risk_flags_with_accepts": 0,
            "unique_stress_failures": 0,
            "stress_failures_with_accepts": 0,
            "unique_models": 0,
            "models_with_accepts": 0,
            "unique_actions": 0,
            "actions_with_accepts": 0,
            "unique_strategies": 0,
            "strategies_with_accepts": 0,
            "unique_symbols": 0,
            "symbols_with_accepts": 0,
            "full_total": self.full_total,
            "full_accepted": full_accepted,
            "full_rejected": full_rejected,
            "full_acceptance_rate": full_acceptance_rate,
            "current_acceptance_streak": 0,
            "current_rejection_streak": 0,
            "longest_acceptance_streak": 0,
            "longest_rejection_streak": 0,
        }

        self.accepted = 0
        self.current_acceptance_streak = 0
        self.current_rejection_streak = 0
        self.longest_acceptance_streak = 0
        self.longest_rejection_streak = 0

        self.rejection_reasons: Counter[str] = Counter()
        self.risk_flag_counts: Counter[str] = Counter()
        self.stress_failure_counts: Counter[str] = Counter()
        self.risk_flag_totals: Counter[str] = Counter()
        self.risk_flag_accepted: Counter[str] = Counter()
        self.stress_failure_totals: Counter[str] = Counter()
        self.stress_failure_accepted: Counter[str] = Counter()

        self.model_usage: Counter[str] = Counter()
        self.action_usage: Counter[str] = Counter()
        self.strategy_usage: Counter[str] = Counter()
        self.symbol_usage: Counter[str] = Counter()

        self.model_totals: Counter[str] = Counter()
        self.model_accepted: Counter[str] = Counter()
        self.action_totals: Counter[str] = Counter()
        self.action_accepted: Counter[str] = Counter()
        self.strategy_totals: Counter[str] = Counter()
        self.strategy_accepted: Counter[str] = Counter()
        self.symbol_totals: Counter[str] = Counter()
        self.symbol_accepted: Counter[str] = Counter()

        self.model_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
        self.action_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
        self.strategy_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
        self.symbol_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}

        self.metrics = {prefix: SegmentedMetricCollector() for prefix in METRIC_SETTINGS}
        self.threshold_collectors: dict[str, ThresholdMarginCollector] = {}

        self.history_generated_at: list[str] = []
        self.latest_payload: Mapping[str, object] | None = None

    def build_summary(self) -> Mapping[str, object]:
        for payload in self.windowed:
            if isinstance(payload, Mapping):
                self.process_payload(payload)
        return self.finalize()

    def process_payload(self, payload: Mapping[str, object]) -> None:
        self.latest_payload = payload
        is_accepted = bool(payload.get("accepted"))
        if is_accepted:
            self.accepted += 1
            self.current_acceptance_streak += 1
            self.current_rejection_streak = 0
        else:
            for reason in _iter_strings(payload.get("reasons")):
                self.rejection_reasons[str(reason)] += 1
            self.current_rejection_streak += 1
            self.current_acceptance_streak = 0

        self.longest_acceptance_streak = max(
            self.longest_acceptance_streak, self.current_acceptance_streak
        )
        self.longest_rejection_streak = max(
            self.longest_rejection_streak, self.current_rejection_streak
        )

        generated_at = _extract_generated_at(payload)
        if generated_at is not None:
            self.history_generated_at.append(generated_at)

        candidate = payload.get("candidate")
        candidate_map: Mapping[str, object] | None = (
            candidate if isinstance(candidate, Mapping) else None
        )

        def candidate_or_payload(key: str) -> float | None:
            if candidate_map and key in candidate_map:
                value = _coerce_float(candidate_map.get(key))
                if value is not None:
                    return value
            return _coerce_float(payload.get(key))
    if total == 0:
        return DecisionEngineSummary.model_validate(summary)

def _extract_generated_at(payload: Mapping[str, object]) -> str | None:
    candidate = payload.get("candidate")
    if isinstance(candidate, Mapping):
        metadata = _extract_candidate_metadata(candidate)
        if metadata:
            generated_at = metadata.get("generated_at") or metadata.get("timestamp")
            if generated_at is not None:
                return str(generated_at)
        candidate_generated = candidate.get("generated_at")
        if candidate_generated is not None:
            return str(candidate_generated)

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        generated_at = metadata.get("generated_at") or metadata.get("timestamp")
        if generated_at is not None:
            return str(generated_at)

    raw = payload.get("generated_at")
    if raw is not None:
        return str(raw)
    return None

    current_acceptance_streak = 0
    current_rejection_streak = 0
    costs: list[float] = []
    probabilities: list[float] = []
    expected_returns: list[float] = []
    notionals: list[float] = []
    model_probabilities: list[float] = []
    model_returns: list[float] = []
    latencies: list[float] = []

def _resolve_history_limit(history_limit: int | None, default: int) -> int:
    if history_limit is None:
        return default
    try:
        limit = int(history_limit)
    except (TypeError, ValueError):  # pragma: no cover - defensywne parsowanie
        return default
    if limit <= 0:
        return default
    return limit


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: value for key, value in counter.most_common()}


def _build_breakdown(
    totals: Counter[str],
    accepted: Counter[str],
    metrics: Mapping[str, Mapping[str, SegmentMetricAccumulator]] | None = None,
) -> Mapping[str, Mapping[str, object]]:
    breakdown: MutableMapping[str, Mapping[str, object]] = {}
    for key, total_count in totals.most_common():
        accepted_count = accepted.get(key, 0)
        rejected_count = max(total_count - accepted_count, 0)
        entry: dict[str, object] = {
            "total": total_count,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "acceptance_rate": (accepted_count / total_count) if total_count else 0.0,
        }
        if metrics and key in metrics:
            metric_entries: dict[str, Mapping[str, float | int]] = {}
            for metric_key, accumulator in sorted(metrics[key].items()):
                metric_entries[metric_key] = accumulator.build_summary()
            if metric_entries:
                entry["metrics"] = metric_entries
        breakdown[key] = entry
    return breakdown


class DecisionSummaryAggregator:
    def __init__(self, evaluations: Sequence[Mapping[str, object]]) -> None:
        self._evaluations = evaluations
        self._metrics = {
            "net_edge_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
                    quantiles_segment=((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_minmax_segment=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "cost_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"), (0.9, "p90")),
                    quantiles_segment=((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_minmax_segment=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "expected_probability": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"),),
                    include_std_total=True,
                    include_std_segment=True,
                    include_sum_total=False,
                    include_sum_segment=False,
                )
            ),
            "expected_return_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "expected_value_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "expected_value_minus_cost_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "notional": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "latency_ms": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
                    quantiles_segment=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "model_success_probability": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"),),
                    include_std_total=True,
                    include_std_segment=True,
                    include_sum_total=False,
                    include_sum_segment=False,
                )
            ),
            "model_expected_return_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "model_expected_value_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
            "model_expected_value_minus_cost_bps": SegmentedMetricCollector(
                MetricSpec(
                    quantiles_total=((0.5, "median"),),
                    quantiles_segment=((0.5, "median"), (0.9, "p90")),
                    include_minmax_total=True,
                    include_std_total=True,
                    include_std_segment=True,
                )
            ),
        }
        self._thresholds = ThresholdAggregator()
        self._accepted = 0
        self._rejection_reasons: Counter[str] = Counter()
        self._risk_flag_counts: Counter[str] = Counter()
        self._risk_flag_totals: Counter[str] = Counter()
        self._risk_flag_accepted: Counter[str] = Counter()
        self._stress_failure_counts: Counter[str] = Counter()
        self._stress_failure_totals: Counter[str] = Counter()
        self._stress_failure_accepted: Counter[str] = Counter()
        self._model_totals: Counter[str] = Counter()
        self._model_accepted: Counter[str] = Counter()
        self._model_metrics: dict[str, dict[str, SegmentMetricAccumulator]] = {}
        self._action_totals: Counter[str] = Counter()
        self._action_accepted: Counter[str] = Counter()
        self._action_metrics: dict[str, dict[str, SegmentMetricAccumulator]] = {}
        self._strategy_totals: Counter[str] = Counter()
        self._strategy_accepted: Counter[str] = Counter()
        self._strategy_metrics: dict[str, dict[str, SegmentMetricAccumulator]] = {}
        self._symbol_totals: Counter[str] = Counter()
        self._symbol_accepted: Counter[str] = Counter()
        self._symbol_metrics: dict[str, dict[str, SegmentMetricAccumulator]] = {}
        self._current_acceptance_streak = 0
        self._current_rejection_streak = 0
        self._longest_acceptance_streak = 0
        self._longest_rejection_streak = 0
        self._history_start_generated_at: str | None = None
        self._latest_payload: Mapping[str, object] | None = None

    def build(self) -> MutableMapping[str, object]:
        for payload in self._evaluations:
            if not isinstance(payload, Mapping):
                continue
            self._latest_payload = payload
            self._process_payload(payload)
        total = len(self._evaluations)
        summary: MutableMapping[str, object] = {
            "total": total,
            "accepted": self._accepted,
            "rejected": total - self._accepted,
            "acceptance_rate": (self._accepted / total) if total else 0.0,
            "rejection_reasons": _sorted_counter(self._rejection_reasons),
            "unique_rejection_reasons": len(self._rejection_reasons),
            "current_acceptance_streak": self._current_acceptance_streak,
            "current_rejection_streak": self._current_rejection_streak,
            "longest_acceptance_streak": self._longest_acceptance_streak,
            "longest_rejection_streak": self._longest_rejection_streak,
            "risk_flags_with_accepts": sum(
                1 for count in self._risk_flag_accepted.values() if count
            ),
            "unique_risk_flags": len(self._risk_flag_counts),
            "stress_failures_with_accepts": sum(
                1 for count in self._stress_failure_accepted.values() if count
            ),
            "unique_stress_failures": len(self._stress_failure_counts),
            "unique_models": len(self._model_totals),
            "models_with_accepts": sum(1 for count in self._model_accepted.values() if count),
            "unique_actions": len(self._action_totals),
            "actions_with_accepts": sum(
                1 for count in self._action_accepted.values() if count
            ),
            "unique_strategies": len(self._strategy_totals),
            "strategies_with_accepts": sum(
                1 for count in self._strategy_accepted.values() if count
            ),
            "unique_symbols": len(self._symbol_totals),
            "symbols_with_accepts": sum(
                1 for count in self._symbol_accepted.values() if count
            ),
        }
        if self._history_start_generated_at is not None:
            summary["history_start_generated_at"] = self._history_start_generated_at
        if self._risk_flag_counts:
            summary["risk_flag_counts"] = _sorted_counter(self._risk_flag_counts)
        if self._risk_flag_totals:
            summary["risk_flag_breakdown"] = _build_breakdown(
                self._risk_flag_totals, self._risk_flag_accepted
            )
        if self._stress_failure_counts:
            summary["stress_failure_counts"] = _sorted_counter(
                self._stress_failure_counts
            )
        if self._stress_failure_totals:
            summary["stress_failure_breakdown"] = _build_breakdown(
                self._stress_failure_totals, self._stress_failure_accepted
            )
        if self._model_totals:
            summary["model_usage"] = _sorted_counter(self._model_totals)
            summary["model_breakdown"] = _build_breakdown(
                self._model_totals, self._model_accepted, self._model_metrics
            )
        if self._action_totals:
            summary["action_usage"] = _sorted_counter(self._action_totals)
            summary["action_breakdown"] = _build_breakdown(
                self._action_totals, self._action_accepted, self._action_metrics
            )
        if self._strategy_totals:
            summary["strategy_usage"] = _sorted_counter(self._strategy_totals)
            summary["strategy_breakdown"] = _build_breakdown(
                self._strategy_totals,
                self._strategy_accepted,
                self._strategy_metrics,
            )
        if self._symbol_totals:
            summary["symbol_usage"] = _sorted_counter(self._symbol_totals)
            summary["symbol_breakdown"] = _build_breakdown(
                self._symbol_totals, self._symbol_accepted, self._symbol_metrics
            )

        for prefix, collector in self._metrics.items():
            collector.inject(summary, prefix)
        self._thresholds.inject(summary)
        _populate_latest_fields(summary, self._latest_payload)
        return summary

    def _process_payload(self, payload: Mapping[str, object]) -> None:
        is_accepted = bool(payload.get("accepted"))
        if is_accepted:
            self._accepted += 1
            self._current_acceptance_streak += 1
            self._current_rejection_streak = 0
        else:
            for reason in _iter_strings(payload.get("reasons")):
                self._rejection_reasons[str(reason)] += 1
            self._current_rejection_streak += 1
            self._current_acceptance_streak = 0
        self._longest_acceptance_streak = max(
            self._longest_acceptance_streak, self._current_acceptance_streak
        )
        self._longest_rejection_streak = max(
            self._longest_rejection_streak, self._current_rejection_streak
        )

        if self._history_start_generated_at is None:
            generated_at = _extract_generated_at(payload)
            if generated_at is not None:
                self._history_start_generated_at = generated_at

        observed_metrics: dict[str, float] = {}

        net_edge = coerce_float(payload.get("net_edge_bps"))
        if net_edge is not None:
            self.metrics["net_edge_bps"].add(net_edge, accepted=is_accepted)
            self._metrics["net_edge_bps"].add(net_edge, accepted=is_accepted)
            observed_metrics["net_edge_bps"] = net_edge

        cost = coerce_float(payload.get("cost_bps"))
        if cost is not None:
            self.metrics["cost_bps"].add(cost, accepted=is_accepted)
            observed_metrics["cost_bps"] = cost

        latency = candidate_or_payload("latency_ms")
        if latency is not None:
            self.metrics["latency_ms"].add(latency, accepted=is_accepted)
            observed_metrics["latency_ms"] = latency

        candidate_probability = candidate_or_payload("expected_probability")
        if candidate_probability is not None:
            self.metrics["expected_probability"].add(
                candidate_probability, accepted=is_accepted
            )
            observed_metrics["expected_probability"] = candidate_probability

        candidate_return = candidate_or_payload("expected_return_bps")
        if candidate_return is not None:
            self.metrics["expected_return_bps"].add(
                candidate_return, accepted=is_accepted
            )

        candidate_notional = candidate_or_payload("notional")
        if candidate_notional is not None:
            self.metrics["notional"].add(candidate_notional, accepted=is_accepted)
            observed_metrics["notional"] = candidate_notional

        if (
            candidate_probability is not None
            and candidate_return is not None
        ):
            candidate_expected_value = candidate_probability * candidate_return
            self.metrics["expected_value_bps"].add(
                candidate_expected_value, accepted=is_accepted
            )
            observed_metrics["expected_value_bps"] = candidate_expected_value
            if cost is not None:
                expected_minus_cost = candidate_expected_value - cost
                self.metrics["expected_value_minus_cost_bps"].add(
                    expected_minus_cost, accepted=is_accepted
                )
                observed_metrics["expected_value_minus_cost_bps"] = expected_minus_cost

        model_probability = _coerce_float(payload.get("model_success_probability"))
        if model_probability is not None:
            self.metrics["model_success_probability"].add(
            self._metrics["cost_bps"].add(cost, accepted=is_accepted)
            observed_metrics["cost_bps"] = cost

        model_probability = coerce_float(payload.get("model_success_probability"))
        if model_probability is not None:
            self._metrics["model_success_probability"].add(
                model_probability, accepted=is_accepted
            )

        model_return = coerce_float(payload.get("model_expected_return_bps"))
        if model_return is not None:
            self.metrics["model_expected_return_bps"].add(
                model_return, accepted=is_accepted
            )

        if model_probability is not None and model_return is not None:
            model_expected_value = model_probability * model_return
            self.metrics["model_expected_value_bps"].add(
                model_expected_value, accepted=is_accepted
            )
            if cost is not None:
                model_expected_minus_cost = model_expected_value - cost
                self.metrics["model_expected_value_minus_cost_bps"].add(
                    model_expected_minus_cost, accepted=is_accepted
                )

        for flag in _iter_strings(payload.get("risk_flags")):
            key = str(flag)
            self.risk_flag_counts[key] += 1
            self.risk_flag_totals[key] += 1
            if is_accepted:
                self.risk_flag_accepted[key] += 1

        for failure in _iter_strings(payload.get("stress_failures")):
            key = str(failure)
            self.stress_failure_counts[key] += 1
            self.stress_failure_totals[key] += 1
            if is_accepted:
                self.stress_failure_accepted[key] += 1

        model_key = payload.get("model_name")
        if model_key is not None:
            key = str(model_key)
            self.model_usage[key] += 1
            self.model_totals[key] += 1
            if is_accepted:
                self.model_accepted[key] += 1
            self._update_breakdown_metrics(
                self.model_metric_totals, key, observed_metrics, accepted=is_accepted
            )

        def _candidate_key(name: str) -> str | None:
            if candidate_map is None:
                return None
            value = candidate_map.get(name)
            return str(value) if value is not None else None

        action_key = _candidate_key("action")
        if action_key is not None:
            self.action_usage[action_key] += 1
            self.action_totals[action_key] += 1
            if is_accepted:
                self.action_accepted[action_key] += 1
            self._update_breakdown_metrics(
                self.action_metric_totals, action_key, observed_metrics, accepted=is_accepted
            )

        strategy_key = _candidate_key("strategy")
        if strategy_key is not None:
            self.strategy_usage[strategy_key] += 1
            self.strategy_totals[strategy_key] += 1
            if is_accepted:
                self.strategy_accepted[strategy_key] += 1
            self._update_breakdown_metrics(
                self.strategy_metric_totals,
                strategy_key,
                observed_metrics,
                accepted=is_accepted,
            )

        symbol_key = _candidate_key("symbol")
        if symbol_key is not None:
            self.symbol_usage[symbol_key] += 1
            self.symbol_totals[symbol_key] += 1
            if is_accepted:
                self.symbol_accepted[symbol_key] += 1
            self._update_breakdown_metrics(
                self.symbol_metric_totals,
                symbol_key,
                observed_metrics,
                accepted=is_accepted,
            )

        thresholds_payload = payload.get("thresholds")
        thresholds_map = (
            _normalize_thresholds(thresholds_payload)
            if isinstance(thresholds_payload, Mapping)
            else None
        )
        if thresholds_map:
            for threshold_key, (metric_key, limit_type, margin_prefix) in _THRESHOLD_DEFINITIONS.items():
                threshold_value = thresholds_map.get(threshold_key)
                observed_value = observed_metrics.get(metric_key)
                if threshold_value is None or observed_value is None:
                    continue
                if limit_type == "min":
                    margin = observed_value - threshold_value
                    breached = observed_value < threshold_value
                else:
                    margin = threshold_value - observed_value
                    breached = observed_value > threshold_value
                collector = self.threshold_collectors.setdefault(
                    margin_prefix, ThresholdMarginCollector()
                )
                collector.add(margin, accepted=is_accepted, breached=breached)

    def finalize(self) -> Mapping[str, object]:
        summary = self.summary
        summary["accepted"] = self.accepted
        summary["rejected"] = self.total - self.accepted
        summary["acceptance_rate"] = self.accepted / self.total if self.total else 0.0
        summary["rejection_reasons"] = dict(
            sorted(self.rejection_reasons.items(), key=lambda item: item[1], reverse=True)
        )
        summary["unique_rejection_reasons"] = len(summary["rejection_reasons"])
        summary["current_acceptance_streak"] = self.current_acceptance_streak
        summary["current_rejection_streak"] = self.current_rejection_streak
        summary["longest_acceptance_streak"] = self.longest_acceptance_streak
        summary["longest_rejection_streak"] = self.longest_rejection_streak

        if self.history_generated_at:
            summary["history_start_generated_at"] = self.history_generated_at[0]

        for prefix, settings in METRIC_SETTINGS.items():
            self.metrics[prefix].inject(summary, prefix, settings)

        for prefix, collector in self.threshold_collectors.items():
            collector.inject(summary, prefix)

        if self.risk_flag_counts:
            summary["risk_flag_counts"] = dict(
                sorted(self.risk_flag_counts.items(), key=lambda item: item[1], reverse=True)
            )
        summary["unique_risk_flags"] = len(self.risk_flag_counts)
        summary["risk_flags_with_accepts"] = sum(
            1 for count in self.risk_flag_accepted.values() if count
            self._metrics["model_expected_return_bps"].add(
                model_return, accepted=is_accepted
            )

        if model_return is not None and model_probability is not None:
            model_expected_value = model_return * model_probability
            self._metrics["model_expected_value_bps"].add(
                model_expected_value, accepted=is_accepted
            )
            if cost is not None:
                self._metrics["model_expected_value_minus_cost_bps"].add(
                    model_expected_value - cost, accepted=is_accepted
                )

        candidate = payload.get("candidate")
        candidate_probability = None
        candidate_return = None
        candidate_notional = None
        latency_value = coerce_float(payload.get("latency_ms"))
        candidate_latency = None
        action_key: object | None = None
        strategy_key: object | None = None
        symbol_key: object | None = None

        if isinstance(candidate, Mapping):
            candidate_probability = coerce_float(candidate.get("expected_probability"))
            if candidate_probability is not None:
                self._metrics["expected_probability"].add(
                    candidate_probability, accepted=is_accepted
                )
                observed_metrics["expected_probability"] = candidate_probability

            candidate_return = coerce_float(candidate.get("expected_return_bps"))
            if candidate_return is not None:
                self._metrics["expected_return_bps"].add(
                    candidate_return, accepted=is_accepted
                )

            if candidate_probability is not None and candidate_return is not None:
                expected_value = candidate_return * candidate_probability
                self._metrics["expected_value_bps"].add(
                    expected_value, accepted=is_accepted
                )
                observed_metrics["expected_value_bps"] = expected_value
        candidate = payload.get("candidate")
        if isinstance(candidate, Mapping):
            probability = _coerce_float(candidate.get("expected_probability"))
            if probability is not None:
                probabilities.append(probability)
                observed_metrics["expected_probability"] = probability
                if is_accepted:
                    accepted_probabilities.append(probability)
                else:
                    rejected_probabilities.append(probability)
            expected_return = _coerce_float(candidate.get("expected_return_bps"))
            candidate_expected_value = None
            if expected_return is not None:
                expected_returns.append(expected_return)
                if is_accepted:
                    accepted_expected_returns.append(expected_return)
                else:
                    rejected_expected_returns.append(expected_return)
            if expected_return is not None and probability is not None:
                candidate_expected_value = expected_return * probability
                expected_values.append(candidate_expected_value)
                observed_metrics["expected_value_bps"] = candidate_expected_value
                if is_accepted:
                    accepted_expected_values.append(candidate_expected_value)
                else:
                    rejected_expected_values.append(candidate_expected_value)
                if cost is not None:
                    expected_minus_cost = expected_value - cost
                    self._metrics["expected_value_minus_cost_bps"].add(
                        expected_minus_cost, accepted=is_accepted
                    )
                    observed_metrics[
                        "expected_value_minus_cost_bps"
                    ] = expected_minus_cost

            candidate_notional = coerce_float(candidate.get("notional"))
            if candidate_notional is not None:
                self._metrics["notional"].add(
                    candidate_notional, accepted=is_accepted
                )
                observed_metrics["notional"] = candidate_notional

            candidate_latency = coerce_float(candidate.get("latency_ms"))
            if latency_value is None and candidate_latency is not None:
                latency_value = candidate_latency

            action_key = candidate.get("action")
            strategy_key = candidate.get("strategy")
            symbol_key = candidate.get("symbol")

        if latency_value is None and candidate_latency is not None:
            latency_value = candidate_latency

        if latency_value is not None:
            self._metrics["latency_ms"].add(latency_value, accepted=is_accepted)
            observed_metrics["latency_ms"] = latency_value

        self._register_dimension(
            action_key,
            self._action_totals,
            self._action_accepted,
            self._action_metrics,
            is_accepted,
            observed_metrics,
                if margin < 0:
                    threshold_breach_counts[base_key] = (
                        threshold_breach_counts.get(base_key, 0) + 1
                    )
                    if is_accepted:
                        accepted_threshold_breach_counts[base_key] = (
                            accepted_threshold_breach_counts.get(base_key, 0) + 1
                        )
                    else:
                        rejected_threshold_breach_counts[base_key] = (
                            rejected_threshold_breach_counts.get(base_key, 0) + 1
                        )
    summary["accepted"] = accepted
    summary["rejected"] = total - accepted
    summary["acceptance_rate"] = accepted / total if total else 0.0
    summary["rejection_reasons"] = dict(
        sorted(rejection_reasons.items(), key=lambda item: item[1], reverse=True)
    )
    summary["unique_rejection_reasons"] = len(summary["rejection_reasons"])
    summary["current_acceptance_streak"] = current_acceptance_streak
    summary["current_rejection_streak"] = current_rejection_streak

    if net_edges:
        total_net_edge = sum(net_edges)
        summary["avg_net_edge_bps"] = total_net_edge / len(net_edges)
        summary["sum_net_edge_bps"] = total_net_edge
    if costs:
        total_cost = sum(costs)
        summary["avg_cost_bps"] = total_cost / len(costs)
        summary["sum_cost_bps"] = total_cost
    if probabilities:
        summary["avg_expected_probability"] = sum(probabilities) / len(probabilities)
    if expected_returns:
        total_expected_return = sum(expected_returns)
        summary["avg_expected_return_bps"] = (
            total_expected_return / len(expected_returns)
        )
        summary["sum_expected_return_bps"] = total_expected_return
    if expected_values:
        total_expected_value = sum(expected_values)
        summary["avg_expected_value_bps"] = total_expected_value / len(
            expected_values
        )
        summary["sum_expected_value_bps"] = total_expected_value
    if expected_values_minus_costs:
        total_expected_value_minus_cost = sum(expected_values_minus_costs)
        summary["avg_expected_value_minus_cost_bps"] = (
            total_expected_value_minus_cost / len(expected_values_minus_costs)
        )
        summary["sum_expected_value_minus_cost_bps"] = (
            total_expected_value_minus_cost
        )
    if notionals:
        total_notional = sum(notionals)
        summary["avg_notional"] = total_notional / len(notionals)
        summary["sum_notional"] = total_notional

    if net_edges:
        summary["avg_net_edge_bps"] = sum(net_edges) / len(net_edges)
    if costs:
        summary["avg_cost_bps"] = sum(costs) / len(costs)
    if probabilities:
        summary["avg_expected_probability"] = sum(probabilities) / len(probabilities)
    if expected_returns:
        summary["avg_expected_return_bps"] = sum(expected_returns) / len(expected_returns)
    if notionals:
        summary["avg_notional"] = sum(notionals) / len(notionals)
    if model_probabilities:
        summary["avg_model_success_probability"] = sum(model_probabilities) / len(
            model_probabilities
        )
    if model_returns:
        total_model_expected_return = sum(model_returns)
        summary["avg_model_expected_return_bps"] = (
            total_model_expected_return / len(model_returns)
        )
        summary["sum_model_expected_return_bps"] = total_model_expected_return
    if model_expected_values:
        total_model_expected_value = sum(model_expected_values)
        summary["avg_model_expected_value_bps"] = (
            total_model_expected_value / len(model_expected_values)
        )
        self._register_dimension(
            strategy_key,
            self._strategy_totals,
            self._strategy_accepted,
            self._strategy_metrics,
            is_accepted,
            observed_metrics,
        )
        self._register_dimension(
            symbol_key,
            self._symbol_totals,
            self._symbol_accepted,
            self._symbol_metrics,
            is_accepted,
            observed_metrics,
        )
        if self.risk_flag_totals:
            summary["risk_flag_breakdown"] = self._build_breakdown(
                self.risk_flag_totals, self.risk_flag_accepted
            )

        if self.stress_failure_counts:
            summary["stress_failure_counts"] = dict(
                sorted(
                    self.stress_failure_counts.items(), key=lambda item: item[1], reverse=True
                )
            )
        summary["unique_stress_failures"] = len(self.stress_failure_counts)
        summary["stress_failures_with_accepts"] = sum(
            1 for count in self.stress_failure_accepted.values() if count
        )
        if self.stress_failure_totals:
            summary["stress_failure_breakdown"] = self._build_breakdown(
                self.stress_failure_totals, self.stress_failure_accepted
            )

        summary["unique_models"] = len(self.model_usage)
        summary["models_with_accepts"] = sum(
            1 for count in self.model_accepted.values() if count
        model_name = payload.get("model_name")
        self._register_dimension(
            model_name,
            self._model_totals,
            self._model_accepted,
            self._model_metrics,
            is_accepted,
            observed_metrics,
        )
        if self.model_usage:
            summary["model_usage"] = dict(
                sorted(self.model_usage.items(), key=lambda item: item[1], reverse=True)
            )
        if self.model_totals:
            summary["model_breakdown"] = self._build_breakdown(
                self.model_totals, self.model_accepted, self.model_metric_totals
            )

        summary["unique_actions"] = len(self.action_usage)
        summary["actions_with_accepts"] = sum(
            1 for count in self.action_accepted.values() if count
        )
        if self.action_usage:
            summary["action_usage"] = dict(
                sorted(self.action_usage.items(), key=lambda item: item[1], reverse=True)
            )
        if self.action_totals:
            summary["action_breakdown"] = self._build_breakdown(
                self.action_totals, self.action_accepted, self.action_metric_totals
            )

        summary["unique_strategies"] = len(self.strategy_usage)
        summary["strategies_with_accepts"] = sum(
            1 for count in self.strategy_accepted.values() if count
        )
        if self.strategy_usage:
            summary["strategy_usage"] = dict(
                sorted(self.strategy_usage.items(), key=lambda item: item[1], reverse=True)
            )
        if self.strategy_totals:
            summary["strategy_breakdown"] = self._build_breakdown(
                self.strategy_totals, self.strategy_accepted, self.strategy_metric_totals
            )

        summary["unique_symbols"] = len(self.symbol_usage)
        summary["symbols_with_accepts"] = sum(
            1 for count in self.symbol_accepted.values() if count
        )
        if self.symbol_usage:
            summary["symbol_usage"] = dict(
                sorted(self.symbol_usage.items(), key=lambda item: item[1], reverse=True)
            )
        if self.symbol_totals:
            summary["symbol_breakdown"] = self._build_breakdown(
                self.symbol_totals, self.symbol_accepted, self.symbol_metric_totals
            )

        if self.latest_payload is not None:
            self._inject_latest_payload_details(self.latest_payload, summary)

        return summary

    def _update_breakdown_metrics(
        self,
        container: dict[str, dict[str, _SegmentMetricAccumulator]],
        key: str,
        observed_metrics: Mapping[str, float],
        *,
        accepted: bool,
    ) -> None:
        metric_map = container.setdefault(key, {})
        for metric in self.breakdown_metric_keys:
            value = observed_metrics.get(metric)
            if value is None:
                continue
            accumulator = metric_map.get(metric)
            if accumulator is None:
                accumulator = _SegmentMetricAccumulator()
                metric_map[metric] = accumulator
            accumulator.update(value, accepted=accepted)

    def _build_breakdown(
        self,
        totals: Counter[str],
        accepted_counts: Counter[str],
        metrics: Mapping[str, Mapping[str, _SegmentMetricAccumulator]] | None = None,
    ) -> Mapping[str, Mapping[str, object]]:
        breakdown: MutableMapping[str, Mapping[str, object]] = {}
        for key, total_count in totals.most_common():
            accepted_count = accepted_counts.get(key, 0)
            rejected_count = max(total_count - accepted_count, 0)
            entry: dict[str, object] = {
                "total": total_count,
                "accepted": accepted_count,
                "rejected": rejected_count,
                "acceptance_rate": (accepted_count / total_count) if total_count else 0.0,
            }
            if metrics and key in metrics:
                metric_entries: dict[str, Mapping[str, float | int]] = {}
                for metric_key, accumulator in metrics[key].items():
                    metric_entries[metric_key] = accumulator.build_summary()
                if metric_entries:
                    entry["metrics"] = metric_entries
            breakdown[key] = entry
        return breakdown

    def _inject_latest_payload_details(
        self,
        payload: Mapping[str, object],
        summary: MutableMapping[str, object],
    ) -> None:
        model_name = payload.get("model_name")
        if model_name is not None:
            summary["latest_model"] = str(model_name)

        summary["latest_status"] = (
            "accepted" if bool(payload.get("accepted")) else "rejected"
        )

        net_edge = _coerce_float(payload.get("net_edge_bps"))
        if net_edge is not None:
            summary["latest_net_edge_bps"] = net_edge

        cost = _coerce_float(payload.get("cost_bps"))
        if cost is not None:
            summary["latest_cost_bps"] = cost

        model_return = _coerce_float(payload.get("model_expected_return_bps"))
        model_probability = _coerce_float(payload.get("model_success_probability"))
        if model_return is not None:
            summary["latest_model_expected_return_bps"] = model_return
        if model_probability is not None:
            summary["latest_model_success_probability"] = model_probability
        if model_return is not None and model_probability is not None:
            latest_model_expected_value = model_return * model_probability
            summary["latest_model_expected_value_bps"] = latest_model_expected_value
            if cost is not None:
                summary["latest_model_expected_value_minus_cost_bps"] = (
                    latest_model_expected_value - cost
        for flag in _iter_strings(payload.get("risk_flags")):
            key = str(flag)
            self._risk_flag_counts[key] += 1
            self._risk_flag_totals[key] += 1
            if is_accepted:
                self._risk_flag_accepted[key] += 1

        for failure in _iter_strings(payload.get("stress_failures")):
            key = str(failure)
            self._stress_failure_counts[key] += 1
            self._stress_failure_totals[key] += 1
            if is_accepted:
                self._stress_failure_accepted[key] += 1

        thresholds_payload = payload.get("thresholds")
        normalized_thresholds = _normalize_thresholds(
            thresholds_payload if isinstance(thresholds_payload, Mapping) else None
        )
        self._thresholds.add(normalized_thresholds, observed_metrics, accepted=is_accepted)

    def _register_dimension(
        self,
        raw_key: object,
        totals: Counter[str],
        accepted: Counter[str],
        metrics: dict[str, dict[str, SegmentMetricAccumulator]],
        is_accepted: bool,
        observed_metrics: Mapping[str, float],
    ) -> None:
        if raw_key is None:
            return
        key = str(raw_key)
        totals[key] += 1
        if is_accepted:
            accepted[key] += 1
        if not observed_metrics:
            return
        metric_map = metrics.setdefault(key, {})
        for metric_key, value in observed_metrics.items():
            if metric_key not in _BREAKDOWN_METRIC_KEYS:
                continue
            accumulator = metric_map.get(metric_key)
            if accumulator is None:
                accumulator = SegmentMetricAccumulator()
                metric_map[metric_key] = accumulator
            accumulator.update(value, accepted=is_accepted)


def _populate_latest_fields(
    summary: MutableMapping[str, object],
    payload: Mapping[str, object] | None,
) -> None:
    if not payload:
        return
    summary["latest_status"] = (
        "accepted" if bool(payload.get("accepted")) else "rejected"
    )
    model_name = payload.get("model_name")
    if model_name is not None:
        summary["latest_model"] = str(model_name)

    latest_net_edge = coerce_float(payload.get("net_edge_bps"))
    if latest_net_edge is not None:
        summary["latest_net_edge_bps"] = latest_net_edge

    latest_cost = coerce_float(payload.get("cost_bps"))
    if latest_cost is not None:
        summary["latest_cost_bps"] = latest_cost

    latest_model_return = coerce_float(payload.get("model_expected_return_bps"))
    if latest_model_return is not None:
        summary["latest_model_expected_return_bps"] = latest_model_return
    latest_payload = windowed[-1]
    if latencies:
        summary["avg_latency_ms"] = sum(latencies) / len(latencies)
    if isinstance(latest_payload, Mapping):
        latest_model = latest_payload.get("model_name")
        if latest_model:
            summary["latest_model"] = str(latest_model)

    latest_model_probability = coerce_float(payload.get("model_success_probability"))
    if latest_model_probability is not None:
        summary["latest_model_success_probability"] = latest_model_probability

    if latest_model_return is not None and latest_model_probability is not None:
        latest_model_expected_value = latest_model_return * latest_model_probability
        summary["latest_model_expected_value_bps"] = latest_model_expected_value
        if latest_cost is not None:
            summary["latest_model_expected_value_minus_cost_bps"] = (
                latest_model_expected_value - latest_cost
            )

    candidate = payload.get("candidate")
    candidate_payload: MutableMapping[str, object] = {}
    candidate_probability = None
    candidate_return = None
    candidate_notional = None
    candidate_latency = None

    if isinstance(candidate, Mapping):
        for key in ("symbol", "action", "strategy"):
            value = candidate.get(key)
            if value is not None:
                candidate_payload[key] = value
        candidate_probability = coerce_float(candidate.get("expected_probability"))
        if candidate_probability is not None:
            summary["latest_expected_probability"] = candidate_probability
        candidate_return = coerce_float(candidate.get("expected_return_bps"))
        if candidate_return is not None:
            summary["latest_expected_return_bps"] = candidate_return
        candidate_notional = coerce_float(candidate.get("notional"))
        if candidate_notional is not None:
            summary["latest_notional"] = candidate_notional
        candidate_latency = coerce_float(candidate.get("latency_ms"))
        metadata = _extract_candidate_metadata(candidate)
        if metadata:
            candidate_payload.setdefault("metadata", metadata)
        if (
            candidate_probability is not None
            and candidate_return is not None
        ):
            expected_value = candidate_return * candidate_probability
            candidate_payload["expected_value_bps"] = expected_value
            summary["latest_expected_value_bps"] = expected_value
            if latest_cost is not None:
                summary["latest_expected_value_minus_cost_bps"] = (
                    expected_value - latest_cost
                )
        if candidate_payload:
            summary["latest_candidate"] = candidate_payload

    latest_latency = coerce_float(payload.get("latency_ms"))
    if latest_latency is None and candidate_latency is not None:
        latest_latency = candidate_latency
    if latest_latency is not None:
        summary["latest_latency_ms"] = latest_latency

    thresholds = _normalize_thresholds(payload.get("thresholds"))
    if thresholds:
        summary["latest_thresholds"] = dict(thresholds)

    reasons = list(_iter_strings(payload.get("reasons")))
    if reasons:
        summary["latest_reasons"] = reasons

    risk_flags = list(_iter_strings(payload.get("risk_flags")))
    if risk_flags:
        summary["latest_risk_flags"] = risk_flags

    stress_failures = list(_iter_strings(payload.get("stress_failures")))
    if stress_failures:
        summary["latest_stress_failures"] = stress_failures

    model_selection = payload.get("model_selection")
    if isinstance(model_selection, Mapping):
        summary["latest_model_selection"] = {
            str(key): model_selection[key] for key in model_selection
        }

        latency = _coerce_float(payload.get("latency_ms"))
        latency_value = latency
        if latency is not None:
            summary["latest_latency_ms"] = latency

        thresholds_payload = payload.get("thresholds")
        thresholds_map = (
            _normalize_thresholds(thresholds_payload)
            if isinstance(thresholds_payload, Mapping)
            else None
        )
        latest_threshold_lookup: Mapping[str, float | None] | None = None
        if thresholds_map:
            latest_threshold_lookup = dict(thresholds_map)
            summary["latest_thresholds"] = dict(thresholds_map)

        latest_reasons = list(_iter_strings(payload.get("reasons")))
        if latest_reasons:
            summary["latest_reasons"] = latest_reasons

        latest_risk_flags = list(_iter_strings(payload.get("risk_flags")))
        if latest_risk_flags:
            summary["latest_risk_flags"] = latest_risk_flags

        latest_failures = list(_iter_strings(payload.get("stress_failures")))
        if latest_failures:
            summary["latest_stress_failures"] = latest_failures

        model_selection = payload.get("model_selection")
        if isinstance(model_selection, Mapping):
            summary["latest_model_selection"] = {
                str(key): model_selection[key] for key in model_selection
            }

        candidate = payload.get("candidate")
        candidate_map: Mapping[str, object] | None = (
            candidate if isinstance(candidate, Mapping) else None
        )

        candidate_probability_value: float | None = None
        candidate_notional_value: float | None = None

        if candidate_map:
            candidate_payload: MutableMapping[str, object] = {}
            for key in ("symbol", "action", "strategy"):
                value = candidate_map.get(key)
                if value is not None:
                    candidate_payload[key] = value

            candidate_probability = _coerce_float(candidate_map.get("expected_probability"))
            candidate_return = _coerce_float(candidate_map.get("expected_return_bps"))
            candidate_notional = _coerce_float(candidate_map.get("notional"))
            candidate_latency = _coerce_float(candidate_map.get("latency_ms"))

            candidate_probability_value = candidate_probability
            candidate_notional_value = candidate_notional

            if candidate_probability is not None:
                summary["latest_expected_probability"] = candidate_probability
            if candidate_return is not None:
                summary["latest_expected_return_bps"] = candidate_return
            if candidate_notional is not None:
                summary["latest_notional"] = candidate_notional
            if candidate_latency is not None and latency_value is None:
                latency_value = candidate_latency
                summary["latest_latency_ms"] = candidate_latency

            if (
                candidate_probability is not None
                and candidate_return is not None
            ):
                candidate_expected_value = candidate_probability * candidate_return
                candidate_payload["expected_value_bps"] = candidate_expected_value
                summary["latest_expected_value_bps"] = candidate_expected_value
                if cost is not None:
                    summary["latest_expected_value_minus_cost_bps"] = (
                        candidate_expected_value - cost
                    )

            metadata = _extract_candidate_metadata(candidate_map)
            generated_at = None
            if metadata:
                generated_at = metadata.get("generated_at") or metadata.get("timestamp")
            if generated_at is None:
                generated_at = candidate_map.get("generated_at")
            if generated_at is not None:
                summary["latest_generated_at"] = str(generated_at)

            if candidate_payload:
                summary["latest_candidate"] = dict(candidate_payload)

        if "latest_generated_at" not in summary:
            generated_at = _extract_generated_at(payload)
            if generated_at is not None:
                summary["latest_generated_at"] = generated_at

        if latest_threshold_lookup:
            min_probability = _coerce_float(
                latest_threshold_lookup.get("min_probability")
            )
            if min_probability is not None and candidate_probability_value is not None:
                summary["latest_probability_threshold_margin"] = (
                    candidate_probability_value - min_probability
                )

            max_cost_threshold = _coerce_float(
                latest_threshold_lookup.get("max_cost_bps")
            )
            if max_cost_threshold is not None and cost is not None:
                summary["latest_cost_threshold_margin"] = max_cost_threshold - cost

            min_net_edge_threshold = _coerce_float(
                latest_threshold_lookup.get("min_net_edge_bps")
            )
            if min_net_edge_threshold is not None and net_edge is not None:
                summary["latest_net_edge_threshold_margin"] = (
                    net_edge - min_net_edge_threshold
                )

            max_latency_threshold = _coerce_float(
                latest_threshold_lookup.get("max_latency_ms")
            )
            if max_latency_threshold is not None and latency_value is not None:
                summary["latest_latency_threshold_margin"] = (
                    max_latency_threshold - latency_value
                )

            max_notional_threshold = _coerce_float(
                latest_threshold_lookup.get("max_trade_notional")
            )
            if (
                max_notional_threshold is not None
                and candidate_notional_value is not None
            ):
                summary["latest_notional_threshold_margin"] = (
                    max_notional_threshold - candidate_notional_value
                )


def summarize_evaluation_payloads(
    evaluations: Sequence[Mapping[str, object]],
    *,
    history_limit: int | None = None,
) -> Mapping[str, object]:
    aggregator = DecisionSummaryAggregator(evaluations, history_limit=history_limit)
    return aggregator.build_summary()


__all__ = [
    "DecisionSummaryAggregator",
    "summarize_evaluation_payloads",
]

    generated_at = _extract_generated_at(payload)
    if generated_at is not None:
        summary["latest_generated_at"] = generated_at

    if not thresholds:
        return

    def _margin(
        threshold_key: str,
        observed_value: float | None,
        mode: str,
    ) -> float | None:
        if observed_value is None:
            return None
        threshold_candidate = thresholds.get(threshold_key)
        threshold_value = coerce_float(threshold_candidate)
        if threshold_value is None:
            return None
        if mode == "min":
            return observed_value - threshold_value
        return threshold_value - observed_value

    probability_margin = _margin(
        "min_probability", candidate_probability, "min"
    )
    if probability_margin is not None:
        summary["latest_probability_threshold_margin"] = probability_margin

    cost_margin = _margin("max_cost_bps", latest_cost, "max")
    if cost_margin is not None:
        summary["latest_cost_threshold_margin"] = cost_margin

    net_edge_margin = _margin("min_net_edge_bps", latest_net_edge, "min")
    if net_edge_margin is not None:
        summary["latest_net_edge_threshold_margin"] = net_edge_margin

    latency_margin = _margin("max_latency_ms", latest_latency, "max")
    if latency_margin is not None:
        summary["latest_latency_threshold_margin"] = latency_margin

    notional_margin = _margin("max_trade_notional", candidate_notional, "max")
    if notional_margin is not None:
        summary["latest_notional_threshold_margin"] = notional_margin


def summarize_evaluation_payloads(
    evaluations: Sequence[Mapping[str, object]],
    *,
    history_limit: int | None = None,
) -> DecisionEngineSummary:
    items = [payload for payload in evaluations if isinstance(payload, Mapping)]
    full_total = len(items)
    effective_limit = _resolve_history_limit(history_limit, full_total)
    if full_total and effective_limit and effective_limit < full_total:
        windowed = items[-effective_limit:]
    else:
        windowed = items

    aggregator = DecisionSummaryAggregator(windowed)
    summary_payload = aggregator.build()
    summary_payload["history_window"] = len(windowed)
    summary_payload["history_limit"] = effective_limit if effective_limit else full_total
    summary_payload["full_total"] = full_total

    if full_total:
        if len(windowed) != full_total:
            accepted_full = sum(
                1
                for payload in items
                if bool(payload.get("accepted"))
            )
            summary_payload["full_accepted"] = accepted_full
            summary_payload["full_rejected"] = full_total - accepted_full
            summary_payload["full_acceptance_rate"] = (
                accepted_full / full_total if full_total else 0.0
            )
        else:
            summary_payload["full_accepted"] = summary_payload["accepted"]
            summary_payload["full_rejected"] = summary_payload["rejected"]
            summary_payload["full_acceptance_rate"] = summary_payload["acceptance_rate"]
    else:
        summary_payload["full_accepted"] = 0
        summary_payload["full_rejected"] = 0
        summary_payload["full_acceptance_rate"] = 0.0

    return DecisionEngineSummary.model_validate(summary_payload)
    return DecisionEngineSummary.model_validate(summary)


__all__ = ["DecisionEngineSummary", "summarize_evaluation_payloads"]
