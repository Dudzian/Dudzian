"""Narzędzia do agregacji i raportowania jakości decyzji AI."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence


def _coerce_float(value: object) -> float | None:
    """Próbuje rzutować dowolną wartość na float."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensywne
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_thresholds(snapshot: Mapping[str, object] | None) -> Mapping[str, float | None] | None:
    if not snapshot or not isinstance(snapshot, Mapping):
        return None
    normalized: MutableMapping[str, float | None] = {}
    for key, value in snapshot.items():
        normalized[str(key)] = _coerce_float(value)
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

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        generated_at = metadata.get("generated_at") or metadata.get("timestamp")
        if generated_at is not None:
            return str(generated_at)

    raw = payload.get("generated_at")
    if raw is not None:
        return str(raw)
    return None


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


def _inject_distribution_metrics(
    summary: MutableMapping[str, object],
    values: Sequence[float],
    *,
    prefix: str,
    quantiles: Sequence[tuple[float, str]] = (),
    include_minmax: bool = True,
) -> None:
    if not values:
        return
    if include_minmax:
        summary[f"min_{prefix}"] = min(values)
        summary[f"max_{prefix}"] = max(values)
    if not quantiles:
        return
    sorted_values = sorted(values)
    for quantile, label in quantiles:
        summary[f"{label}_{prefix}"] = _compute_quantile(sorted_values, quantile)


def _compute_std(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Brak wartości do policzenia odchylenia standardowego")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _inject_std_metric(
    summary: MutableMapping[str, object],
    values: Sequence[float],
    *,
    prefix: str,
) -> None:
    if not values:
        return
    summary[f"std_{prefix}"] = _compute_std(values)


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

    __slots__ = (
        "total_sum",
        "total_count",
        "accepted_sum",
        "accepted_count",
        "rejected_sum",
        "rejected_count",
    )

    def __init__(self) -> None:
        self.total_sum = 0.0
        self.total_count = 0
        self.accepted_sum = 0.0
        self.accepted_count = 0
        self.rejected_sum = 0.0
        self.rejected_count = 0

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


def _inject_segmented_threshold_metrics(
    summary: MutableMapping[str, object],
    values: Sequence[float],
    *,
    prefix: str,
    segment: str,
    quantiles: Sequence[tuple[float, str]] = ((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
) -> None:
    if not values:
        return
    summary[f"{segment}_min_{prefix}"] = min(values)
    summary[f"{segment}_max_{prefix}"] = max(values)
    if quantiles:
        sorted_values = sorted(values)
        for quantile, label in quantiles:
            summary[f"{segment}_{label}_{prefix}"] = _compute_quantile(
                sorted_values, quantile
            )
    summary[f"{segment}_std_{prefix}"] = _compute_std(values)


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

        observed_metrics: dict[str, float] = {}

        net_edge = _coerce_float(payload.get("net_edge_bps"))
        if net_edge is not None:
            self.metrics["net_edge_bps"].add(net_edge, accepted=is_accepted)
            observed_metrics["net_edge_bps"] = net_edge

        cost = _coerce_float(payload.get("cost_bps"))
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
                model_probability, accepted=is_accepted
            )

        model_return = _coerce_float(payload.get("model_expected_return_bps"))
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
                )

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

