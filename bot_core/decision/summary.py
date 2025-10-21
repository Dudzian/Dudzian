"""Utilities for aggregating Decision Engine evaluation payloads."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.decision.models import DecisionEngineSummary

from .utils import coerce_float

__all__ = ["DecisionSummaryAggregator", "summarize_evaluation_payloads", "DecisionEngineSummary"]


def _summary_mapping(model: DecisionEngineSummary) -> dict[str, object]:
    return model.model_dump(exclude_none=False)

DecisionEngineSummary.keys = lambda self: _summary_mapping(self).keys()  # type: ignore[attr-defined]
DecisionEngineSummary.__iter__ = lambda self: iter(_summary_mapping(self))  # type: ignore[attr-defined]
DecisionEngineSummary.__getitem__ = lambda self, item: _summary_mapping(self)[item]  # type: ignore[attr-defined]
DecisionEngineSummary.__len__ = lambda self: len(_summary_mapping(self))  # type: ignore[attr-defined]


@dataclass
class _DistributionAccumulator:
    """Przechowuje wartości całkowite oraz segmentowe dla metryk liczbowych."""

    total: list[float] = field(default_factory=list)
    accepted: list[float] = field(default_factory=list)
    rejected: list[float] = field(default_factory=list)

    def add(self, value: float, *, accepted: bool) -> None:
        self.total.append(value)
        if accepted:
            self.accepted.append(value)
        else:
            self.rejected.append(value)


@dataclass(frozen=True)
class _MetricSpec:
    overall_quantiles: tuple[tuple[float, str], ...] = ()
    overall_include_minmax: bool = True
    overall_include_sum: bool = True
    overall_include_std: bool = True
    segment_quantiles: tuple[tuple[float, str], ...] = (
        (0.1, "p10"),
        (0.5, "median"),
        (0.9, "p90"),
    )
    segment_include_minmax: bool = True
    segment_include_sum: bool = True
    segment_include_std: bool = True


class _AttrDict(dict):
    """Dictionary subclass that also exposes keys as attributes."""

    __slots__ = ()

    def __getattr__(self, item: str) -> object:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


_METRIC_SPECS: Mapping[str, _MetricSpec] = {
    "net_edge_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
    ),
    "cost_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
    "expected_probability": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        overall_include_minmax=False,
        overall_include_sum=False,
        segment_quantiles=((0.5, "median"),),
        segment_include_minmax=False,
        segment_include_sum=False,
    ),
    "expected_return_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        segment_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
    "expected_value_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        segment_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
    "expected_value_minus_cost_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
@@ -126,759 +110,155 @@ _METRIC_SPECS: Mapping[str, _MetricSpec] = {
        overall_quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
        segment_quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
    ),
    "model_success_probability": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        overall_include_minmax=False,
        overall_include_sum=False,
        segment_quantiles=((0.5, "median"),),
        segment_include_minmax=False,
        segment_include_sum=False,
    ),
    "model_expected_return_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        segment_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
    "model_expected_value_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        segment_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
    "model_expected_value_minus_cost_bps": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        segment_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
}

_THRESHOLD_DISTRIBUTION_QUANTILES: tuple[tuple[float, str], ...] = (
    (0.1, "p10"),
    (0.5, "median"),
    (0.9, "p90"),
)

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


_THRESHOLD_DEFINITIONS: Mapping[str, tuple[str, str, str]] = {
    "min_probability": ("expected_probability", "min", "probability_threshold_margin"),
    "min_net_edge_bps": ("net_edge_bps", "min", "net_edge_threshold_margin"),
    "max_cost_bps": ("cost_bps", "max", "cost_threshold_margin"),
    "max_latency_ms": ("latency_ms", "max", "latency_threshold_margin"),
    "max_trade_notional": ("notional", "max", "notional_threshold_margin"),
}

def summarize_evaluation_payloads(
    evaluations: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    history_limit: int | None = None,
) -> DecisionEngineSummary:
    """Build a validated ``DecisionEngineSummary`` for the provided evaluations."""

    aggregator = DecisionSummaryAggregator(evaluations, history_limit=history_limit)
    summary = dict(aggregator.build_summary())
    model_payload = dict(summary)
    model_payload["type"] = "decision_engine_summary"
    model = DecisionEngineSummary.model_validate(model_payload)
    _attach_breakdown_namespaces(model)
    return model


class DecisionSummaryAggregator:
    def __init__(
        self,
        evaluations: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        *,
        history_limit: int | None,
    ) -> None:
        self._raw_items = list(evaluations)
        self._items = [payload for payload in self._raw_items if isinstance(payload, Mapping)]
        self._full_total = len(self._items)
        self._effective_limit = _resolve_history_limit(history_limit, self._full_total)
        if (
            self._full_total
            and self._effective_limit
            and self._effective_limit < self._full_total
        ):
            window_start = self._full_total - self._effective_limit
            self._window = self._items[window_start:]
        else:
            self._window = self._items
        self._history_window = len(self._window)
        self._full_accepted = sum(1 for payload in self._items if bool(payload.get("accepted")))

    def build_summary(self) -> MutableMapping[str, object]:
        summary: MutableMapping[str, object] = {
            "total": self._history_window,
            "accepted": 0,
            "rejected": 0,
            "acceptance_rate": 0.0,
            "history_limit": self._effective_limit if self._effective_limit else self._full_total,
            "history_window": self._history_window,
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
            "full_total": self._full_total,
            "full_accepted": self._full_accepted,
            "full_rejected": self._full_total - self._full_accepted,
            "full_acceptance_rate": (
@@ -918,822 +298,109 @@ class DecisionSummaryAggregator:
        action_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
        strategy_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
        symbol_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
        history_generated_at: list[str] = []
        threshold_margin_values: dict[str, list[float]] = {}
        accepted_threshold_margin_values: dict[str, list[float]] = {}
        rejected_threshold_margin_values: dict[str, list[float]] = {}
        threshold_breach_counts: dict[str, int] = {}
        accepted_threshold_breach_counts: dict[str, int] = {}
        rejected_threshold_breach_counts: dict[str, int] = {}

        accepted = 0
        current_acceptance_streak = 0
        current_rejection_streak = 0

        for payload in self._window:
            if not isinstance(payload, Mapping):
                continue
            is_accepted = bool(payload.get("accepted"))
            if is_accepted:
                accepted += 1
                current_acceptance_streak += 1
                current_rejection_streak = 0
            else:
                current_rejection_streak += 1
                current_acceptance_streak = 0
                for reason in _iter_strings(payload.get("reasons")):
                    rejection_reasons[reason] += 1

            summary["longest_acceptance_streak"] = max(
                summary["longest_acceptance_streak"], current_acceptance_streak
            )
            summary["longest_rejection_streak"] = max(
                summary["longest_rejection_streak"], current_rejection_streak
            )

            observed_metrics: dict[str, float] = {}
            dimension_keys: dict[str, str] = {}

            net_edge = coerce_float(payload.get("net_edge_bps"))
            if net_edge is not None:
                distributions["net_edge_bps"].add(net_edge, accepted=is_accepted)
                observed_metrics["net_edge_bps"] = net_edge

            cost = coerce_float(payload.get("cost_bps"))
            if cost is not None:
                distributions["cost_bps"].add(cost, accepted=is_accepted)
                observed_metrics["cost_bps"] = cost

            model_probability = coerce_float(payload.get("model_success_probability"))
            if model_probability is not None:
                distributions["model_success_probability"].add(
                    model_probability, accepted=is_accepted
                )

            model_return = coerce_float(payload.get("model_expected_return_bps"))
            if model_return is not None:
                distributions["model_expected_return_bps"].add(
                    model_return, accepted=is_accepted
                )
            model_expected_value: float | None = None
            if model_return is not None and model_probability is not None:
                model_expected_value = model_return * model_probability
                distributions["model_expected_value_bps"].add(
                    model_expected_value, accepted=is_accepted
                )
                if cost is not None:
                    distributions["model_expected_value_minus_cost_bps"].add(
                        model_expected_value - cost,
                        accepted=is_accepted,
                    )
            elif model_return is not None and cost is not None:
                distributions["model_expected_value_minus_cost_bps"].add(
                    model_return - cost,
                    accepted=is_accepted,
                )

            latency_value = coerce_float(payload.get("latency_ms"))
            if latency_value is not None:
                distributions["latency_ms"].add(latency_value, accepted=is_accepted)
                observed_metrics["latency_ms"] = latency_value

            for flag in _iter_strings(payload.get("risk_flags")):
                risk_flag_counts[flag] += 1
                if is_accepted:
                    risk_flag_accepted[flag] += 1

            for failure in _iter_strings(payload.get("stress_failures")):
                stress_failure_counts[failure] += 1
                if is_accepted:
                    stress_failure_accepted[failure] += 1

            model_name = payload.get("model_name")
            if model_name is not None:
                model_key = str(model_name)
                model_usage[model_key] += 1
                model_totals[model_key] += 1
                if is_accepted:
                    model_accepted[model_key] += 1
                dimension_keys["model"] = model_key

            thresholds_map = _normalize_thresholds(payload.get("thresholds"))

            candidate = payload.get("candidate")
            candidate_probability: float | None = None
            candidate_return: float | None = None
            candidate_notional: float | None = None
            candidate_latency: float | None = None
            if isinstance(candidate, Mapping):
@@ -1879,142 +546,50 @@ class DecisionSummaryAggregator:
                    )
                    _update_breakdown_metric(
                        strategy_metric_totals,
                        dimension_keys.get("strategy"),
                        metric_name,
                        metric_value,
                        accepted=is_accepted,
                    )
                    _update_breakdown_metric(
                        symbol_metric_totals,
                        dimension_keys.get("symbol"),
                        metric_name,
                        metric_value,
                        accepted=is_accepted,
                    )

            generated_at = _extract_generated_at(payload)
            if generated_at is not None:
                history_generated_at.append(generated_at)

        summary["accepted"] = accepted
        summary["rejected"] = summary["total"] - accepted
        summary["acceptance_rate"] = accepted / summary["total"] if summary["total"] else 0.0
        summary["rejection_reasons"] = dict(
            sorted(rejection_reasons.items(), key=lambda item: item[1], reverse=True)
        )
        summary["unique_rejection_reasons"] = len(summary["rejection_reasons"])
        summary["current_acceptance_streak"] = current_acceptance_streak
        summary["current_rejection_streak"] = current_rejection_streak

        for prefix, accumulator in distributions.items():
            _inject_metric_summary(summary, prefix, accumulator, _METRIC_SPECS[prefix])

        summary["unique_models"] = len(model_usage)
        summary["models_with_accepts"] = sum(1 for count in model_accepted.values() if count)
        if model_usage:
            summary["model_usage"] = dict(
                sorted(model_usage.items(), key=lambda item: item[1], reverse=True)
            )

        summary["unique_actions"] = len(action_usage)
        summary["actions_with_accepts"] = sum(
            1 for count in action_accepted.values() if count
        )
        if action_usage:
            summary["action_usage"] = dict(
                sorted(action_usage.items(), key=lambda item: item[1], reverse=True)
            )

        summary["unique_strategies"] = len(strategy_usage)
@@ -2071,52 +646,100 @@ class DecisionSummaryAggregator:
            )
        if strategy_totals:
            summary["strategy_breakdown"] = _build_breakdown(
                strategy_totals, strategy_accepted, strategy_metric_totals
            )
        if symbol_totals:
            summary["symbol_breakdown"] = _build_breakdown(
                symbol_totals, symbol_accepted, symbol_metric_totals
            )

        if history_generated_at:
            summary["history_start_generated_at"] = history_generated_at[0]

        _inject_threshold_metrics(
            summary,
            threshold_margin_values,
            accepted_threshold_margin_values,
            rejected_threshold_margin_values,
            threshold_breach_counts,
            accepted_threshold_breach_counts,
            rejected_threshold_breach_counts,
        )

        _inject_latest_snapshot(summary, self._window[-1] if self._window else None)

        if self._window:
            summary["history"] = [
                _sanitize_payload(payload)
                for payload in self._window
                if isinstance(payload, Mapping)
            ]

        return summary

def _attach_breakdown_namespaces(model: DecisionEngineSummary) -> None:
    for key in (
        "model_breakdown",
        "action_breakdown",
        "strategy_breakdown",
        "symbol_breakdown",
    ):
        breakdown = getattr(model, key, None)
        if isinstance(breakdown, dict):
            setattr(
                model,
                key,
                {
                    name: _namespace_breakdown(entry)
                    for name, entry in breakdown.items()
                    if isinstance(entry, Mapping)
                },
            )


def _namespace_breakdown(entry: Mapping[str, object]) -> _AttrDict:
    data = dict(entry)
    metrics = data.get("metrics")
    if isinstance(metrics, Mapping):
        data["metrics"] = {
            metric_name: _AttrDict(metric_values)
            for metric_name, metric_values in metrics.items()
            if isinstance(metric_values, Mapping)
        }
    return _AttrDict(data)

def _sanitize_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _sanitize_value(value) for key, value in payload.items()}


def _sanitize_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _sanitize_payload(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_value(item) for item in value]
    return value


def _update_breakdown_metric(
    container: MutableMapping[str, dict[str, _SegmentMetricAccumulator]] | None,
    key: str | None,
    metric: str,
    value: float,
    *,
    accepted: bool,
) -> None:
    if container is None or key is None:
        return
    metric_map = container.setdefault(key, {})
    accumulator = metric_map.get(metric)
    if accumulator is None:
        accumulator = _SegmentMetricAccumulator()
        metric_map[metric] = accumulator
    accumulator.update(value, accepted=accepted)


def _inject_metric_summary(
    summary: MutableMapping[str, object],
    prefix: str,
    accumulator: _DistributionAccumulator,
    spec: _MetricSpec,
@@ -2248,51 +871,50 @@ def _inject_segment_stats(
        (0.9, "p90"),
    ),
    include_minmax: bool = True,
    include_std: bool = True,
) -> None:
    if not values:
        return
    count = len(values)
    total_value = sum(values)
    summary[f"{segment}_{prefix}_count"] = count
    summary[f"{segment}_avg_{prefix}"] = total_value / count
    if include_sum:
        summary[f"{segment}_sum_{prefix}"] = total_value
    sorted_values = sorted(values)
    if include_minmax:
        summary[f"{segment}_min_{prefix}"] = sorted_values[0]
        summary[f"{segment}_max_{prefix}"] = sorted_values[-1]
    for quantile, label in quantiles:
        summary[f"{segment}_{label}_{prefix}"] = _compute_quantile(
            sorted_values, quantile
        )
    if include_std:
        summary[f"{segment}_std_{prefix}"] = _compute_std(values)


def _inject_latest_snapshot(
    summary: MutableMapping[str, object],
    payload: Mapping[str, object] | None,
) -> None:
    if not payload or not isinstance(payload, Mapping):
        return

    is_accepted = bool(payload.get("accepted"))
    summary["latest_status"] = "accepted" if is_accepted else "rejected"

    latest_model = payload.get("model_name")
    if latest_model is not None:
        summary["latest_model"] = str(latest_model)

    latest_net_edge = coerce_float(payload.get("net_edge_bps"))
    if latest_net_edge is not None:
        summary["latest_net_edge_bps"] = latest_net_edge

    latest_cost = coerce_float(payload.get("cost_bps"))
    if latest_cost is not None:
        summary["latest_cost_bps"] = latest_cost

    latest_model_return = coerce_float(payload.get("model_expected_return_bps"))
    if latest_model_return is not None:
        summary["latest_model_expected_return_bps"] = latest_model_return
@@ -2324,360 +946,71 @@ def _inject_latest_snapshot(
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
    candidate_probability: float | None = None
    candidate_return: float | None = None
    candidate_notional: float | None = None
    candidate_latency: float | None = None
    if isinstance(candidate, Mapping):
        candidate_payload: MutableMapping[str, object] = {}
        for key in ("symbol", "action", "strategy"):
            value = candidate.get(key)
            if value is not None:
                candidate_payload[key] = value
        candidate_probability = coerce_float(candidate.get("expected_probability"))
        if candidate_probability is not None:
            candidate_payload["expected_probability"] = candidate_probability
            summary["latest_expected_probability"] = candidate_probability
        candidate_return = coerce_float(candidate.get("expected_return_bps"))
        if candidate_return is not None:
            candidate_payload["expected_return_bps"] = candidate_return
            summary["latest_expected_return_bps"] = candidate_return
        candidate_notional = coerce_float(candidate.get("notional"))
        if candidate_notional is not None:
            candidate_payload["notional"] = candidate_notional
            summary["latest_notional"] = candidate_notional
        candidate_latency = coerce_float(candidate.get("latency_ms"))
        if candidate_latency is not None:
            candidate_payload["latency_ms"] = candidate_latency
            if "latest_latency_ms" not in summary:
                summary["latest_latency_ms"] = candidate_latency
        if (
            candidate_probability is not None
            and candidate_return is not None
        ):
            candidate_expected_value = candidate_probability * candidate_return
            candidate_payload["expected_value_bps"] = candidate_expected_value
            summary["latest_expected_value_bps"] = candidate_expected_value
            if latest_cost is not None:
                summary["latest_expected_value_minus_cost_bps"] = (
                    candidate_expected_value - latest_cost
                )
        if candidate_payload:
            summary["latest_candidate"] = dict(candidate_payload)
        metadata = _extract_candidate_metadata(candidate)
        generated_at = None
        if metadata:
            generated_at = metadata.get("generated_at") or metadata.get("timestamp")
        if generated_at is None:
            generated_at = candidate.get("generated_at")
        if generated_at is not None:
            summary["latest_generated_at"] = str(generated_at)

    if "latest_generated_at" not in summary:
        generated_at = _extract_generated_at(payload)
        if generated_at is not None:
@@ -2836,307 +1169,25 @@ def _inject_distribution_metrics(
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

