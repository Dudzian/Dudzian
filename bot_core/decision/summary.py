"""Agregacja ewaluacji Decision Engine."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

from .utils import coerce_float

__all__ = ["summarize_evaluation_payloads"]


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
        segment_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
    "notional": _MetricSpec(
        overall_quantiles=((0.5, "median"),),
        segment_quantiles=((0.5, "median"), (0.9, "p90")),
    ),
    "latency_ms": _MetricSpec(
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
    evaluations: Sequence[Mapping[str, object]],
    *,
    history_limit: int | None = None,
) -> MutableMapping[str, object]:
    """Buduje zagregowane podsumowanie Decision Engine na podstawie ewaluacji."""

    aggregator = _EvaluationAggregator(evaluations, history_limit=history_limit)
    return aggregator.build()


class _EvaluationAggregator:
    def __init__(
        self,
        evaluations: Sequence[Mapping[str, object]],
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

    def build(self) -> MutableMapping[str, object]:
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
                self._full_accepted / self._full_total if self._full_total else 0.0
            ),
            "current_acceptance_streak": 0,
            "current_rejection_streak": 0,
            "longest_acceptance_streak": 0,
            "longest_rejection_streak": 0,
        }

        if self._history_window == 0:
            return summary

        distributions = {
            prefix: _DistributionAccumulator() for prefix in _METRIC_SPECS
        }

        rejection_reasons: Counter[str] = Counter()
        risk_flag_counts: Counter[str] = Counter()
        risk_flag_accepted: Counter[str] = Counter()
        stress_failure_counts: Counter[str] = Counter()
        stress_failure_accepted: Counter[str] = Counter()
        model_usage: Counter[str] = Counter()
        action_usage: Counter[str] = Counter()
        strategy_usage: Counter[str] = Counter()
        symbol_usage: Counter[str] = Counter()
        model_totals: Counter[str] = Counter()
        model_accepted: Counter[str] = Counter()
        action_totals: Counter[str] = Counter()
        action_accepted: Counter[str] = Counter()
        strategy_totals: Counter[str] = Counter()
        strategy_accepted: Counter[str] = Counter()
        symbol_totals: Counter[str] = Counter()
        symbol_accepted: Counter[str] = Counter()
        model_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
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
                action = candidate.get("action")
                if action is not None:
                    action_key = str(action)
                    action_usage[action_key] += 1
                    action_totals[action_key] += 1
                    if is_accepted:
                        action_accepted[action_key] += 1
                    dimension_keys.setdefault("action", action_key)
                strategy = candidate.get("strategy")
                if strategy is not None:
                    strategy_key = str(strategy)
                    strategy_usage[strategy_key] += 1
                    strategy_totals[strategy_key] += 1
                    if is_accepted:
                        strategy_accepted[strategy_key] += 1
                    dimension_keys.setdefault("strategy", strategy_key)
                symbol = candidate.get("symbol")
                if symbol is not None:
                    symbol_key = str(symbol)
                    symbol_usage[symbol_key] += 1
                    symbol_totals[symbol_key] += 1
                    if is_accepted:
                        symbol_accepted[symbol_key] += 1
                    dimension_keys.setdefault("symbol", symbol_key)

                candidate_probability = coerce_float(candidate.get("expected_probability"))
                if candidate_probability is not None:
                    distributions["expected_probability"].add(
                        candidate_probability, accepted=is_accepted
                    )
                    observed_metrics["expected_probability"] = candidate_probability

                candidate_return = coerce_float(candidate.get("expected_return_bps"))
                if candidate_return is not None:
                    distributions["expected_return_bps"].add(
                        candidate_return, accepted=is_accepted
                    )

                if (
                    candidate_probability is not None
                    and candidate_return is not None
                ):
                    candidate_expected_value = candidate_probability * candidate_return
                    distributions["expected_value_bps"].add(
                        candidate_expected_value, accepted=is_accepted
                    )
                    observed_metrics["expected_value_bps"] = candidate_expected_value
                    if cost is not None:
                        candidate_minus_cost = candidate_expected_value - cost
                        distributions["expected_value_minus_cost_bps"].add(
                            candidate_minus_cost,
                            accepted=is_accepted,
                        )
                        observed_metrics[
                            "expected_value_minus_cost_bps"
                        ] = candidate_minus_cost

                candidate_notional = coerce_float(candidate.get("notional"))
                if candidate_notional is not None:
                    distributions["notional"].add(
                        candidate_notional, accepted=is_accepted
                    )
                    observed_metrics["notional"] = candidate_notional

                candidate_latency = coerce_float(candidate.get("latency_ms"))
                if (
                    candidate_latency is not None
                    and "latency_ms" not in observed_metrics
                ):
                    distributions["latency_ms"].add(
                        candidate_latency, accepted=is_accepted
                    )
                    observed_metrics["latency_ms"] = candidate_latency

            if thresholds_map:
                for threshold_key, raw_value in thresholds_map.items():
                    definition = _THRESHOLD_DEFINITIONS.get(threshold_key)
                    if not definition or raw_value is None:
                        continue
                    metric_name, mode, margin_prefix = definition
                    observed_value = observed_metrics.get(metric_name)
                    if observed_value is None:
                        continue
                    if mode == "min":
                        margin = observed_value - raw_value
                    else:
                        margin = raw_value - observed_value
                    threshold_margin_values.setdefault(margin_prefix, []).append(margin)
                    if is_accepted:
                        accepted_threshold_margin_values.setdefault(margin_prefix, []).append(
                            margin
                        )
                    else:
                        rejected_threshold_margin_values.setdefault(margin_prefix, []).append(
                            margin
                        )
                    base_key = (
                        margin_prefix.rsplit("_margin", 1)[0]
                        if margin_prefix.endswith("_margin")
                        else margin_prefix
                    )
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

            if dimension_keys:
                for metric_name in (
                    "net_edge_bps",
                    "cost_bps",
                    "expected_value_bps",
                    "expected_value_minus_cost_bps",
                    "notional",
                    "latency_ms",
                ):
                    metric_value = observed_metrics.get(metric_name)
                    if metric_value is None:
                        continue
                    _update_breakdown_metric(
                        model_metric_totals,
                        dimension_keys.get("model"),
                        metric_name,
                        metric_value,
                        accepted=is_accepted,
                    )
                    _update_breakdown_metric(
                        action_metric_totals,
                        dimension_keys.get("action"),
                        metric_name,
                        metric_value,
                        accepted=is_accepted,
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
        summary["strategies_with_accepts"] = sum(
            1 for count in strategy_accepted.values() if count
        )
        if strategy_usage:
            summary["strategy_usage"] = dict(
                sorted(strategy_usage.items(), key=lambda item: item[1], reverse=True)
            )

        summary["unique_symbols"] = len(symbol_usage)
        summary["symbols_with_accepts"] = sum(
            1 for count in symbol_accepted.values() if count
        )
        if symbol_usage:
            summary["symbol_usage"] = dict(
                sorted(symbol_usage.items(), key=lambda item: item[1], reverse=True)
            )

        if risk_flag_counts:
            summary["risk_flag_counts"] = dict(
                sorted(risk_flag_counts.items(), key=lambda item: item[1], reverse=True)
            )
            summary["risk_flag_breakdown"] = _build_breakdown(
                risk_flag_counts, risk_flag_accepted
            )
        summary["unique_risk_flags"] = len(risk_flag_counts)
        summary["risk_flags_with_accepts"] = sum(
            1 for count in risk_flag_accepted.values() if count
        )

        if stress_failure_counts:
            summary["stress_failure_counts"] = dict(
                sorted(
                    stress_failure_counts.items(), key=lambda item: item[1], reverse=True
                )
            )
            summary["stress_failure_breakdown"] = _build_breakdown(
                stress_failure_counts, stress_failure_accepted
            )
        summary["unique_stress_failures"] = len(stress_failure_counts)
        summary["stress_failures_with_accepts"] = sum(
            1 for count in stress_failure_accepted.values() if count
        )

        if model_totals:
            summary["model_breakdown"] = _build_breakdown(
                model_totals, model_accepted, model_metric_totals
            )
        if action_totals:
            summary["action_breakdown"] = _build_breakdown(
                action_totals, action_accepted, action_metric_totals
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

        return summary



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
) -> None:
    values = accumulator.total
    if values:
        total_sum = sum(values)
        summary[f"avg_{prefix}"] = total_sum / len(values)
        if spec.overall_include_sum:
            summary[f"sum_{prefix}"] = total_sum
        _inject_distribution_metrics(
            summary,
            values,
            prefix=prefix,
            quantiles=spec.overall_quantiles,
            include_minmax=spec.overall_include_minmax,
        )
        if spec.overall_include_std:
            _inject_std_metric(summary, values, prefix=prefix)
    if accumulator.accepted:
        _inject_segment_stats(
            summary,
            accumulator.accepted,
            prefix=prefix,
            segment="accepted",
            include_sum=spec.segment_include_sum,
            include_minmax=spec.segment_include_minmax,
            quantiles=spec.segment_quantiles,
            include_std=spec.segment_include_std,
        )
    if accumulator.rejected:
        _inject_segment_stats(
            summary,
            accumulator.rejected,
            prefix=prefix,
            segment="rejected",
            include_sum=spec.segment_include_sum,
            include_minmax=spec.segment_include_minmax,
            quantiles=spec.segment_quantiles,
            include_std=spec.segment_include_std,
        )


def _inject_threshold_metrics(
    summary: MutableMapping[str, object],
    totals: Mapping[str, Sequence[float]],
    accepted_values: Mapping[str, Sequence[float]],
    rejected_values: Mapping[str, Sequence[float]],
    breach_counts: Mapping[str, int],
    accepted_breaches: Mapping[str, int],
    rejected_breaches: Mapping[str, int],
) -> None:
    for prefix, values in totals.items():
        if not values:
            continue
        count = len(values)
        total_value = sum(values)
        summary[f"{prefix}_count"] = count
        summary[f"avg_{prefix}"] = total_value / count
        summary[f"sum_{prefix}"] = total_value
        _inject_distribution_metrics(
            summary,
            values,
            prefix=prefix,
            quantiles=_THRESHOLD_DISTRIBUTION_QUANTILES,
        )
        _inject_std_metric(summary, values, prefix=prefix)

        base_key = (
            prefix.rsplit("_margin", 1)[0] if prefix.endswith("_margin") else prefix
        )
        breaches = breach_counts.get(base_key, 0)
        summary[f"{base_key}_breaches"] = breaches
        summary[f"{base_key}_breach_rate"] = breaches / count if count else 0.0

        accepted_series = accepted_values.get(prefix, [])
        accepted_count = len(accepted_series)
        if accepted_series:
            _inject_segment_stats(
                summary,
                accepted_series,
                prefix=prefix,
                segment="accepted",
            )
            accepted_total = sum(accepted_series)
            summary[f"accepted_sum_{prefix}"] = accepted_total
            summary[f"accepted_avg_{prefix}"] = accepted_total / accepted_count
        else:
            accepted_total = 0.0
        summary[f"accepted_{prefix}_count"] = accepted_count
        accepted_breach = accepted_breaches.get(base_key, 0)
        summary[f"accepted_{base_key}_breaches"] = accepted_breach
        summary[f"accepted_{base_key}_breach_rate"] = (
            accepted_breach / accepted_count if accepted_count else 0.0
        )

        rejected_series = rejected_values.get(prefix, [])
        rejected_count = len(rejected_series)
        if rejected_series:
            _inject_segment_stats(
                summary,
                rejected_series,
                prefix=prefix,
                segment="rejected",
            )
            rejected_total = sum(rejected_series)
            summary[f"rejected_sum_{prefix}"] = rejected_total
            summary[f"rejected_avg_{prefix}"] = rejected_total / rejected_count
        else:
            rejected_total = 0.0
        summary[f"rejected_{prefix}_count"] = rejected_count
        rejected_breach = rejected_breaches.get(base_key, 0)
        summary[f"rejected_{base_key}_breaches"] = rejected_breach
        summary[f"rejected_{base_key}_breach_rate"] = (
            rejected_breach / rejected_count if rejected_count else 0.0
        )


def _inject_segment_stats(
    summary: MutableMapping[str, object],
    values: Sequence[float],
    *,
    prefix: str,
    segment: str,
    include_sum: bool = True,
    quantiles: Sequence[tuple[float, str]] = (
        (0.1, "p10"),
        (0.5, "median"),
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

    latest_model_probability = coerce_float(payload.get("model_success_probability"))
    if latest_model_probability is not None:
        summary["latest_model_success_probability"] = latest_model_probability

    latest_model_expected_value: float | None = None
    if (
        latest_model_return is not None
        and latest_model_probability is not None
    ):
        latest_model_expected_value = latest_model_return * latest_model_probability
        summary["latest_model_expected_value_bps"] = latest_model_expected_value
        if latest_cost is not None:
            summary["latest_model_expected_value_minus_cost_bps"] = (
                latest_model_expected_value - latest_cost
            )

    latest_latency = coerce_float(payload.get("latency_ms"))
    if latest_latency is not None:
        summary["latest_latency_ms"] = latest_latency

    thresholds_map = _normalize_thresholds(payload.get("thresholds"))
    if thresholds_map:
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
            summary["latest_generated_at"] = generated_at

    if thresholds_map:
        probability_margin_value: float | None = candidate_probability
        notional_value: float | None = candidate_notional
        latency_value = summary.get("latest_latency_ms")
        if isinstance(latency_value, (int, float)):
            latency_value = float(latency_value)
        margin_inputs = {
            "expected_probability": probability_margin_value,
            "net_edge_bps": latest_net_edge,
            "cost_bps": latest_cost,
            "latency_ms": latency_value,
            "notional": notional_value,
        }
        for threshold_key, (metric_name, mode, margin_prefix) in _THRESHOLD_DEFINITIONS.items():
            threshold_value = thresholds_map.get(threshold_key)
            if threshold_value is None:
                continue
            observed_value = margin_inputs.get(metric_name)
            if observed_value is None:
                continue
            if mode == "min":
                margin = observed_value - threshold_value
            else:
                margin = threshold_value - observed_value
            summary[f"latest_{margin_prefix}"] = margin


def _build_breakdown(
    totals: Counter[str],
    accepted_counts: Counter[str],
    metrics: Mapping[str, Mapping[str, _SegmentMetricAccumulator]] | None = None,
) -> Mapping[str, Mapping[str, object]]:
    breakdown: MutableMapping[str, Mapping[str, object]] = {}
    for key, total_count in totals.most_common():
        accepted_count = accepted_counts.get(key, 0)
        rejected_count = max(total_count - accepted_count, 0)
        entry: MutableMapping[str, object] = {
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


def _normalize_thresholds(snapshot: object) -> Mapping[str, float | None] | None:
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
    except (TypeError, ValueError):
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


