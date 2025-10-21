"""Narzędzia do agregacji i raportowania jakości decyzji AI."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, MutableMapping, Sequence

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
            self.accepted_sum / self.accepted_count
            if self.accepted_count
            else 0.0
        )
        rejected_avg = (
            self.rejected_sum / self.rejected_count
            if self.rejected_count
            else 0.0
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
) -> Mapping[str, object]:
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
    total = len(evaluations)
    limit = history_limit if history_limit is not None else total
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
        summary["full_accepted"] = full_accepted
        summary["full_rejected"] = full_total - full_accepted
        summary["full_acceptance_rate"] = (
            full_accepted / full_total if full_total else 0.0
        )
        "history_limit": limit,
        "history_window": total,
        "rejection_reasons": {},
    }
    if total == 0:
        return summary

    accepted = 0
    rejection_reasons: Counter[str] = Counter()
    net_edges: list[float] = []
    accepted_net_edges: list[float] = []
    rejected_net_edges: list[float] = []
    costs: list[float] = []
    accepted_costs: list[float] = []
    rejected_costs: list[float] = []
    probabilities: list[float] = []
    accepted_probabilities: list[float] = []
    rejected_probabilities: list[float] = []
    expected_returns: list[float] = []
    accepted_expected_returns: list[float] = []
    rejected_expected_returns: list[float] = []
    expected_values: list[float] = []
    accepted_expected_values: list[float] = []
    rejected_expected_values: list[float] = []
    expected_values_minus_costs: list[float] = []
    accepted_expected_values_minus_costs: list[float] = []
    rejected_expected_values_minus_costs: list[float] = []
    notionals: list[float] = []
    accepted_notionals: list[float] = []
    rejected_notionals: list[float] = []
    model_probabilities: list[float] = []
    accepted_model_probabilities: list[float] = []
    rejected_model_probabilities: list[float] = []
    model_returns: list[float] = []
    accepted_model_returns: list[float] = []
    rejected_model_returns: list[float] = []
    model_expected_values: list[float] = []
    accepted_model_expected_values: list[float] = []
    rejected_model_expected_values: list[float] = []
    model_expected_values_minus_costs: list[float] = []
    accepted_model_expected_values_minus_costs: list[float] = []
    rejected_model_expected_values_minus_costs: list[float] = []
    latencies: list[float] = []
    accepted_latencies: list[float] = []
    rejected_latencies: list[float] = []
    risk_flag_counts: Counter[str] = Counter()
    stress_failure_counts: Counter[str] = Counter()
    risk_flag_totals: Counter[str] = Counter()
    risk_flag_accepted: Counter[str] = Counter()
    stress_failure_totals: Counter[str] = Counter()
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
    history_generated_at: list[str] = []
    threshold_margin_values: dict[str, list[float]] = {}
    accepted_threshold_margin_values: dict[str, list[float]] = {}
    rejected_threshold_margin_values: dict[str, list[float]] = {}
    threshold_breach_counts: dict[str, int] = {}
    accepted_threshold_breach_counts: dict[str, int] = {}
    rejected_threshold_breach_counts: dict[str, int] = {}

    model_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
    action_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
    strategy_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}
    symbol_metric_totals: dict[str, dict[str, _SegmentMetricAccumulator]] = {}

    breakdown_metric_keys = (
        "net_edge_bps",
        "cost_bps",
        "expected_value_bps",
        "expected_value_minus_cost_bps",
        "notional",
        "latency_ms",
    )

    def _update_breakdown_metric(
        container: dict[str, dict[str, _SegmentMetricAccumulator]],
        key: str,
        metric: str,
        value: float,
        *,
        accepted: bool,
    ) -> None:
        metric_map = container.setdefault(key, {})
        accumulator = metric_map.get(metric)
        if accumulator is None:
            accumulator = _SegmentMetricAccumulator()
            metric_map[metric] = accumulator
        accumulator.update(value, accepted=accepted)

    breakdown_containers = (
        ("model", model_metric_totals),
        ("action", action_metric_totals),
        ("strategy", strategy_metric_totals),
        ("symbol", symbol_metric_totals),
    )

    current_acceptance_streak = 0
    current_rejection_streak = 0

    for payload in windowed:
        if not isinstance(payload, Mapping):
            continue
        thresholds_map: Mapping[str, float | None] | None = None
        thresholds_payload = payload.get("thresholds")
        if isinstance(thresholds_payload, Mapping):
            thresholds_map = _normalize_thresholds(thresholds_payload)
        observed_metrics: dict[str, float] = {}
        dimension_keys: dict[str, str] = {}
        is_accepted = bool(payload.get("accepted"))
        if is_accepted:
            accepted += 1
            current_acceptance_streak += 1
            current_rejection_streak = 0
        else:
            for reason in _iter_strings(payload.get("reasons")):
                rejection_reasons[str(reason)] += 1
            current_rejection_streak += 1
            current_acceptance_streak = 0

        summary["longest_acceptance_streak"] = max(
            summary["longest_acceptance_streak"], current_acceptance_streak
        )
        summary["longest_rejection_streak"] = max(
            summary["longest_rejection_streak"], current_rejection_streak
        )
    costs: list[float] = []
    probabilities: list[float] = []
    expected_returns: list[float] = []
    notionals: list[float] = []
    model_probabilities: list[float] = []
    model_returns: list[float] = []
    latencies: list[float] = []

    for payload in evaluations:
        if not isinstance(payload, Mapping):
            continue
        if bool(payload.get("accepted")):
            accepted += 1
        else:
            for reason in payload.get("reasons", ()):
                rejection_reasons[str(reason)] += 1

        net_edge = _coerce_float(payload.get("net_edge_bps"))
        if net_edge is not None:
            net_edges.append(net_edge)
            observed_metrics["net_edge_bps"] = net_edge
            if is_accepted:
                accepted_net_edges.append(net_edge)
            else:
                rejected_net_edges.append(net_edge)

        cost = _coerce_float(payload.get("cost_bps"))
        if cost is not None:
            costs.append(cost)
            observed_metrics["cost_bps"] = cost
            if is_accepted:
                accepted_costs.append(cost)
            else:
                rejected_costs.append(cost)

        model_prob = _coerce_float(payload.get("model_success_probability"))
        if model_prob is not None:
            model_probabilities.append(model_prob)
            if is_accepted:
                accepted_model_probabilities.append(model_prob)
            else:
                rejected_model_probabilities.append(model_prob)

        model_return = _coerce_float(payload.get("model_expected_return_bps"))
        if model_return is not None:
            model_returns.append(model_return)
            if is_accepted:
                accepted_model_returns.append(model_return)
            else:
                rejected_model_returns.append(model_return)
        if model_return is not None and model_prob is not None:
            model_expected_value = model_return * model_prob
            model_expected_values.append(model_expected_value)
            if is_accepted:
                accepted_model_expected_values.append(model_expected_value)
            else:
                rejected_model_expected_values.append(model_expected_value)
            if cost is not None:
                model_expected_values_minus_costs.append(
                    model_expected_value - cost
                )
                if is_accepted:
                    accepted_model_expected_values_minus_costs.append(
                        model_expected_value - cost
                    )
                else:
                    rejected_model_expected_values_minus_costs.append(
                        model_expected_value - cost
                    )

        latency = _coerce_float(payload.get("latency_ms"))
        latency_recorded = False
        if latency is not None:
            latencies.append(latency)
            latency_recorded = True
            observed_metrics.setdefault("latency_ms", latency)
            if is_accepted:
                accepted_latencies.append(latency)
            else:
                rejected_latencies.append(latency)

        for flag in _iter_strings(payload.get("risk_flags")):
            key = str(flag)
            risk_flag_counts[key] += 1
            risk_flag_totals[key] += 1
            if is_accepted:
                risk_flag_accepted[key] += 1

        for failure in _iter_strings(payload.get("stress_failures")):
            key = str(failure)
            stress_failure_counts[key] += 1
            stress_failure_totals[key] += 1
            if is_accepted:
                stress_failure_accepted[key] += 1

        model_name = payload.get("model_name")
        if model_name is not None:
            model_key = str(model_name)
            model_usage[model_key] += 1
            model_totals[model_key] += 1
            if is_accepted:
                model_accepted[model_key] += 1
            dimension_keys["model"] = model_key

        latency = _coerce_float(payload.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)

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
                    expected_value_minus_cost = candidate_expected_value - cost
                    expected_values_minus_costs.append(
                        expected_value_minus_cost
                    )
                    observed_metrics[
                        "expected_value_minus_cost_bps"
                    ] = expected_value_minus_cost
                    if is_accepted:
                        accepted_expected_values_minus_costs.append(
                            expected_value_minus_cost
                        )
                    else:
                        rejected_expected_values_minus_costs.append(
                            expected_value_minus_cost
                        )
            notional = _coerce_float(candidate.get("notional"))
            if notional is not None:
                notionals.append(notional)
                observed_metrics["notional"] = notional
                if is_accepted:
                    accepted_notionals.append(notional)
                else:
                    rejected_notionals.append(notional)

            candidate_latency = _coerce_float(candidate.get("latency_ms"))
            if candidate_latency is not None and not latency_recorded:
                latencies.append(candidate_latency)
                observed_metrics.setdefault("latency_ms", candidate_latency)
                if is_accepted:
                    accepted_latencies.append(candidate_latency)
                else:
                    rejected_latencies.append(candidate_latency)

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

        if dimension_keys:
            for metric_name in breakdown_metric_keys:
                metric_value = observed_metrics.get(metric_name)
                if metric_value is None:
                    continue
                for dimension, container in breakdown_containers:
                    key = dimension_keys.get(dimension)
                    if key is None:
                        continue
                    _update_breakdown_metric(
                        container,
                        key,
                        metric_name,
                        metric_value,
                        accepted=is_accepted,
                    )

        generated_at = _extract_generated_at(payload)
        if generated_at is not None:
            history_generated_at.append(generated_at)

        if thresholds_map:
            for threshold_key, raw_value in thresholds_map.items():
                definition = _THRESHOLD_DEFINITIONS.get(threshold_key)
                if not definition:
                    continue
                metric_name, mode, margin_prefix = definition
                threshold_value = _coerce_float(raw_value)
                if threshold_value is None:
                    continue
                observed_value = observed_metrics.get(metric_name)
                if observed_value is None:
                    continue
                if mode == "min":
                    margin = observed_value - threshold_value
                else:
                    margin = threshold_value - observed_value
                threshold_margin_values.setdefault(margin_prefix, []).append(margin)
                if is_accepted:
                    accepted_threshold_margin_values.setdefault(margin_prefix, []).append(margin)
                else:
                    rejected_threshold_margin_values.setdefault(margin_prefix, []).append(margin)
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
            expected_return = _coerce_float(candidate.get("expected_return_bps"))
            if expected_return is not None:
                expected_returns.append(expected_return)
            notional = _coerce_float(candidate.get("notional"))
            if notional is not None:
                notionals.append(notional)

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
        summary["sum_model_expected_value_bps"] = total_model_expected_value
    if model_expected_values_minus_costs:
        total_model_expected_value_minus_cost = sum(
            model_expected_values_minus_costs
        )
        summary["avg_model_expected_value_minus_cost_bps"] = (
            total_model_expected_value_minus_cost
            / len(model_expected_values_minus_costs)
        )
        summary["sum_model_expected_value_minus_cost_bps"] = (
            total_model_expected_value_minus_cost
        )
    if latencies:
        total_latency = sum(latencies)
        summary["avg_latency_ms"] = total_latency / len(latencies)
        summary["sum_latency_ms"] = total_latency

    for prefix, values in threshold_margin_values.items():
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
            quantiles=((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
        )
        _inject_std_metric(summary, values, prefix=prefix)
        base_key = (
            prefix.rsplit("_margin", 1)[0] if prefix.endswith("_margin") else prefix
        )
        breaches = threshold_breach_counts.get(base_key, 0)
        summary[f"{base_key}_breaches"] = breaches
        summary[f"{base_key}_breach_rate"] = breaches / count if count else 0.0
        accepted_values = accepted_threshold_margin_values.get(prefix, [])
        if accepted_values:
            accepted_total = sum(accepted_values)
            accepted_count = len(accepted_values)
            summary[f"accepted_avg_{prefix}"] = accepted_total / accepted_count
            summary[f"accepted_sum_{prefix}"] = accepted_total
            summary[f"accepted_{prefix}_count"] = accepted_count
            _inject_segmented_threshold_metrics(
                summary,
                accepted_values,
                prefix=prefix,
                segment="accepted",
            )
        else:
            accepted_count = 0
        rejected_values = rejected_threshold_margin_values.get(prefix, [])
        if rejected_values:
            rejected_total = sum(rejected_values)
            rejected_count = len(rejected_values)
            summary[f"rejected_avg_{prefix}"] = rejected_total / rejected_count
            summary[f"rejected_sum_{prefix}"] = rejected_total
            summary[f"rejected_{prefix}_count"] = rejected_count
            _inject_segmented_threshold_metrics(
                summary,
                rejected_values,
                prefix=prefix,
                segment="rejected",
            )
        else:
            rejected_count = 0
        accepted_breaches = accepted_threshold_breach_counts.get(base_key, 0)
        rejected_breaches = rejected_threshold_breach_counts.get(base_key, 0)
        summary[f"accepted_{base_key}_breaches"] = accepted_breaches
        summary[f"rejected_{base_key}_breaches"] = rejected_breaches
        summary[f"accepted_{base_key}_breach_rate"] = (
            accepted_breaches / accepted_count if accepted_count else 0.0
        )
        summary[f"rejected_{base_key}_breach_rate"] = (
            rejected_breaches / rejected_count if rejected_count else 0.0
        )

    _inject_distribution_metrics(
        summary,
        net_edges,
        prefix="net_edge_bps",
        quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
    )
    _inject_std_metric(summary, net_edges, prefix="net_edge_bps")
    _inject_distribution_metrics(
        summary,
        costs,
        prefix="cost_bps",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_std_metric(summary, costs, prefix="cost_bps")
    _inject_distribution_metrics(
        summary,
        latencies,
        prefix="latency_ms",
        quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
    )
    _inject_std_metric(summary, latencies, prefix="latency_ms")
    _inject_distribution_metrics(
        summary,
        probabilities,
        prefix="expected_probability",
        quantiles=((0.5, "median"),),
        include_minmax=False,
    )
    _inject_std_metric(summary, probabilities, prefix="expected_probability")
    _inject_distribution_metrics(
        summary,
        expected_returns,
        prefix="expected_return_bps",
        quantiles=((0.5, "median"),),
    )
    _inject_std_metric(summary, expected_returns, prefix="expected_return_bps")
    _inject_distribution_metrics(
        summary,
        expected_values,
        prefix="expected_value_bps",
        quantiles=((0.5, "median"),),
    )
    _inject_std_metric(summary, expected_values, prefix="expected_value_bps")
    _inject_distribution_metrics(
        summary,
        expected_values_minus_costs,
        prefix="expected_value_minus_cost_bps",
        quantiles=((0.5, "median"),),
    )
    _inject_std_metric(
        summary,
        expected_values_minus_costs,
        prefix="expected_value_minus_cost_bps",
    )
    _inject_distribution_metrics(
        summary,
        notionals,
        prefix="notional",
        quantiles=((0.5, "median"),),
    )
    _inject_std_metric(summary, notionals, prefix="notional")
    _inject_distribution_metrics(
        summary,
        model_probabilities,
        prefix="model_success_probability",
        quantiles=((0.5, "median"),),
        include_minmax=False,
    )
    _inject_std_metric(
        summary,
        model_probabilities,
        prefix="model_success_probability",
    )
    _inject_distribution_metrics(
        summary,
        model_returns,
        prefix="model_expected_return_bps",
        quantiles=((0.5, "median"),),
    )
    _inject_std_metric(
        summary,
        model_returns,
        prefix="model_expected_return_bps",
    )
    _inject_distribution_metrics(
        summary,
        model_expected_values,
        prefix="model_expected_value_bps",
        quantiles=((0.5, "median"),),
    )
    _inject_std_metric(
        summary,
        model_expected_values,
        prefix="model_expected_value_bps",
    )
    _inject_distribution_metrics(
        summary,
        model_expected_values_minus_costs,
        prefix="model_expected_value_minus_cost_bps",
        quantiles=((0.5, "median"),),
    )
    _inject_std_metric(
        summary,
        model_expected_values_minus_costs,
        prefix="model_expected_value_minus_cost_bps",
    )

    def _inject_segment_stats(
        values: Sequence[float],
        *,
        prefix: str,
        segment: str,
        include_sum: bool = True,
        quantiles: Sequence[tuple[float, str]] = ((0.1, "p10"), (0.5, "median"), (0.9, "p90")),
        include_minmax: bool = True,
        include_std: bool = True,
    ) -> None:
        if not values:
            return
        total_value = sum(values)
        count = len(values)
        summary[f"{segment}_avg_{prefix}"] = total_value / count
        summary[f"{segment}_{prefix}_count"] = count
        if include_sum:
            summary[f"{segment}_sum_{prefix}"] = total_value
        sorted_values = sorted(values)
        if include_minmax:
            summary[f"{segment}_min_{prefix}"] = sorted_values[0]
            summary[f"{segment}_max_{prefix}"] = sorted_values[-1]
        if quantiles:
            for quantile, label in quantiles:
                summary[f"{segment}_{label}_{prefix}"] = _compute_quantile(
                    sorted_values, quantile
                )
        if include_std:
            summary[f"{segment}_std_{prefix}"] = _compute_std(values)

    _inject_segment_stats(accepted_net_edges, prefix="net_edge_bps", segment="accepted")
    _inject_segment_stats(rejected_net_edges, prefix="net_edge_bps", segment="rejected")
    _inject_segment_stats(accepted_costs, prefix="cost_bps", segment="accepted")
    _inject_segment_stats(rejected_costs, prefix="cost_bps", segment="rejected")
    _inject_segment_stats(
        accepted_probabilities,
        prefix="expected_probability",
        segment="accepted",
        include_sum=False,
        include_minmax=False,
        quantiles=((0.5, "median"),),
    )
    _inject_segment_stats(
        rejected_probabilities,
        prefix="expected_probability",
        segment="rejected",
        include_sum=False,
        include_minmax=False,
        quantiles=((0.5, "median"),),
    )
    _inject_segment_stats(
        accepted_expected_returns,
        prefix="expected_return_bps",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        rejected_expected_returns,
        prefix="expected_return_bps",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        accepted_expected_values,
        prefix="expected_value_bps",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        rejected_expected_values,
        prefix="expected_value_bps",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        accepted_expected_values_minus_costs,
        prefix="expected_value_minus_cost_bps",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        rejected_expected_values_minus_costs,
        prefix="expected_value_minus_cost_bps",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        accepted_notionals,
        prefix="notional",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        rejected_notionals,
        prefix="notional",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        accepted_latencies,
        prefix="latency_ms",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
    )
    _inject_segment_stats(
        rejected_latencies,
        prefix="latency_ms",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90"), (0.95, "p95")),
    )
    _inject_segment_stats(
        accepted_model_probabilities,
        prefix="model_success_probability",
        segment="accepted",
        include_sum=False,
        include_minmax=False,
        quantiles=((0.5, "median"),),
    )
    _inject_segment_stats(
        rejected_model_probabilities,
        prefix="model_success_probability",
        segment="rejected",
        include_sum=False,
        include_minmax=False,
        quantiles=((0.5, "median"),),
    )
    _inject_segment_stats(
        accepted_model_returns,
        prefix="model_expected_return_bps",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        rejected_model_returns,
        prefix="model_expected_return_bps",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        accepted_model_expected_values,
        prefix="model_expected_value_bps",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        rejected_model_expected_values,
        prefix="model_expected_value_bps",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        accepted_model_expected_values_minus_costs,
        prefix="model_expected_value_minus_cost_bps",
        segment="accepted",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )
    _inject_segment_stats(
        rejected_model_expected_values_minus_costs,
        prefix="model_expected_value_minus_cost_bps",
        segment="rejected",
        quantiles=((0.5, "median"), (0.9, "p90")),
    )

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

    def _build_breakdown(
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

    if risk_flag_totals:
        summary["risk_flag_breakdown"] = _build_breakdown(
            risk_flag_totals, risk_flag_accepted
        )
    summary["risk_flags_with_accepts"] = sum(
        1 for count in risk_flag_accepted.values() if count
    )
    summary["unique_risk_flags"] = len(risk_flag_counts)
    if risk_flag_counts:
        summary["risk_flag_counts"] = dict(
            sorted(risk_flag_counts.items(), key=lambda item: item[1], reverse=True)
        )

    if stress_failure_totals:
        summary["stress_failure_breakdown"] = _build_breakdown(
            stress_failure_totals, stress_failure_accepted
        )
    summary["stress_failures_with_accepts"] = sum(
        1 for count in stress_failure_accepted.values() if count
    )
    summary["unique_stress_failures"] = len(stress_failure_counts)
    if stress_failure_counts:
        summary["stress_failure_counts"] = dict(
            sorted(
                stress_failure_counts.items(), key=lambda item: item[1], reverse=True
            )
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

    latest_payload = windowed[-1]
        summary["avg_model_expected_return_bps"] = sum(model_returns) / len(model_returns)
    if latencies:
        summary["avg_latency_ms"] = sum(latencies) / len(latencies)

    latest_payload = evaluations[-1]
    if isinstance(latest_payload, Mapping):
        latest_model = latest_payload.get("model_name")
        if latest_model:
            summary["latest_model"] = str(latest_model)

        summary["latest_status"] = (
            "accepted" if bool(latest_payload.get("accepted")) else "rejected"
        )

        latest_net_edge = _coerce_float(latest_payload.get("net_edge_bps"))
        if latest_net_edge is not None:
            summary["latest_net_edge_bps"] = latest_net_edge

        latest_cost = _coerce_float(latest_payload.get("cost_bps"))
        latest_model_return = _coerce_float(
            latest_payload.get("model_expected_return_bps")
        )
        latest_model_probability = _coerce_float(
            latest_payload.get("model_success_probability")
        )
        if latest_cost is not None:
            summary["latest_cost_bps"] = latest_cost
        if latest_model_return is not None:
            summary["latest_model_expected_return_bps"] = latest_model_return
        if latest_model_probability is not None:
            summary["latest_model_success_probability"] = (
                latest_model_probability
            )
        if (
            latest_model_return is not None
            and latest_model_probability is not None
        ):
            latest_model_expected_value = (
                latest_model_return * latest_model_probability
            )
            summary["latest_model_expected_value_bps"] = (
                latest_model_expected_value
            )
            if latest_cost is not None:
                summary["latest_model_expected_value_minus_cost_bps"] = (
                    latest_model_expected_value - latest_cost
                )

        latest_latency = _coerce_float(latest_payload.get("latency_ms"))
        if latest_latency is not None:
            summary["latest_latency_ms"] = latest_latency

        thresholds_snapshot = latest_payload.get("thresholds")
        normalized = _normalize_thresholds(
            thresholds_snapshot if isinstance(thresholds_snapshot, Mapping) else None
        )
        latest_threshold_lookup: Mapping[str, float | None] | None = None
        if normalized:
            latest_threshold_lookup = dict(normalized)
            summary["latest_thresholds"] = dict(normalized)

        latest_reasons = list(_iter_strings(latest_payload.get("reasons")))
        if latest_reasons:
            summary["latest_reasons"] = latest_reasons

        latest_risk_flags = list(_iter_strings(latest_payload.get("risk_flags")))
        if latest_risk_flags:
            summary["latest_risk_flags"] = latest_risk_flags

        latest_failures = list(_iter_strings(latest_payload.get("stress_failures")))
        if latest_failures:
            summary["latest_stress_failures"] = latest_failures

        model_selection = latest_payload.get("model_selection")
        if isinstance(model_selection, Mapping):
            summary["latest_model_selection"] = {
                str(key): model_selection[key] for key in model_selection
            }

        candidate_probability_value: float | None = None
        candidate_notional_value: float | None = None
        if normalized:
            summary["latest_thresholds"] = dict(normalized)

        candidate = latest_payload.get("candidate")
        if isinstance(candidate, Mapping):
            candidate_payload: MutableMapping[str, object] = {}
            for key in ("symbol", "action", "strategy"):
                value = candidate.get(key)
                if value is not None:
                    candidate_payload[key] = value
            for key in ("expected_probability", "expected_return_bps"):
                value = candidate.get(key)
                if value is not None:
                    candidate_payload[key] = value
            candidate_probability = _coerce_float(candidate.get("expected_probability"))
            candidate_return = _coerce_float(candidate.get("expected_return_bps"))
            candidate_notional = _coerce_float(candidate.get("notional"))
            candidate_latency = _coerce_float(candidate.get("latency_ms"))
            candidate_probability_value = candidate_probability
            candidate_notional_value = candidate_notional
            if candidate_latency is not None and latest_latency is None:
                latest_latency = candidate_latency
            if (
                candidate_probability is not None
                and candidate_return is not None
            ):
                candidate_expected_value = (
                    candidate_return * candidate_probability
                )
                candidate_payload["expected_value_bps"] = (
                    candidate_expected_value
                )
                summary["latest_expected_value_bps"] = (
                    candidate_payload["expected_value_bps"]
                )
                if latest_cost is not None:
                    summary[
                        "latest_expected_value_minus_cost_bps"
                    ] = candidate_expected_value - latest_cost
            if candidate_payload:
                summary["latest_candidate"] = candidate_payload

            if candidate_probability is not None:
                summary["latest_expected_probability"] = candidate_probability
            if candidate_return is not None:
                summary["latest_expected_return_bps"] = candidate_return
            if candidate_notional is not None:
                summary["latest_notional"] = candidate_notional
            if candidate_latency is not None and "latest_latency_ms" not in summary:
                summary["latest_latency_ms"] = candidate_latency

            if candidate_payload:
                summary["latest_candidate"] = candidate_payload

            metadata = _extract_candidate_metadata(candidate)
            generated_at = None
            if metadata:
                generated_at = metadata.get("generated_at") or metadata.get("timestamp")
            if generated_at is None:
                generated_at = candidate.get("generated_at")
            if generated_at is not None:
                summary["latest_generated_at"] = generated_at
        if latest_threshold_lookup:
            min_probability = _coerce_float(
                latest_threshold_lookup.get("min_probability")
            )
            if (
                min_probability is not None
                and candidate_probability_value is not None
            ):
                summary["latest_probability_threshold_margin"] = (
                    candidate_probability_value - min_probability
                )
            max_cost_threshold = _coerce_float(
                latest_threshold_lookup.get("max_cost_bps")
            )
            if max_cost_threshold is not None and latest_cost is not None:
                summary["latest_cost_threshold_margin"] = (
                    max_cost_threshold - latest_cost
                )
            min_net_edge_threshold = _coerce_float(
                latest_threshold_lookup.get("min_net_edge_bps")
            )
            if (
                min_net_edge_threshold is not None
                and latest_net_edge is not None
            ):
                summary["latest_net_edge_threshold_margin"] = (
                    latest_net_edge - min_net_edge_threshold
                )
            max_latency_threshold = _coerce_float(
                latest_threshold_lookup.get("max_latency_ms")
            )
            if (
                max_latency_threshold is not None
                and latest_latency is not None
            ):
                summary["latest_latency_threshold_margin"] = (
                    max_latency_threshold - latest_latency
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
        elif "latest_generated_at" not in summary:
            generated_at = _extract_generated_at(latest_payload)
            if generated_at is not None:
                summary["latest_generated_at"] = generated_at

    return summary


__all__ = [
    "summarize_evaluation_payloads",
]
