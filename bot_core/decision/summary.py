"""Utilities for aggregating Decision Engine evaluation payloads."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from bot_core.decision.models import DecisionEngineSummary

__all__ = ["DecisionSummaryAggregator", "summarize_evaluation_payloads", "DecisionEngineSummary"]


def _coerce_float(value: object) -> float | None:
    """Best-effort conversion of ``value`` to ``float``."""

    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int, guard explicitly
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_thresholds(
    snapshot: Mapping[str, object] | None,
) -> dict[str, float | None] | None:
    if snapshot is None or not isinstance(snapshot, Mapping):
        return None
    normalized: dict[str, float | None] = {}
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
            for key in ("generated_at", "timestamp"):
                if metadata.get(key) is not None:
                    return str(metadata[key])
        generated_at = candidate.get("generated_at")
        if generated_at is not None:
            return str(generated_at)
    return None


def _iter_strings(values: object) -> Iterable[str]:
    if values is None:
        return ()
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else ()
    if isinstance(values, Mapping):
        return (str(key) for key in values.keys())
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        for item in values:
            if item in (None, ""):
                continue
            yield str(item)
        return
    return (str(values),)


def _resolve_history_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    try:
        coerced = int(limit)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if coerced <= 0:
        return 0
    return coerced


@dataclass
class _MetricAccumulator:
    total_sum: float = 0.0
    total_count: int = 0
    accepted_sum: float = 0.0
    accepted_count: int = 0
    rejected_sum: float = 0.0
    rejected_count: int = 0

    def add(self, value: float, *, accepted: bool) -> None:
        self.total_sum += value
        self.total_count += 1
        if accepted:
            self.accepted_sum += value
            self.accepted_count += 1
        else:
            self.rejected_sum += value
            self.rejected_count += 1

    def as_mapping(self) -> dict[str, float | int]:
        def _avg(total: float, count: int) -> float:
            return total / count if count else 0.0

        return {
            "total_sum": self.total_sum,
            "total_avg": _avg(self.total_sum, self.total_count),
            "total_count": self.total_count,
            "accepted_sum": self.accepted_sum,
            "accepted_avg": _avg(self.accepted_sum, self.accepted_count),
            "accepted_count": self.accepted_count,
            "rejected_sum": self.rejected_sum,
            "rejected_avg": _avg(self.rejected_sum, self.rejected_count),
            "rejected_count": self.rejected_count,
        }


@dataclass
class _BreakdownAccumulator:
    metrics: dict[str, _MetricAccumulator] = field(default_factory=dict)
    total: int = 0
    accepted: int = 0
    rejected: int = 0

    def add(self, metrics: Mapping[str, float], *, accepted: bool) -> None:
        self.total += 1
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        for key, value in metrics.items():
            if value is None:
                continue
            self.metrics.setdefault(key, _MetricAccumulator()).add(value, accepted=accepted)

    def as_mapping(self) -> dict[str, object]:
        acceptance_rate = self.accepted / self.total if self.total else 0.0
        payload: dict[str, object] = {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": acceptance_rate,
        }
        if self.metrics:
            payload["metrics"] = {
                key: accumulator.as_mapping() for key, accumulator in self.metrics.items()
            }
        return payload


_THRESHOLD_DEFINITIONS: Mapping[str, tuple[str, str, str]] = {
    "min_probability": ("expected_probability", "min", "probability_threshold_margin"),
    "min_net_edge_bps": ("net_edge_bps", "min", "net_edge_threshold_margin"),
    "max_cost_bps": ("cost_bps", "max", "cost_threshold_margin"),
    "max_latency_ms": ("latency_ms", "max", "latency_threshold_margin"),
    "max_trade_notional": ("notional", "max", "notional_threshold_margin"),
}


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = position - lower_index
    return lower_value + fraction * (upper_value - lower_value)


def _compute_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    count = len(values)
    if not count:
        return {
            "count": 0,
            "sum": 0.0,
            "avg": 0.0,
            "median": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
            "std": 0.0,
        }
    sorted_values = sorted(values)
    total = math.fsum(sorted_values)
    average = total / count
    variance = math.fsum((value - average) ** 2 for value in sorted_values) / count
    std = math.sqrt(variance)
    return {
        "count": count,
        "sum": total,
        "avg": average,
        "median": _percentile(sorted_values, 0.5),
        "p10": _percentile(sorted_values, 0.1),
        "p90": _percentile(sorted_values, 0.9),
        "p95": _percentile(sorted_values, 0.95),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "std": std,
    }


@dataclass
class _StatAccumulator:
    total: list[float] = field(default_factory=list)
    accepted: list[float] = field(default_factory=list)
    rejected: list[float] = field(default_factory=list)

    def add(self, value: float, *, accepted: bool) -> None:
        self.total.append(value)
        if accepted:
            self.accepted.append(value)
        else:
            self.rejected.append(value)

    def finalize(
        self,
        name: str,
        summary: dict[str, object],
        *,
        include_p95: bool = True,
    ) -> None:
        total_stats = _compute_stats(self.total)
        summary[f"{name}_count"] = total_stats["count"]
        summary[f"sum_{name}"] = total_stats["sum"]
        summary[f"avg_{name}"] = total_stats["avg"]
        summary[f"median_{name}"] = total_stats["median"]
        summary[f"p10_{name}"] = total_stats["p10"]
        summary[f"p90_{name}"] = total_stats["p90"]
        if include_p95:
            summary[f"p95_{name}"] = total_stats["p95"]
        summary[f"min_{name}"] = total_stats["min"]
        summary[f"max_{name}"] = total_stats["max"]
        summary[f"std_{name}"] = total_stats["std"]

        accepted_stats = _compute_stats(self.accepted)
        summary[f"accepted_{name}_count"] = accepted_stats["count"]
        summary[f"accepted_sum_{name}"] = accepted_stats["sum"]
        summary[f"accepted_avg_{name}"] = accepted_stats["avg"]
        summary[f"accepted_median_{name}"] = accepted_stats["median"]
        summary[f"accepted_p10_{name}"] = accepted_stats["p10"]
        summary[f"accepted_p90_{name}"] = accepted_stats["p90"]
        if include_p95:
            summary[f"accepted_p95_{name}"] = accepted_stats["p95"]
        summary[f"accepted_min_{name}"] = accepted_stats["min"]
        summary[f"accepted_max_{name}"] = accepted_stats["max"]
        summary[f"accepted_std_{name}"] = accepted_stats["std"]

        rejected_stats = _compute_stats(self.rejected)
        summary[f"rejected_{name}_count"] = rejected_stats["count"]
        summary[f"rejected_sum_{name}"] = rejected_stats["sum"]
        summary[f"rejected_avg_{name}"] = rejected_stats["avg"]
        summary[f"rejected_median_{name}"] = rejected_stats["median"]
        summary[f"rejected_p10_{name}"] = rejected_stats["p10"]
        summary[f"rejected_p90_{name}"] = rejected_stats["p90"]
        if include_p95:
            summary[f"rejected_p95_{name}"] = rejected_stats["p95"]
        summary[f"rejected_min_{name}"] = rejected_stats["min"]
        summary[f"rejected_max_{name}"] = rejected_stats["max"]
        summary[f"rejected_std_{name}"] = rejected_stats["std"]


@dataclass
class _ThresholdAccumulator:
    margin_key: str
    total: list[float] = field(default_factory=list)
    accepted: list[float] = field(default_factory=list)
    rejected: list[float] = field(default_factory=list)
    total_breaches: int = 0
    accepted_breaches: int = 0
    rejected_breaches: int = 0

    def add(self, value: float, *, accepted: bool) -> None:
        self.total.append(value)
        breached = value < 0.0
        if accepted:
            self.accepted.append(value)
            if breached:
                self.accepted_breaches += 1
        else:
            self.rejected.append(value)
            if breached:
                self.rejected_breaches += 1
        if breached:
            self.total_breaches += 1

    def finalize(self, base_name: str, summary: dict[str, object]) -> None:
        total_stats = _compute_stats(self.total)
        margin_key = self.margin_key
        summary[f"{margin_key}_count"] = total_stats["count"]
        summary[f"sum_{margin_key}"] = total_stats["sum"]
        summary[f"avg_{margin_key}"] = total_stats["avg"]
        summary[f"median_{margin_key}"] = total_stats["median"]
        summary[f"p10_{margin_key}"] = total_stats["p10"]
        summary[f"p90_{margin_key}"] = total_stats["p90"]
        summary[f"min_{margin_key}"] = total_stats["min"]
        summary[f"max_{margin_key}"] = total_stats["max"]
        summary[f"std_{margin_key}"] = total_stats["std"]
        summary[f"{base_name}_threshold_breaches"] = self.total_breaches
        summary[f"{base_name}_threshold_breach_rate"] = (
            self.total_breaches / total_stats["count"]
            if total_stats["count"]
            else 0.0
        )

        accepted_stats = _compute_stats(self.accepted)
        summary[f"accepted_{margin_key}_count"] = accepted_stats["count"]
        summary[f"accepted_sum_{margin_key}"] = accepted_stats["sum"]
        summary[f"accepted_avg_{margin_key}"] = accepted_stats["avg"]
        summary[f"accepted_median_{margin_key}"] = accepted_stats["median"]
        summary[f"accepted_p10_{margin_key}"] = accepted_stats["p10"]
        summary[f"accepted_p90_{margin_key}"] = accepted_stats["p90"]
        summary[f"accepted_min_{margin_key}"] = accepted_stats["min"]
        summary[f"accepted_max_{margin_key}"] = accepted_stats["max"]
        summary[f"accepted_std_{margin_key}"] = accepted_stats["std"]
        summary[f"accepted_{base_name}_threshold_breaches"] = self.accepted_breaches
        summary[f"accepted_{base_name}_threshold_breach_rate"] = (
            self.accepted_breaches / accepted_stats["count"]
            if accepted_stats["count"]
            else 0.0
        )

        rejected_stats = _compute_stats(self.rejected)
        summary[f"rejected_{margin_key}_count"] = rejected_stats["count"]
        summary[f"rejected_sum_{margin_key}"] = rejected_stats["sum"]
        summary[f"rejected_avg_{margin_key}"] = rejected_stats["avg"]
        summary[f"rejected_median_{margin_key}"] = rejected_stats["median"]
        summary[f"rejected_p10_{margin_key}"] = rejected_stats["p10"]
        summary[f"rejected_p90_{margin_key}"] = rejected_stats["p90"]
        summary[f"rejected_min_{margin_key}"] = rejected_stats["min"]
        summary[f"rejected_max_{margin_key}"] = rejected_stats["max"]
        summary[f"rejected_std_{margin_key}"] = rejected_stats["std"]
        summary[f"rejected_{base_name}_threshold_breaches"] = self.rejected_breaches
        summary[f"rejected_{base_name}_threshold_breach_rate"] = (
            self.rejected_breaches / rejected_stats["count"]
            if rejected_stats["count"]
            else 0.0
        )


class DecisionSummaryAggregator:
    """Aggregates raw evaluation payloads into a summary mapping."""

    def __init__(self, evaluations: Iterable[Mapping[str, object]], *, history_limit: int | None = None) -> None:
        self._items = [payload for payload in evaluations if isinstance(payload, Mapping)]
        self._full_total = len(self._items)
        self._history_limit = _resolve_history_limit(history_limit)
        if self._history_limit is None:
            self._window = list(self._items)
        elif self._history_limit <= 0:
            self._window = []
        elif self._history_limit < self._full_total:
            self._window = self._items[-self._history_limit :]
        else:
            self._window = list(self._items)

    @property
    def full_total(self) -> int:
        return self._full_total

    @property
    def history_window(self) -> int:
        return len(self._window)

    @property
    def history_limit(self) -> int:
        """Effective history limit for the aggregated window."""

        if self._history_limit is None:
            return self.history_window
        return self._history_limit

    def build_summary(self) -> dict[str, object]:
        summary: dict[str, object] = {}

        metric_accumulators: dict[str, _StatAccumulator] = {
            "net_edge_bps": _StatAccumulator(),
            "cost_bps": _StatAccumulator(),
            "latency_ms": _StatAccumulator(),
            "expected_probability": _StatAccumulator(),
            "expected_return_bps": _StatAccumulator(),
            "expected_value_bps": _StatAccumulator(),
            "expected_value_minus_cost_bps": _StatAccumulator(),
            "notional": _StatAccumulator(),
            "model_success_probability": _StatAccumulator(),
            "model_expected_return_bps": _StatAccumulator(),
            "model_expected_value_bps": _StatAccumulator(),
            "model_expected_value_minus_cost_bps": _StatAccumulator(),
        }
        metrics_without_p95 = {"expected_probability", "model_success_probability"}

        threshold_accumulators = {
            "probability": _ThresholdAccumulator("probability_threshold_margin"),
            "net_edge": _ThresholdAccumulator("net_edge_threshold_margin"),
            "cost": _ThresholdAccumulator("cost_threshold_margin"),
            "latency": _ThresholdAccumulator("latency_threshold_margin"),
            "notional": _ThresholdAccumulator("notional_threshold_margin"),
        }
        margin_to_base = {
            accumulator.margin_key: base
            for base, accumulator in threshold_accumulators.items()
        }

        accepted_count = 0
        rejected_count = 0
        latest_payload: Mapping[str, object] | None = None
        history_start_generated_at: str | None = None

        risk_counts: Counter[str] = Counter()
        stress_counts: Counter[str] = Counter()
        risk_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "accepted": 0, "rejected": 0})
        stress_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "accepted": 0, "rejected": 0})

        model_usage: Counter[str] = Counter()
        action_usage: Counter[str] = Counter()
        strategy_usage: Counter[str] = Counter()
        symbol_usage: Counter[str] = Counter()

        model_breakdown: dict[str, _BreakdownAccumulator] = {}
        action_breakdown: dict[str, _BreakdownAccumulator] = {}
        strategy_breakdown: dict[str, _BreakdownAccumulator] = {}
        symbol_breakdown: dict[str, _BreakdownAccumulator] = {}

        longest_acceptance_streak = 0
        longest_rejection_streak = 0
        current_acceptance_streak = 0
        current_rejection_streak = 0

        for index, payload in enumerate(self._window):
            accepted = bool(payload.get("accepted"))
            if accepted:
                accepted_count += 1
                current_acceptance_streak += 1
                longest_acceptance_streak = max(longest_acceptance_streak, current_acceptance_streak)
                current_rejection_streak = 0
            else:
                rejected_count += 1
                current_rejection_streak += 1
                longest_rejection_streak = max(longest_rejection_streak, current_rejection_streak)
                current_acceptance_streak = 0

            candidate_raw = payload.get("candidate")
            candidate = dict(candidate_raw) if isinstance(candidate_raw, Mapping) else {}

            if index == 0:
                history_start_generated_at = _extract_generated_at(payload)

            generated_at = _extract_generated_at(payload)
            if generated_at is not None:
                summary_generated = summary.get("generated_at")
                if summary_generated is None or index == len(self._window) - 1:
                    summary["generated_at"] = generated_at

            cost = _coerce_float(payload.get("cost_bps"))
            if cost is not None:
                metric_accumulators["cost_bps"].add(cost, accepted=accepted)

            net_edge = _coerce_float(payload.get("net_edge_bps"))
            if net_edge is not None:
                metric_accumulators["net_edge_bps"].add(net_edge, accepted=accepted)

            expected_probability = _coerce_float(candidate.get("expected_probability"))
            if expected_probability is not None:
                metric_accumulators["expected_probability"].add(
                    expected_probability, accepted=accepted
                )

            expected_return = _coerce_float(candidate.get("expected_return_bps"))
            expected_value = (
                expected_probability * expected_return
                if expected_probability is not None and expected_return is not None
                else None
            )

            notional = _coerce_float(candidate.get("notional"))
            latency = _coerce_float(payload.get("latency_ms"))
            if latency is None:
                latency = _coerce_float(candidate.get("latency_ms"))

            expected_value_minus_cost = (
                expected_value - cost if expected_value is not None and cost is not None else None
            )

            model_expected_return = _coerce_float(payload.get("model_expected_return_bps"))
            model_success_probability = _coerce_float(payload.get("model_success_probability"))
            model_expected_value = (
                model_success_probability * model_expected_return
                if model_success_probability is not None and model_expected_return is not None
                else None
            )
            model_expected_value_minus_cost = (
                model_expected_value - cost
                if model_expected_value is not None and cost is not None
                else None
            )

            if expected_return is not None:
                metric_accumulators["expected_return_bps"].add(
                    expected_return, accepted=accepted
                )
            if expected_value is not None:
                metric_accumulators["expected_value_bps"].add(
                    expected_value, accepted=accepted
                )
            if expected_value_minus_cost is not None:
                metric_accumulators["expected_value_minus_cost_bps"].add(
                    expected_value_minus_cost, accepted=accepted
                )
            if notional is not None:
                metric_accumulators["notional"].add(notional, accepted=accepted)
            if latency is not None:
                metric_accumulators["latency_ms"].add(latency, accepted=accepted)
            if model_success_probability is not None:
                metric_accumulators["model_success_probability"].add(
                    model_success_probability, accepted=accepted
                )
            if model_expected_return is not None:
                metric_accumulators["model_expected_return_bps"].add(
                    model_expected_return, accepted=accepted
                )
            if model_expected_value is not None:
                metric_accumulators["model_expected_value_bps"].add(
                    model_expected_value, accepted=accepted
                )
            if model_expected_value_minus_cost is not None:
                metric_accumulators["model_expected_value_minus_cost_bps"].add(
                    model_expected_value_minus_cost, accepted=accepted
                )

            breakdown_metrics: dict[str, float] = {}
            if net_edge is not None:
                breakdown_metrics["net_edge_bps"] = net_edge
            if expected_value_minus_cost is not None:
                breakdown_metrics["expected_value_minus_cost_bps"] = expected_value_minus_cost
            if expected_value is not None:
                breakdown_metrics["expected_value_bps"] = expected_value
            if cost is not None:
                breakdown_metrics["cost_bps"] = cost
            if notional is not None:
                breakdown_metrics["notional"] = notional
            if latency is not None:
                breakdown_metrics["latency_ms"] = latency

            model_name = payload.get("model_name")
            if isinstance(model_name, str) and model_name:
                model_usage[model_name] += 1
                model_breakdown.setdefault(model_name, _BreakdownAccumulator()).add(
                    breakdown_metrics, accepted=accepted
                )

            action = candidate.get("action")
            if isinstance(action, str) and action:
                action_usage[action] += 1
                action_breakdown.setdefault(action, _BreakdownAccumulator()).add(
                    breakdown_metrics, accepted=accepted
                )

            strategy = candidate.get("strategy")
            if isinstance(strategy, str) and strategy:
                strategy_usage[strategy] += 1
                strategy_breakdown.setdefault(strategy, _BreakdownAccumulator()).add(
                    breakdown_metrics, accepted=accepted
                )

            symbol = candidate.get("symbol")
            if isinstance(symbol, str) and symbol:
                symbol_usage[symbol] += 1
                symbol_breakdown.setdefault(symbol, _BreakdownAccumulator()).add(
                    breakdown_metrics, accepted=accepted
                )

            reasons = list(_iter_strings(payload.get("reasons")))
            if reasons:
                summary.setdefault("rejection_reasons", Counter())
                reasons_counter: Counter[str] = summary["rejection_reasons"]  # type: ignore[assignment]
                for reason in reasons:
                    reasons_counter[reason] += 1

            risk_flags = list(_iter_strings(payload.get("risk_flags")))
            for flag in risk_flags:
                risk_counts[flag] += 1
                entry = risk_breakdown[flag]
                entry["total"] += 1
                if accepted:
                    entry["accepted"] += 1
                else:
                    entry["rejected"] += 1

            stress_failures = list(_iter_strings(payload.get("stress_failures")))
            for failure in stress_failures:
                stress_counts[failure] += 1
                entry = stress_breakdown[failure]
                entry["total"] += 1
                if accepted:
                    entry["accepted"] += 1
                else:
                    entry["rejected"] += 1

            latest_payload = payload
            candidate_summary = dict(candidate)
            if expected_value is not None:
                candidate_summary["expected_value_bps"] = expected_value
            if expected_value_minus_cost is not None:
                candidate_summary["expected_value_minus_cost_bps"] = expected_value_minus_cost

            model_selection = payload.get("model_selection")
            if isinstance(model_selection, Mapping):
                model_selection_value: object = dict(model_selection)
            else:
                model_selection_value = model_selection

            summary.update(
                {
                    "latest_model": model_name if isinstance(model_name, str) else None,
                    "latest_status": "accepted" if accepted else "rejected",
                    "latest_thresholds": _normalize_thresholds(payload.get("thresholds")),
                    "latest_reasons": reasons or None,
                    "latest_risk_flags": risk_flags or None,
                    "latest_stress_failures": stress_failures or None,
                    "latest_model_selection": model_selection_value,
                    "latest_candidate": candidate_summary or None,
                    "latest_generated_at": generated_at,
                    "latest_expected_value_bps": expected_value,
                    "latest_expected_value_minus_cost_bps": expected_value_minus_cost,
                    "latest_net_edge_bps": net_edge,
                    "latest_cost_bps": cost,
                    "latest_latency_ms": latency,
                    "latest_expected_probability": expected_probability,
                    "latest_expected_return_bps": expected_return,
                    "latest_notional": notional,
                    "latest_model_expected_value_bps": model_expected_value,
                    "latest_model_expected_value_minus_cost_bps": model_expected_value_minus_cost,
                    "latest_model_expected_return_bps": model_expected_return,
                    "latest_model_success_probability": model_success_probability,
                }
            )

            thresholds = _normalize_thresholds(payload.get("thresholds"))
            if thresholds:
                observed_values = {
                    "expected_probability": expected_probability,
                    "net_edge_bps": net_edge,
                    "cost_bps": cost,
                    "latency_ms": latency,
                    "notional": notional,
                }
                for key, (metric_key, direction, margin_key) in _THRESHOLD_DEFINITIONS.items():
                    threshold_value = thresholds.get(key)
                    observed = observed_values.get(metric_key)
                    if threshold_value is None or observed is None:
                        continue
                    if direction == "min":
                        margin = observed - threshold_value
                    else:
                        margin = threshold_value - observed
                    summary[f"latest_{margin_key}"] = margin
                    base_name = margin_to_base.get(margin_key)
                    if base_name is not None:
                        threshold_accumulators[base_name].add(
                            margin, accepted=accepted
                        )

        if "rejection_reasons" in summary:
            rejection_counter: Counter[str] = summary["rejection_reasons"]  # type: ignore[assignment]
            summary["rejection_reasons"] = dict(rejection_counter)
            summary["unique_rejection_reasons"] = len(rejection_counter)
        else:
            summary["rejection_reasons"] = {}
            summary["unique_rejection_reasons"] = 0

        summary.update(
            {
                "total": self.history_window,
                "accepted": accepted_count,
                "rejected": rejected_count,
                "acceptance_rate": accepted_count / self.history_window if self.history_window else 0.0,
                "history_limit": self.history_limit,
                "history_window": self.history_window,
                "full_total": self.full_total,
            }
        )

        full_accepted = sum(1 for payload in self._items if bool(payload.get("accepted")))
        full_rejected = self.full_total - full_accepted
        summary["full_accepted"] = full_accepted
        summary["full_rejected"] = full_rejected
        summary["full_acceptance_rate"] = full_accepted / self.full_total if self.full_total else 0.0

        summary["risk_flag_counts"] = dict(risk_counts)
        summary["unique_risk_flags"] = len(risk_counts)
        summary["risk_flags_with_accepts"] = sum(1 for flag, data in risk_breakdown.items() if data["accepted"])
        summary["risk_flag_breakdown"] = {
            flag: {
                **data,
                "acceptance_rate": data["accepted"] / data["total"] if data["total"] else 0.0,
            }
            for flag, data in risk_breakdown.items()
        }

        summary["stress_failure_counts"] = dict(stress_counts)
        summary["unique_stress_failures"] = len(stress_counts)
        summary["stress_failures_with_accepts"] = sum(
            1 for failure, data in stress_breakdown.items() if data["accepted"]
        )
        summary["stress_failure_breakdown"] = {
            failure: {
                **data,
                "acceptance_rate": data["accepted"] / data["total"] if data["total"] else 0.0,
            }
            for failure, data in stress_breakdown.items()
        }

        summary["model_usage"] = dict(model_usage)
        summary["unique_models"] = len(model_usage)
        summary["models_with_accepts"] = sum(
            1 for name, accumulator in model_breakdown.items() if accumulator.accepted
        )
        summary["model_breakdown"] = {
            name: accumulator.as_mapping() for name, accumulator in model_breakdown.items()
        }

        summary["action_usage"] = dict(action_usage)
        summary["unique_actions"] = len(action_usage)
        summary["actions_with_accepts"] = sum(
            1 for name, accumulator in action_breakdown.items() if accumulator.accepted
        )
        summary["action_breakdown"] = {
            name: accumulator.as_mapping() for name, accumulator in action_breakdown.items()
        }

        summary["strategy_usage"] = dict(strategy_usage)
        summary["unique_strategies"] = len(strategy_usage)
        summary["strategies_with_accepts"] = sum(
            1 for name, accumulator in strategy_breakdown.items() if accumulator.accepted
        )
        summary["strategy_breakdown"] = {
            name: accumulator.as_mapping() for name, accumulator in strategy_breakdown.items()
        }

        summary["symbol_usage"] = dict(symbol_usage)
        summary["unique_symbols"] = len(symbol_usage)
        summary["symbols_with_accepts"] = sum(
            1 for name, accumulator in symbol_breakdown.items() if accumulator.accepted
        )
        summary["symbol_breakdown"] = {
            name: accumulator.as_mapping() for name, accumulator in symbol_breakdown.items()
        }

        summary["current_acceptance_streak"] = current_acceptance_streak
        summary["current_rejection_streak"] = current_rejection_streak
        summary["longest_acceptance_streak"] = longest_acceptance_streak
        summary["longest_rejection_streak"] = longest_rejection_streak

        summary["history_start_generated_at"] = history_start_generated_at

        for name, accumulator in metric_accumulators.items():
            include_p95 = name not in metrics_without_p95
            accumulator.finalize(name, summary, include_p95=include_p95)

        for base_name, accumulator in threshold_accumulators.items():
            accumulator.finalize(base_name, summary)

        if latest_payload is None:
            summary.setdefault("generated_at", None)
            summary.setdefault("latest_model", None)
            summary.setdefault("latest_status", None)
            summary.setdefault("latest_thresholds", None)
            summary.setdefault("latest_reasons", None)
            summary.setdefault("latest_risk_flags", None)
            summary.setdefault("latest_stress_failures", None)
            summary.setdefault("latest_model_selection", None)
            summary.setdefault("latest_candidate", None)
            summary.setdefault("latest_generated_at", None)

        return summary


def summarize_evaluation_payloads(
    evaluations: Iterable[Mapping[str, object]], *, history_limit: int | None = None
) -> DecisionEngineSummary:
    aggregator = DecisionSummaryAggregator(evaluations, history_limit=history_limit)
    summary = aggregator.build_summary()
    summary["type"] = "decision_engine_summary"
    return DecisionEngineSummary.model_validate(summary)
