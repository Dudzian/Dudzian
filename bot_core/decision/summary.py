"""Utilities for aggregating Decision Engine evaluation payloads."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from bot_core.decision.schemas import DecisionEngineSummary

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
        if self._history_limit is None:
            return self._full_total
        return self._history_limit

    def build_summary(self) -> dict[str, object]:
        summary: dict[str, object] = {}

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

        cost_sum = 0.0
        cost_count = 0
        net_edge_sum = 0.0
        net_edge_count = 0
        expected_probability_sum = 0.0
        expected_probability_count = 0

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
                cost_sum += cost
                cost_count += 1

            net_edge = _coerce_float(payload.get("net_edge_bps"))
            if net_edge is not None:
                net_edge_sum += net_edge
                net_edge_count += 1

            expected_probability = _coerce_float(candidate.get("expected_probability"))
            if expected_probability is not None:
                expected_probability_sum += expected_probability
                expected_probability_count += 1

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

        summary["avg_expected_probability"] = (
            expected_probability_sum / expected_probability_count if expected_probability_count else 0.0
        )
        summary["avg_cost_bps"] = cost_sum / cost_count if cost_count else 0.0
        summary["avg_net_edge_bps"] = net_edge_sum / net_edge_count if net_edge_count else 0.0
        summary["sum_cost_bps"] = cost_sum
        summary["sum_net_edge_bps"] = net_edge_sum

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
