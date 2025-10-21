"""Narzędzia do agregacji i raportowania jakości decyzji AI."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, MutableMapping, Sequence


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


def summarize_evaluation_payloads(
    evaluations: Sequence[Mapping[str, object]],
    *,
    history_limit: int | None = None,
) -> Mapping[str, object]:
    """Buduje zagregowane podsumowanie Decision Engine na podstawie ewaluacji."""

    total = len(evaluations)
    limit = history_limit if history_limit is not None else total
    summary: MutableMapping[str, object] = {
        "total": total,
        "accepted": 0,
        "rejected": 0,
        "acceptance_rate": 0.0,
        "history_limit": limit,
        "history_window": total,
        "rejection_reasons": {},
    }
    if total == 0:
        return summary

    accepted = 0
    rejection_reasons: Counter[str] = Counter()
    net_edges: list[float] = []
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

        cost = _coerce_float(payload.get("cost_bps"))
        if cost is not None:
            costs.append(cost)

        model_prob = _coerce_float(payload.get("model_success_probability"))
        if model_prob is not None:
            model_probabilities.append(model_prob)

        model_return = _coerce_float(payload.get("model_expected_return_bps"))
        if model_return is not None:
            model_returns.append(model_return)

        latency = _coerce_float(payload.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)

        candidate = payload.get("candidate")
        if isinstance(candidate, Mapping):
            probability = _coerce_float(candidate.get("expected_probability"))
            if probability is not None:
                probabilities.append(probability)
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
        summary["avg_model_expected_return_bps"] = sum(model_returns) / len(model_returns)
    if latencies:
        summary["avg_latency_ms"] = sum(latencies) / len(latencies)

    latest_payload = evaluations[-1]
    if isinstance(latest_payload, Mapping):
        latest_model = latest_payload.get("model_name")
        if latest_model:
            summary["latest_model"] = str(latest_model)

        thresholds_snapshot = latest_payload.get("thresholds")
        normalized = _normalize_thresholds(
            thresholds_snapshot if isinstance(thresholds_snapshot, Mapping) else None
        )
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

    return summary


__all__ = [
    "summarize_evaluation_payloads",
]
