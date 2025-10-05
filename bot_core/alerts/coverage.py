"""Pomocnicze funkcje do budowy kontekstu alertów dla jakości danych."""
from __future__ import annotations

import json
from typing import Mapping, Sequence

from bot_core.data.ohlcv import CoverageSummary, coerce_summary_mapping
from bot_core.data.ohlcv.coverage_check import SummaryThresholdResult


def _format_float(value: float, *, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def build_coverage_alert_context(
    *,
    summary: CoverageSummary | Mapping[str, object] | None,
    threshold_result: SummaryThresholdResult | Mapping[str, object] | None = None,
    extra_issues: Sequence[str] | None = None,
) -> dict[str, str]:
    """Serializuje zagregowane metryki pokrycia do kontekstu alertu."""

    normalized = coerce_summary_mapping(summary)
    context: dict[str, str] = {
        "summary": json.dumps(normalized, ensure_ascii=False),
        "summary_status": str(normalized.get("status")),
        "summary_total": str(normalized.get("total")),
        "summary_ok": str(normalized.get("ok")),
        "summary_warning": str(normalized.get("warning")),
        "summary_error": str(normalized.get("error")),
        "summary_stale_entries": str(normalized.get("stale_entries")),
    }

    ok_ratio = normalized.get("ok_ratio")
    if isinstance(ok_ratio, (int, float)):
        context["summary_ok_ratio"] = _format_float(float(ok_ratio))
    else:
        try:
            context["summary_ok_ratio"] = _format_float(float(ok_ratio))
        except (TypeError, ValueError):
            context["summary_ok_ratio"] = "n/a"

    worst_gap = normalized.get("worst_gap")
    if isinstance(worst_gap, Mapping):
        symbol = worst_gap.get("symbol")
        interval = worst_gap.get("interval")
        gap_value = worst_gap.get("gap_minutes")
        if symbol is not None:
            context["summary_worst_gap_symbol"] = str(symbol)
        if interval is not None:
            context["summary_worst_gap_interval"] = str(interval)
        if gap_value is not None:
            try:
                context["summary_worst_gap_minutes"] = _format_float(float(gap_value))
            except (TypeError, ValueError):
                context["summary_worst_gap_minutes"] = str(gap_value)
        else:
            context["summary_worst_gap_minutes"] = "n/a"
    else:
        context["summary_worst_gap_minutes"] = "n/a"

    threshold_mapping: Mapping[str, object] | None = None
    if isinstance(threshold_result, SummaryThresholdResult):
        threshold_mapping = threshold_result.to_mapping()
    elif isinstance(threshold_result, Mapping):
        threshold_mapping = threshold_result

    if threshold_mapping is not None:
        thresholds = threshold_mapping.get("thresholds") or {}
        if isinstance(thresholds, Mapping) and thresholds:
            normalized_thresholds: dict[str, float] = {}
            for key, value in thresholds.items():
                try:
                    normalized_thresholds[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if normalized_thresholds:
                context["thresholds"] = json.dumps(normalized_thresholds, ensure_ascii=False)
                for key, value in normalized_thresholds.items():
                    precision = 4 if key == "min_ok_ratio" else 2
                    context[f"threshold_{key}"] = _format_float(value, precision=precision)

        issues = threshold_mapping.get("issues") or ()
        if isinstance(issues, Sequence) and issues:
            context["threshold_issues"] = json.dumps(list(issues), ensure_ascii=False)

        observed = threshold_mapping.get("observed") or {}
        if isinstance(observed, Mapping):
            for key, value in observed.items():
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                precision = 4 if key in {"ok_ratio"} else 2
                context[f"observed_{key}"] = _format_float(numeric, precision=precision)

    if extra_issues:
        context["additional_issues"] = json.dumps(list(extra_issues), ensure_ascii=False)

    return context


__all__ = ["build_coverage_alert_context"]
