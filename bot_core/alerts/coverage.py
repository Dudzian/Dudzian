"""Pomocnicze funkcje do budowy kontekstu alertów dla jakości danych."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.alerts.base import AlertMessage
from bot_core.alerts.router import DefaultAlertRouter
from bot_core.config.models import CoreConfig, EnvironmentConfig
from bot_core.data.ohlcv import (
    CoverageReportPayload,
    CoverageSummary,
    coerce_summary_mapping,
    compute_gap_statistics,
    compute_gap_statistics_by_interval,
    evaluate_coverage,
    summarize_coverage,
    summarize_issues,
    status_to_mapping,
)
from bot_core.data.ohlcv.coverage_check import (
    SummaryThresholdResult,
    evaluate_summary_thresholds,
)


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


def _resolve_thresholds(
    config: CoreConfig,
    environment: EnvironmentConfig,
) -> tuple[float | None, float | None]:
    max_gap_minutes: float | None = None
    min_ok_ratio: float | None = None

    data_quality = getattr(environment, "data_quality", None)
    if data_quality is not None:
        if getattr(data_quality, "max_gap_minutes", None) is not None:
            max_gap_minutes = float(data_quality.max_gap_minutes)  # type: ignore[arg-type]
        if getattr(data_quality, "min_ok_ratio", None) is not None:
            min_ok_ratio = float(data_quality.min_ok_ratio)  # type: ignore[arg-type]

    if (max_gap_minutes is None or min_ok_ratio is None) and getattr(environment, "risk_profile", None):
        profile = config.risk_profiles.get(environment.risk_profile)
        if profile and profile.data_quality:
            profile_quality = profile.data_quality
            if max_gap_minutes is None and getattr(profile_quality, "max_gap_minutes", None) is not None:
                max_gap_minutes = float(profile_quality.max_gap_minutes)  # type: ignore[arg-type]
            if min_ok_ratio is None and getattr(profile_quality, "min_ok_ratio", None) is not None:
                min_ok_ratio = float(profile_quality.min_ok_ratio)  # type: ignore[arg-type]

    return max_gap_minutes, min_ok_ratio


def _extend_summary(summary: Mapping[str, object], statuses: Sequence[object]) -> dict[str, object]:
    payload = dict(summary)

    def _breakdown(key: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for status in statuses:
            token = str(getattr(status, key, None) or "unknown")
            bucket = result.setdefault(token, {})
            bucket["total"] = bucket.get("total", 0) + 1
            state = str(getattr(status, "status", None) or "unknown")
            bucket[state] = bucket.get(state, 0) + 1
        return result

    payload.setdefault("by_interval", _breakdown("interval"))
    payload.setdefault("by_symbol", _breakdown("symbol"))
    return payload


def build_environment_coverage_report(
    *,
    config: CoreConfig,
    environment: EnvironmentConfig,
    as_of: datetime | None = None,
) -> CoverageReportPayload:
    if not environment.instrument_universe:
        raise ValueError("Środowisko nie ma przypisanego instrument_universe")

    universe = config.instrument_universes.get(environment.instrument_universe)
    if universe is None:
        raise KeyError(
            f"Brak definicji uniwersum instrumentów: {environment.instrument_universe}"
        )

    evaluation_time = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
    manifest_path = Path(environment.data_cache_path) / "ohlcv_manifest.sqlite"

    statuses = list(
        evaluate_coverage(
            manifest_path=manifest_path,
            universe=universe,
            exchange_name=environment.exchange,
            as_of=evaluation_time,
        )
    )

    issues = tuple(summarize_issues(statuses))
    summary = summarize_coverage(statuses)
    summary_payload = coerce_summary_mapping(summary)
    summary_payload = _extend_summary(summary_payload, statuses)

    max_gap_minutes, min_ok_ratio = _resolve_thresholds(config, environment)
    threshold_result: SummaryThresholdResult | None = None
    threshold_payload: Mapping[str, object] | None = None
    threshold_issues: tuple[str, ...] = ()
    if max_gap_minutes is not None or min_ok_ratio is not None:
        threshold_result = evaluate_summary_thresholds(
            summary_payload,
            max_gap_minutes=max_gap_minutes,
            min_ok_ratio=min_ok_ratio,
        )
        threshold_payload = threshold_result.to_mapping()
        threshold_issues = threshold_result.issues

    status_token = str(summary_payload.get("status") or "unknown")
    if issues or threshold_issues:
        status_token = "error"

    gap_stats = compute_gap_statistics(statuses)
    interval_stats = compute_gap_statistics_by_interval(statuses)

    payload: dict[str, object] = {
        "environment": environment.name,
        "exchange": environment.exchange,
        "manifest_path": str(manifest_path),
        "as_of": evaluation_time.isoformat(),
        "entries": [status_to_mapping(status) for status in statuses],
        "issues": list(issues),
        "summary": summary_payload,
        "status": status_token,
        "threshold_issues": list(threshold_issues),
        "gap_statistics": gap_stats.to_mapping(),
    }
    if threshold_payload is not None:
        payload["threshold_evaluation"] = threshold_payload
    if interval_stats:
        payload["gap_statistics_by_interval"] = {
            interval: stats.to_mapping() for interval, stats in interval_stats.items()
        }

    return CoverageReportPayload(
        payload=payload,
        statuses=tuple(statuses),
        summary=summary_payload,
        threshold_result=threshold_result,
        threshold_issues=threshold_issues,
        issues=issues,
        gap_statistics=gap_stats,
        gap_statistics_by_interval=interval_stats,
    )


def dispatch_coverage_alert(
    router: DefaultAlertRouter,
    payload: Mapping[str, object],
    *,
    severity_override: str | None = None,
    category: str = "data.ohlcv",
) -> bool:
    issues = [str(entry) for entry in payload.get("issues", [])]
    threshold_issues = [str(entry) for entry in payload.get("threshold_issues", [])]

    if not issues and not threshold_issues and severity_override is None:
        return False

    summary = payload.get("summary")
    threshold_mapping = payload.get("threshold_evaluation")
    context = build_coverage_alert_context(
        summary=summary,
        threshold_result=threshold_mapping,
        extra_issues=threshold_issues,
    )

    environment = str(payload.get("environment", "unknown"))
    exchange = str(payload.get("exchange", "unknown"))
    status = str(payload.get("status", "unknown"))

    severity = severity_override or "info"
    if issues or threshold_issues:
        severity = "warning"
    if any(issue.startswith("ok_ratio_below_threshold") for issue in threshold_issues):
        severity = "critical"
    if any("missing_metadata" in issue for issue in issues):
        severity = "critical"

    lines = [
        f"Środowisko: {environment} ({exchange})",
        f"Status manifestu: {status}",
    ]

    summary_payload = summary if isinstance(summary, Mapping) else {}
    if isinstance(summary_payload, Mapping):
        total = summary_payload.get("total")
        ok_count = summary_payload.get("ok")
        warning_count = summary_payload.get("warning")
        error_count = summary_payload.get("error")
        lines.append(
            "Podsumowanie: total={total} ok={ok} warning={warning} error={error}".format(
                total=total,
                ok=ok_count,
                warning=warning_count,
                error=error_count,
            )
        )
        worst_gap = summary_payload.get("worst_gap")
        if isinstance(worst_gap, Mapping):
            gap_symbol = worst_gap.get("symbol")
            gap_interval = worst_gap.get("interval")
            gap_minutes = worst_gap.get("gap_minutes")
            lines.append(
                "Największa luka: {symbol}/{interval} = {gap}".format(
                    symbol=gap_symbol or "?",
                    interval=gap_interval or "?",
                    gap=gap_minutes,
                )
            )

    if issues:
        lines.append("Problemy manifestu:")
        lines.extend(f" - {issue}" for issue in issues)
    if threshold_issues:
        lines.append("Alert progów jakości:")
        lines.extend(f" - {issue}" for issue in threshold_issues)

    message = AlertMessage(
        category=category,
        title=f"Pokrycie danych OHLCV ({environment})",
        body="\n".join(lines),
        severity=severity,
        context=context,
    )
    router.dispatch(message)
    return True


__all__ = [
    "build_coverage_alert_context",
    "build_environment_coverage_report",
    "dispatch_coverage_alert",
]
