"""Pomocnicze funkcje i integracje alertów dla jakości danych OHLCV."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.alerts.base import AlertMessage, AlertRouter
from bot_core.config import CoreConfig, EnvironmentConfig, load_core_config
from bot_core.data.ohlcv import (
    CoverageGapStatistics,
    CoverageReportPayload,
    CoverageSummary,
    build_coverage_report_payload,
    coerce_summary_mapping,
    evaluate_coverage,
)
from bot_core.data.ohlcv.coverage_check import SummaryThresholdResult


def _format_float(value: float, *, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def build_environment_coverage_report(
    *,
    config: CoreConfig,
    environment: EnvironmentConfig,
    as_of: datetime | None = None,
) -> CoverageReportPayload:
    """Buduje raport pokrycia dla wskazanego środowiska."""

    if not environment.instrument_universe:
        raise ValueError(
            "Środowisko nie posiada przypisanego instrument_universe — uzupełnij konfigurację."
        )

    as_of_dt = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)

    try:
        universe = config.instrument_universes[environment.instrument_universe]
    except KeyError as exc:
        raise KeyError(
            f"Środowisko {environment.name} wskazuje nieistniejące uniwersum "
            f"{environment.instrument_universe}."
        ) from exc

    manifest_path = Path(environment.data_cache_path) / "ohlcv_manifest.sqlite"

    statuses = evaluate_coverage(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=environment.exchange,
        as_of=as_of_dt,
    )

    data_quality = environment.data_quality
    if data_quality is None:
        profile = config.risk_profiles.get(environment.risk_profile)
        if profile and profile.data_quality is not None:
            data_quality = profile.data_quality

    return build_coverage_report_payload(
        statuses=tuple(statuses),
        manifest_path=manifest_path,
        environment_name=environment.name,
        exchange_name=environment.exchange,
        as_of=as_of_dt,
        data_quality=data_quality,
    )


def _coerce_gap_statistics(
    gap_statistics: CoverageGapStatistics | Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if isinstance(gap_statistics, CoverageGapStatistics):
        return gap_statistics.to_mapping()
    if isinstance(gap_statistics, Mapping):
        return dict(gap_statistics)
    return None


def build_coverage_alert_context(
    *,
    summary: CoverageSummary | Mapping[str, object] | None,
    threshold_result: SummaryThresholdResult | Mapping[str, object] | None = None,
    extra_issues: Sequence[str] | None = None,
    gap_statistics: CoverageGapStatistics | Mapping[str, object] | None = None,
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

    gap_mapping = _coerce_gap_statistics(gap_statistics)
    if gap_mapping:
        context["gap_statistics"] = json.dumps(gap_mapping, ensure_ascii=False)
        for key, precision in (
            ("max_gap_minutes", 2),
            ("median_gap_minutes", 2),
            ("percentile_95_gap_minutes", 2),
        ):
            value = gap_mapping.get(key)
            if value is None:
                continue
            try:
                context[f"gap_{key}"] = _format_float(float(value), precision=precision)
            except (TypeError, ValueError):
                context[f"gap_{key}"] = str(value)

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


def _derive_severity(*, status: str, threshold_issues: Sequence[str], has_manifest_issues: bool) -> str:
    """Wyznacza poziom ważności alertu na podstawie statusu i progów."""

    normalized_status = status.lower()
    severity_map = {
        "error": "critical",
        "warning": "warning",
        "ok": "info",
        "unknown": "warning",
    }

    severity = severity_map.get(normalized_status, "warning")

    if has_manifest_issues and severity == "info":
        severity = "warning"

    if threshold_issues:
        if any(issue.startswith("ok_ratio_below_threshold") for issue in threshold_issues):
            severity = "critical"
        elif severity == "info":
            severity = "warning"

    return severity


def build_coverage_alert_message(
    *,
    payload: Mapping[str, object],
    severity_override: str | None = None,
    category: str = "data.ohlcv",
) -> AlertMessage:
    """Tworzy obiekt :class:`AlertMessage` na podstawie wyniku CLI."""

    environment = str(payload.get("environment") or "unknown")
    exchange = str(payload.get("exchange") or "unknown")
    summary = payload.get("summary")
    threshold_payload = payload.get("threshold_evaluation")
    threshold_issues = tuple(payload.get("threshold_issues") or ())
    issues = tuple(payload.get("issues") or ())
    manifest_path = payload.get("manifest_path")
    as_of = payload.get("as_of")

    context = build_coverage_alert_context(
        summary=summary,
        threshold_result=threshold_payload,
        extra_issues=issues,
        gap_statistics=payload.get("gap_statistics"),
    )
    context = {
        **context,
        "environment": environment,
        "exchange": exchange,
    }
    if manifest_path is not None:
        context["manifest_path"] = str(manifest_path)
    if as_of is not None:
        context["as_of"] = str(as_of)
    if threshold_issues:
        context["threshold_issue_count"] = str(len(threshold_issues))

    summary_mapping = coerce_summary_mapping(summary)

    lines: list[str] = [
        f"Środowisko: {environment} ({exchange})",
    ]
    if manifest_path:
        lines.append(f"Manifest: {manifest_path}")
    if as_of:
        lines.append(f"Ocena na: {as_of}")

    lines.append(
        "Podsumowanie: status={status} ok={ok}/{total} warning={warning} "
        "error={error} stale_entries={stale_entries} ok_ratio={ok_ratio}".format(
            status=summary_mapping.get("status"),
            ok=summary_mapping.get("ok"),
            total=summary_mapping.get("total"),
            warning=summary_mapping.get("warning"),
            error=summary_mapping.get("error"),
            stale_entries=summary_mapping.get("stale_entries"),
            ok_ratio=summary_mapping.get("ok_ratio"),
        )
    )

    gap_stats_payload = payload.get("gap_statistics")
    if isinstance(gap_stats_payload, Mapping):
        lines.append(
            "Statystyki luk: count={with_gap}/{total} median={median} p95={p95} max={max_gap}".format(
                with_gap=gap_stats_payload.get("with_gap_measurement"),
                total=gap_stats_payload.get("total_entries"),
                median=gap_stats_payload.get("median_gap_minutes"),
                p95=gap_stats_payload.get("percentile_95_gap_minutes"),
                max_gap=gap_stats_payload.get("max_gap_minutes"),
            )
        )

    worst_gap = summary_mapping.get("worst_gap")
    if isinstance(worst_gap, Mapping):
        lines.append(
            "Największa luka: {symbol}/{interval} gap={gap}min threshold={threshold}".format(
                symbol=worst_gap.get("symbol", "?"),
                interval=worst_gap.get("interval", "?"),
                gap=worst_gap.get("gap_minutes", "?"),
                threshold=worst_gap.get("threshold_minutes", "-"),
            )
        )

    if issues:
        lines.append("Problemy manifestu:")
        for issue in issues[:10]:
            lines.append(f"- {issue}")
        if len(issues) > 10:
            lines.append(f"… (+{len(issues) - 10}) kolejnych wpisów")

    if threshold_issues:
        lines.append("Naruszenia progów jakości danych:")
        for issue in threshold_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("Brak naruszeń progów jakości danych")

    threshold_mapping = {}
    if isinstance(threshold_payload, SummaryThresholdResult):
        threshold_mapping = threshold_payload.to_mapping()
    elif isinstance(threshold_payload, Mapping):
        threshold_mapping = dict(threshold_payload)
    thresholds = threshold_mapping.get("thresholds") if isinstance(threshold_mapping, Mapping) else None
    if isinstance(thresholds, Mapping) and thresholds:
        lines.append(
            "Skonfigurowane progi: "
            + ", ".join(f"{name}={value}" for name, value in thresholds.items())
        )
    observed = threshold_mapping.get("observed") if isinstance(threshold_mapping, Mapping) else None
    if isinstance(observed, Mapping) and observed:
        lines.append(
            "Wartości obserwowane: "
            + ", ".join(f"{name}={value}" for name, value in observed.items())
        )

    lines.append("Szczegóły znajdziesz w logach i raporcie coverage CLI.")

    status_token = str(payload.get("status") or summary_mapping.get("status") or "warning")
    severity = severity_override or _derive_severity(
        status=status_token,
        threshold_issues=threshold_issues,
        has_manifest_issues=bool(issues),
    )

    title = f"Alert pokrycia danych OHLCV ({environment})"

    return AlertMessage(
        category=category,
        title=title,
        body="\n".join(lines),
        severity=severity,
        context=context,
    )


def dispatch_coverage_alert(
    router: AlertRouter,
    *,
    payload: Mapping[str, object],
    severity_override: str | None = None,
    category: str = "data.ohlcv",
) -> bool:
    """Buduje i wysyła alert na podstawie wyniku CLI.

    Zwraca ``True``, jeśli alert został wysłany, lub ``False`` w przeciwnym wypadku.
    """

    issues = tuple(payload.get("issues") or ())
    threshold_issues = tuple(payload.get("threshold_issues") or ())
    if not issues and not threshold_issues:
        return False

    message = build_coverage_alert_message(
        payload=payload,
        severity_override=severity_override,
        category=category,
    )
    router.dispatch(message)
    return True


def run_coverage_check_and_alert(
    *,
    config_path: str | Path,
    environment_name: str,
    router: AlertRouter,
    as_of: datetime | None = None,
    category: str = "data.ohlcv",
    severity_override: str | None = None,
) -> tuple[CoverageReportPayload, bool]:
    """Uruchamia walidację pokrycia i wysyła alert przy wykryciu problemów.

    Zwraca dwuelementową krotkę: obiekt raportu z kompletem metryk oraz flagę
    wskazującą, czy alert został wysłany. Funkcja jest przeznaczona do użycia w
    pipeline'ach CI lub nocnych zadaniach kontrolnych, które po wykonaniu
    `scripts/check_data_coverage.py` powinny natychmiast eskalować wynik.
    """

    config = load_core_config(Path(config_path))

    try:
        environment = config.environments[environment_name]
    except KeyError as exc:  # pragma: no cover - walidowane testami integracyjnymi
        raise KeyError(f"Nie znaleziono środowiska '{environment_name}' w konfiguracji") from exc

    report = build_environment_coverage_report(
        config=config,
        environment=environment,
        as_of=as_of,
    )

    dispatched = dispatch_coverage_alert(
        router,
        payload=report.payload,
        severity_override=severity_override,
        category=category,
    )
    return report, dispatched


__all__ = [
    "build_environment_coverage_report",
    "build_coverage_alert_context",
    "build_coverage_alert_message",
    "dispatch_coverage_alert",
    "run_coverage_check_and_alert",
]
