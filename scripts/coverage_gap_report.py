"""Raport statystyk luk pokrycia danych OHLCV."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from bot_core.alerts import build_environment_coverage_report
from bot_core.config import CoreConfig, EnvironmentConfig, load_core_config
from bot_core.data.ohlcv import (
    CoverageReportPayload,
    compute_gap_statistics,
    compute_gap_statistics_by_interval,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza luk czasowych w manifestach OHLCV dla wskazanych środowisk. "
            "Skrypt pomaga kalibrować progi data_quality na podstawie rzeczywistych danych."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/core.yaml"),
        help="Ścieżka do pliku konfiguracyjnego CoreConfig (domyślnie config/core.yaml).",
    )
    parser.add_argument(
        "--environment",
        dest="environments",
        action="append",
        help="Nazwa środowiska do analizy; opcja może być powtórzona wielokrotnie.",
    )
    parser.add_argument(
        "--all-configured",
        action="store_true",
        help=(
            "Jeśli ustawione, analizuje wszystkie środowiska z sekcji coverage_monitoring "
            "w konfiguracji."
        ),
    )
    parser.add_argument(
        "--as-of",
        type=str,
        help="Znacznik czasu ISO 8601 używany do walidacji manifestu (domyślnie teraz UTC).",
    )
    parser.add_argument(
        "--group-by-interval",
        action="store_true",
        help="Dodatkowo wyświetla statystyki z rozbiciem na interwały.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Zwraca wynik w formacie JSON zamiast tekstowego podsumowania.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_environments(config: CoreConfig, args: argparse.Namespace) -> list[EnvironmentConfig]:
    environments: list[str] = []
    if args.all_configured:
        monitoring = config.coverage_monitoring
        if monitoring and monitoring.targets:
            environments = [target.environment for target in monitoring.targets]
        else:
            raise SystemExit(
                "Brak zdefiniowanych targetów coverage_monitoring w konfiguracji."
            )

    if args.environments:
        environments.extend(args.environments)

    if not environments:
        raise SystemExit(
            "Należy podać co najmniej jedno środowisko (--environment) lub --all-configured."
        )

    unique_names: list[str] = []
    for name in environments:
        if name not in unique_names:
            unique_names.append(name)

    resolved: list[EnvironmentConfig] = []
    for name in unique_names:
        try:
            resolved.append(config.environments[name])
        except KeyError as exc:
            raise SystemExit(f"Środowisko {name} nie istnieje w konfiguracji.") from exc
    return resolved


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensywne logowanie
        raise SystemExit(f"Nieprawidłowy format as_of: {value}") from exc


def _report_to_mapping(
    report: CoverageReportPayload,
    *,
    include_interval_breakdown: bool,
) -> dict[str, object]:
    if report.gap_statistics is not None:
        statistics = report.gap_statistics
    else:
        statistics = compute_gap_statistics(report.statuses)
    payload: dict[str, object] = {
        "environment": report.payload.get("environment"),
        "exchange": report.payload.get("exchange"),
        "summary": report.summary,
        "gap_statistics": statistics.to_mapping(),
        "threshold_issues": list(report.threshold_issues),
        "issues": list(report.issues),
        "status": report.payload.get("status"),
    }

    if include_interval_breakdown:
        interval_source = report.gap_statistics_by_interval
        if not interval_source:
            interval_source = compute_gap_statistics_by_interval(report.statuses)
        interval_stats = {
            interval: stats.to_mapping()
            for interval, stats in interval_source.items()
        }
        payload["gap_statistics_by_interval"] = interval_stats
    return payload


def _print_text_report(reports: list[dict[str, object]]) -> None:
    for entry in reports:
        environment = entry.get("environment", "unknown")
        exchange = entry.get("exchange", "unknown")
        status = entry.get("status", "unknown")
        stats = entry.get("gap_statistics", {})
        summary = entry.get("summary", {})

        print(f"Środowisko: {environment} ({exchange})")
        print(f"  Status manifestu: {status}")
        print(
            "  Wpisy OK: {ok}/{total} (ok_ratio={ratio})".format(
                ok=summary.get("ok"),
                total=summary.get("total"),
                ratio=summary.get("ok_ratio"),
            )
        )
        print(
            "  Luki: count={with_gap}/{total}, min={min_gap}, median={median}, "
            "p95={p95}, max={max_gap}".format(
                with_gap=stats.get("with_gap_measurement"),
                total=stats.get("total_entries"),
                min_gap=stats.get("min_gap_minutes"),
                median=stats.get("median_gap_minutes"),
                p95=stats.get("percentile_95_gap_minutes"),
                max_gap=stats.get("max_gap_minutes"),
            )
        )

        interval_breakdown = entry.get("gap_statistics_by_interval")
        if isinstance(interval_breakdown, dict) and interval_breakdown:
            print("  Rozbicie na interwały:")
            for interval, stats_map in sorted(interval_breakdown.items()):
                print(
                    "    {interval}: median={median} min={min_gap} max={max_gap} count={count}".format(
                        interval=interval,
                        median=stats_map.get("median_gap_minutes"),
                        min_gap=stats_map.get("min_gap_minutes"),
                        max_gap=stats_map.get("max_gap_minutes"),
                        count=stats_map.get("with_gap_measurement"),
                    )
                )

        issues = entry.get("issues") or []
        threshold_issues = entry.get("threshold_issues") or []
        if issues or threshold_issues:
            print("  Ostrzeżenia: {}".format(list(issues) + list(threshold_issues)))
        print()


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_core_config(args.config)
    environments = _resolve_environments(config, args)
    as_of = _parse_as_of(args.as_of)

    report_entries: list[dict[str, object]] = []
    for environment in environments:
        coverage_report = build_environment_coverage_report(
            config=config,
            environment=environment,
            as_of=as_of,
        )
        report_entries.append(
            _report_to_mapping(
                coverage_report,
                include_interval_breakdown=args.group_by_interval,
            )
        )

    if args.json:
        print(json.dumps({"reports": report_entries}, ensure_ascii=False, indent=2))
    else:
        _print_text_report(report_entries)

    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())

