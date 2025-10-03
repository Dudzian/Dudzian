"""Preflight weryfikacji przed uruchomieniem smoke testu paper."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot_core.config.loader import load_core_config
from bot_core.config.validation import validate_core_config
from bot_core.data.intervals import normalize_interval_token
from bot_core.data.ohlcv import (
    CoverageStatus,
    evaluate_coverage,
    summarize_by_interval,
    summarize_by_symbol,
    summarize_coverage,
    summarize_issues,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uruchamia podstawowe walidacje przed smoke testem paper – "
            "kontrolę konfiguracji oraz pokrycia danych w manifeście."
        )
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Środowisko docelowe z sekcji environments",
    )
    parser.add_argument(
        "--manifest",
        help="Opcjonalna ścieżka do manifestu SQLite (domyślnie katalog cache środowiska)",
    )
    parser.add_argument(
        "--as-of",
        help="Znacznik czasu ISO8601 wykorzystywany przy ocenie świeżości danych (domyślnie teraz, UTC)",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        default=None,
        help=(
            "Filtruj kontrolę pokrycia do wskazanych instrumentów. "
            "Można używać zarówno nazw z konfiguracji (np. BTC_USDT) jak i symboli giełdowych."
        ),
    )
    parser.add_argument(
        "--interval",
        dest="intervals",
        action="append",
        default=None,
        help="Filtruj kontrolę pokrycia do wskazanych interwałów (np. 1d, D1, 1h).",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=None,
        help="Maksymalna dopuszczalna luka czasowa (minuty) – nadpisuje ustawienie środowiska.",
    )
    parser.add_argument(
        "--min-ok-ratio",
        type=float,
        default=None,
        help="Minimalny udział poprawnych wpisów manifestu (0-1) – nadpisuje konfigurację środowiska.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Zwróć wynik w formacie JSON zamiast tekstowego podsumowania.",
    )
    parser.add_argument(
        "--output",
        help="Ścieżka pliku, do którego zostanie zapisany wynik w formacie JSON.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Zakończ komendę kodem błędu, jeśli pojawią się ostrzeżenia.",
    )
    return parser.parse_args(argv)


def _parse_as_of(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _filter_statuses_by_symbols(
    statuses: Iterable[CoverageStatus],
    *,
    universe,
    exchange: str,
    filters: Sequence[str] | None,
) -> tuple[list[CoverageStatus], list[str]]:
    if not filters:
        return list(statuses), []

    alias_map: dict[str, str] = {}
    for instrument in getattr(universe, "instruments", ()):  # type: ignore[attr-defined]
        symbol = instrument.exchange_symbols.get(exchange)
        if not symbol:
            continue
        alias_map[instrument.name.upper()] = symbol
        alias_map[symbol.upper()] = symbol

    available = {status.symbol.upper(): status.symbol for status in statuses}

    resolved: set[str] = set()
    unknown: list[str] = []
    for raw in filters:
        token = str(raw).upper()
        symbol = alias_map.get(token)
        if symbol is None:
            symbol = available.get(token)
        if symbol is None:
            unknown.append(raw)
            continue
        resolved.add(symbol)

    if not resolved:
        return [], unknown

    filtered = [status for status in statuses if status.symbol in resolved]
    return filtered, unknown


def _filter_statuses_by_intervals(
    statuses: Iterable[CoverageStatus],
    *,
    filters: Sequence[str] | None,
) -> tuple[list[CoverageStatus], list[str]]:
    if not filters:
        return list(statuses), []

    normalized_filters: list[tuple[str, str]] = []
    invalid: list[str] = []
    for raw in filters:
        normalized = normalize_interval_token(raw)
        if not normalized:
            invalid.append(str(raw))
            continue
        normalized_filters.append((str(raw), normalized))

    if invalid:
        return [], invalid

    available_map: dict[str, set[str]] = {}
    for status in statuses:
        normalized = normalize_interval_token(status.interval)
        if not normalized:
            continue
        available_map.setdefault(normalized, set()).add(status.interval)

    resolved_variants: set[str] = set()
    missing: list[str] = []
    for raw, normalized in normalized_filters:
        variants = available_map.get(normalized)
        if not variants:
            missing.append(raw)
            continue
        resolved_variants.update(variants)

    if missing:
        return [], missing

    filtered = [status for status in statuses if status.interval in resolved_variants]
    return filtered, []


def _coverage_payload(
    *,
    manifest_path: Path,
    universe,
    exchange: str,
    as_of: datetime,
    symbols: Sequence[str] | None,
    intervals: Sequence[str] | None,
    max_gap_minutes: float | None,
    min_ok_ratio: float | None,
) -> Mapping[str, object]:
    statuses = list(
        evaluate_coverage(
            manifest_path=manifest_path,
            universe=universe,
            exchange_name=exchange,
            as_of=as_of,
            intervals=intervals,
        )
    )

    statuses, unknown_symbols = _filter_statuses_by_symbols(
        statuses,
        universe=universe,
        exchange=exchange,
        filters=symbols,
    )
    if unknown_symbols:
        return {
            "status": "error",
            "issues": [
                "unknown_symbols:" + ",".join(sorted(str(token) for token in unknown_symbols))
            ],
            "summary": {"status": "error"},
        }
    statuses, interval_errors = _filter_statuses_by_intervals(statuses, filters=intervals)
    if interval_errors:
        return {
            "status": "error",
            "issues": [
                "unknown_intervals:" + ",".join(sorted(str(token) for token in interval_errors))
            ],
            "summary": {"status": "error"},
        }

    issues = summarize_issues(statuses)
    summary_payload = dict(summarize_coverage(statuses))

    interval_breakdown = summarize_by_interval(statuses)
    if interval_breakdown:
        summary_payload["by_interval"] = interval_breakdown
    symbol_breakdown = summarize_by_symbol(statuses)
    if symbol_breakdown:
        summary_payload["by_symbol"] = symbol_breakdown

    thresholds: dict[str, float] = {}
    worst_gap = summary_payload.get("worst_gap") if isinstance(summary_payload, Mapping) else None

    if max_gap_minutes is not None:
        thresholds["max_gap_minutes"] = float(max_gap_minutes)
        if isinstance(worst_gap, Mapping):
            try:
                worst_gap_minutes = float(worst_gap.get("gap_minutes", 0.0))
            except (TypeError, ValueError):
                worst_gap_minutes = 0.0
            if worst_gap_minutes > float(max_gap_minutes):
                issues.append(
                    f"max_gap_exceeded:{worst_gap_minutes}>{float(max_gap_minutes)}"
                )

    ok_ratio_value: float | None = None
    if min_ok_ratio is not None:
        thresholds["min_ok_ratio"] = float(min_ok_ratio)
        ok_ratio = summary_payload.get("ok_ratio")
        if isinstance(ok_ratio, (int, float)):
            ok_ratio_value = float(ok_ratio)
        else:
            try:
                ok_ratio_value = float(ok_ratio) if ok_ratio is not None else None
            except (TypeError, ValueError):
                ok_ratio_value = None
        if ok_ratio_value is not None and ok_ratio_value < float(min_ok_ratio):
            issues.append(
                f"ok_ratio_below_threshold:{ok_ratio_value:.4f}<{float(min_ok_ratio):.4f}"
            )
        elif ok_ratio_value is None and float(min_ok_ratio) > 0:
            total_entries = summary_payload.get("total")
            if isinstance(total_entries, (int, float)) and float(total_entries) <= 0:
                issues.append("manifest_empty_for_threshold")

    status = "error" if issues else summary_payload.get("status", "ok")
    payload: dict[str, object] = {
        "status": status,
        "issues": issues,
        "summary": summary_payload,
        "thresholds": thresholds,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_core_config(Path(args.config))
    validation = validate_core_config(config)

    environment = config.environments.get(args.environment)
    if environment is None:
        print(f"Nie znaleziono środowiska: {args.environment}", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest) if args.manifest else Path(environment.data_cache_path) / "ohlcv_manifest.sqlite"

    as_of = _parse_as_of(args.as_of)

    coverage_result: Mapping[str, object] | None = None
    coverage_status = "skipped"
    coverage_warnings: list[str] = []

    if not environment.instrument_universe:
        coverage_warnings.append("environment_missing_universe")
    else:
        universe = config.instrument_universes.get(environment.instrument_universe)
        if universe is None:
            coverage_warnings.append("universe_not_defined")
        elif not manifest_path.exists():
            coverage_warnings.append("manifest_missing")
        else:
            max_gap = args.max_gap_minutes
            min_ok_ratio = args.min_ok_ratio
            data_quality = getattr(environment, "data_quality", None)
            if data_quality is not None:
                if max_gap is None:
                    candidate = getattr(data_quality, "max_gap_minutes", None)
                    if candidate is not None:
                        try:
                            max_gap = float(candidate)
                        except (TypeError, ValueError):
                            coverage_warnings.append("invalid_max_gap_in_config")
                if min_ok_ratio is None:
                    candidate_ratio = getattr(data_quality, "min_ok_ratio", None)
                    if candidate_ratio is not None:
                        try:
                            min_ok_ratio = float(candidate_ratio)
                        except (TypeError, ValueError):
                            coverage_warnings.append("invalid_min_ok_ratio_in_config")

            if min_ok_ratio is not None and not 0 <= float(min_ok_ratio) <= 1:
                print("--min-ok-ratio musi mieścić się w zakresie 0-1", file=sys.stderr)
                return 2

            coverage_result = _coverage_payload(
                manifest_path=manifest_path,
                universe=universe,
                exchange=environment.exchange,
                as_of=as_of,
                symbols=args.symbols,
                intervals=args.intervals,
                max_gap_minutes=max_gap,
                min_ok_ratio=min_ok_ratio,
            )
            coverage_status = str(coverage_result.get("status", "unknown"))

    config_payload = {
        "valid": validation.is_valid(),
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
    }

    result_payload: dict[str, object] = {
        "status": "ok",
        "config": config_payload,
        "coverage": coverage_result,
        "coverage_warnings": coverage_warnings,
        "environment": environment.name,
        "manifest_path": str(manifest_path),
        "as_of": as_of.isoformat(),
    }

    final_status = "ok"
    exit_code = 0

    if not validation.is_valid():
        final_status = "error"
        exit_code = 2
    elif coverage_result is not None and coverage_status == "error":
        final_status = "error"
        exit_code = 3
    elif args.fail_on_warnings and (
        validation.warnings or coverage_status == "warning" or coverage_warnings
    ):
        final_status = "error"
        exit_code = 4
    elif validation.warnings or coverage_status == "warning" or coverage_warnings:
        final_status = "warning"

    result_payload["status"] = final_status
    result_payload["coverage_status"] = coverage_status

    if args.json or args.output:
        serialized = json.dumps(result_payload, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
        if args.json:
            print(serialized)

    if not args.json:
        print(f"Config: {'OK' if validation.is_valid() else 'ERROR'}")
        if validation.errors:
            for entry in validation.errors:
                print(f"  ✖ {entry}")
        if validation.warnings:
            for entry in validation.warnings:
                print(f"  ⚠ {entry}")

        if coverage_result is None:
            print("Pokrycie danych: pominięto (brak manifestu lub uniwersum)")
        else:
            summary = coverage_result.get("summary", {}) if isinstance(coverage_result, Mapping) else {}
            print(f"Pokrycie danych: {coverage_status}")
            if isinstance(summary, Mapping):
                total = summary.get("total", 0)
                ok = summary.get("ok", 0)
                error = summary.get("error", 0)
                print(f"  łącznie={total} ok={ok} błędy={error}")
                ok_ratio = summary.get("ok_ratio")
                if isinstance(ok_ratio, (int, float)):
                    print(f"  ok_ratio={float(ok_ratio):.4f}")
                manifest_counts = summary.get("manifest_status_counts", {})
                if isinstance(manifest_counts, Mapping) and manifest_counts:
                    counts = ", ".join(
                        f"{status}={count}" for status, count in sorted(manifest_counts.items())
                    )
                    print(f"  statusy manifestu: {counts}")
                worst_gap = summary.get("worst_gap")
                if isinstance(worst_gap, Mapping):
                    print(
                        "  największa luka: {symbol}/{interval} ({gap} min)".format(
                            symbol=worst_gap.get("symbol", "?"),
                            interval=worst_gap.get("interval", "?"),
                            gap=worst_gap.get("gap_minutes", "?"),
                        )
                    )
            issues = coverage_result.get("issues") if isinstance(coverage_result, Mapping) else None
            if issues:
                print("  Problemy:")
                for issue in issues:  # type: ignore[assignment]
                    print(f"    - {issue}")

        if coverage_warnings:
            print("Ostrzeżenia pokrycia:")
            for warning in coverage_warnings:
                print(f"  - {warning}")

        print(f"Status końcowy: {final_status.upper()}")

    return exit_code


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())

