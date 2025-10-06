"""CLI do weryfikacji pokrycia danych OHLCV względem wymagań backfillu."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.config import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig
from bot_core.data.intervals import normalize_interval_token as _normalize_interval_token
from bot_core.data.ohlcv import (
    CoverageStatus,
    coerce_summary_mapping,
    compute_gap_statistics,
    compute_gap_statistics_by_interval,
    evaluate_coverage,
    summarize_coverage,
    summarize_issues,
)
from bot_core.data.ohlcv.coverage_check import (
    SummaryThresholdResult,
    evaluate_summary_thresholds,
    status_to_mapping,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprawdza manifest danych OHLCV dla środowiska paper/testnet.",
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        required=True,
        help="Nazwa środowiska z sekcji environments",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Znacznik czasu ISO8601 używany do oceny opóźnień danych (domyślnie teraz, UTC)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Zwróć wynik w formacie JSON (łatwy do integracji z CI)",
    )
    parser.add_argument(
        "--output",
        help=(
            "Ścieżka do pliku, w którym zostanie zapisany wynik w formacie JSON. "
            "Katalogi zostaną utworzone automatycznie."
        ),
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        default=None,
        help=(
            "Filtruj wynik do wskazanego symbolu (można podać wiele razy). "
            "Obsługiwane są zarówno nazwy instrumentów z konfiguracji (np. BTC_USDT), "
            "jak i symbole giełdowe (np. BTCUSDT)."
        ),
    )
    parser.add_argument(
        "--interval",
        dest="intervals",
        action="append",
        default=None,
        help=(
            "Filtruj wynik do wskazanych interwałów (np. 1d, 1h, D1). "
            "Można podać wiele razy; nazwy są nieczułe na wielkość liter i akceptują "
            "format zarówno 1d, jak i D1."
        ),
    )
    return parser.parse_args(argv)


def _parse_as_of(arg: str | None) -> datetime:
    if not arg:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(arg)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_thresholds(config: CoreConfig, environment: EnvironmentConfig) -> tuple[float | None, float | None]:
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


def _filter_statuses_by_symbols(
    statuses: Sequence[CoverageStatus],
    *,
    universe,
    exchange: str,
    requested: Sequence[str] | None,
) -> tuple[list[CoverageStatus], list[str]]:
    if not requested:
        return list(statuses), []

    filter_tokens = [token.upper() for token in requested]
    alias_map: dict[str, str] = {}
    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange)
        if symbol:
            alias_map[instrument.name.upper()] = symbol

    available_symbols: dict[str, str] = {status.symbol.upper(): status.symbol for status in statuses}

    resolved: set[str] = set()
    unknown: list[str] = []
    for token, raw in zip(filter_tokens, requested, strict=True):
        symbol = alias_map.get(token)
        if symbol is None:
            symbol = available_symbols.get(token)
        if symbol is None:
            unknown.append(raw)
            continue
        resolved.add(symbol)

    if unknown:
        return [], unknown

    filtered = [status for status in statuses if status.symbol in resolved]
    return filtered, []


def _filter_statuses_by_intervals(
    statuses: Sequence[CoverageStatus],
    *,
    requested: Sequence[str] | None,
) -> tuple[list[CoverageStatus], list[str]]:
    if not requested:
        return list(statuses), []

    filter_tokens = [_normalize_interval_token(token) for token in requested]
    available_map: dict[str, set[str]] = {}
    for status in statuses:
        normalized = _normalize_interval_token(status.interval)
        if not normalized:
            continue
        available_map.setdefault(normalized, set()).add(status.interval)

    resolved: set[str] = set()
    unknown: list[str] = []
    for raw, normalized in zip(requested, filter_tokens, strict=True):
        if not normalized:
            unknown.append(raw)
            continue
        variants = available_map.get(normalized)
        if not variants:
            unknown.append(raw)
            continue
        resolved.update(variants)

    if unknown:
        return [], unknown

    filtered = [status for status in statuses if status.interval in resolved]
    return filtered, []


def _extend_summary_with_breakdowns(
    summary_payload: Mapping[str, object],
    statuses: Sequence[CoverageStatus],
) -> dict[str, object]:
    payload = dict(summary_payload)

    def _breakdown(key_name: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for status in statuses:
            key = str(getattr(status, key_name, None) or "unknown")
            bucket = result.setdefault(key, {})
            bucket["total"] = bucket.get("total", 0) + 1
            state = str(getattr(status, "status", None) or "unknown")
            bucket[state] = bucket.get(state, 0) + 1
        return result

    payload["by_interval"] = _breakdown("interval")
    payload["by_symbol"] = _breakdown("symbol")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_core_config(Path(args.config))

    environment = config.environments.get(args.environment)
    if environment is None:
        print(f"Nie znaleziono środowiska: {args.environment}", file=sys.stderr)
        return 2

    if not environment.instrument_universe:
        print(
            f"Środowisko {environment.name} nie ma przypisanego instrument_universe",
            file=sys.stderr,
        )
        return 2

    universe = config.instrument_universes.get(environment.instrument_universe)
    if universe is None:
        print(
            f"Brak definicji uniwersum instrumentów: {environment.instrument_universe}",
            file=sys.stderr,
        )
        return 2

    as_of = _parse_as_of(args.as_of)
    manifest_path = Path(environment.data_cache_path) / "ohlcv_manifest.sqlite"

    statuses = list(
        evaluate_coverage(
            manifest_path=manifest_path,
            universe=universe,
            exchange_name=environment.exchange,
            as_of=as_of,
        )
    )

    statuses, unknown_symbols = _filter_statuses_by_symbols(
        statuses,
        universe=universe,
        exchange=environment.exchange,
        requested=args.symbols or None,
    )
    if unknown_symbols:
        print("Nieznane symbole: " + ", ".join(unknown_symbols), file=sys.stderr)
        return 2
    if not statuses:
        print("Brak wpisów w manifeście dla wskazanych symboli.", file=sys.stderr)
        return 2

    statuses, unknown_intervals = _filter_statuses_by_intervals(
        statuses,
        requested=args.intervals or None,
    )
    if unknown_intervals:
        print("Nieznane interwały: " + ", ".join(unknown_intervals), file=sys.stderr)
        return 2
    if not statuses:
        print("Brak wpisów w manifeście dla wskazanych interwałów.", file=sys.stderr)
        return 2

    issues = summarize_issues(statuses)
    summary = summarize_coverage(statuses)
    summary_payload = coerce_summary_mapping(summary)
    summary_payload = _extend_summary_with_breakdowns(summary_payload, statuses)

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

    payload: dict[str, object] = {
        "environment": environment.name,
        "exchange": environment.exchange,
        "manifest_path": str(manifest_path),
        "as_of": as_of.isoformat(),
        "entries": [status_to_mapping(status) for status in statuses],
        "issues": issues,
        "summary": summary_payload,
        "status": status_token,
        "threshold_issues": list(threshold_issues),
    }
    if threshold_payload is not None:
        payload["threshold_evaluation"] = threshold_payload

    gap_stats = compute_gap_statistics(statuses)
    payload["gap_statistics"] = gap_stats.to_mapping()
    interval_stats = compute_gap_statistics_by_interval(statuses)
    if interval_stats:
        payload["gap_statistics_by_interval"] = {
            interval: stats.to_mapping() for interval, stats in interval_stats.items()
        }

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    if args.json:
        print(serialized)
    else:
        print(f"Manifest: {payload['manifest_path']}")
        print(f"Środowisko: {payload['environment']} ({payload['exchange']})")
        print(f"Ocena na: {payload['as_of']}")
        for entry in payload["entries"]:
            print(
                " - {symbol} {interval}: status={status} row_count={row_count} "
                "required={required_rows} gap={gap_minutes}".format(**entry)
            )
        summary_map = payload["summary"]
        print(
            "Podsumowanie: status={status} ok={ok}/{total} warning={warning} "
            "error={error} stale_entries={stale_entries}".format(**summary_map)
        )
        issue_counts = summary_map.get("issue_counts") or {}
        issue_examples = summary_map.get("issue_examples") or {}
        if issue_counts:
            print("Kody problemów:")
            for code in sorted(issue_counts):
                count = issue_counts[code]
                example = issue_examples.get(code)
                if example:
                    print(f" * {code}: count={count} example={example}")
                else:
                    print(f" * {code}: count={count}")
        worst_gap = summary_map.get("worst_gap")
        if isinstance(worst_gap, Mapping):
            details = {
                "symbol": worst_gap.get("symbol", "?"),
                "interval": worst_gap.get("interval", "?"),
                "gap": worst_gap.get("gap_minutes"),
                "threshold": worst_gap.get("threshold_minutes"),
                "manifest_status": worst_gap.get("manifest_status"),
                "last": worst_gap.get("last_timestamp_iso"),
            }
            print(
                "Największa luka: {symbol} {interval} gap={gap}min "
                "threshold={threshold} manifest_status={manifest_status} last={last}".format(
                    **details
                )
            )
        if issues or threshold_issues:
            print("Problemy:")
            for issue in list(issues) + list(threshold_issues):
                print(f" * {issue}")
        else:
            print("Brak problemów z pokryciem danych")

    return 0 if not issues and not threshold_issues else 1


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
