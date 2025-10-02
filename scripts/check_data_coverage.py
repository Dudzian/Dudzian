"""CLI do weryfikacji pokrycia danych OHLCV względem wymagań backfillu."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.config import load_core_config
from bot_core.data.intervals import normalize_interval_token as _normalize_interval_token
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
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=None,
        help=(
            "Maksymalna dopuszczalna luka czasowa (minuty) w danych. Jeśli największa luka "
            "przekroczy ten próg, komenda zakończy się kodem błędu."
        ),
    )
    parser.add_argument(
        "--min-ok-ratio",
        type=float,
        default=None,
        help=(
            "Minimalny udział poprawnych wpisów manifestu (0-1). Gdy ok_ratio spadnie poniżej "
            "progu, komenda zakończy się kodem błędu."
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


def _format_status(status: CoverageStatus) -> dict[str, object]:
    entry = status.manifest_entry
    return {
        "symbol": status.symbol,
        "interval": status.interval,
        "status": status.status,
        "manifest_status": entry.status,
        "row_count": entry.row_count,
        "required_rows": status.required_rows,
        "last_timestamp_iso": entry.last_timestamp_iso,
        "gap_minutes": entry.gap_minutes,
        "issues": list(status.issues),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_core_config(Path(args.config))

    environment = config.environments.get(args.environment)
    if environment is None:
        print(f"Nie znaleziono środowiska: {args.environment}", file=sys.stderr)
        return 2

    if not environment.instrument_universe:
        print(
            f"Środowisko {environment.name} nie ma przypisanego instrument_universe", file=sys.stderr
        )
        return 2

    universe = config.instrument_universes.get(environment.instrument_universe)
    if universe is None:
        print(
            f"Brak definicji uniwersum instrumentów: {environment.instrument_universe}",
            file=sys.stderr,
        )
        return 2

    data_quality = getattr(environment, "data_quality", None)
    max_gap_minutes = args.max_gap_minutes
    if max_gap_minutes is None and data_quality is not None:
        candidate = getattr(data_quality, "max_gap_minutes", None)
        if candidate is not None:
            try:
                max_gap_minutes = float(candidate)
            except (TypeError, ValueError):
                print(
                    "Nieprawidłowa wartość data_quality.max_gap_minutes w konfiguracji środowiska.",
                    file=sys.stderr,
                )
                return 2

    min_ok_ratio = args.min_ok_ratio
    if min_ok_ratio is None and data_quality is not None:
        candidate = getattr(data_quality, "min_ok_ratio", None)
        if candidate is not None:
            try:
                min_ok_ratio = float(candidate)
            except (TypeError, ValueError):
                print(
                    "Nieprawidłowa wartość data_quality.min_ok_ratio w konfiguracji środowiska.",
                    file=sys.stderr,
                )
                return 2

    if min_ok_ratio is not None and not 0 <= float(min_ok_ratio) <= 1:
        print("--min-ok-ratio musi zawierać się w przedziale 0-1", file=sys.stderr)
        return 2

    as_of = _parse_as_of(args.as_of)
    manifest_path = Path(environment.data_cache_path) / "ohlcv_manifest.sqlite"
    interval_filters: list[tuple[str, str]] | None = None
    interval_tokens: list[str] | None = None
    if args.intervals:
        interval_filters = []
        invalid: list[str] = []
        for raw in args.intervals:
            normalized = _normalize_interval_token(raw)
            if not normalized:
                invalid.append(raw)
                continue
            interval_filters.append((raw, normalized))
        if invalid:
            print("Nieznane interwały: " + ", ".join(invalid), file=sys.stderr)
            return 2
        if not interval_filters:
            print("Brak poprawnych interwałów w filtrze.", file=sys.stderr)
            return 2
        interval_tokens = [normalized for _, normalized in interval_filters]

    statuses = evaluate_coverage(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=environment.exchange,
        as_of=as_of,
        intervals=interval_tokens,
    )

    if args.symbols:
        filter_tokens = [token.upper() for token in args.symbols]
        alias_map: dict[str, str] = {}
        for instrument in universe.instruments:
            symbol = instrument.exchange_symbols.get(environment.exchange)
            if symbol:
                alias_map[instrument.name.upper()] = symbol

        available_symbols: dict[str, str] = {
            status.symbol.upper(): status.symbol for status in statuses
        }

        resolved: set[str] = set()
        unknown: list[str] = []
        for token, raw in zip(filter_tokens, args.symbols, strict=True):
            symbol = alias_map.get(token)
            if symbol is None:
                symbol = available_symbols.get(token)
            if symbol is None:
                unknown.append(raw)
                continue
            resolved.add(symbol)

        if unknown:
            print(
                "Nieznane symbole: " + ", ".join(unknown),
                file=sys.stderr,
            )
            return 2

        statuses = [status for status in statuses if status.symbol in resolved]
        if not statuses:
            print(
                "Brak wpisów w manifeście dla wskazanych symboli.",
                file=sys.stderr,
            )
            return 2

    if interval_filters is not None:
        available_map: dict[str, set[str]] = {}
        for status in statuses:
            normalized = _normalize_interval_token(status.interval)
            if not normalized:
                continue
            available_map.setdefault(normalized, set()).add(status.interval)

        resolved: set[str] = set()
        missing: list[str] = []
        for raw, normalized in interval_filters:
            variants = available_map.get(normalized)
            if not variants:
                missing.append(raw)
                continue
            resolved.update(variants)

        if missing:
            print(
                "Brak wpisów w manifeście dla wskazanych interwałów: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 2

        statuses = [status for status in statuses if status.interval in resolved]

    issues = summarize_issues(statuses)
    summary_payload = dict(summarize_coverage(statuses))
    interval_breakdown = summarize_by_interval(statuses)
    if interval_breakdown:
        summary_payload["by_interval"] = interval_breakdown
    symbol_breakdown = summarize_by_symbol(statuses)
    if symbol_breakdown:
        summary_payload["by_symbol"] = symbol_breakdown
    payload = {
        "environment": environment.name,
        "exchange": environment.exchange,
        "manifest_path": str(manifest_path),
        "as_of": as_of.isoformat(),
        "entries": [_format_status(status) for status in statuses],
        "summary": summary_payload,
    }

    if max_gap_minutes is not None:
        threshold_value = float(max_gap_minutes)
        payload.setdefault("thresholds", {})["max_gap_minutes"] = threshold_value
        summary_payload.setdefault("thresholds", {})["max_gap_minutes"] = threshold_value

    worst_gap = summary_payload.get("worst_gap") if isinstance(summary_payload, dict) else None
    if max_gap_minutes is not None and isinstance(worst_gap, dict):
        try:
            worst_gap_minutes = float(worst_gap.get("gap_minutes", 0.0))
        except (TypeError, ValueError):
            worst_gap_minutes = 0.0
        if worst_gap_minutes > float(max_gap_minutes):
            issues.append(
                f"max_gap_exceeded:{worst_gap_minutes}>{float(max_gap_minutes)}"
            )
            issues.append(
                f"Największa luka ({worst_gap_minutes:.2f} min) przekracza próg {float(max_gap_minutes):.2f} min"
            )

    if min_ok_ratio is not None:
        threshold_ratio = float(min_ok_ratio)
        payload.setdefault("thresholds", {})["min_ok_ratio"] = threshold_ratio
        summary_payload.setdefault("thresholds", {})["min_ok_ratio"] = threshold_ratio
        ok_ratio = summary_payload.get("ok_ratio")
        if ok_ratio is not None:
            try:
                ok_ratio_value = float(ok_ratio)
            except (TypeError, ValueError):
                ok_ratio_value = None
            if ok_ratio_value is not None and ok_ratio_value < threshold_ratio:
                issues.append(
                    f"ok_ratio_below_threshold:{ok_ratio_value:.4f}<{threshold_ratio:.4f}"
                )
                issues.append(
                    "Udział poprawnych wpisów (ok_ratio) spadł poniżej wymaganego progu"
                )
        else:
            total_entries = summary_payload.get("total")
            if isinstance(total_entries, (int, float)) and threshold_ratio > 0:
                if float(total_entries) <= 0:
                    issues.append(
                        "Brak wpisów w manifeście uniemożliwia ocenę ok_ratio przy aktywnym progu"
                    )

    payload["issues"] = issues
    payload["status"] = "error" if issues else summary_payload.get("status", "ok")

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
        print(
            "Podsumowanie: łącznie={total} OK={ok} błędy={error}".format(
                **summary_payload
            )
        )
        if "ok_ratio" in summary_payload:
            try:
                ratio_value = float(summary_payload["ok_ratio"])
            except (TypeError, ValueError):
                ratio_value = summary_payload["ok_ratio"]
                print(f"ok_ratio: {ratio_value}")
            else:
                print(f"ok_ratio: {ratio_value:.4f}")
        manifest_counts = summary_payload.get("manifest_status_counts", {})
        if manifest_counts:
            counts_str = ", ".join(
                f"{status}={count}" for status, count in sorted(manifest_counts.items())
            )
            print(f"Statusy manifestu: {counts_str}")
        interval_breakdown = summary_payload.get("by_interval")
        if isinstance(interval_breakdown, Mapping) and interval_breakdown:
            print("Rozkład interwałów:")
            for interval_name, payload_interval in sorted(interval_breakdown.items()):
                total = payload_interval.get("total", 0)
                ok = payload_interval.get("ok", 0)
                error = payload_interval.get("error", 0)
                ratio = payload_interval.get("ok_ratio")
                if isinstance(ratio, (int, float)):
                    ratio_text = f"{float(ratio):.4f}"
                else:
                    ratio_text = str(ratio) if ratio is not None else "n/a"
                print(
                    f" * {interval_name}: OK={ok} błędy={error} łącznie={total} ok_ratio={ratio_text}"
                )
        symbol_breakdown = summary_payload.get("by_symbol")
        if isinstance(symbol_breakdown, Mapping) and symbol_breakdown:
            problematic = [
                (symbol, payload_symbol)
                for symbol, payload_symbol in symbol_breakdown.items()
                if (payload_symbol.get("error") or 0)
            ]
            if problematic:
                print("Symbole z problemami:")
                for symbol, payload_symbol in sorted(problematic):
                    total = payload_symbol.get("total", 0)
                    error = payload_symbol.get("error", 0)
                    ratio = payload_symbol.get("ok_ratio")
                    if isinstance(ratio, (int, float)):
                        ratio_text = f"{float(ratio):.4f}"
                    else:
                        ratio_text = str(ratio) if ratio is not None else "n/a"
                    print(
                        f" * {symbol}: błędy={error} / {total} wpisów (ok_ratio={ratio_text})"
                    )
        worst_gap = summary_payload.get("worst_gap")
        if isinstance(worst_gap, dict):
            print(
                "Największa luka: {symbol}/{interval} ({gap_minutes} min)".format(
                    symbol=worst_gap.get("symbol", "?"),
                    interval=worst_gap.get("interval", "?"),
                    gap_minutes=worst_gap.get("gap_minutes", "?"),
                )
            )
        for entry in payload["entries"]:
            print(
                " - {symbol} {interval}: status={status} row_count={row_count} required={required_rows} gap={gap_minutes}".format(
                    **entry
                )
            )
        if issues:
            print("Problemy:")
            for issue in issues:
                print(f" * {issue}")
        else:
            print("Brak problemów z pokryciem danych")

    return 0 if not issues else 1


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
