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
    CoverageSummary,
    coerce_summary_mapping,
    evaluate_coverage,
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
    return parser.parse_args(argv)


def _parse_as_of(arg: str | None) -> datetime:
    if not arg:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(arg)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
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
    statuses = evaluate_coverage(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=environment.exchange,
        as_of=as_of,
    )

    # Filtrowanie po symbolach (opcjonalnie)
    if args.symbols:
        filter_tokens = [token.upper() for token in args.symbols]
        # mapowanie aliasów z nazw instrumentów na symbole giełdowe
        alias_map: dict[str, str] = {}
        for instrument in universe.instruments:
            symbol = instrument.exchange_symbols.get(environment.exchange)
            if symbol:
                alias_map[instrument.name.upper()] = symbol

        available_symbols: dict[str, str] = {s.symbol.upper(): s.symbol for s in statuses}

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
            print("Nieznane symbole: " + ", ".join(unknown), file=sys.stderr)
            return 2

        statuses = [status for status in statuses if status.symbol in resolved]
        if not statuses:
            print("Brak wpisów w manifeście dla wskazanych symboli.", file=sys.stderr)
            return 2

    # Filtrowanie po interwałach (opcjonalnie)
    if args.intervals:
        filter_tokens = [_normalize_interval_token(token) for token in args.intervals]
        available_map: dict[str, set[str]] = {}
        for status in statuses:
            normalized = _normalize_interval_token(status.interval)
            if not normalized:
                continue
            available_map.setdefault(normalized, set()).add(status.interval)

        unknown: list[str] = []
        resolved: set[str] = set()
        for raw, normalized in zip(args.intervals, filter_tokens, strict=True):
            if not normalized:
                unknown.append(raw)
                continue
            variants = available_map.get(normalized)
            if not variants:
                unknown.append(raw)
                continue
            resolved.update(variants)

        if unknown:
            print("Nieznane interwały: " + ", ".join(unknown), file=sys.stderr)
            return 2

        statuses = [status for status in statuses if status.interval in resolved]
        if not statuses:
            print("Brak wpisów w manifeście dla wskazanych interwałów.", file=sys.stderr)
            return 2

    issues = summarize_issues(statuses)
    summary_payload = coerce_summary_mapping(summarize_coverage(statuses))
    status_token = str(summary_payload.get("status") or ("ok" if not issues else "error"))
    payload = {
        "environment": environment.name,
        "exchange": environment.exchange,
        "manifest_path": str(manifest_path),
        "as_of": as_of.isoformat(),
        "entries": [_format_status(status) for status in statuses],
        "issues": issues,
        "summary": summary_payload,
        "status": status_token,
    }
    if threshold_payload is not None:
        payload["threshold_evaluation"] = threshold_payload
        payload["threshold_issues"] = list(threshold_issues)

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    # Zapis do pliku, jeśli wskazano --output
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
        summary = payload["summary"]
        for entry in payload["entries"]:
            print(
                " - {symbol} {interval}: status={status} row_count={row_count} "
                "required={required_rows} gap={gap_minutes}".format(**entry)
            )
        print(
            "Podsumowanie: status={status} ok={ok}/{total} warning={warning} "
            "error={error} stale_entries={stale_entries}".format(**summary)
        )
        issue_counts = summary.get("issue_counts") or {}
        issue_examples = summary.get("issue_examples") or {}
        if issue_counts:
            print("Kody problemów:")
            for code in sorted(issue_counts):
                count = issue_counts[code]
                example = issue_examples.get(code)
                if example:
                    print(f" * {code}: count={count} example={example}")
                else:
                    print(f" * {code}: count={count}")
        worst_gap = summary.get("worst_gap")
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
        if issues:
            print("Problemy:")
            for issue in issues:
                print(f" * {issue}")
        else:
            print("Brak problemów z pokryciem danych")

    return 0 if not issues and not threshold_issues else 1


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
