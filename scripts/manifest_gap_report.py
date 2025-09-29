"""CLI raportujący stan manifestu SQLite dla danych OHLCV."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig, InstrumentUniverseConfig
from bot_core.data.ohlcv.manifest_report import generate_manifest_report, summarize_status


def _resolve_environment(config: CoreConfig, environment_name: str) -> EnvironmentConfig:
    try:
        return config.environments[environment_name]
    except KeyError as exc:
        raise SystemExit(f"Nie znaleziono środowiska '{environment_name}' w konfiguracji") from exc


def _resolve_universe(config: CoreConfig, environment: EnvironmentConfig) -> InstrumentUniverseConfig:
    if not environment.instrument_universe:
        raise SystemExit(
            "Środowisko nie ma przypisanego instrument_universe – uzupełnij config/core.yaml."
        )
    try:
        return config.instrument_universes[environment.instrument_universe]
    except KeyError as exc:
        raise SystemExit(
            f"Uniwersum '{environment.instrument_universe}' nie istnieje w konfiguracji."
        ) from exc


def _parse_as_of(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SystemExit(
            "Parametr --as-of musi być w formacie ISO 8601, np. 2024-05-18T12:00:00Z"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_threshold_overrides(environment: EnvironmentConfig) -> Mapping[str, int]:
    settings = environment.adapter_settings or {}
    raw_policy = settings.get("ohlcv_gap_alerts") if isinstance(settings, Mapping) else None
    if not isinstance(raw_policy, Mapping):
        return {}
    warning_cfg = raw_policy.get("warning_gap_minutes")
    if not isinstance(warning_cfg, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, value in warning_cfg.items():
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            continue
        if minutes > 0:
            result[str(key)] = minutes
    return result


def _format_table(entries) -> str:
    headers = [
        "Symbol",
        "Interwał",
        "Wiersze",
        "Ostatni timestamp (UTC)",
        "Luka [min]",
        "Próg [min]",
        "Status",
    ]
    rows = []
    for entry in entries:
        rows.append(
            [
                entry.symbol,
                entry.interval,
                "-" if entry.row_count is None else str(entry.row_count),
                entry.last_timestamp_iso or "-",
                "-" if entry.gap_minutes is None else f"{entry.gap_minutes:.1f}",
                "-" if entry.threshold_minutes is None else str(entry.threshold_minutes),
                entry.status,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    parts = [
        _format_row(headers),
        "-+-".join("-" * width for width in widths),
    ]
    parts.extend(_format_row(row) for row in rows)
    return "\n".join(parts)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raport manifestu OHLCV z SQLite")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument("--environment", required=True, help="Środowisko do analizy")
    parser.add_argument(
        "--manifest-path",
        help="Ścieżka do pliku manifestu. Domyślnie data_cache_path/ohlcv_manifest.sqlite",
    )
    parser.add_argument(
        "--as-of",
        help="Czas odniesienia w ISO 8601 (domyślnie teraz w UTC)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Zwróć wynik w formacie JSON zamiast tabeli tekstowej",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_core_config(args.config)
    environment = _resolve_environment(config, args.environment)
    universe = _resolve_universe(config, environment)

    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else Path(environment.data_cache_path) / "ohlcv_manifest.sqlite"
    )

    if not manifest_path.exists():
        raise SystemExit(f"Plik manifestu {manifest_path} nie istnieje")

    overrides = _extract_threshold_overrides(environment)
    as_of = _parse_as_of(args.as_of)
    entries = generate_manifest_report(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=environment.exchange,
        as_of=as_of,
        warning_thresholds=overrides,
    )

    if args.json:
        payload = [entry.__dict__ for entry in entries]
        print(json.dumps({"entries": payload, "summary": summarize_status(entries)}, indent=2, ensure_ascii=False))
        return 0

    if not entries:
        print("Brak instrumentów do raportowania dla wybranego uniwersum")
        return 0

    table = _format_table(entries)
    summary = summarize_status(entries)
    print(table)
    print()
    print("Podsumowanie statusów:")
    for status, count in sorted(summary.items()):
        print(f"- {status}: {count}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
