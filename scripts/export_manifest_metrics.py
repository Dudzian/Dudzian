from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Sequence

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig, InstrumentUniverseConfig
from bot_core.data.ohlcv import (
    ManifestMetricsExporter,
    generate_manifest_report,
    summarize_status,
)
from bot_core.security.signing import build_hmac_signature
from bot_core.observability.metrics import MetricsRegistry


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eksportuje metryki z manifestu OHLCV do formatu Prometheusa.",
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument("--environment", required=True, help="Środowisko z konfiguracji")
    parser.add_argument(
        "--manifest-path",
        help="Ścieżka do pliku manifestu. Domyślnie <data_cache_path>/ohlcv_manifest.sqlite",
    )
    parser.add_argument(
        "--as-of",
        help="Znacznik czasu ISO8601 używany do oceny luk danych (domyślnie teraz w UTC)",
    )
    parser.add_argument(
        "--output",
        help="Plik z wyeksportowanymi metrykami w formacie Prometheusa",
    )
    parser.add_argument(
        "--summary-output",
        help="Plik JSON z podsumowaniem statusów manifestu",
    )
    parser.add_argument(
        "--stage",
        help="Etykieta etapu pipeline'u (domyślnie wartość pola environment)",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        default=None,
        help="Ogranicz raport do wskazanych symboli (można podać wiele razy).",
    )
    parser.add_argument(
        "--interval",
        dest="intervals",
        action="append",
        default=None,
        help="Ogranicz raport do wskazanych interwałów (np. 1h, 1d).",
    )
    parser.add_argument(
        "--deny-status",
        dest="deny_status",
        action="append",
        default=None,
        help="Zwróć kod wyjścia 2 jeśli pojawi się dany status (można podać wiele razy).",
    )
    parser.add_argument(
        "--summary-hmac-key",
        dest="summary_hmac_key",
        help="Klucz HMAC (UTF-8) używany do podpisu JSON podsumowania manifestu.",
    )
    parser.add_argument(
        "--summary-hmac-key-file",
        dest="summary_hmac_key_file",
        help="Plik zawierający klucz HMAC do podpisu podsumowania manifestu.",
    )
    parser.add_argument(
        "--summary-hmac-key-env",
        dest="summary_hmac_key_env",
        help="Nazwa zmiennej środowiskowej z kluczem HMAC do podpisu podsumowania.",
    )
    parser.add_argument(
        "--summary-hmac-key-id",
        dest="summary_hmac_key_id",
        help="Opcjonalny identyfikator klucza HMAC do zapisania w podpisie.",
    )
    parser.add_argument(
        "--require-summary-signature",
        action="store_true",
        help="Zakończ działanie błędem, jeśli nie wygenerowano podpisu podsumowania.",
    )
    return parser.parse_args(argv)


def _load_summary_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    """Zwraca klucz HMAC i identyfikator do podpisu podsumowania manifestu."""

    provided = [
        bool(args.summary_hmac_key),
        bool(args.summary_hmac_key_file),
        bool(args.summary_hmac_key_env),
    ]
    if sum(provided) > 1:
        print(
            "Błąd: opcje --summary-hmac-key, --summary-hmac-key-file i --summary-hmac-key-env "
            "są wzajemnie wykluczające.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    key_material: str | None = None
    if args.summary_hmac_key:
        key_material = args.summary_hmac_key
    elif args.summary_hmac_key_file:
        try:
            key_material = (
                Path(args.summary_hmac_key_file).expanduser().read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            print(
                f"Nie znaleziono pliku z kluczem HMAC: {args.summary_hmac_key_file}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        except OSError as exc:  # pragma: no cover - zależne od platformy
            print(
                f"Nie udało się odczytać klucza HMAC z {args.summary_hmac_key_file}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
    elif args.summary_hmac_key_env:
        env_value = os.getenv(args.summary_hmac_key_env)
        if env_value is None:
            print(
                f"Zmienna środowiskowa {args.summary_hmac_key_env} nie zawiera klucza HMAC",
                file=sys.stderr,
            )
            raise SystemExit(2)
        key_material = env_value

    if key_material is None:
        if args.require_summary_signature:
            print(
                "Wymagano podpisania podsumowania manifestu, ale nie dostarczono klucza HMAC.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return None, None

    key_bytes = key_material.strip().encode("utf-8")
    if not key_bytes:
        print("Klucz HMAC podsumowania manifestu nie może być pusty", file=sys.stderr)
        raise SystemExit(2)

    if len(key_bytes) < 16:  # pragma: no cover - ostrzeżenie nie wpływa na testy
        print(
            "Ostrzeżenie: klucz HMAC ma mniej niż 16 bajtów – rozważ użycie dłuższego klucza.",
            file=sys.stderr,
        )

    return key_bytes, (args.summary_hmac_key_id or None)


def _resolve_environment(config: CoreConfig, name: str) -> EnvironmentConfig:
    try:
        return config.environments[name]
    except KeyError as exc:  # pragma: no cover - defensywne
        raise SystemExit(f"Środowisko '{name}' nie istnieje w konfiguracji") from exc


def _resolve_universe(config: CoreConfig, environment: EnvironmentConfig) -> InstrumentUniverseConfig:
    universe_name = environment.instrument_universe
    if not universe_name:
        raise SystemExit("Środowisko nie ma przypisanego instrument_universe – uzupełnij konfigurację.")
    try:
        return config.instrument_universes[universe_name]
    except KeyError as exc:  # pragma: no cover - defensywne
        raise SystemExit(f"Uniwersum '{universe_name}' nie istnieje w konfiguracji") from exc


def _parse_as_of(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _filter_entries(entries, symbols: Sequence[str] | None, intervals: Sequence[str] | None):
    filtered = list(entries)
    if symbols:
        allowed = {symbol.upper() for symbol in symbols}
        filtered = [entry for entry in filtered if entry.symbol.upper() in allowed]
    if intervals:
        allowed_intervals = {interval.lower() for interval in intervals}
        filtered = [entry for entry in filtered if entry.interval.lower() in allowed_intervals]
    return filtered


def main(argv: Sequence[str] | None = None) -> int:
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

    as_of = _parse_as_of(args.as_of)
    entries = generate_manifest_report(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=environment.exchange,
        as_of=as_of,
    )

    entries = _filter_entries(entries, args.symbols, args.intervals)

    registry = MetricsRegistry()
    exporter = ManifestMetricsExporter(
        registry=registry,
        environment=args.environment,
        exchange=environment.exchange,
        stage=args.stage or environment.environment.value,
        risk_profile=environment.risk_profile,
    )
    summary = exporter.observe(entries)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(registry.render_prometheus(), encoding="utf-8")
    else:
        print(registry.render_prometheus(), end="")

    status_summary = summarize_status(entries)
    json_payload = {
        "status_counts": status_summary,
        "total_entries": summary["total_entries"],
        "worst_status": summary["worst_status"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "environment": args.environment,
        "exchange": environment.exchange,
    }

    signing_key, signing_key_id = _load_summary_signing_key(args)

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if signing_key:
            json_payload["summary_signature"] = build_hmac_signature(
                {key: value for key, value in json_payload.items() if key != "summary_signature"},
                key=signing_key,
                key_id=signing_key_id,
            )
        summary_path.write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        payload = dict(json_payload)
        if signing_key:
            payload["summary_signature"] = build_hmac_signature(
                {key: value for key, value in payload.items() if key != "summary_signature"},
                key=signing_key,
                key_id=signing_key_id,
            )
        print(json.dumps(payload, ensure_ascii=False))

    denied = {status.lower() for status in (args.deny_status or ())}
    if any(status.lower() in denied for status in status_summary.keys()):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
