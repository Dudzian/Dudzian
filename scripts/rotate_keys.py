"""Aktualizacja rotacji kluczy API oraz zapis podpisanego raportu Stage5."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig
from bot_core.security.rotation import RotationRegistry
from bot_core.security.rotation_report import RotationRecord, RotationSummary, write_rotation_summary


def _decode_key(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # pragma: no cover - walidacja wejścia CLI
        raise SystemExit(f"Niepoprawny klucz HMAC: {exc}") from exc


def _resolve_signing_key(args: argparse.Namespace) -> bytes | None:
    if args.signing_key:
        return _decode_key(args.signing_key)
    if args.signing_key_file:
        return _decode_key(Path(args.signing_key_file).read_text(encoding="utf-8"))
    if args.signing_key_env:
        return _decode_key(os.environ.get(args.signing_key_env))
    return None


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        now = datetime.now(timezone.utc)
        return now.replace(microsecond=0)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - walidacja wejścia CLI
        raise SystemExit(f"Niepoprawna data ISO8601: {value!r} ({exc})") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aktualizuje rejestr rotacji kluczy i zapisuje podpisany raport Stage5."
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku CoreConfig")
    parser.add_argument(
        "--environment",
        action="append",
        dest="environments",
        help="Nazwa środowiska do rotacji (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--interval-days",
        type=float,
        default=90.0,
        help="Interwał rotacji kluczy w dniach (domyślnie 90).",
    )
    parser.add_argument("--operator", required=True, help="Imię i nazwisko lub identyfikator operatora")
    parser.add_argument("--notes", help="Dodatkowe uwagi do raportu")
    parser.add_argument(
        "--artifact-root",
        default="var/audit/stage5/key_rotation",
        help="Katalog docelowy raportów rotacji",
    )
    parser.add_argument("--output", help="Pełna ścieżka pliku z raportem (opcjonalne)")
    parser.add_argument("--executed-at", help="Nadpisanie znacznika czasu rotacji (ISO8601)")
    parser.add_argument("--dry-run", action="store_true", help="Nie zapisuj zmian w rejestrze rotacji")
    parser.add_argument("--signing-key", help="Klucz HMAC (Base64)")
    parser.add_argument("--signing-key-file", help="Plik zawierający klucz HMAC (Base64)")
    parser.add_argument("--signing-key-env", help="Zmienna środowiskowa z kluczem HMAC (Base64)")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza podpisującego")
    return parser


def _resolve_environments(config: CoreConfig, names: Iterable[str] | None) -> list[EnvironmentConfig]:
    if not names:
        return list(config.environments.values())

    resolved: list[EnvironmentConfig] = []
    for name in names:
        try:
            resolved.append(config.environments[name])
        except KeyError as exc:  # pragma: no cover - walidacja CLI
            raise SystemExit(f"Środowisko '{name}' nie istnieje w konfiguracji") from exc
    return resolved


def _registry_path(environment: EnvironmentConfig) -> Path:
    return Path(environment.data_cache_path) / "security" / "rotation_log.json"


def _resolve_output_path(
    args: argparse.Namespace,
    *,
    executed_at: datetime,
    environments: Sequence[EnvironmentConfig],
) -> Path:
    if args.output:
        return Path(args.output)

    base = Path(args.artifact_root)
    timestamp = executed_at.strftime("%Y%m%dT%H%M%SZ")
    if len(environments) == 1:
        suffix = environments[0].name.replace("/", "-")
        filename = f"key_rotation_{suffix}_{timestamp}.json"
    else:
        filename = f"key_rotation_batch_{timestamp}.json"
    return base / filename


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    executed_at = _parse_datetime(args.executed_at)

    try:
        config = load_core_config(args.config)
    except FileNotFoundError:
        print(json.dumps({"error": f"Plik konfiguracji {args.config} nie istnieje"}))
        return 2

    environments = _resolve_environments(config, args.environments)
    if not environments:
        print(json.dumps({"error": "Brak środowisk do rotacji"}))
        return 1

    signing_key = _resolve_signing_key(args)

    records: list[RotationRecord] = []

    for environment in environments:
        purpose = getattr(environment, "credential_purpose", "trading")
        registry = RotationRegistry(_registry_path(environment))
        status = registry.status(
            environment.keychain_key,
            purpose,
            interval_days=args.interval_days,
            now=executed_at,
        )

        records.append(
            RotationRecord(
                environment=environment.name,
                key=environment.keychain_key,
                purpose=purpose,
                registry_path=_registry_path(environment),
                status_before=status,
                rotated_at=executed_at,
                interval_days=args.interval_days,
                metadata={"exchange": environment.exchange},
            )
        )

        if not args.dry_run:
            registry.mark_rotated(environment.keychain_key, purpose, timestamp=executed_at)

    summary = RotationSummary(
        operator=args.operator,
        executed_at=executed_at,
        records=records,
        notes=args.notes,
        metadata={"dry_run": bool(args.dry_run)},
    )

    output_path = _resolve_output_path(args, executed_at=executed_at, environments=environments)
    path = write_rotation_summary(
        summary,
        output=output_path,
        signing_key=signing_key,
        signing_key_id=args.signing_key_id,
    )

    print(
        json.dumps(
            {
                "output": str(path),
                "signed": bool(signing_key),
                "environments": [env.name for env in environments],
                "dry_run": bool(args.dry_run),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())

