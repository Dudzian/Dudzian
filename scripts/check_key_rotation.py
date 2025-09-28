"""Sprawdza terminy rotacji kluczy API na podstawie rejestrów środowisk."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig
from bot_core.security.rotation import RotationRegistry


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Weryfikuje terminy rotacji kluczy API i raportuje wpisy wymagające odświeżenia."
        )
    )
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku konfiguracji CoreConfig",
    )
    parser.add_argument(
        "--environment",
        action="append",
        dest="environments",
        help="Nazwa środowiska do sprawdzenia (można podać wiele parametrów).",
    )
    parser.add_argument(
        "--interval-days",
        type=float,
        default=90.0,
        help="Docelowy interwał rotacji kluczy w dniach (domyślnie 90).",
    )
    parser.add_argument(
        "--warn-days",
        type=float,
        default=14.0,
        help="Liczba dni przed terminem, przy której zgłaszamy ostrzeżenie (domyślnie 14).",
    )
    parser.add_argument(
        "--mark-rotated",
        action="store_true",
        help="Zapisz bieżącą datę jako nową rotację dla sprawdzanych środowisk.",
    )
    return parser.parse_args(argv)


def _resolve_environments(config: CoreConfig, names: Iterable[str] | None) -> list[EnvironmentConfig]:
    if not names:
        return list(config.environments.values())

    resolved: list[EnvironmentConfig] = []
    for name in names:
        try:
            resolved.append(config.environments[name])
        except KeyError as exc:
            raise SystemExit(f"Środowisko '{name}' nie istnieje w konfiguracji") from exc
    return resolved


def _registry_path(environment: EnvironmentConfig) -> Path:
    return Path(environment.data_cache_path) / "security" / "rotation_log.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        core_config = load_core_config(args.config)
    except FileNotFoundError:
        print(f"Plik konfiguracji {args.config} nie istnieje", file=sys.stderr)
        return 2

    environments = _resolve_environments(core_config, args.environments)
    if not environments:
        print("Brak środowisk do sprawdzenia", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    exit_code = 0

    for environment in environments:
        purpose = getattr(environment, "credential_purpose", "trading")
        registry = RotationRegistry(_registry_path(environment))
        status = registry.status(
            environment.keychain_key,
            purpose,
            interval_days=args.interval_days,
            now=now,
        )

        last_rotated = (
            status.last_rotated.isoformat().replace("+00:00", "Z")
            if status.last_rotated
            else "nigdy"
        )
        days_since = (
            f"{status.days_since_rotation:.1f}" if status.days_since_rotation is not None else "n/d"
        )
        due_in = f"{status.due_in_days:.1f}"

        print(
            f"[{environment.name}] key={environment.keychain_key} purpose={purpose} "
            f"last_rotated={last_rotated} days_since={days_since} due_in={due_in}"
        )

        if status.is_overdue:
            print("  ⚠️  Rotacja jest przeterminowana – wymagana natychmiastowa wymiana klucza.")
            exit_code = max(exit_code, 2)
        elif status.is_due or status.due_in_days <= args.warn_days:
            print("  ⚠️  Zbliża się termin rotacji – zaplanuj wymianę klucza.")
            exit_code = max(exit_code, 1)
        else:
            print("  ✅  Rotacja w bezpiecznym oknie.")

        if args.mark_rotated:
            registry.mark_rotated(environment.keychain_key, purpose, timestamp=now)
            print("  ℹ️  Zapisano bieżącą datę jako nową rotację.")

    return exit_code


if __name__ == "__main__":  # pragma: no cover - uruchomienie z CLI
    raise SystemExit(main())

