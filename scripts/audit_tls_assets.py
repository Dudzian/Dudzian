"""Audyt TLS dla usług runtime (MetricsService, RiskService)."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:  # pragma: no cover - opcjonalny moduł konfiguracji
    from bot_core.config import load_core_config  # type: ignore
except Exception:  # pragma: no cover - minimalne środowiska testowe
    load_core_config = None  # type: ignore

from bot_core.security.tls_audit import audit_tls_assets
from scripts._cli_common import env_flag, env_value, should_print
from scripts._json_utils import dump_json

LOGGER = logging.getLogger("bot_core.scripts.audit_tls_assets")

_ENV_PREFIX = "BOT_CORE_TLS_AUDIT_"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Ścieżka do core.yaml (domyślnie config/core.yaml lub zmienna środowiskowa)",
    )
    parser.add_argument(
        "--json-output",
        help="Zapisz wynik audytu w pliku JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Formatuj JSON z wcięciami i sortowaniem kluczy",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Zwróć kod zakończenia 1 gdy pojawią się ostrzeżenia",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Zwróć kod zakończenia 2 gdy wykryto błędy",
    )
    parser.add_argument(
        "--warn-expiring-days",
        type=float,
        default=None,
        help="Liczba dni do wygaśnięcia certyfikatu powodująca ostrzeżenie (domyślnie 30)",
    )
    parser.add_argument(
        "--print",
        dest="print_stdout",
        action="store_true",
        help="Wypisz raport na stdout (domyślne jeśli brak pliku wyjściowego)",
    )
    return parser.parse_args(argv)


def _resolve_config_path(args: argparse.Namespace) -> Path:
    env_override = env_value(_ENV_PREFIX, "CONFIG")
    if args.config:
        return Path(args.config).expanduser()
    if env_override:
        return Path(env_override).expanduser()
    return Path("config/core.yaml")


def _resolve_json_output(args: argparse.Namespace) -> str | None:
    if args.json_output:
        return args.json_output
    return env_value(_ENV_PREFIX, "JSON_OUTPUT")


def _resolve_warn_days(args: argparse.Namespace) -> float:
    if args.warn_expiring_days is not None:
        return float(args.warn_expiring_days)
    env_override = env_value(_ENV_PREFIX, "WARN_EXPIRING_DAYS")
    if env_override is not None:
        try:
            return float(env_override)
        except ValueError:  # pragma: no cover - diagnostyka konfiguracji
            LOGGER.warning("Nieprawidłowa wartość WARN_EXPIRING_DAYS='%s'", env_override)
    return 30.0


def _should_print(args: argparse.Namespace) -> bool:
    return should_print(
        _ENV_PREFIX,
        json_output=args.json_output,
        cli_flag=args.print_stdout,
        default_when_unspecified=not bool(args.json_output),
    )


def _should_fail_on_warning(args: argparse.Namespace) -> bool:
    if args.fail_on_warning:
        return True
    return env_flag(_ENV_PREFIX, "FAIL_ON_WARNING", False)


def _should_fail_on_error(args: argparse.Namespace) -> bool:
    if args.fail_on_error:
        return True
    return env_flag(_ENV_PREFIX, "FAIL_ON_ERROR", False)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    config_path = _resolve_config_path(args)
    json_output = _resolve_json_output(args)
    warn_days = _resolve_warn_days(args)
    pretty = args.pretty or env_flag(_ENV_PREFIX, "PRETTY", False)
    print_stdout = _should_print(args)
    fail_on_warning = _should_fail_on_warning(args)
    fail_on_error = _should_fail_on_error(args)

    if load_core_config is None:
        raise RuntimeError("Moduł konfiguracji nie jest dostępny – zainstaluj zależność bot_core.config")

    if not config_path.exists():
        raise FileNotFoundError(f"Plik konfiguracji '{config_path}' nie istnieje")

    core_config = load_core_config(str(config_path))
    report = audit_tls_assets(core_config, warn_expiring_within_days=warn_days)

    serialized = dump_json(report, pretty=pretty)

    if json_output:
        output_path = Path(json_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    if print_stdout or not json_output:
        print(serialized)

    exit_code = 0
    if fail_on_error and report.get("errors"):
        exit_code = 2
    elif fail_on_warning and (report.get("warnings") or report.get("errors")):
        exit_code = 1 if exit_code == 0 else exit_code
    return exit_code


if __name__ == "__main__":  # pragma: no cover - obsługa CLI
    raise SystemExit(main())
