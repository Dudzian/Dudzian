"""Audyt konfiguracji tokenów usługowych dla MetricsService i RiskService."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - opcjonalny moduł konfiguracji
    from bot_core.config import load_core_config  # type: ignore
except Exception:  # pragma: no cover - środowisko minimalne
    load_core_config = None  # type: ignore

from bot_core.security.token_audit import audit_service_tokens
from scripts._cli_common import env_flag, env_value, normalize_scopes, should_print
from scripts._json_utils import dump_json

LOGGER = logging.getLogger("bot_core.scripts.audit_service_tokens")
_ENV_PREFIX = "BOT_CORE_TOKEN_AUDIT_"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Ścieżka do core.yaml (domyślnie config/core.yaml lub zmienna środowiskowa)",
    )
    parser.add_argument(
        "--json-output",
        help="Zapisz raport audytu w pliku JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Formatuj JSON z wcięciami i sortowaniem kluczy",
    )
    parser.add_argument(
        "--print",
        dest="print_stdout",
        action="store_true",
        help="Wypisz raport na stdout (domyślne jeśli brak pliku wyjściowego)",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Zwróć kod 1 gdy wykryto ostrzeżenia",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Zwróć kod 2 gdy wykryto błędy",
    )
    parser.add_argument(
        "--allow-shared-secret",
        action="store_true",
        help="Nie traktuj statycznego auth_token jako ostrzeżenia",
    )
    parser.add_argument(
        "--metrics-scope",
        action="append",
        help="Wymagany scope dla MetricsService (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--risk-scope",
        action="append",
        help="Wymagany scope dla RiskService (można podać wielokrotnie)",
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


def _should_allow_shared_secret(args: argparse.Namespace) -> bool:
    if args.allow_shared_secret:
        return True
    return env_flag(_ENV_PREFIX, "ALLOW_SHARED_SECRET", False)


def _resolve_scopes(args: argparse.Namespace, *, kind: str) -> tuple[str, ...]:
    env_override = env_value(_ENV_PREFIX, f"{kind.upper()}_SCOPES")
    from_args = args.metrics_scope if kind == "metrics" else args.risk_scope
    combined: list[str] = []
    if env_override:
        combined.extend(part.strip() for part in env_override.split(",") if part.strip())
    if from_args:
        combined.extend(from_args)
    scopes = normalize_scopes(combined)
    if scopes:
        return scopes
    if kind == "metrics":
        return ("metrics.read",)
    if kind == "risk":
        return ("risk.read",)
    return tuple()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    config_path = _resolve_config_path(args)
    json_output = _resolve_json_output(args)
    pretty = args.pretty or env_flag(_ENV_PREFIX, "PRETTY", False)
    print_stdout = _should_print(args)
    fail_on_warning = _should_fail_on_warning(args)
    fail_on_error = _should_fail_on_error(args)
    allow_shared_secret = _should_allow_shared_secret(args)
    metrics_scopes = _resolve_scopes(args, kind="metrics")
    risk_scopes = _resolve_scopes(args, kind="risk")

    if load_core_config is None:
        raise RuntimeError("Moduł konfiguracji nie jest dostępny – zainstaluj bot_core.config")

    if not config_path.exists():
        raise FileNotFoundError(f"Plik konfiguracji '{config_path}' nie istnieje")

    LOGGER.info("Audytuję tokeny usługowe na podstawie %s", config_path)
    core_config = load_core_config(str(config_path))
    report = audit_service_tokens(
        core_config,
        env=os.environ,
        metrics_required_scopes=metrics_scopes,
        risk_required_scopes=risk_scopes,
        warn_on_shared_secret=not allow_shared_secret,
    )

    payload = report.as_dict()
    serialized = dump_json(payload, pretty=pretty)

    if json_output:
        output_path = Path(json_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    if print_stdout or not json_output:
        print(serialized)

    warnings = payload.get("warnings") or []
    errors = payload.get("errors") or []

    exit_code = 0
    if fail_on_error and errors:
        exit_code = 2
    elif fail_on_warning and (warnings or errors):
        exit_code = 1 if exit_code == 0 else exit_code

    return exit_code


if __name__ == "__main__":  # pragma: no cover - obsługa CLI
    raise SystemExit(main())
