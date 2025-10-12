"""Audyt zbiorczy bezpieczeństwa (TLS + RBAC) dla usług runtime."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - opcjonalne środowiska testowe
    from bot_core.config import load_core_config  # type: ignore
except Exception:  # pragma: no cover - minimalne instalacje
    load_core_config = None  # type: ignore

from bot_core.security.baseline import generate_security_baseline_report
from bot_core.security.signing import build_hmac_signature

LOGGER = logging.getLogger("bot_core.scripts.audit_security_baseline")

_ENV_PREFIX = "BOT_CORE_SECURITY_BASELINE_"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(f"{_ENV_PREFIX}{name}")
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"{_ENV_PREFIX}{name}")
    if value is None:
        return default
    return value.strip() or default


def _env_list(name: str) -> list[str]:
    value = _env_value(name)
    if not value:
        return []
    return [entry for entry in value.split(";") if entry] if ";" in value else value.split(",")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Ścieżka do core.yaml (domyślnie config/core.yaml lub wartość zmiennej środowiskowej)",
    )
    parser.add_argument(
        "--json-output",
        help="Zapisz raport bezpieczeństwa do pliku JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Formatuj JSON z wcięciami i sortowaniem kluczy",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Zakończ kodem 1 gdy wykryto ostrzeżenia",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Zakończ kodem 2 gdy wykryto błędy",
    )
    parser.add_argument(
        "--warn-expiring-days",
        type=float,
        default=None,
        help="Liczba dni przed wygaśnięciem certyfikatu generująca ostrzeżenie (domyślnie 30)",
    )
    parser.add_argument(
        "--metrics-required-scope",
        dest="metrics_scopes",
        action="append",
        default=[],
        help="Wymagany scope tokenu RBAC dla MetricsService (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--risk-required-scope",
        dest="risk_scopes",
        action="append",
        default=[],
        help="Wymagany scope tokenu RBAC dla RiskService (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--print",
        dest="print_stdout",
        action="store_true",
        help="Wypisz raport na stdout (domyślnie jeśli nie podano pliku wyjściowego)",
    )
    parser.add_argument(
        "--summary-hmac-key",
        dest="summary_hmac_key",
        help="Bezpośrednia wartość klucza HMAC do podpisu raportu bezpieczeństwa.",
    )
    parser.add_argument(
        "--summary-hmac-key-file",
        dest="summary_hmac_key_file",
        help="Plik zawierający klucz HMAC do podpisu raportu bezpieczeństwa.",
    )
    parser.add_argument(
        "--summary-hmac-key-env",
        dest="summary_hmac_key_env",
        help="Nazwa zmiennej środowiskowej z kluczem HMAC do podpisu raportu bezpieczeństwa.",
    )
    parser.add_argument(
        "--summary-hmac-key-id",
        dest="summary_hmac_key_id",
        help="Opcjonalny identyfikator klucza HMAC dodawany do podpisu raportu bezpieczeństwa.",
    )
    parser.add_argument(
        "--require-summary-signature",
        action="store_true",
        help="Zakończ działanie błędem, jeśli raport bezpieczeństwa nie został podpisany.",
    )
    return parser.parse_args(argv)


def _resolve_config_path(args: argparse.Namespace) -> Path:
    env_override = _env_value("CONFIG")
    if args.config:
        return Path(args.config).expanduser()
    if env_override:
        return Path(env_override).expanduser()
    return Path("config/core.yaml")


def _resolve_json_output(args: argparse.Namespace) -> str | None:
    if args.json_output:
        return args.json_output
    return _env_value("JSON_OUTPUT")


def _resolve_warn_days(args: argparse.Namespace) -> float:
    if args.warn_expiring_days is not None:
        return float(args.warn_expiring_days)
    env_value = _env_value("WARN_EXPIRING_DAYS")
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:  # pragma: no cover - diagnostyka konfiguracji
            LOGGER.warning("Nieprawidłowa wartość WARN_EXPIRING_DAYS='%s'", env_value)
    return 30.0


def _normalize_scopes(values: Iterable[str], *, default: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        for entry in str(value).replace(";", ",").split(","):
            scope = entry.strip().lower()
            if scope:
                normalized.append(scope)
    if not normalized:
        normalized = list(default)
    return tuple(dict.fromkeys(normalized))


def _resolve_metrics_scopes(args: argparse.Namespace) -> tuple[str, ...]:
    env_scopes = _env_list("METRICS_SCOPES")
    return _normalize_scopes([*args.metrics_scopes, *env_scopes], default=("metrics.read",))


def _resolve_risk_scopes(args: argparse.Namespace) -> tuple[str, ...]:
    env_scopes = _env_list("RISK_SCOPES")
    return _normalize_scopes([*args.risk_scopes, *env_scopes], default=("risk.read",))


def _should_print(args: argparse.Namespace) -> bool:
    if args.print_stdout:
        return True
    return _env_flag("PRINT", not bool(args.json_output))


def _should_fail_on_warning(args: argparse.Namespace) -> bool:
    if args.fail_on_warning:
        return True
    return _env_flag("FAIL_ON_WARNING", False)


def _should_fail_on_error(args: argparse.Namespace) -> bool:
    if args.fail_on_error:
        return True
    return _env_flag("FAIL_ON_ERROR", False)


def _apply_signature_env_overrides(args: argparse.Namespace) -> None:
    if not getattr(args, "summary_hmac_key", None):
        env_value = _env_value("SUMMARY_HMAC_KEY")
        if env_value:
            args.summary_hmac_key = env_value
    if not getattr(args, "summary_hmac_key_file", None):
        env_file = _env_value("SUMMARY_HMAC_KEY_FILE")
        if env_file:
            args.summary_hmac_key_file = env_file
    if not getattr(args, "summary_hmac_key_env", None):
        env_name = _env_value("SUMMARY_HMAC_KEY_ENV")
        if env_name:
            args.summary_hmac_key_env = env_name
    if not getattr(args, "summary_hmac_key_id", None):
        env_id = _env_value("SUMMARY_HMAC_KEY_ID")
        if env_id:
            args.summary_hmac_key_id = env_id
    if not getattr(args, "require_summary_signature", False):
        args.require_summary_signature = _env_flag("REQUIRE_SUMMARY_SIGNATURE", False)


def _load_summary_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    provided = [
        bool(getattr(args, "summary_hmac_key", None)),
        bool(getattr(args, "summary_hmac_key_file", None)),
        bool(getattr(args, "summary_hmac_key_env", None)),
    ]
    if sum(provided) > 1:
        print(
            "Błąd: opcje --summary-hmac-key, --summary-hmac-key-file i --summary-hmac-key-env są wzajemnie wykluczające.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    key_material: str | None = None
    if getattr(args, "summary_hmac_key", None):
        key_material = args.summary_hmac_key
    elif getattr(args, "summary_hmac_key_file", None):
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
        except OSError as exc:
            print(
                f"Nie udało się odczytać klucza HMAC z {args.summary_hmac_key_file}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
    elif getattr(args, "summary_hmac_key_env", None):
        env_value = os.getenv(args.summary_hmac_key_env)
        if env_value is None:
            print(
                f"Zmienna środowiskowa {args.summary_hmac_key_env} nie zawiera klucza HMAC",
                file=sys.stderr,
            )
            raise SystemExit(2)
        key_material = env_value

    if key_material is None:
        if getattr(args, "require_summary_signature", False):
            print(
                "Wymagano podpisania raportu bezpieczeństwa, ale nie dostarczono klucza HMAC.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return None, getattr(args, "summary_hmac_key_id", None)

    key_bytes = key_material.strip().encode("utf-8")
    if not key_bytes:
        print("Klucz HMAC raportu bezpieczeństwa nie może być pusty", file=sys.stderr)
        raise SystemExit(2)

    if len(key_bytes) < 16:  # pragma: no cover - ostrzeżenie informacyjne
        print(
            "Ostrzeżenie: klucz HMAC ma mniej niż 16 bajtów – rozważ użycie dłuższego klucza.",
            file=sys.stderr,
        )

    key_id = getattr(args, "summary_hmac_key_id", None)
    if key_id is not None:
        key_id = key_id.strip() or None

    return key_bytes, key_id


def _dump_json(payload: Any, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    _apply_signature_env_overrides(args)

    config_path = _resolve_config_path(args)
    json_output = _resolve_json_output(args)
    pretty = args.pretty or _env_flag("PRETTY", False)
    warn_days = _resolve_warn_days(args)
    print_stdout = _should_print(args)
    fail_on_warning = _should_fail_on_warning(args)
    fail_on_error = _should_fail_on_error(args)
    metrics_scopes = _resolve_metrics_scopes(args)
    risk_scopes = _resolve_risk_scopes(args)
    signing_key, signing_key_id = _load_summary_signing_key(args)

    if load_core_config is None:
        raise RuntimeError(
            "Moduł konfiguracji nie jest dostępny – zainstaluj zależność bot_core.config"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Plik konfiguracji '{config_path}' nie istnieje")

    core_config = load_core_config(str(config_path))
    report = generate_security_baseline_report(
        core_config,
        env=os.environ,
        warn_expiring_within_days=warn_days,
        metrics_required_scopes=metrics_scopes,
        risk_required_scopes=risk_scopes,
    )

    payload = dict(report.as_dict())
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["config_path"] = str(config_path)
    if signing_key:
        payload["summary_signature"] = build_hmac_signature(
            {key: value for key, value in payload.items() if key != "summary_signature"},
            key=signing_key,
            key_id=signing_key_id,
        )
    serialized = _dump_json(payload, pretty=pretty)

    if json_output:
        output_path = Path(json_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    if print_stdout or not json_output:
        print(serialized)

    exit_code = 0
    if fail_on_error and report.has_errors:
        exit_code = 2
    elif fail_on_warning and (report.has_warnings or report.has_errors):
        exit_code = 1 if exit_code == 0 else exit_code
    return exit_code


if __name__ == "__main__":  # pragma: no cover - obsługa CLI
    raise SystemExit(main())

