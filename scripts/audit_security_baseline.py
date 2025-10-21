"""Audyt zbiorczy bezpieczeństwa (TLS + RBAC) dla usług runtime."""

from __future__ import annotations

import argparse
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
from scripts._cli_common import env_flag, env_list, env_value, normalize_scopes, should_print
from scripts._json_utils import dump_json

LOGGER = logging.getLogger("bot_core.scripts.audit_security_baseline")

_ENV_PREFIX = "BOT_CORE_SECURITY_BASELINE_"


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
        "--scheduler-required-scope",
        dest="scheduler_scopes",
        action="append",
        default=[],
        help=(
            "Wymagany scope tokenu RBAC dla scheduler-a (można wskazać wartość globalną"
            " lub pary <scheduler>:<scope>)"
        ),
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


def _resolve_metrics_scopes(args: argparse.Namespace) -> tuple[str, ...]:
    env_scopes = env_list(_ENV_PREFIX, "METRICS_SCOPES")
    combined = [*args.metrics_scopes, *env_scopes]
    scopes = normalize_scopes(combined)
    return scopes or ("metrics.read",)


def _resolve_risk_scopes(args: argparse.Namespace) -> tuple[str, ...]:
    env_scopes = env_list(_ENV_PREFIX, "RISK_SCOPES")
    combined = [*args.risk_scopes, *env_scopes]
    scopes = normalize_scopes(combined)
    return scopes or ("risk.read",)


def _normalize_scheduler_scope_entries(
    entries: Iterable[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    default_entries: list[str] = []
    overrides: dict[str, list[str]] = {}
    for raw_entry in entries:
        if raw_entry is None:
            continue
        text = str(raw_entry).strip()
        if not text:
            continue
        if ":" in text:
            scheduler_name, scope_part = text.split(":", 1)
            scheduler_key = scheduler_name.strip()
            scope_value = scope_part.strip()
            if not scheduler_key:
                default_entries.append(scope_value)
                continue
            overrides.setdefault(scheduler_key, []).append(scope_value)
        else:
            default_entries.append(text)

    default_scopes = normalize_scopes(
        default_entries,
        default=("runtime.schedule.read", "runtime.schedule.write"),
    )
    normalized_overrides = {
        name: normalize_scopes(values, default=default_scopes)
        for name, values in overrides.items()
    }
    return default_scopes, normalized_overrides


def _resolve_scheduler_scopes(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    env_entries = env_list(_ENV_PREFIX, "SCHEDULER_SCOPES")
    return _normalize_scheduler_scope_entries([*args.scheduler_scopes, *env_entries])


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


def _apply_signature_env_overrides(args: argparse.Namespace) -> None:
    if not getattr(args, "summary_hmac_key", None):
        override_value = env_value(_ENV_PREFIX, "SUMMARY_HMAC_KEY")
        if override_value:
            args.summary_hmac_key = override_value
    if not getattr(args, "summary_hmac_key_file", None):
        override_file = env_value(_ENV_PREFIX, "SUMMARY_HMAC_KEY_FILE")
        if override_file:
            args.summary_hmac_key_file = override_file
    if not getattr(args, "summary_hmac_key_env", None):
        override_env = env_value(_ENV_PREFIX, "SUMMARY_HMAC_KEY_ENV")
        if override_env:
            args.summary_hmac_key_env = override_env
    if not getattr(args, "summary_hmac_key_id", None):
        override_id = env_value(_ENV_PREFIX, "SUMMARY_HMAC_KEY_ID")
        if override_id:
            args.summary_hmac_key_id = override_id
    if not getattr(args, "require_summary_signature", False):
        args.require_summary_signature = env_flag(_ENV_PREFIX, "REQUIRE_SUMMARY_SIGNATURE", False)


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    _apply_signature_env_overrides(args)

    config_path = _resolve_config_path(args)
    json_output = _resolve_json_output(args)
    pretty = args.pretty or env_flag(_ENV_PREFIX, "PRETTY", False)
    warn_days = _resolve_warn_days(args)
    print_stdout = _should_print(args)
    fail_on_warning = _should_fail_on_warning(args)
    fail_on_error = _should_fail_on_error(args)
    metrics_scopes = _resolve_metrics_scopes(args)
    risk_scopes = _resolve_risk_scopes(args)
    scheduler_default_scopes, scheduler_overrides = _resolve_scheduler_scopes(args)
    signing_key, signing_key_id = _load_summary_signing_key(args)

    if load_core_config is None:
        raise RuntimeError(
            "Moduł konfiguracji nie jest dostępny – zainstaluj zależność bot_core.config"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Plik konfiguracji '{config_path}' nie istnieje")

    core_config = load_core_config(str(config_path))
    scheduler_scope_payload: dict[str, tuple[str, ...]] = {"*": scheduler_default_scopes}
    scheduler_scope_payload.update(scheduler_overrides)
    report = generate_security_baseline_report(
        core_config,
        env=os.environ,
        warn_expiring_within_days=warn_days,
        metrics_required_scopes=metrics_scopes,
        risk_required_scopes=risk_scopes,
        scheduler_required_scopes=scheduler_scope_payload,
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
    serialized = dump_json(payload, pretty=pretty)

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

