"""Mini-audyt zgodności Stage4/Stage5: RBAC, mTLS, TCO oraz observability."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config.loader import load_core_config  # noqa: E402
from bot_core.config.models import (  # noqa: E402
    CoreConfig,
    DecisionEngineTCOConfig,
    KeyRotationConfig,
    KeyRotationEntryConfig,
    MultiStrategySchedulerConfig,
    ObservabilityConfig,
    ServiceTokenConfig,
)
from bot_core.security.rotation import RotationRegistry  # noqa: E402

from scripts._cli_common import now_iso


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Weryfikuje krytyczne wymogi Stage4 (RBAC, mTLS, rotacje kluczy) "
            "na podstawie konfiguracji core oraz środowiska uruchomieniowego."
        )
    )
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku CoreConfig (domyślnie config/core.yaml)",
    )
    parser.add_argument(
        "--profile",
        choices=("stage4", "stage5"),
        default="stage4",
        help=(
            "Określa profil kontroli: 'stage4' (domyślny) ogranicza się do RBAC/mTLS,"
            " 'stage5' dodaje walidacje TCO i observability."
        ),
    )
    parser.add_argument(
        "--allow-missing-env",
        action="store_true",
        help="Nie kończ audytu błędem, gdy zmienne środowiskowe nie są ustawione (zamiast tego ostrzeżenie).",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Sprawdź istnienie plików certyfikatów, kluczy i rejestru rotacji.",
    )
    parser.add_argument(
        "--mtls-bundle-name",
        help="Identyfikator bundla mTLS oczekiwanego w rejestrze rotacji (np. core-oem).",
    )
    parser.add_argument(
        "--rotation-interval-days",
        type=float,
        default=90.0,
        help="Maksymalny dopuszczalny okres między rotacjami certyfikatów/kluczy (domyślnie 90 dni).",
    )
    parser.add_argument(
        "--rotation-warn-days",
        type=float,
        default=14.0,
        help="Okno ostrzeżeń przed kolejną rotacją (domyślnie 14 dni).",
    )
    parser.add_argument(
        "--min-secret-length",
        type=int,
        default=24,
        help="Minimalna długość sekretów RBAC/HMAC (domyślnie 24 znaki).",
    )
    parser.add_argument(
        "--output-json",
        help="Opcjonalna ścieżka zapisu raportu audytu w formacie JSON.",
    )
    return parser.parse_args(argv)
def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _record(target: list[Mapping[str, Any]], *, check: str, message: str) -> None:
    target.append({"check": check, "message": message})


def _check_env_secret(
    *,
    env_name: str,
    allow_missing: bool,
    min_length: int,
    issues: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
    context: str,
) -> None:
    value = os.environ.get(env_name)
    if value is None:
        message = f"Zmienna środowiskowa {env_name} wymagana przez {context} nie jest ustawiona."
        target = warnings if allow_missing else issues
        _record(target, check=f"env:{env_name}", message=message)
        return
    if len(value.strip()) < min_length:
        _record(
            warnings,
            check=f"env:{env_name}",
            message=(
                f"Sekret {env_name} używany przez {context} ma długość < {min_length} znaków – "
                "rozważ rotację/utwardzenie."
            ),
        )


def _validate_metrics_service(
    *,
    raw: Mapping[str, Any],
    base_dir: Path,
    allow_missing_env: bool,
    min_secret_length: int,
    check_paths: bool,
    issues: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
) -> None:
    metrics = raw.get("metrics_service")
    if not isinstance(metrics, Mapping):
        _record(issues, check="metrics_service", message="Brak sekcji metrics_service w konfiguracji core.")
        return

    auth_sources = [metrics.get("auth_token"), metrics.get("auth_token_env"), metrics.get("auth_token_file")]
    if not any(auth_sources):
        _record(
            issues,
            check="metrics_service.auth",
            message="metrics_service wymaga ustawienia auth_token/auth_token_env/auth_token_file.",
        )
    env_name = metrics.get("auth_token_env")
    if isinstance(env_name, str) and env_name.strip():
        _check_env_secret(
            env_name=env_name.strip(),
            allow_missing=allow_missing_env,
            min_length=min_secret_length,
            issues=issues,
            warnings=warnings,
            context="metrics_service auth_token_env",
        )
    token_file = metrics.get("auth_token_file")
    if check_paths and isinstance(token_file, str) and token_file.strip():
        resolved = _resolve_path(base_dir, token_file.strip())
        if not resolved or not resolved.exists():
            _record(
                issues,
                check="metrics_service.auth_token_file",
                message=f"Plik tokenu telemetryki nie istnieje: {token_file}",
            )

    tls = metrics.get("tls")
    if not isinstance(tls, Mapping):
        _record(issues, check="metrics_service.tls", message="Brak sekcji metrics_service.tls lub niepoprawny format.")
        return

    if not tls.get("enabled"):
        _record(issues, check="metrics_service.tls", message="TLS dla MetricsService musi być włączony.")
    if not tls.get("require_client_auth"):
        _record(
            issues,
            check="metrics_service.tls",
            message="MetricsService powinien wymagać uwierzytelnienia klienta (mTLS).",
        )

    for field in ("certificate_path", "private_key_path", "client_ca_path"):
        value = tls.get(field)
        if not value:
            _record(
                issues,
                check=f"metrics_service.tls.{field}",
                message=f"Pole {field} w metrics_service.tls jest wymagane.",
            )
            continue
        if check_paths:
            resolved = _resolve_path(base_dir, str(value))
            if not resolved or not resolved.exists():
                _record(
                    issues,
                    check=f"metrics_service.tls.{field}",
                    message=f"Ścieżka {value} nie istnieje.",
                )


def _validate_scheduler_tokens(
    *,
    config: CoreConfig,
    allow_missing_env: bool,
    min_secret_length: int,
    issues: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
) -> None:
    if not config.multi_strategy_schedulers:
        _record(
            issues,
            check="multi_strategy_schedulers",
            message="Brak zdefiniowanych scheduler-ów multi-strategy.",
        )
        return

    required_scopes = {"runtime.schedule.write", "runtime.schedule.read"}
    for name, scheduler in config.multi_strategy_schedulers.items():
        if not isinstance(scheduler, MultiStrategySchedulerConfig):
            continue
        tokens = scheduler.rbac_tokens
        if not tokens:
            _record(
                issues,
                check=f"scheduler:{name}",
                message="Scheduler multi-strategy wymaga co najmniej jednego tokenu RBAC.",
            )
            continue
        for token in tokens:
            if not isinstance(token, ServiceTokenConfig):
                continue
            token_label = token.token_id or "<anonymous>"
            has_source = any((token.token_env, token.token_value, token.token_hash))
            if not has_source:
                _record(
                    issues,
                    check=f"scheduler:{name}:{token_label}",
                    message="Token RBAC musi mieć ustawione token_env/token_value/token_hash.",
                )
            if token.token_env:
                _check_env_secret(
                    env_name=token.token_env,
                    allow_missing=allow_missing_env,
                    min_length=min_secret_length,
                    issues=issues,
                    warnings=warnings,
                    context=f"scheduler {name} ({token_label})",
                )
            missing_scopes = required_scopes.difference(token.scopes)
            if missing_scopes:
                _record(
                    issues,
                    check=f"scheduler:{name}:{token_label}:scopes",
                    message=(
                        "Token RBAC wymaga scopes runtime.schedule.read/write – brak: "
                        + ", ".join(sorted(missing_scopes))
                    ),
                )


def _validate_live_router(
    *,
    raw: Mapping[str, Any],
    base_dir: Path,
    allow_missing_env: bool,
    min_secret_length: int,
    check_paths: bool,
    mtls_bundle_name: str | None,
    rotation_interval_days: float,
    rotation_warn_days: float,
    issues: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
) -> None:
    execution = raw.get("execution")
    if not isinstance(execution, Mapping):
        _record(issues, check="execution", message="Brak sekcji execution w konfiguracji core.")
        return

    live_router = execution.get("live_router")
    if not isinstance(live_router, Mapping):
        _record(issues, check="execution.live_router", message="Brak konfiguracji live_router.")
    else:
        decision = live_router.get("decision_log")
        if not isinstance(decision, Mapping):
            _record(
                issues,
                check="execution.live_router.decision_log",
                message="Sekcja decision_log jest wymagana dla live_router.",
            )
        else:
            path = decision.get("path")
            if not path:
                _record(
                    issues,
                    check="execution.live_router.decision_log.path",
                    message="Decision log live routera wymaga pola path.",
                )
            key_env = decision.get("hmac_key_env")
            key_file = decision.get("hmac_key_path") or decision.get("signing_key_path")
            if not key_env and not key_file:
                _record(
                    issues,
                    check="execution.live_router.decision_log",
                    message="Decision log musi posiadać hmac_key_env lub hmac_key_path.",
                )
            if key_env:
                _check_env_secret(
                    env_name=str(key_env),
                    allow_missing=allow_missing_env,
                    min_length=min_secret_length,
                    issues=issues,
                    warnings=warnings,
                    context="decision log live router",
                )
            if check_paths and key_file:
                resolved = _resolve_path(base_dir, str(key_file))
                if not resolved or not resolved.exists():
                    _record(
                        issues,
                        check="execution.live_router.decision_log.hmac_key_path",
                        message=f"Plik klucza HMAC nie istnieje: {key_file}",
                    )

    mtls = execution.get("mtls") if isinstance(execution, Mapping) else None
    if not isinstance(mtls, Mapping):
        _record(issues, check="execution.mtls", message="Brak konfiguracji execution.mtls (wymagane dla Stage4).")
        return

    for field in (
        "bundle_directory",
        "ca_certificate",
        "server_certificate",
        "server_key",
        "client_certificate",
        "client_key",
        "rotation_registry",
    ):
        value = mtls.get(field)
        if not value:
            _record(
                issues,
                check=f"execution.mtls.{field}",
                message=f"Pole {field} w konfiguracji mTLS jest wymagane.",
            )
            continue
        if check_paths:
            resolved = _resolve_path(base_dir, str(value))
            if not resolved or not resolved.exists():
                _record(
                    issues,
                    check=f"execution.mtls.{field}",
                    message=f"Ścieżka {value} nie istnieje.",
                )

    registry_path_value = mtls.get("rotation_registry")
    if registry_path_value:
        registry_path = _resolve_path(base_dir, str(registry_path_value))
        if not registry_path:
            _record(
                issues,
                check="execution.mtls.rotation_registry",
                message="Nie można znormalizować ścieżki do rejestru rotacji.",
            )
            return
        if not registry_path.exists():
            _record(
                issues,
                check="execution.mtls.rotation_registry",
                message=f"Rejestr rotacji TLS nie istnieje: {registry_path}",
            )
            return
        registry = RotationRegistry(registry_path)
        has_entries = False
        now = _dt.datetime.now(_dt.timezone.utc)
        if mtls_bundle_name:
            for purpose in ("tls_ca", "tls_server", "tls_client"):
                status = registry.status(
                    mtls_bundle_name,
                    purpose,
                    interval_days=rotation_interval_days,
                    now=now,
                )
                has_entries = True
                if status.is_overdue:
                    _record(
                        issues,
                        check=f"execution.mtls.rotation_registry.{purpose}",
                        message=(
                            f"Wpis rotacji {mtls_bundle_name}/{purpose} jest przeterminowany – "
                            "natychmiastowa rotacja wymagana."
                        ),
                    )
                elif status.is_due or status.due_in_days <= rotation_warn_days:
                    _record(
                        warnings,
                        check=f"execution.mtls.rotation_registry.{purpose}",
                        message=(
                            f"Wpis rotacji {mtls_bundle_name}/{purpose} wymaga odświeżenia w ciągu "
                            f"{status.due_in_days:.1f} dni."
                        ),
                    )
        if not has_entries and not any(registry.entries()):
            _record(
                warnings,
                check="execution.mtls.rotation_registry",
                message="Rejestr rotacji istnieje, lecz nie zawiera żadnych wpisów.",
            )
    else:
        _record(
            issues,
            check="execution.mtls.rotation_registry",
            message="Konfiguracja mTLS wymaga pola rotation_registry.",
        )


def _validate_stage5_tco(
    *,
    config: CoreConfig,
    base_dir: Path,
    issues: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
) -> None:
    decision_engine = getattr(config, "decision_engine", None)
    if decision_engine is None:
        _record(
            issues,
            check="decision_engine",
            message="Profil Stage5 wymaga konfiguracji decision_engine.",
        )
        return

    tco_config = getattr(decision_engine, "tco", None)
    if not isinstance(tco_config, DecisionEngineTCOConfig):
        _record(
            issues,
            check="decision_engine.tco",
            message="Profil Stage5 wymaga sekcji decision_engine.tco (raporty kosztów).",
        )
        return

    report_paths = tuple(str(path) for path in getattr(tco_config, "report_paths", ()) if str(path).strip())
    if not report_paths:
        _record(
            issues,
            check="decision_engine.tco.report_paths",
            message="Sekcja decision_engine.tco musi definiować co najmniej jeden raport TCO.",
        )
        return

    now = _dt.datetime.now(_dt.timezone.utc)
    warn_age = getattr(tco_config, "warn_report_age_hours", None)
    max_age = getattr(tco_config, "max_report_age_hours", None)

    for raw_path in report_paths:
        resolved = _resolve_path(base_dir, raw_path)
        if not resolved or not resolved.exists():
            _record(
                issues,
                check=f"decision_engine.tco.report:{raw_path}",
                message=f"Raport TCO {raw_path} nie istnieje – dostarcz aktualny artefakt.",
            )
            continue

        mtime = _dt.datetime.fromtimestamp(resolved.stat().st_mtime, tz=_dt.timezone.utc)
        age_hours = (now - mtime).total_seconds() / 3600.0

        if max_age is not None and age_hours > max_age:
            _record(
                issues,
                check=f"decision_engine.tco.report:{raw_path}",
                message=(
                    f"Raport TCO {raw_path} jest starszy niż dozwolone {max_age:.1f} h (wiek {age_hours:.1f} h)."
                ),
            )
        elif warn_age is not None and age_hours > warn_age:
            _record(
                warnings,
                check=f"decision_engine.tco.report:{raw_path}",
                message=(
                    f"Raport TCO {raw_path} przekroczył próg ostrzeżenia {warn_age:.1f} h (wiek {age_hours:.1f} h)."
                ),
            )


def _validate_stage5_observability(
    *,
    config: CoreConfig,
    base_dir: Path,
    issues: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
) -> None:
    observability = getattr(config, "observability", None)
    if not isinstance(observability, ObservabilityConfig):
        _record(
            issues,
            check="observability",
            message="Profil Stage5 wymaga sekcji observability z definicjami SLO i rotacją kluczy.",
        )
        return

    slo_definitions = getattr(observability, "slo", {}) or {}
    required_slos = {"decision_latency", "trade_cost", "fill_rate"}
    missing_slos = sorted(required_slos.difference(slo_definitions))
    if missing_slos:
        _record(
            issues,
            check="observability.slo",
            message="Brak wymaganych SLO Stage5: " + ", ".join(missing_slos),
        )

    key_rotation = getattr(observability, "key_rotation", None)
    if not isinstance(key_rotation, KeyRotationConfig):
        _record(
            issues,
            check="observability.key_rotation",
            message="Sekcja observability.key_rotation jest wymagana w profilu Stage5.",
        )
        return

    registry_path_value = getattr(key_rotation, "registry_path", "")
    registry_path = _resolve_path(base_dir, str(registry_path_value))
    if not registry_path or not registry_path.exists():
        _record(
            issues,
            check="observability.key_rotation.registry_path",
            message=f"Rejestr rotacji Stage5 nie istnieje: {registry_path_value}",
        )
        return

    registry = RotationRegistry(registry_path)
    entries = tuple(getattr(key_rotation, "entries", ()) or ())
    if not entries:
        _record(
            warnings,
            check="observability.key_rotation.entries",
            message="Rejestr rotacji Stage5 istnieje, lecz nie zdefiniowano żadnych wpisów.",
        )
        return

    default_interval = getattr(key_rotation, "default_interval_days", 90.0) or 90.0
    default_warn = getattr(key_rotation, "default_warn_within_days", 14.0) or 14.0
    now = _dt.datetime.now(_dt.timezone.utc)

    for entry in entries:
        if not isinstance(entry, KeyRotationEntryConfig):
            continue
        interval_days = float(entry.interval_days or default_interval)
        warn_within = float(entry.warn_within_days or default_warn)
        status = registry.status(entry.key, entry.purpose, interval_days=interval_days, now=now)
        label = f"{entry.key}:{entry.purpose}"

        if status.is_overdue:
            if status.last_rotated is None:
                message = f"Brak historii rotacji dla {label} – wpis nigdy nie został wykonany."
            else:
                overdue_days = abs(status.due_in_days)
                message = (
                    f"Rotacja {label} jest przeterminowana o {overdue_days:.1f} dnia/dni (interwał {interval_days:.1f} dni)."
                )
            _record(
                issues,
                check=f"observability.key_rotation:{label}",
                message=message,
            )
        elif status.is_due or status.due_in_days <= warn_within:
            message = (
                f"Rotacja {label} będzie wymagana w ciągu {status.due_in_days:.1f} dnia/dni (interwał {interval_days:.1f} dni)."
            )
            _record(
                warnings,
                check=f"observability.key_rotation:{label}",
                message=message,
            )

def _build_report(
    *,
    config_path: Path,
    issues: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if issues:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "ok"
    return {
        "checked_at": now_iso(),
        "config": str(config_path),
        "status": status,
        "issues": issues,
        "warnings": warnings,
    }


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    profile = str(getattr(args, "profile", "stage4")).lower()

    try:
        core_config = load_core_config(config_path)
    except FileNotFoundError:
        print(f"Nie znaleziono pliku konfiguracji: {config_path}", file=sys.stderr)
        return 2

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
        if not isinstance(raw_config, Mapping):
            print("Plik konfiguracji nie zawiera poprawnej struktury YAML", file=sys.stderr)
            return 2

    base_dir = config_path.parent
    issues: list[Mapping[str, Any]] = []
    warnings: list[Mapping[str, Any]] = []

    _validate_metrics_service(
        raw=raw_config,
        base_dir=base_dir,
        allow_missing_env=args.allow_missing_env,
        min_secret_length=args.min_secret_length,
        check_paths=args.check_paths,
        issues=issues,
        warnings=warnings,
    )
    _validate_scheduler_tokens(
        config=core_config,
        allow_missing_env=args.allow_missing_env,
        min_secret_length=args.min_secret_length,
        issues=issues,
        warnings=warnings,
    )
    _validate_live_router(
        raw=raw_config,
        base_dir=base_dir,
        allow_missing_env=args.allow_missing_env,
        min_secret_length=args.min_secret_length,
        check_paths=args.check_paths,
        mtls_bundle_name=args.mtls_bundle_name,
        rotation_interval_days=args.rotation_interval_days,
        rotation_warn_days=args.rotation_warn_days,
        issues=issues,
        warnings=warnings,
    )

    if profile == "stage5":
        _validate_stage5_tco(
            config=core_config,
            base_dir=base_dir,
            issues=issues,
            warnings=warnings,
        )
        _validate_stage5_observability(
            config=core_config,
            base_dir=base_dir,
            issues=issues,
            warnings=warnings,
        )

    report = _build_report(config_path=config_path, issues=issues, warnings=warnings)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0 if report["status"] != "fail" else 1


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    try:
        return run(argv)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Błąd audytu Stage4: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
