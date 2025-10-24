"""Local OEM orchestrator managing demo/paper/live environments."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    StrategyDefinition,
)

from bot_core.config.loader import load_core_config
from bot_core.security.guards import CapabilityGuard, LicenseCapabilityError
from bot_core.security.license import validate_license_from_config
from bot_core.security.license_service import LicenseService, LicenseServiceError
from bot_core.security.update import verify_update_bundle
from deploy.packaging.build_pyinstaller_bundle import build_bundle as build_pyinstaller_bundle

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EnvironmentDefinition:
    name: str
    environment_key: str
    config_path: Path
    scheduler_name: str | None = None


DEFAULT_ENVIRONMENTS: Mapping[str, EnvironmentDefinition] = {
    "demo": EnvironmentDefinition(
        name="demo",
        environment_key="demo",
        config_path=Path("config/core.yaml"),
        scheduler_name="demo",
    ),
    "paper": EnvironmentDefinition(
        name="paper",
        environment_key="binance_paper",
        config_path=Path("config/core.yaml"),
        scheduler_name="paper_default",
    ),
    "live": EnvironmentDefinition(
        name="live",
        environment_key="binance_live",
        config_path=Path("config/core.yaml"),
        scheduler_name="live_default",
    ),
}


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"environments": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"environments": {}}
    return data


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _definition_from_config(cfg: Any) -> StrategyDefinition:
    """Buduje obiekt `StrategyDefinition` na podstawie wpisu konfiguracji."""

    metadata = dict(getattr(cfg, "metadata", {}) or {})
    capability = getattr(cfg, "capability", None)
    if capability and "capability" not in metadata:
        metadata["capability"] = capability

    try:
        spec = DEFAULT_STRATEGY_CATALOG.get(getattr(cfg, "engine"))
    except KeyError:
        spec = None

    license_tier = getattr(cfg, "license_tier", None) or (spec.license_tier if spec else "unspecified")
    risk_classes = tuple(getattr(cfg, "risk_classes", ()) or (spec.risk_classes if spec else ("unspecified",)))
    required_data = tuple(
        getattr(cfg, "required_data", ()) or (spec.required_data if spec else ("unspecified",))
    )

    if spec and spec.capability and "capability" not in metadata:
        metadata.setdefault("capability", spec.capability)

    return StrategyDefinition(
        name=getattr(cfg, "name"),
        engine=getattr(cfg, "engine"),
        license_tier=license_tier,
        risk_classes=risk_classes,
        required_data=required_data,
        parameters=dict(getattr(cfg, "parameters", {}) or {}),
        risk_profile=getattr(cfg, "risk_profile", None),
        tags=tuple(getattr(cfg, "tags", ()) or ()),
        metadata=metadata,
    )


def _ensure_environment(name: str, *, environments: Mapping[str, EnvironmentDefinition]) -> EnvironmentDefinition:
    try:
        return environments[name]
    except KeyError as exc:
        raise SystemExit(f"Nieznane środowisko: {name}") from exc


def cmd_prepare(args: argparse.Namespace, environments: Mapping[str, EnvironmentDefinition]) -> None:
    env = _ensure_environment(args.environment, environments=environments)
    base_dir = Path(args.base_dir).expanduser().resolve()
    env_dir = base_dir / env.name
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "logs").mkdir(exist_ok=True)
    (env_dir / "bundles").mkdir(exist_ok=True)
    print(f"Przygotowano katalog środowiska {env.name}: {env_dir}")


def cmd_bundle(args: argparse.Namespace, environments: Mapping[str, EnvironmentDefinition], *, state_path: Path) -> None:
    env = _ensure_environment(args.environment, environments=environments)
    license_result = _resolve_license_result(env.config_path)
    guard = _require_capability_guard(license_result)
    capabilities = guard.capabilities

    if args.allowed_profile and license_result.profile:
        allowed = {profile.strip().lower() for profile in args.allowed_profile}
        if license_result.profile.lower() not in allowed:
            raise SystemExit(
                "Profil licencji OEM nie jest dozwolony w manifeście aktualizacji – zaktualizuj --allowed-profile."
            )

    if not capabilities.is_maintenance_active():
        raise SystemExit("Licencja utrzymaniowa wygasła – budowanie bundla jest zablokowane.")

    try:
        guard.require_module(
            "oem_updater",
            message="Licencja nie zawiera modułu OEM Updater wymaganego do dystrybucji bundli.",
        )
    except LicenseCapabilityError as exc:
        raise SystemExit(str(exc)) from exc

    base_dir = Path(args.base_dir).expanduser().resolve()
    env_dir = base_dir / env.name
    env_dir.mkdir(parents=True, exist_ok=True)
    workdir = env_dir / "build"
    output_dir = env_dir / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    bundle_args = argparse.Namespace(
        entrypoint=args.entrypoint,
        qt_dist=args.qt_dist,
        briefcase_project=args.briefcase_project,
        platform=args.platform,
        version=args.version,
        output_dir=str(output_dir),
        workdir=str(workdir),
        hidden_import=args.hidden_import,
        runtime_name=args.runtime_name,
        include=args.include,
        signing_key=args.signing_key,
        signing_key_id=args.signing_key_id,
        allowed_profile=args.allowed_profile,
        metadata=args.metadata,
        metadata_file=args.metadata_file,
        metadata_url=args.metadata_url,
        metadata_url_header=args.metadata_url_header,
        metadata_url_timeout=args.metadata_url_timeout,
        metadata_url_max_size=args.metadata_url_max_size,
        metadata_url_allow_http=args.metadata_url_allow_http,
        metadata_url_allowed_host=args.metadata_url_allowed_host,
        metadata_url_cert_fingerprint=args.metadata_url_cert_fingerprint,
        metadata_url_cert_subject=args.metadata_url_cert_subject,
        metadata_url_cert_issuer=args.metadata_url_cert_issuer,
        metadata_url_cert_san=args.metadata_url_cert_san,
        metadata_url_cert_eku=args.metadata_url_cert_eku,
        metadata_url_cert_policy=args.metadata_url_cert_policy,
        metadata_url_cert_serial=args.metadata_url_cert_serial,
        metadata_url_ca=args.metadata_url_ca,
        metadata_url_capath=args.metadata_url_capath,
        metadata_url_client_cert=args.metadata_url_client_cert,
        metadata_url_client_key=args.metadata_url_client_key,
        metadata_ini=args.metadata_ini,
        metadata_toml=args.metadata_toml,
        metadata_yaml=args.metadata_yaml,
        metadata_dotenv=args.metadata_dotenv,
        metadata_env_prefix=args.metadata_env_prefix,
    )

    archive = build_pyinstaller_bundle(bundle_args)
    state = _load_state(state_path)
    env_state = state.setdefault("environments", {}).setdefault(env.name, {})
    env_state["last_bundle"] = {
        "path": str(archive),
        "timestamp": _timestamp(),
        "version": args.version,
        "platform": args.platform,
    }
    _save_state(state_path, state)
    print(f"Zbudowano bundla środowiska {env.name}: {archive}")


def cmd_launch(args: argparse.Namespace, environments: Mapping[str, EnvironmentDefinition], *, state_path: Path) -> None:
    env = _ensure_environment(args.environment, environments=environments)
    license_result = _resolve_license_result(env.config_path)
    guard = _require_capability_guard(license_result)

    try:
        guard.require_environment(
            env.name,
            message=f"Licencja nie pozwala na uruchomienie środowiska '{env.name}'.",
        )
    except LicenseCapabilityError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        guard.require_runtime(
            "multi_strategy_scheduler",
            message="Multi-strategy scheduler jest zablokowany przez licencję.",
        )
    except LicenseCapabilityError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        guard.require_module(
            "walk_forward",
            message="Scheduler multi-strategy wymaga modułu Walk Forward w licencji.",
        )
    except LicenseCapabilityError as exc:
        raise SystemExit(str(exc)) from exc

    if env.name == "live":
        try:
            guard.require_edition(
                "pro",
                message="Uruchomienie środowiska live wymaga licencji w edycji Pro lub wyższej.",
            )
        except LicenseCapabilityError as exc:
            raise SystemExit(str(exc)) from exc

    config_path = env.config_path.expanduser().resolve()
    command = [
        sys.executable,
        "scripts/run_multi_strategy_scheduler.py",
        "--config",
        str(config_path),
        "--environment",
        env.environment_key,
    ]
    if env.scheduler_name:
        command.extend(["--scheduler", env.scheduler_name])
    if args.run_once:
        command.append("--run-once")
    if args.extra_arg:
        command.extend(args.extra_arg)

    if args.dry_run:
        print("[dry-run]", " ".join(command))
        return

    result = subprocess.run(command, check=False)
    state = _load_state(state_path)
    env_state = state.setdefault("environments", {}).setdefault(env.name, {})
    env_state["last_launch"] = {
        "timestamp": _timestamp(),
        "exit_code": result.returncode,
    }
    _save_state(state_path, state)
    if result.returncode != 0:
        raise SystemExit(f"Proces zakończył się kodem {result.returncode}")


def _resolve_license_result(config_path: Path):
    core_config = load_core_config(config_path)
    license_config = getattr(getattr(core_config, "security", None), "license", None)
    if license_config is None:
        raise SystemExit("Konfiguracja core nie zawiera sekcji security.license")
    result = validate_license_from_config(license_config)

    offline_license_path = os.environ.get("BOT_CORE_LICENSE_PATH")
    offline_public_key = os.environ.get("BOT_CORE_LICENSE_PUBLIC_KEY")
    if not offline_license_path or not offline_public_key:
        LOGGER.error(
            "Brak zmiennych BOT_CORE_LICENSE_PATH/BOT_CORE_LICENSE_PUBLIC_KEY – licencja offline jest wymagana."
        )
        raise SystemExit("Nie znaleziono licencji offline wymaganej przez orchestratora.")

    try:
        service = LicenseService(verify_key_hex=offline_public_key)
        snapshot = service.load_from_file(
            offline_license_path,
            expected_hwid=result.fingerprint,
        )
    except FileNotFoundError as exc:
        LOGGER.error("Plik licencji offline nie istnieje: %s", offline_license_path)
        raise SystemExit("Plik licencji offline nie został znaleziony.") from exc
    except LicenseServiceError as exc:
        LOGGER.error("Błąd podczas ładowania licencji offline: %s", exc)
        raise SystemExit(f"Nie udało się odczytać licencji offline: {exc}") from exc

    guard = CapabilityGuard(snapshot.capabilities)
    result.capabilities = snapshot.capabilities
    result.capability_guard = guard
    return result


def _require_capability_guard(result) -> CapabilityGuard:
    guard = getattr(result, "capability_guard", None)
    capabilities = getattr(result, "capabilities", None)
    if guard is None or capabilities is None:
        raise SystemExit("Licencja offline nie dostarczyła capabilities – operacja zabroniona.")
    return guard


def cmd_verify_update(args: argparse.Namespace, environments: Mapping[str, EnvironmentDefinition]) -> None:
    env = _ensure_environment(args.environment, environments=environments)
    manifest_path = Path(args.manifest).expanduser().resolve()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    signature_path = Path(args.signature).expanduser().resolve() if args.signature else None

    license_result = None
    if args.core_config:
        license_result = _resolve_license_result(Path(args.core_config).expanduser().resolve())
    else:
        license_result = _resolve_license_result(env.config_path)

    key_bytes = None
    if args.signing_key:
        key_candidate = Path(args.signing_key)
        if key_candidate.exists():
            key_bytes = key_candidate.read_bytes()
        else:
            key_bytes = args.signing_key.encode("utf-8")

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=bundle_dir,
        signature_path=signature_path,
        hmac_key=key_bytes,
        license_result=license_result,
    )

    print(json.dumps(
        {
            "manifest": {
                "version": result.manifest.version,
                "platform": result.manifest.platform,
                "runtime": result.manifest.runtime,
                "allowed_profiles": result.manifest.allowed_profiles,
            },
            "signature_valid": result.signature_valid,
            "license_ok": result.license_ok,
            "artifact_checks": result.artifact_checks,
            "errors": result.errors,
            "warnings": result.warnings,
        },
        ensure_ascii=False,
        indent=2,
    ))

    if not result.is_successful:
        raise SystemExit("Aktualizacja nie przeszła weryfikacji")


def cmd_status(args: argparse.Namespace, environments: Mapping[str, EnvironmentDefinition], *, state_path: Path) -> None:
    state = _load_state(state_path)
    env_states = state.get("environments", {})
    report: Dict[str, Any] = {}
    for name, env in environments.items():
        entry: Dict[str, Any] = {
            "config_path": str(env.config_path),
            "environment_key": env.environment_key,
            "scheduler": env.scheduler_name,
            "state": env_states.get(name, {}),
        }
        try:
            core_config = load_core_config(env.config_path)
        except Exception as exc:  # pragma: no cover - defensywny fallback
            entry["error"] = str(exc)
            report[name] = entry
            continue

        definitions_cfg = getattr(core_config, "strategy_definitions", {})
        definitions: dict[str, StrategyDefinition] = {}
        for definition_name, cfg in definitions_cfg.items():
            try:
                definition = _definition_from_config(cfg)
            except ValueError as exc:
                entry.setdefault("definition_errors", {})[definition_name] = str(exc)
                continue
            definitions[definition_name] = definition

        if definitions:
            definition_summaries = DEFAULT_STRATEGY_CATALOG.describe_definitions(
                definitions, include_metadata=True
            )
            entry["strategies"] = definition_summaries
            entry["license_tiers"] = sorted(
                {payload.get("license_tier") for payload in definition_summaries if payload.get("license_tier")}
            )
            capabilities = sorted(
                {payload.get("capability") for payload in definition_summaries if payload.get("capability")}
            )
            if capabilities:
                entry["capabilities"] = capabilities
        else:
            entry["strategies"] = []

        summaries_by_name = {payload.get("name"): payload for payload in entry.get("strategies", [])}
        if env.scheduler_name:
            scheduler_cfg = getattr(core_config, "multi_strategy_schedulers", {}).get(env.scheduler_name)
            if scheduler_cfg:
                schedule_entries: list[dict[str, Any]] = []
                for schedule in getattr(scheduler_cfg, "schedules", []):
                    schedule_payload: dict[str, Any] = {
                        "name": schedule.name,
                        "strategy": schedule.strategy,
                        "cadence_seconds": schedule.cadence_seconds,
                        "max_drift_seconds": schedule.max_drift_seconds,
                        "warmup_bars": schedule.warmup_bars,
                        "risk_profile": schedule.risk_profile,
                        "max_signals": schedule.max_signals,
                    }
                    definition_payload = summaries_by_name.get(schedule.strategy)
                    if definition_payload:
                        schedule_payload["engine"] = definition_payload.get("engine")
                        schedule_payload["license_tier"] = definition_payload.get("license_tier")
                        schedule_payload["risk_classes"] = definition_payload.get("risk_classes")
                        schedule_payload["required_data"] = definition_payload.get("required_data")
                        if definition_payload.get("capability"):
                            schedule_payload["capability"] = definition_payload["capability"]
                        if definition_payload.get("tags"):
                            schedule_payload["tags"] = definition_payload["tags"]
                    schedule_entries.append(schedule_payload)

                license_tiers = sorted(
                    {
                        entry.get("license_tier")
                        for entry in schedule_entries
                        if entry.get("license_tier")
                    }
                )
                capability_set = sorted(
                    {entry.get("capability") for entry in schedule_entries if entry.get("capability")}
                )
                scheduler_summary: dict[str, Any] = {
                    "name": scheduler_cfg.name,
                    "strategies": schedule_entries,
                }
                if license_tiers:
                    scheduler_summary["license_tiers"] = license_tiers
                if capability_set:
                    scheduler_summary["capabilities"] = capability_set
                entry["scheduler_plan"] = scheduler_summary

        report[name] = entry

    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local orchestrator for OEM environments")
    parser.add_argument("--state-file", default="var/orchestrator/state.json", help="Ścieżka pliku stanu orchestratora")
    parser.add_argument("--base-dir", default="var/orchestrator", help="Katalog z artefaktami orchestratora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Przygotuj katalog środowiska")
    prepare_parser.add_argument("environment", choices=sorted(DEFAULT_ENVIRONMENTS))
    prepare_parser.set_defaults(func=cmd_prepare)

    bundle_parser = subparsers.add_parser("bundle", help="Zbuduj bundla PyInstaller dla środowiska")
    bundle_parser.add_argument("environment", choices=sorted(DEFAULT_ENVIRONMENTS))
    bundle_parser.add_argument("--platform", required=True)
    bundle_parser.add_argument("--version", required=True)
    bundle_parser.add_argument("--entrypoint", default="scripts/run_multi_strategy_scheduler.py")
    bundle_parser.add_argument("--qt-dist")
    bundle_parser.add_argument("--briefcase-project")
    bundle_parser.add_argument("--hidden-import", action="append")
    bundle_parser.add_argument("--runtime-name")
    bundle_parser.add_argument("--include", action="append")
    bundle_parser.add_argument("--signing-key")
    bundle_parser.add_argument("--signing-key-id")
    bundle_parser.add_argument(
        "--allowed-profile",
        action="append",
        help="Dodaj profil OEM uprawniony do aktualizacji (można powtarzać)",
    )
    bundle_parser.add_argument(
        "--metadata",
        action="append",
        help="Dodaj wpis metadata w formacie klucz=wartość (wartość może być JSON, klucz może być kropkowany)",
    )
    bundle_parser.add_argument(
        "--metadata-file",
        action="append",
        help="Wczytaj dodatkowe metadane z pliku JSON",
    )
    bundle_parser.add_argument(
        "--metadata-url",
        action="append",
        help="Pobierz metadane z adresu URL zwracającego obiekt JSON",
    )
    bundle_parser.add_argument(
        "--metadata-url-header",
        action="append",
        help="Dołącz nagłówek HTTP do zapytań --metadata-url w formacie Nazwa=Wartość",
    )
    bundle_parser.add_argument(
        "--metadata-url-timeout",
        type=float,
        help="Limit czasu (sekundy) na pobranie metadanych z URL (domyślnie 10)",
    )
    bundle_parser.add_argument(
        "--metadata-url-max-size",
        type=int,
        help="Maksymalny rozmiar (w bajtach) odpowiedzi metadanych pobieranej z URL",
    )
    bundle_parser.add_argument(
        "--metadata-url-allow-http",
        action="store_true",
        help="Zezwól na pobieranie metadanych przez HTTP (domyślnie tylko HTTPS)",
    )
    bundle_parser.add_argument(
        "--metadata-url-allowed-host",
        action="append",
        help="Ogranicz pobieranie metadanych do wskazanych hostów (można podać wiele)",
    )
    bundle_parser.add_argument(
        "--metadata-url-cert-fingerprint",
        action="append",
        help="Dodatkowa weryfikacja TLS: lista odcisków certyfikatów w formacie algorytm:HEX",
    )
    bundle_parser.add_argument(
        "--metadata-url-cert-subject",
        action="append",
        help=(
            "Wymagaj, aby certyfikat TLS źródła metadanych zawierał wskazane atrybuty tematu w "
            "formacie Atrybut=Wartość (np. commonName=updates.example.com)"
        ),
    )
    bundle_parser.add_argument(
        "--metadata-url-cert-issuer",
        action="append",
        help=(
            "Wymagaj, aby certyfikat TLS źródła metadanych został wystawiony przez CA zawierające "
            "wskazane atrybuty w formacie Atrybut=Wartość (np. organizationName=Trusted CA)"
        ),
    )
    bundle_parser.add_argument(
        "--metadata-url-cert-san",
        action="append",
        help=(
            "Wymagaj, aby certyfikat TLS źródła metadanych zawierał wpisy subjectAltName "
            "w formacie Typ=Wartość (np. DNS=updates.example.com)"
        ),
    )
    bundle_parser.add_argument(
        "--metadata-url-cert-eku",
        action="append",
        help=(
            "Wymagaj, aby certyfikat TLS źródła metadanych zawierał rozszerzenia Extended Key Usage "
            "(np. serverAuth)"
        ),
    )
    bundle_parser.add_argument(
        "--metadata-url-cert-policy",
        action="append",
        help=(
            "Wymagaj, aby certyfikat TLS źródła metadanych zawierał identyfikatory certificatePolicies "
            "(np. anyPolicy lub OID)"
        ),
    )
    bundle_parser.add_argument(
        "--metadata-url-cert-serial",
        action="append",
        help=(
            "Wymagaj, aby certyfikat TLS źródła metadanych posiadał numer seryjny z listy dozwolonych "
            "(format dziesiętny, szesnastkowy lub z dwukropkami)"
        ),
    )
    bundle_parser.add_argument(
        "--metadata-url-ca",
        help="Ścieżka do dodatkowego pliku CA używanego przy pobieraniu metadanych",
    )
    bundle_parser.add_argument(
        "--metadata-url-capath",
        help="Katalog z dodatkowymi certyfikatami CA dla pobierania metadanych",
    )
    bundle_parser.add_argument(
        "--metadata-url-client-cert",
        help="Certyfikat klienta (PEM) wykorzystywany przy uwierzytelnieniu TLS metadanych",
    )
    bundle_parser.add_argument(
        "--metadata-url-client-key",
        help="Klucz prywatny klienta (PEM) używany wraz z certyfikatem metadanych",
    )
    bundle_parser.add_argument(
        "--metadata-ini",
        action="append",
        help="Wczytaj metadane z pliku INI (obsługa sekcji oraz kluczy z __ jako separatora kropki)",
    )
    bundle_parser.add_argument(
        "--metadata-toml",
        action="append",
        help="Wczytaj metadane z pliku TOML (obsługa zagnieżdżonych tabel i kluczy kropkowanych)",
    )
    bundle_parser.add_argument(
        "--metadata-yaml",
        action="append",
        help="Wczytaj metadane z pliku YAML (obsługa kluczy kropkowanych i wartości JSON)",
    )
    bundle_parser.add_argument(
        "--metadata-dotenv",
        action="append",
        help="Wczytaj metadane z pliku .env (wspiera __ jako separator kropkowanych kluczy)",
    )
    bundle_parser.add_argument(
        "--metadata-env-prefix",
        action="append",
        help=(
            "Zaczytaj metadane ze zmiennych środowiskowych rozpoczynających się od prefiksu; "
            "po prefiksie użyj __ jako separatora segmentów klucza"
        ),
    )
    bundle_parser.set_defaults(func=cmd_bundle)

    launch_parser = subparsers.add_parser("launch", help="Uruchom runtime środowiska")
    launch_parser.add_argument("environment", choices=sorted(DEFAULT_ENVIRONMENTS))
    launch_parser.add_argument("--run-once", action="store_true")
    launch_parser.add_argument("--extra-arg", action="append")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.set_defaults(func=cmd_launch)

    verify_parser = subparsers.add_parser("verify-update", help="Zweryfikuj podpisane paczki aktualizacji")
    verify_parser.add_argument("environment", choices=sorted(DEFAULT_ENVIRONMENTS))
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--bundle-dir", required=True)
    verify_parser.add_argument("--signature")
    verify_parser.add_argument("--signing-key")
    verify_parser.add_argument("--core-config")
    verify_parser.set_defaults(func=cmd_verify_update)

    status_parser = subparsers.add_parser("status", help="Wyświetl stan orchestratora")
    status_parser.set_defaults(func=cmd_status)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    environments = DEFAULT_ENVIRONMENTS
    command = args.command
    state_path = Path(args.state_file).expanduser().resolve()

    kwargs = {"environments": environments}
    if command in {"bundle", "launch", "status"}:
        kwargs["state_path"] = state_path

    args.func(args, **kwargs)  # type: ignore[misc]
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
