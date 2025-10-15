"""Koordynator akceptacji end-to-end dla dystrybucji OEM."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, List, Mapping, Sequence

from deploy.packaging.build_core_bundle import SUPPORTED_PLATFORMS, build_from_cli
from scripts.export_observability_bundle import main as run_observability_bundle
from scripts.generate_mtls_bundle import main as generate_mtls_bundle
from scripts.oem_provision_license import main as provision_license
from scripts.rotate_keys import run as run_rotate_keys
from scripts.run_decision_engine_smoke import main as run_decision_engine_smoke
from scripts.run_risk_simulation_lab import main as run_risk_simulation
from scripts.run_tco_analysis import run as run_tco_analysis
from scripts.slo_monitor import run as run_slo_monitor
from bot_core.config.loader import load_core_config
from bot_core.security.signing import build_hmac_signature


@dataclass(slots=True)
class StepOutcome:
    """Reprezentuje wynik pojedynczego kroku akceptacyjnego."""

    step: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


class AcceptanceError(RuntimeError):
    """Wyjątek podnoszony w przypadku niepowodzenia kroku."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uruchamia pakiet kroków akceptacyjnych OEM (bundle -> licencja -> Paper Labs -> mTLS)."
    )

    parser.add_argument("--summary-path", help="Ścieżka do pliku JSON z podsumowaniem akceptacji")
    parser.add_argument("--print-summary", action="store_true", help="Wypisz podsumowanie na stdout")
    parser.add_argument("--fail-fast", action="store_true", help="Zatrzymaj proces po pierwszym błędzie")

    parser.add_argument(
        "--artifact-root",
        default="var/audit/acceptance",
        help="Katalog bazowy, w którym zostaną zapisane artefakty akceptacyjne",
    )

    parser.add_argument("--skip-bundle", action="store_true", help="Pomiń etap budowania bundla")
    parser.add_argument("--skip-license", action="store_true", help="Pomiń provisioning licencji")
    parser.add_argument("--skip-risk", action="store_true", help="Pomiń symulacje Paper Labs")
    parser.add_argument("--skip-mtls", action="store_true", help="Pomiń generowanie pakietu mTLS")
    parser.add_argument("--skip-tco", action="store_true", help="Pomiń analizę kosztów transakcyjnych (TCO)")
    parser.add_argument(
        "--skip-decision",
        action="store_true",
        help="Pomiń smoke test DecisionOrchestratora",
    )
    parser.add_argument(
        "--skip-slo",
        action="store_true",
        help="Pomiń generowanie raportu SLO observability",
    )
    parser.add_argument(
        "--skip-rotation",
        action="store_true",
        help="Pomiń plan rotacji kluczy Stage5",
    )
    parser.add_argument(
        "--skip-observability",
        action="store_true",
        help="Pomiń budowanie paczki obserwowalności Stage5",
    )

    # Parametry decision logu
    parser.add_argument("--decision-log-path", help="Ścieżka do decision logu JSONL")
    parser.add_argument(
        "--decision-log-hmac-key",
        help="Wartość klucza HMAC (ciąg znaków) do podpisu decision logu",
    )
    parser.add_argument(
        "--decision-log-hmac-key-file",
        help="Plik zawierający klucz HMAC do podpisu decision logu",
    )
    parser.add_argument("--decision-log-key-id", help="Identyfikator klucza decision logu")
    parser.add_argument(
        "--decision-log-category",
        default="release.oem.acceptance",
        help="Kategoria wpisu decision logu",
    )
    parser.add_argument("--decision-log-notes", help="Notatka dołączana do wpisu decision logu")
    parser.add_argument(
        "--decision-log-allow-unsigned",
        action="store_true",
        help="Pozwól na dodanie wpisu decision logu bez podpisu (niezalecane)",
    )

    # Parametry bundla
    parser.add_argument("--bundle-platform", choices=sorted(SUPPORTED_PLATFORMS))
    parser.add_argument("--bundle-version", help="Wersja bundla Core OEM")
    parser.add_argument("--bundle-signing-key", help="Plik z kluczem HMAC do podpisu bundla")
    parser.add_argument("--bundle-daemon", action="append", default=[], help="Ścieżka do artefaktu demona (wielokrotna)")
    parser.add_argument("--bundle-ui", action="append", default=[], help="Ścieżka do artefaktu UI (wielokrotna)")
    parser.add_argument(
        "--bundle-config",
        action="append",
        default=[],
        help="Wpis konfiguracyjny w formacie relatywna_sciezka=plik",
    )
    parser.add_argument(
        "--bundle-resource",
        action="append",
        default=[],
        help="Dodatkowy zasób w formacie katalog=plik",
    )
    parser.add_argument(
        "--bundle-output-dir",
        help="Katalog docelowy dla archiwum bundla (domyślnie var/dist)",
    )
    parser.add_argument(
        "--bundle-fingerprint-placeholder",
        default="UNPROVISIONED",
        help="Wartość fingerprintu zapisywana w bundlu",
    )

    # Parametry licencji
    parser.add_argument("--license-signing-key", help="Plik z kluczem HMAC licencji")
    parser.add_argument("--license-key-id", help="Identyfikator klucza licencyjnego")
    parser.add_argument("--license-fingerprint", help="Fingerprint urządzenia OEM")
    parser.add_argument("--license-fingerprint-file", help="Plik z fingerprintem urządzenia")
    parser.add_argument("--license-profile", default="paper", help="Profil pracy zapisywany w licencji")
    parser.add_argument(
        "--license-valid-days",
        type=int,
        default=365,
        help="Okres ważności licencji w dniach",
    )
    parser.add_argument("--license-registry", help="Ścieżka rejestru licencji JSONL")
    parser.add_argument(
        "--license-rotation-log",
        help="Plik logu rotacji klucza licencji (jeśli wymagany)",
    )
    parser.add_argument(
        "--license-rotation-interval-days",
        type=float,
        default=90.0,
        help="Interwał rotacji klucza licencyjnego",
    )
    parser.add_argument("--license-notes", help="Notatka dołączana do licencji")
    parser.add_argument(
        "--license-feature",
        action="append",
        dest="license_features",
        default=[],
        help="Flagi funkcjonalne w licencji (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--license-bundle-version",
        help="Wersja bundla wpisywana do licencji (domyślnie taka jak budowany bundle)",
    )

    # Parametry Paper Labs
    parser.add_argument("--risk-config", help="Plik config/core.yaml używany przez Paper Labs")
    parser.add_argument("--risk-environment", help="Nazwa środowiska konfiguracyjnego (np. paper)")
    parser.add_argument("--risk-dataset-root", help="Nadpisanie ścieżki do danych Parquet")
    parser.add_argument("--risk-namespace", default="binance_spot", help="Namespace danych Parquet")
    parser.add_argument(
        "--risk-symbol",
        action="append",
        dest="risk_symbols",
        default=[],
        help="Symbol instrumentu (można podać wielokrotnie)",
    )
    parser.add_argument("--risk-interval", default="1h", help="Interwał świec do symulacji")
    parser.add_argument("--risk-max-bars", type=int, default=720, help="Maks. liczba świec")
    parser.add_argument(
        "--risk-base-equity",
        type=float,
        default=100_000.0,
        help="Kapitał początkowy w USD",
    )
    parser.add_argument("--risk-output-dir", help="Katalog na raporty Paper Labs")
    parser.add_argument("--risk-json-name", default="risk_simulation_report.json", help="Nazwa pliku JSON")
    parser.add_argument("--risk-pdf-name", default="risk_simulation_report.pdf", help="Nazwa pliku PDF")
    parser.add_argument("--risk-fail-on-breach", action="store_true", help="Przerwij jeśli symulacje wykryją naruszenia")
    parser.add_argument(
        "--risk-synthetic-fallback",
        action="store_true",
        dest="risk_synthetic_fallback",
        help="Wymuś generowanie syntetycznych danych",
    )
    parser.add_argument(
        "--risk-disable-synthetic-fallback",
        action="store_false",
        dest="risk_synthetic_fallback",
        help="Wyłącz generowanie syntetycznych danych",
    )
    parser.set_defaults(risk_synthetic_fallback=True)

    # Parametry mTLS
    parser.add_argument("--mtls-output-dir", help="Katalog docelowy pakietu mTLS")
    parser.add_argument("--mtls-bundle-name", default="core-oem", help="Prefiks plików mTLS")
    parser.add_argument("--mtls-common-name", default="Dudzian OEM", help="CN certyfikatu")
    parser.add_argument("--mtls-organization", default="Dudzian", help="Pole O certyfikatu")
    parser.add_argument("--mtls-valid-days", type=int, default=365, help="Ważność certyfikatów")
    parser.add_argument("--mtls-key-size", type=int, default=4096, help="Rozmiar klucza RSA")
    parser.add_argument(
        "--mtls-server-hostname",
        action="append",
        dest="mtls_server_hostnames",
        default=[],
        help="Hostname/IP dodawane do SAN certyfikatu serwera",
    )
    parser.add_argument(
        "--mtls-rotation-registry",
        help="Plik rejestru rotacji TLS (opcjonalny)",
    )
    parser.add_argument(
        "--mtls-ca-passphrase-env",
        help="Zmienne ENV z passphrase dla klucza CA",
    )
    parser.add_argument(
        "--mtls-server-passphrase-env",
        help="Zmienne ENV z passphrase klucza serwera",
    )
    parser.add_argument(
        "--mtls-client-passphrase-env",
        help="Zmienne ENV z passphrase klucza klienta",
    )

    # Parametry TCO
    parser.add_argument("--tco-fill", action="append", dest="tco_fills", default=[], help="Plik JSONL z fillami (wielokrotnie)")
    parser.add_argument(
        "--tco-output-dir",
        default="var/audit/tco",
        help="Katalog docelowy raportu TCO",
    )
    parser.add_argument(
        "--tco-basename",
        default="oem_acceptance_tco",
        help="Nazwa bazowa artefaktów raportu TCO",
    )
    parser.add_argument("--tco-signing-key", help="Klucz HMAC do podpisu raportu TCO")
    parser.add_argument("--tco-signing-key-id", help="Identyfikator klucza TCO")
    parser.add_argument(
        "--tco-cost-limit-bps",
        type=float,
        default=None,
        help="Limit kosztów w bps generujący alert w raporcie TCO",
    )
    parser.add_argument(
        "--tco-metadata",
        action="append",
        default=[],
        help="Dodatkowe metadane TCO (klucz=wartość)",
    )

    # Parametry DecisionOrchestratora
    parser.add_argument("--decision-config", help="Konfiguracja core.yaml używana przez decision engine")
    parser.add_argument("--decision-risk-snapshot", help="Plik JSON z snapshotem ryzyka profili")
    parser.add_argument("--decision-candidates", help="Plik JSON z kandydatami decyzji")
    parser.add_argument(
        "--decision-output",
        help="Ścieżka wynikowego raportu DecisionOrchestratora",
    )
    parser.add_argument(
        "--decision-allow-empty",
        action="store_true",
        help="Nie traktuj braku akceptowanych decyzji jako błędu",
    )
    parser.add_argument("--decision-signing-key", help="Wartość klucza podpisu raportu decision engine")
    parser.add_argument("--decision-signing-key-env", help="Nazwa zmiennej ENV z kluczem podpisu")
    parser.add_argument("--decision-signing-key-file", help="Plik z kluczem podpisu decision engine")
    parser.add_argument("--decision-signing-key-id", help="Identyfikator klucza decision engine")
    parser.add_argument(
        "--decision-tco-report",
        help="Opcjonalna ścieżka do raportu TCO wykorzystywana przez orchestrator",
    )

    # Parametry SLO observability
    parser.add_argument("--slo-config", help="Konfiguracja core.yaml z definicjami SLO")
    parser.add_argument("--slo-metric", action="append", dest="slo_metrics", default=[], help="Plik JSONL z metrykami SLO (wielokrotnie)")
    parser.add_argument("--slo-output-dir", default="var/audit/slo", help="Katalog raportów SLO")
    parser.add_argument("--slo-basename", default="oem_slo_report", help="Nazwa bazowa raportu SLO")
    parser.add_argument("--slo-signing-key", help="Klucz podpisu raportu SLO")
    parser.add_argument("--slo-signing-key-id", help="Identyfikator klucza raportu SLO")
    parser.add_argument(
        "--slo-metadata",
        action="append",
        default=[],
        help="Metadane raportu SLO (klucz=wartość)",
    )

    # Parametry rotacji kluczy Stage5
    parser.add_argument(
        "--rotation-config",
        help="Konfiguracja core.yaml używana do planu rotacji kluczy",
    )
    parser.add_argument(
        "--rotation-registry",
        help="Ścieżka rejestru rotacji (nadpisuje konfigurację)",
    )
    parser.add_argument(
        "--rotation-output-dir",
        help="Katalog docelowy raportu rotacji (domyślnie z konfiguracji)",
    )
    parser.add_argument(
        "--rotation-basename",
        default="rotation_plan_oem",
        help="Nazwa bazowa raportu rotacji",
    )
    parser.add_argument(
        "--rotation-execute",
        action="store_true",
        help="Zapisz aktualizację rejestru rotacji podczas akceptacji",
    )
    parser.add_argument(
        "--rotation-interval-override",
        type=float,
        default=None,
        help="Globalne nadpisanie interwału rotacji (dni)",
    )
    parser.add_argument(
        "--rotation-warn-override",
        type=float,
        default=None,
        help="Globalne nadpisanie progu ostrzeżeń rotacji (dni)",
    )

    # Parametry paczki obserwowalności
    parser.add_argument("--observability-version", help="Wersja paczki observability Stage5")
    parser.add_argument(
        "--observability-output-dir",
        help="Katalog docelowy paczki observability",
    )
    parser.add_argument(
        "--observability-signing-key",
        help="Plik z kluczem podpisu paczki observability",
    )
    parser.add_argument("--observability-key-id", help="Identyfikator klucza paczki observability")
    parser.add_argument(
        "--observability-dashboard",
        action="append",
        dest="observability_dashboards",
        default=[],
        help="Dodatkowy plik dashboardu Grafany",
    )
    parser.add_argument(
        "--observability-alert",
        action="append",
        dest="observability_alerts",
        default=[],
        help="Dodatkowy plik reguł alertowych Prometheusa",
    )

    return parser


def _ensure_paths_exist(values: Iterable[str], description: str) -> None:
    missing = [candidate for candidate in values if candidate and not Path(candidate).expanduser().exists()]
    if missing:
        raise AcceptanceError(f"{description}: brakujące ścieżki: {', '.join(missing)}")


def _run_bundle_step(args: argparse.Namespace) -> dict[str, Any]:
    required_fields = {
        "--bundle-platform": args.bundle_platform,
        "--bundle-version": args.bundle_version,
        "--bundle-signing-key": args.bundle_signing_key,
    }
    for flag, value in required_fields.items():
        if not value:
            raise AcceptanceError(f"Brak parametru {flag} dla etapu bundla")
    if not args.bundle_daemon:
        raise AcceptanceError("Należy wskazać co najmniej jeden artefakt demona (--bundle-daemon)")
    if not args.bundle_ui:
        raise AcceptanceError("Należy wskazać co najmniej jeden artefakt UI (--bundle-ui)")
    if not args.bundle_config:
        raise AcceptanceError("Należy dodać przynajmniej jeden plik konfiguracyjny (--bundle-config)")

    _ensure_paths_exist(
        [args.bundle_signing_key, *(args.bundle_daemon), *(args.bundle_ui)],
        "Etap bundla",
    )
    for entry in args.bundle_config + args.bundle_resource:
        path = entry.split("=", 1)[-1] if "=" in entry else entry
        _ensure_paths_exist([path], "Etap bundla (config/resource)")

    cli_args: List[str] = [
        "--platform",
        args.bundle_platform,
        "--version",
        args.bundle_version,
        "--signing-key-path",
        args.bundle_signing_key,
    ]
    for path in args.bundle_daemon:
        cli_args.extend(["--daemon", path])
    for path in args.bundle_ui:
        cli_args.extend(["--ui", path])
    for config_entry in args.bundle_config:
        cli_args.extend(["--config", config_entry])
    for resource_entry in args.bundle_resource:
        cli_args.extend(["--resource", resource_entry])
    if args.bundle_output_dir:
        cli_args.extend(["--output-dir", args.bundle_output_dir])
    if args.bundle_fingerprint_placeholder:
        cli_args.extend(["--fingerprint-placeholder", args.bundle_fingerprint_placeholder])

    bundle_path = build_from_cli(cli_args)
    return {"archive": str(bundle_path), "platform": args.bundle_platform, "version": args.bundle_version}


def _run_license_step(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--license-signing-key": args.license_signing_key,
        "--license-registry": args.license_registry,
    }
    if not (args.license_fingerprint or args.license_fingerprint_file):
        required["--license-fingerprint"] = None
    for flag, value in required.items():
        if not value:
            raise AcceptanceError(f"Brak parametru {flag} dla provisioning licencji")

    _ensure_paths_exist([args.license_signing_key], "Etap licencji")
    if args.license_fingerprint_file:
        _ensure_paths_exist([args.license_fingerprint_file], "Etap licencji (fingerprint)")

    bundle_version = args.license_bundle_version or args.bundle_version or "0.0.0"

    cli_args: List[str] = [
        "--signing-key-path",
        args.license_signing_key,
        "--profile",
        args.license_profile,
        "--bundle-version",
        bundle_version,
        "--output",
        args.license_registry,
        "--valid-days",
        str(args.license_valid_days),
    ]
    if args.license_key_id:
        cli_args.extend(["--key-id", args.license_key_id])
    if args.license_rotation_log:
        cli_args.extend(["--rotation-log", args.license_rotation_log])
        cli_args.extend([
            "--rotation-interval-days",
            str(args.license_rotation_interval_days),
        ])
    if args.license_notes:
        cli_args.extend(["--notes", args.license_notes])
    for feature in args.license_features:
        cli_args.extend(["--feature", feature])
    if args.license_fingerprint_file:
        cli_args.extend(["--fingerprint-file", args.license_fingerprint_file])
    elif args.license_fingerprint:
        cli_args.extend(["--fingerprint", args.license_fingerprint])

    exit_code = provision_license(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"Provisioning licencji zakończony kodem {exit_code}")

    return {
        "registry": str(Path(args.license_registry).expanduser().resolve()),
        "fingerprint_source": "file" if args.license_fingerprint_file else "value",
        "bundle_version": bundle_version,
    }


def _run_risk_step(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--risk-config": args.risk_config,
        "--risk-output-dir": args.risk_output_dir,
    }
    for flag, value in required.items():
        if not value:
            raise AcceptanceError(f"Brak parametru {flag} dla Paper Labs")

    _ensure_paths_exist([args.risk_config], "Etap Paper Labs")
    if args.risk_dataset_root:
        _ensure_paths_exist([args.risk_dataset_root], "Paper Labs dataset")

    cli_args: List[str] = [
        "--config",
        args.risk_config,
        "--output-dir",
        args.risk_output_dir,
        "--interval",
        args.risk_interval,
        "--max-bars",
        str(args.risk_max_bars),
        "--base-equity",
        str(args.risk_base_equity),
        "--json-output",
        args.risk_json_name,
        "--pdf-output",
        args.risk_pdf_name,
    ]
    if args.risk_environment:
        cli_args.extend(["--environment", args.risk_environment])
    if args.risk_dataset_root:
        cli_args.extend(["--dataset-root", args.risk_dataset_root])
    if args.risk_namespace:
        cli_args.extend(["--namespace", args.risk_namespace])
    symbols = args.risk_symbols or []
    if symbols:
        cli_args.append("--symbols")
        cli_args.extend(symbols)
    if args.risk_synthetic_fallback:
        cli_args.append("--synthetic-fallback")
    if args.risk_fail_on_breach:
        cli_args.append("--fail-on-breach")

    exit_code = run_risk_simulation(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"Paper Labs zakończyło się kodem {exit_code}")

    output_dir = Path(args.risk_output_dir).expanduser().resolve()
    return {
        "json_report": str(output_dir / args.risk_json_name),
        "pdf_report": str(output_dir / args.risk_pdf_name),
    }


def _run_mtls_step(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--mtls-output-dir": args.mtls_output_dir,
    }
    for flag, value in required.items():
        if not value:
            raise AcceptanceError(f"Brak parametru {flag} dla generowania mTLS")

    cli_args: List[str] = [
        "--output-dir",
        args.mtls_output_dir,
        "--bundle-name",
        args.mtls_bundle_name,
        "--common-name",
        args.mtls_common_name,
        "--organization",
        args.mtls_organization,
        "--valid-days",
        str(args.mtls_valid_days),
        "--key-size",
        str(args.mtls_key_size),
    ]
    for host in args.mtls_server_hostnames or []:
        cli_args.extend(["--server-hostname", host])
    if args.mtls_rotation_registry:
        cli_args.extend(["--rotation-registry", args.mtls_rotation_registry])
    if args.mtls_ca_passphrase_env:
        cli_args.extend(["--ca-key-passphrase-env", args.mtls_ca_passphrase_env])
    if args.mtls_server_passphrase_env:
        cli_args.extend(["--server-key-passphrase-env", args.mtls_server_passphrase_env])
    if args.mtls_client_passphrase_env:
        cli_args.extend(["--client-key-passphrase-env", args.mtls_client_passphrase_env])

    exit_code = generate_mtls_bundle(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"Generowanie pakietu mTLS zakończyło się kodem {exit_code}")

    output_dir = Path(args.mtls_output_dir).expanduser().resolve()
    bundle_name = args.mtls_bundle_name
    return {
        "metadata": str(output_dir / f"{bundle_name}-metadata.json"),
        "ca_certificate": str(output_dir / f"{bundle_name}-ca.pem"),
        "server_certificate": str(output_dir / f"{bundle_name}-server.pem"),
        "client_certificate": str(output_dir / f"{bundle_name}-client.pem"),
    }


def _run_tco_step(args: argparse.Namespace) -> dict[str, Any]:
    if not args.tco_fills:
        raise AcceptanceError("Należy wskazać co najmniej jeden plik z transakcjami (--tco-fill)")
    if not args.tco_signing_key:
        raise AcceptanceError("Wymagany jest klucz podpisu raportu TCO (--tco-signing-key)")

    _ensure_paths_exist(args.tco_fills, "Etap TCO (fills)")
    _ensure_paths_exist([args.tco_signing_key], "Etap TCO (klucz podpisu)")

    cli_args: List[str] = []
    for fill in args.tco_fills:
        cli_args.extend(["--fills", fill])
    cli_args.extend(["--output-dir", args.tco_output_dir])
    basename = args.tco_basename or "oem_acceptance_tco"
    cli_args.extend(["--basename", basename])
    cli_args.extend(["--signing-key-path", args.tco_signing_key])
    if args.tco_signing_key_id:
        cli_args.extend(["--signing-key-id", args.tco_signing_key_id])
    if args.tco_cost_limit_bps is not None:
        cli_args.extend(["--cost-limit-bps", str(args.tco_cost_limit_bps)])
    for metadata in args.tco_metadata:
        cli_args.extend(["--metadata", metadata])

    exit_code = run_tco_analysis(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"Analiza TCO zakończyła się kodem {exit_code}")

    output_dir = Path(args.tco_output_dir).expanduser().resolve()
    csv_path = output_dir / f"{basename}.csv"
    pdf_path = output_dir / f"{basename}.pdf"
    json_path = output_dir / f"{basename}.json"
    for candidate in (csv_path, pdf_path, json_path):
        if not candidate.exists():
            raise AcceptanceError(f"Nie znaleziono artefaktu TCO: {candidate}")

    args._tco_report_path = str(json_path)

    details: dict[str, Any] = {
        "csv": str(csv_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
    }
    for label, path in ("csv", csv_path), ("pdf", pdf_path), ("json", json_path):
        signature = path.with_suffix(path.suffix + ".sig")
        if signature.exists():
            details[f"{label}_signature"] = str(signature)
    if args.tco_cost_limit_bps is not None:
        details["cost_limit_bps"] = args.tco_cost_limit_bps
    return details


def _run_decision_step(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--decision-risk-snapshot": args.decision_risk_snapshot,
        "--decision-candidates": args.decision_candidates,
        "--decision-output": args.decision_output,
    }
    for flag, value in required.items():
        if not value:
            raise AcceptanceError(f"Brak parametru {flag} dla DecisionOrchestratora")

    config_path = args.decision_config or args.risk_config or str(Path("config") / "core.yaml")
    _ensure_paths_exist([config_path, args.decision_risk_snapshot, args.decision_candidates], "Etap DecisionOrchestrator")
    if args.decision_signing_key_file:
        _ensure_paths_exist([args.decision_signing_key_file], "DecisionOrchestrator (klucz plikowy)")

    cli_args: List[str] = [
        "--config",
        str(Path(config_path).expanduser()),
        "--risk-snapshot",
        str(Path(args.decision_risk_snapshot).expanduser()),
        "--candidates",
        str(Path(args.decision_candidates).expanduser()),
        "--output",
        str(Path(args.decision_output).expanduser()),
    ]

    tco_report = args.decision_tco_report or getattr(args, "_tco_report_path", None)
    if tco_report:
        cli_args.extend(["--tco-report", str(Path(tco_report).expanduser())])
    if args.decision_allow_empty:
        cli_args.append("--allow-empty")

    signing_args = [value for value in (args.decision_signing_key, args.decision_signing_key_env, args.decision_signing_key_file) if value]
    if len(signing_args) > 1:
        raise AcceptanceError(
            "Podaj klucz decision engine w jednej formie: wartość, zmienna ENV lub plik"
        )
    if args.decision_signing_key:
        cli_args.extend(["--signing-key", args.decision_signing_key])
    elif args.decision_signing_key_env:
        cli_args.extend(["--signing-key-env", args.decision_signing_key_env])
    elif args.decision_signing_key_file:
        cli_args.extend(["--signing-key-file", args.decision_signing_key_file])
    if args.decision_signing_key_id:
        cli_args.extend(["--signing-key-id", args.decision_signing_key_id])

    exit_code = run_decision_engine_smoke(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"DecisionOrchestrator zakończył się kodem {exit_code}")

    output_path = Path(args.decision_output).expanduser().resolve()
    if not output_path.exists():
        raise AcceptanceError(f"Brak raportu DecisionOrchestratora: {output_path}")
    details: dict[str, Any] = {"report": str(output_path)}
    signature = output_path.with_suffix(output_path.suffix + ".sig")
    if signature.exists():
        details["signature"] = str(signature)
    return details


def _run_slo_step(args: argparse.Namespace) -> dict[str, Any]:
    if not args.slo_metrics:
        raise AcceptanceError("Należy wskazać co najmniej jeden plik metryk (--slo-metric)")

    config_path = args.slo_config or args.risk_config or str(Path("config") / "core.yaml")
    _ensure_paths_exist([config_path], "Etap SLO (konfiguracja)")
    _ensure_paths_exist(args.slo_metrics, "Etap SLO (metryki)")
    if args.slo_signing_key:
        _ensure_paths_exist([args.slo_signing_key], "Etap SLO (klucz podpisu)")

    cli_args: List[str] = []
    for metric in args.slo_metrics:
        cli_args.extend(["--metrics", metric])
    cli_args.extend(["--config", str(Path(config_path).expanduser())])
    cli_args.extend(["--output-dir", args.slo_output_dir])
    cli_args.extend(["--basename", args.slo_basename])
    if args.slo_signing_key:
        cli_args.extend(["--signing-key-path", args.slo_signing_key])
        if args.slo_signing_key_id:
            cli_args.extend(["--signing-key-id", args.slo_signing_key_id])
    for metadata in args.slo_metadata:
        cli_args.extend(["--metadata", metadata])

    exit_code = run_slo_monitor(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"Monitor SLO zakończył się kodem {exit_code}")

    output_dir = Path(args.slo_output_dir).expanduser().resolve()
    report_path = output_dir / f"{args.slo_basename}.json"
    if not report_path.exists():
        raise AcceptanceError(f"Raport SLO nie został wygenerowany: {report_path}")
    details: dict[str, Any] = {"report": str(report_path)}
    signature_path = report_path.with_suffix(report_path.suffix + ".sig")
    if signature_path.exists():
        details["signature"] = str(signature_path)
    return details


def _run_rotation_step(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.rotation_config or args.slo_config or args.risk_config or str(Path("config") / "core.yaml")
    _ensure_paths_exist([config_path], "Etap rotacji kluczy (konfiguracja)")
    if args.rotation_registry:
        _ensure_paths_exist([args.rotation_registry], "Etap rotacji kluczy (rejestr)")

    cli_args: List[str] = ["--config", str(Path(config_path).expanduser())]
    if args.rotation_registry:
        cli_args.extend(["--registry-path", args.rotation_registry])
    if args.rotation_output_dir:
        cli_args.extend(["--output-dir", args.rotation_output_dir])
    if args.rotation_basename:
        cli_args.extend(["--basename", args.rotation_basename])
    if args.rotation_execute:
        cli_args.append("--execute")
    if args.rotation_interval_override is not None:
        cli_args.extend(["--interval-override", str(args.rotation_interval_override)])
    if args.rotation_warn_override is not None:
        cli_args.extend(["--warn-within-override", str(args.rotation_warn_override)])

    exit_code = run_rotate_keys(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"Plan rotacji kluczy zakończył się kodem {exit_code}")

    config_resolved = Path(config_path).expanduser().resolve()
    core_config = load_core_config(str(config_resolved))
    observability = getattr(core_config, "observability", None)
    rotation_cfg = getattr(observability, "key_rotation", None)
    if rotation_cfg is None:
        raise AcceptanceError("Konfiguracja nie zawiera sekcji observability.key_rotation")

    if args.rotation_output_dir:
        output_root = Path(args.rotation_output_dir).expanduser()
    else:
        output_root = Path(rotation_cfg.audit_directory)
    if not output_root.is_absolute():
        output_root = (config_resolved.parent / output_root).resolve()
    else:
        output_root = output_root.resolve()

    basename = args.rotation_basename or "rotation_plan_oem"
    plan_path = output_root / f"{basename}.json"
    if not plan_path.exists():
        raise AcceptanceError(f"Nie udało się odnaleźć wygenerowanego planu rotacji: {plan_path}")
    return {"plan": str(plan_path)}


def _run_observability_step(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--observability-version": args.observability_version,
        "--observability-output-dir": args.observability_output_dir,
        "--observability-signing-key": args.observability_signing_key,
    }
    for flag, value in required.items():
        if not value:
            raise AcceptanceError(f"Brak parametru {flag} dla paczki obserwowalności")

    _ensure_paths_exist([args.observability_signing_key], "Paczka obserwowalności (klucz)")
    _ensure_paths_exist(args.observability_dashboards or [], "Paczka obserwowalności (dashboard)")
    _ensure_paths_exist(args.observability_alerts or [], "Paczka obserwowalności (alert)")

    cli_args: List[str] = [
        "--version",
        args.observability_version,
        "--output-dir",
        args.observability_output_dir,
        "--signing-key",
        args.observability_signing_key,
    ]
    if args.observability_key_id:
        cli_args.extend(["--key-id", args.observability_key_id])
    for dashboard in args.observability_dashboards:
        cli_args.extend(["--dashboard", dashboard])
    for alert in args.observability_alerts:
        cli_args.extend(["--alert-rule", alert])

    exit_code = run_observability_bundle(cli_args)
    if exit_code != 0:
        raise AcceptanceError(f"Budowanie paczki obserwowalności zakończyło się kodem {exit_code}")

    output_dir = Path(args.observability_output_dir).expanduser().resolve()
    archive_name = f"observability-bundle-{args.observability_version}.tar.gz"
    archive_path = output_dir / archive_name
    if not archive_path.exists():
        raise AcceptanceError(f"Nie znaleziono paczki obserwowalności: {archive_path}")
    return {"archive": str(archive_path)}


def _record(summary: list[StepOutcome], outcome: StepOutcome) -> None:
    summary.append(outcome)


def _dump_summary(summary: list[StepOutcome], *, args: argparse.Namespace) -> list[Mapping[str, Any]]:
    payload = _serialize_summary(summary)
    if args.summary_path:
        path = Path(args.summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_summary:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _serialize_summary(summary: Sequence[StepOutcome]) -> list[Mapping[str, Any]]:
    return [asdict(item) for item in summary]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_decision_log_key(args: argparse.Namespace) -> bytes | None:
    inline = args.decision_log_hmac_key
    path = args.decision_log_hmac_key_file
    if inline and path:
        raise AcceptanceError("Podaj klucz HMAC jako wartość lub plik, nie oba jednocześnie")
    if inline:
        key = inline.encode("utf-8").strip()
    elif path:
        candidate_path = Path(path).expanduser()
        if not candidate_path.exists():
            raise AcceptanceError(f"Plik klucza decision logu nie istnieje: {candidate_path}")
        key = candidate_path.read_bytes().strip()
    else:
        return None
    if len(key) < 32:
        raise AcceptanceError("Klucz HMAC decision logu musi mieć co najmniej 32 bajty")
    return key


def _build_decision_log_entry(
    summary: Sequence[StepOutcome],
    *,
    args: argparse.Namespace,
    exit_code: int,
) -> dict[str, Any]:
    serialized_steps = _serialize_summary(summary)
    status = "ok" if exit_code == 0 and all(item.status == "ok" for item in summary) else "failed"

    entry: dict[str, Any] = {
        "schema": "core.oem.acceptance",
        "schema_version": "1.0",
        "timestamp": _now_iso(),
        "status": status,
        "steps": serialized_steps,
        "exit_code": exit_code,
    }
    if args.decision_log_category:
        entry["category"] = args.decision_log_category
    context: dict[str, Any] = {}
    if args.bundle_version:
        context["bundle_version"] = args.bundle_version
    if args.bundle_platform:
        context["bundle_platform"] = args.bundle_platform
    if args.license_profile:
        context["license_profile"] = args.license_profile
    if context:
        entry["context"] = context
    if args.decision_log_notes:
        entry["notes"] = args.decision_log_notes
    return entry


def _append_decision_log_entry(
    summary: Sequence[StepOutcome],
    *,
    args: argparse.Namespace,
    exit_code: int,
) -> tuple[Path, Mapping[str, Any]] | None:
    if not args.decision_log_path:
        return None

    entry = _build_decision_log_entry(summary, args=args, exit_code=exit_code)
    key = _resolve_decision_log_key(args)
    signed_entry = dict(entry)
    if key is None:
        if not args.decision_log_allow_unsigned:
            raise AcceptanceError(
                "Brak klucza HMAC decision logu – podaj --decision-log-hmac-key lub --decision-log-hmac-key-file",
            )
    else:
        signed_entry["signature"] = build_hmac_signature(
            entry,
            key=key,
            algorithm="HMAC-SHA256",
            key_id=args.decision_log_key_id,
        )

    path = Path(args.decision_log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(signed_entry, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return path, signed_entry


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _prepare_artifact_directory(root: str | None) -> Path | None:
    if not root:
        return None
    base = Path(root).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    slug = _timestamp_slug()
    candidate = base / slug
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = base / f"{slug}-{counter:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_bytes(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def _safe_member_path(name: str) -> Path:
    member = PurePosixPath(name)
    if member.is_absolute() or any(part in {"", ".."} for part in member.parts):
        raise AcceptanceError(f"Niebezpieczna ścieżka w archiwum bundla: {name}")
    return Path(*member.parts)


def _extract_bundle_metadata(archive_path: Path, destination: Path) -> list[str]:
    extracted: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    signatures_dir = destination / "signatures"
    signatures_dir.mkdir(parents=True, exist_ok=True)

    def _store_member(member_name: str, data: bytes) -> None:
        safe_path = _safe_member_path(member_name)
        parts = list(safe_path.parts)
        if parts and parts[0] == "core_oem_staging":
            parts = parts[1:]
        if not parts:
            return
        normalized = Path(*parts)
        if normalized.suffix == ".sig" or normalized.name == "manifest.json":
            target_base = signatures_dir if normalized.suffix == ".sig" else destination
            target = target_base / normalized
            _write_bytes(target, data)
            extracted.append(str(target))

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                try:
                    _store_member(info.filename, archive.read(info))
                except AcceptanceError:
                    raise
    else:
        mode = "r:gz" if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz" else "r"
        with tarfile.open(archive_path, mode) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue
                data = file_obj.read()
                _store_member(member.name, data)
    return extracted


def _copy_artifact(source: str | Path, destination: Path) -> str:
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise AcceptanceError(f"Artefakt nie istnieje: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination)


def _publish_artifacts(
    *,
    summary: Sequence[StepOutcome],
    serialized_summary: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    exit_code: int,
    decision_log: tuple[Path, Mapping[str, Any]] | None,
) -> Path | None:
    artifact_dir = _prepare_artifact_directory(args.artifact_root)
    if artifact_dir is None:
        return None

    metadata: dict[str, Any] = {
        "generated_at": _now_iso(),
        "exit_code": exit_code,
        "steps": list(serialized_summary),
    }

    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        json.dumps(serialized_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["summary_path"] = str(summary_path)
    if args.summary_path:
        metadata["source_summary_path"] = str(Path(args.summary_path).expanduser())

    details_by_step = {item.step: item.details for item in summary}

    bundle_details = details_by_step.get("bundle")
    if bundle_details and "archive" in bundle_details:
        bundle_dir = artifact_dir / "bundle"
        archive_path = Path(bundle_details["archive"]).expanduser()
        copied_archive = bundle_dir / archive_path.name
        copied_path = _copy_artifact(archive_path, copied_archive)
        metadata.setdefault("bundle", {})["archive"] = copied_path
        extracted = _extract_bundle_metadata(archive_path, bundle_dir)
        metadata["bundle"]["extracted"] = extracted

    license_details = details_by_step.get("license")
    if license_details and "registry" in license_details:
        license_dir = artifact_dir / "license"
        registry_path = Path(license_details["registry"]).expanduser()
        copied_registry = license_dir / registry_path.name
        copied_path = _copy_artifact(registry_path, copied_registry)
        metadata.setdefault("license", {})["registry"] = copied_path

    risk_details = details_by_step.get("risk")
    if risk_details:
        risk_dir = artifact_dir / "paper_labs"
        risk_metadata: dict[str, str] = {}
        for key in ("json_report", "pdf_report"):
            path_value = risk_details.get(key)
            if not path_value:
                continue
            risk_path = Path(path_value).expanduser()
            copied = risk_dir / risk_path.name
            copied_path = _copy_artifact(risk_path, copied)
            risk_metadata[key] = copied_path
        if risk_metadata:
            metadata["paper_labs"] = risk_metadata

    mtls_details = details_by_step.get("mtls")
    if mtls_details:
        mtls_dir = artifact_dir / "mtls"
        mtls_metadata: dict[str, str] = {}
        for key, value in mtls_details.items():
            if not value:
                continue
            source_path = Path(value).expanduser()
            copied = mtls_dir / source_path.name
            copied_path = _copy_artifact(source_path, copied)
            mtls_metadata[key] = copied_path
        if mtls_metadata:
            metadata["mtls"] = mtls_metadata

    if decision_log is not None:
        decision_dir = artifact_dir / "decision_log"
        log_path, entry = decision_log
        copied_log = decision_dir / log_path.name
        copied_log_path = _copy_artifact(log_path, copied_log)
        entry_path = decision_dir / "entry.json"
        entry_path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata["decision_log"] = {
            "log_path": copied_log_path,
            "entry_path": str(entry_path),
        }

    tco_details = details_by_step.get("tco")
    if tco_details:
        tco_dir = artifact_dir / "tco"
        tco_meta: dict[str, str] = {}
        for key in ("csv", "pdf", "json", "csv_signature", "pdf_signature", "json_signature"):
            value = tco_details.get(key)
            if not value:
                continue
            copied = tco_dir / Path(value).name
            copied_path = _copy_artifact(value, copied)
            tco_meta[key] = copied_path
        if tco_meta:
            metadata["tco"] = tco_meta

    decision_details = details_by_step.get("decision")
    if decision_details:
        decision_dir = artifact_dir / "decision"
        decision_meta: dict[str, str] = {}
        for key in ("report", "signature"):
            value = decision_details.get(key)
            if not value:
                continue
            copied = decision_dir / Path(value).name
            copied_path = _copy_artifact(value, copied)
            decision_meta[key] = copied_path
        if decision_meta:
            metadata["decision"] = decision_meta

    slo_details = details_by_step.get("slo")
    if slo_details:
        slo_dir = artifact_dir / "slo"
        slo_meta: dict[str, str] = {}
        for key in ("report", "signature"):
            value = slo_details.get(key)
            if not value:
                continue
            copied = slo_dir / Path(value).name
            copied_path = _copy_artifact(value, copied)
            slo_meta[key] = copied_path
        if slo_meta:
            metadata["slo"] = slo_meta

    rotation_details = details_by_step.get("rotation")
    if rotation_details:
        rotation_dir = artifact_dir / "rotation"
        plan_path = rotation_details.get("plan")
        if plan_path:
            copied = rotation_dir / Path(plan_path).name
            copied_path = _copy_artifact(plan_path, copied)
            metadata.setdefault("rotation", {})["plan"] = copied_path

    observability_details = details_by_step.get("observability")
    if observability_details:
        observability_dir = artifact_dir / "observability"
        archive_path = observability_details.get("archive")
        if archive_path:
            copied = observability_dir / Path(archive_path).name
            copied_path = _copy_artifact(archive_path, copied)
            metadata.setdefault("observability", {})["archive"] = copied_path

    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary: list[StepOutcome] = []
    exit_code = 0

    steps = []
    if not args.skip_bundle:
        steps.append(("bundle", _run_bundle_step))
    if not args.skip_license:
        steps.append(("license", _run_license_step))
    if not args.skip_risk:
        steps.append(("risk", _run_risk_step))
    if not args.skip_mtls:
        steps.append(("mtls", _run_mtls_step))
    if not args.skip_tco:
        steps.append(("tco", _run_tco_step))
    if not args.skip_decision:
        steps.append(("decision", _run_decision_step))
    if not args.skip_slo:
        steps.append(("slo", _run_slo_step))
    if not args.skip_rotation:
        steps.append(("rotation", _run_rotation_step))
    if not args.skip_observability:
        steps.append(("observability", _run_observability_step))

    for name, handler in steps:
        try:
            details = handler(args)
        except AcceptanceError as exc:  # noqa: PERF203 - chcemy zatrzymać się na pierwszym błędzie
            _record(summary, StepOutcome(step=name, status="failed", details={"error": str(exc)}))
            exit_code = 1
            if args.fail_fast:
                break
        except Exception as exc:  # pragma: no cover - zabezpieczenie przed nieoczekiwanymi wyjątkami
            _record(summary, StepOutcome(step=name, status="failed", details={"error": str(exc)}))
            exit_code = 1
            if args.fail_fast:
                break
        else:
            _record(summary, StepOutcome(step=name, status="ok", details=details))

    serialized_summary = _dump_summary(summary, args=args)
    decision_log_info = _append_decision_log_entry(summary, args=args, exit_code=exit_code)
    artifact_dir = _publish_artifacts(
        summary=summary,
        serialized_summary=serialized_summary,
        args=args,
        exit_code=exit_code,
        decision_log=decision_log_info,
    )
    if artifact_dir is not None:
        print(f"Akceptacja OEM – artefakty zapisane w {artifact_dir}")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
