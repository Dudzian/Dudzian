"""Automatyzuje kompletny cykl Observability Stage6."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.observability import (  # noqa: E402 - import po modyfikacji sys.path
    BundleConfig,
    DashboardSyncConfig,
    ObservabilityCycleConfig,
    ObservabilityHypercareCycle,
    OverridesOutputConfig,
    SLOOutputConfig,
)
from bot_core.observability.bundle import AssetSource  # noqa: E402


def _parse_metadata(values: list[str] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not values:
        return metadata
    for item in values:
        if "=" not in item:
            raise ValueError("Metadane muszą mieć format klucz=wartość")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Klucz metadanych nie może być pusty")
        metadata[key] = value.strip()
    return metadata


def _parse_sources(values: list[str] | None) -> list[AssetSource]:
    sources: list[AssetSource] = []
    if not values:
        return sources
    for item in values:
        if "=" not in item:
            raise ValueError("Źródło musi być w formacie kategoria=ścieżka")
        category, raw_path = item.split("=", 1)
        category = category.strip()
        if not category:
            raise ValueError("Kategoria źródła nie może być pusta")
        path = Path(raw_path.strip())
        sources.append(AssetSource(category=category, root=path))
    return sources


def _parse_severity_map(values: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not values:
        return mapping
    for item in values:
        if "=" not in item:
            raise ValueError("Mapowanie severity musi mieć format status=severity")
        status, severity = item.split("=", 1)
        status = status.strip()
        severity = severity.strip()
        if not status or not severity:
            raise ValueError("Status i severity nie mogą być puste")
        mapping[status] = severity
    return mapping


def _load_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    if args.signing_key and args.signing_key_path:
        raise ValueError("Nie można jednocześnie podać klucza HMAC i ścieżki do pliku")
    if args.signing_key:
        return args.signing_key.encode("utf-8"), args.signing_key_id
    if args.signing_key_env:
        env_value = os.environ.get(args.signing_key_env)
        if not env_value:
            raise ValueError(f"Zmienna środowiskowa {args.signing_key_env} nie zawiera klucza HMAC")
        return env_value.encode("utf-8"), args.signing_key_id
    if args.signing_key_path:
        path = Path(args.signing_key_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Plik z kluczem HMAC nie istnieje: {path}")
        return path.read_bytes().strip(), args.signing_key_id
    return None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wykonuje pełny cykl Observability Stage6")
    parser.add_argument("--definitions", required=True, help="Plik z definicjami SLO (YAML/JSON)")
    parser.add_argument("--metrics", required=True, help="Plik z pomiarami metryk (JSON)")
    parser.add_argument(
        "--slo-json",
        default="var/audit/observability/slo_report.json",
        help="Ścieżka raportu SLO w formacie JSON",
    )
    parser.add_argument(
        "--slo-csv",
        default="var/audit/observability/slo_report.csv",
        help="Ścieżka raportu SLO w formacie CSV",
    )
    parser.add_argument("--slo-signature", help="Opcjonalna ścieżka podpisu raportu SLO")
    parser.add_argument("--slo-pretty", action="store_true", help="Zapisz SLO JSON z wcięciami")

    parser.add_argument("--skip-overrides", action="store_true", help="Pomiń generowanie override'ów")
    parser.add_argument(
        "--overrides-json",
        default="var/audit/observability/alert_overrides.json",
        help="Plik wynikowy override'ów",
    )
    parser.add_argument("--overrides-signature", help="Ścieżka podpisu override'ów")
    parser.add_argument("--overrides-ttl", type=int, default=120, help="Czas ważności override'ów w minutach")
    parser.add_argument("--skip-warning", action="store_true", help="Pomiń statusy warning")
    parser.add_argument(
        "--overrides-requested-by",
        default="PortfolioGovernor",
        help="Pole requested_by w override'ach",
    )
    parser.add_argument(
        "--overrides-source",
        default="slo_monitor",
        help="Źródło override zapisywane w raporcie",
    )
    parser.add_argument("--tag", action="append", help="Dodatkowe tagi dodawane do override'ów")
    parser.add_argument("--severity", action="append", help="Mapowanie status=severity")
    parser.add_argument("--existing-overrides", help="Plik z istniejącymi override'ami do połączenia")

    parser.add_argument("--dashboard", help="Plik JSON z dashboardem Grafana Stage6")
    parser.add_argument(
        "--annotations-output",
        default="var/audit/observability/dashboard_annotations.json",
        help="Plik wynikowy z anotacjami",
    )
    parser.add_argument("--annotations-signature", help="Podpis anotacji dashboardu")
    parser.add_argument("--annotations-panel-id", type=int, help="Panel ID dla anotacji Grafana")
    parser.add_argument("--annotations-pretty", action="store_true", help="Formatowanie JSON anotacji")

    parser.add_argument("--skip-bundle", action="store_true", help="Pomiń budowanie paczki obserwowalności")
    parser.add_argument(
        "--bundle-output-dir",
        default=str(REPO_ROOT / "var" / "observability"),
        help="Katalog docelowy paczki",
    )
    parser.add_argument("--bundle-name", default="stage6-observability", help="Prefiks nazwy paczki")
    parser.add_argument("--bundle-source", action="append", help="Źródło paczki w formacie kategoria=ścieżka")
    parser.add_argument("--bundle-include", action="append", help="Wzorce plików do uwzględnienia")
    parser.add_argument("--bundle-exclude", action="append", help="Wzorce plików do pominięcia")
    parser.add_argument("--bundle-metadata", action="append", help="Metadane w formacie klucz=wartość")
    parser.add_argument("--no-verify-bundle", action="store_true", help="Nie weryfikuj paczki po zbudowaniu")

    parser.add_argument("--signing-key", help="Klucz HMAC podany wprost")
    parser.add_argument("--signing-key-env", help="Nazwa zmiennej środowiskowej z kluczem HMAC")
    parser.add_argument("--signing-key-path", help="Plik z kluczem HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        signing_key, key_id = _load_signing_key(args)

        slo_config = SLOOutputConfig(
            json_path=Path(args.slo_json),
            csv_path=Path(args.slo_csv) if args.slo_csv else None,
            signature_path=Path(args.slo_signature) if args.slo_signature else None,
            pretty_json=args.slo_pretty,
        )

        overrides_config: OverridesOutputConfig | None = None
        if not args.skip_overrides:
            ttl_minutes = max(0, int(args.overrides_ttl))
            overrides_config = OverridesOutputConfig(
                json_path=Path(args.overrides_json),
                signature_path=Path(args.overrides_signature) if args.overrides_signature else None,
                include_warning=not args.skip_warning,
                ttl=timedelta(minutes=ttl_minutes),
                requested_by=args.overrides_requested_by,
                source=args.overrides_source,
                tags=tuple(str(tag) for tag in (args.tag or [])),
                severity_overrides=_parse_severity_map(args.severity),
                existing_path=Path(args.existing_overrides) if args.existing_overrides else None,
            )

        dashboard_config: DashboardSyncConfig | None = None
        if args.dashboard:
            dashboard_config = DashboardSyncConfig(
                dashboard_path=Path(args.dashboard),
                output_path=Path(args.annotations_output),
                signature_path=Path(args.annotations_signature) if args.annotations_signature else None,
                panel_id=args.annotations_panel_id,
                pretty=args.annotations_pretty,
            )

        bundle_config: BundleConfig | None = None
        if not args.skip_bundle:
            sources = _parse_sources(args.bundle_source)
            bundle_config = BundleConfig(
                output_dir=Path(args.bundle_output_dir),
                bundle_name=args.bundle_name,
                sources=sources or None,
                include=args.bundle_include or None,
                exclude=args.bundle_exclude or None,
                metadata=_parse_metadata(args.bundle_metadata),
                verify=not args.no_verify_bundle,
            )

        config = ObservabilityCycleConfig(
            definitions_path=Path(args.definitions),
            metrics_path=Path(args.metrics),
            slo=slo_config,
            overrides=overrides_config,
            dashboard=dashboard_config,
            bundle=bundle_config,
            signing_key=signing_key,
            signing_key_id=key_id,
        )

        cycle = ObservabilityHypercareCycle(config)
        result = cycle.run()
    except Exception as exc:  # noqa: BLE001 - komunikat CLI
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    summary: dict[str, Any] = {
        "slo_report": result.slo_report_path.as_posix(),
    }
    if result.slo_signature_path:
        summary["slo_signature"] = result.slo_signature_path.as_posix()
    if result.slo_csv_path:
        summary["slo_csv"] = result.slo_csv_path.as_posix()
    if result.overrides_path:
        summary["alert_overrides"] = result.overrides_path.as_posix()
    if result.overrides_signature_path:
        summary["alert_overrides_signature"] = result.overrides_signature_path.as_posix()
    if result.dashboard_annotations_path:
        summary["dashboard_annotations"] = result.dashboard_annotations_path.as_posix()
    if result.dashboard_signature_path:
        summary["dashboard_annotations_signature"] = result.dashboard_signature_path.as_posix()
    if result.bundle_path:
        summary["bundle"] = result.bundle_path.as_posix()
    if result.bundle_manifest_path:
        summary["bundle_manifest"] = result.bundle_manifest_path.as_posix()
    if result.bundle_signature_path:
        summary["bundle_signature"] = result.bundle_signature_path.as_posix()
    if result.bundle_verification:
        summary["bundle_verification"] = result.bundle_verification

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run())

