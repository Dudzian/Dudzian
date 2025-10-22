"""Procedura synchronizacji przed migracją środowiska paper → live."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

if __package__ is None:  # pragma: no cover - uruchomienie jako skrypt
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig, RiskProfileConfig
from bot_core.runtime.bootstrap import (
    build_live_readiness_checklist,
    extract_live_readiness_metadata,
)
from bot_core.security.license import (
    LicenseValidationError,
    LicenseValidationResult,
    validate_license_from_config,
)


class _DummyAuditLog:
    """Prosty znacznik audytu wykorzystywany w raportach checklisty."""

    def __repr__(self) -> str:  # pragma: no cover - pomoc diagnostyczna
        return "<promotion.audit-log>"


def _build_alert_stub(environment: EnvironmentConfig) -> tuple[Mapping[str, object], Any, _DummyAuditLog]:
    """Buduje minimalne obiekty kompatybilne z checklistą live."""

    channels = {
        str(name): object() for name in (environment.alert_channels or ()) if str(name).strip()
    }

    throttle = getattr(environment, "alert_throttle", None)
    if throttle is not None:
        window_seconds = float(getattr(throttle, "window_seconds", 60.0) or 60.0)
        throttle_ns = SimpleNamespace(window=timedelta(seconds=window_seconds))
    else:
        throttle_ns = None

    alert_router = SimpleNamespace(throttle=throttle_ns)
    audit_log = _DummyAuditLog()
    return channels, alert_router, audit_log


def _extract_risk_profile(core_config: CoreConfig, environment: EnvironmentConfig) -> RiskProfileConfig | None:
    return core_config.risk_profiles.get(environment.risk_profile)


def _build_license_summary(
    core_config: CoreConfig, *, skip_license: bool
) -> Mapping[str, Any]:
    license_config = getattr(core_config, "license", None)
    if skip_license or not license_config:
        return {"status": "skipped" if skip_license else "not_configured"}

    try:
        result = validate_license_from_config(license_config)
    except LicenseValidationError as exc:
        payload: Mapping[str, Any] = {"status": "error", "message": str(exc)}
        if exc.result is not None:
            payload["details"] = exc.result.to_context()
        return payload
    except Exception as exc:  # pragma: no cover - defensywne logowanie środowiskowe
        return {"status": "error", "message": str(exc)}

    assert isinstance(result, LicenseValidationResult)
    context = result.to_context()
    context["status"] = result.status
    context["errors"] = list(result.errors)
    context["warnings"] = list(result.warnings)
    context["is_valid"] = result.is_valid
    return context


def build_promotion_report(
    environment_name: str,
    *,
    config_path: str | Path,
    skip_license: bool = False,
) -> Mapping[str, Any]:
    """Buduje raport synchronizacji przed uruchomieniem środowiska live."""

    config_path_obj = Path(config_path)
    core_config = load_core_config(config_path_obj)
    try:
        environment = core_config.environments[environment_name]
    except KeyError as exc:
        raise KeyError(f"Środowisko '{environment_name}' nie istnieje w konfiguracji") from exc

    risk_profile = _extract_risk_profile(core_config, environment)
    alert_channels, alert_router, audit_log = _build_alert_stub(environment)

    checklist = build_live_readiness_checklist(
        core_config=core_config,
        environment=environment,
        risk_profile_name=environment.risk_profile,
        risk_profile_config=risk_profile,
        alert_router=alert_router,
        alert_channels=alert_channels,
        audit_log=audit_log,
        document_root=config_path_obj.parent,
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment.name,
        "exchange": environment.exchange,
        "risk_profile": environment.risk_profile,
        "risk_profile_details": None,
        "alerting": {
            "channels": list(alert_channels.keys()),
            "throttle_configured": bool(getattr(environment, "alert_throttle", None)),
            "audit_backend": getattr(environment.alert_audit, "backend", None)
            if getattr(environment, "alert_audit", None)
            else None,
        },
        "license": _build_license_summary(core_config, skip_license=skip_license),
        "live_readiness_checklist": checklist,
    }

    if risk_profile is not None:
        report["risk_profile_details"] = {
            "max_daily_loss_pct": getattr(risk_profile, "max_daily_loss_pct", None),
            "max_position_pct": getattr(risk_profile, "max_position_pct", None),
            "hard_drawdown_pct": getattr(risk_profile, "hard_drawdown_pct", None),
            "max_open_positions": getattr(risk_profile, "max_open_positions", None),
        }

    readiness = getattr(environment, "live_readiness", None)
    metadata = extract_live_readiness_metadata(checklist)
    if readiness is not None or metadata is not None:
        base_metadata: dict[str, Any] = {}
        if readiness is not None:
            base_documents = []
            for document in getattr(readiness, "documents", ()) or ():
                name = getattr(document, "name", None)
                base_documents.append(
                    {
                        "name": name,
                        "required": bool(getattr(document, "required", True)),
                        "signed": bool(getattr(document, "signed", False)),
                        "signed_by": tuple(getattr(document, "signed_by", ()) or ()),
                        "signature_path": getattr(document, "signature_path", None),
                        "sha256": getattr(document, "sha256", None),
                    }
                )
            base_metadata.update(
                {
                    "checklist_id": getattr(readiness, "checklist_id", None),
                    "signed": bool(getattr(readiness, "signed", False)),
                    "signed_by": tuple(getattr(readiness, "signed_by", ()) or ()),
                    "signature_path": getattr(readiness, "signature_path", None),
                    "signed_at": getattr(readiness, "signed_at", None),
                    "required_documents": tuple(
                        getattr(readiness, "required_documents", ()) or ()
                    ),
                    "documents": base_documents,
                }
            )

        if metadata is not None:
            if metadata.get("status") is not None:
                base_metadata["status"] = metadata["status"]
            if metadata.get("reasons"):
                base_metadata["reasons"] = metadata["reasons"]
            checklist_meta = metadata.get("checklist") or {}
            if checklist_meta:
                for key in (
                    "checklist_id",
                    "signed",
                    "signed_by",
                    "signed_at",
                    "signature_path",
                    "resolved_signature_path",
                    "required_documents",
                ):
                    value = checklist_meta.get(key)
                    if value is not None:
                        base_metadata[key] = value
                if checklist_meta.get("status") is not None:
                    base_metadata["checklist_status"] = checklist_meta["status"]
                if checklist_meta.get("reasons"):
                    base_metadata["checklist_reasons"] = checklist_meta["reasons"]
            documents = metadata.get("documents")
            if documents is not None:
                base_metadata["documents"] = documents

        report["live_readiness_metadata"] = base_metadata

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", help="Nazwa środowiska live do synchronizacji")
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku core.yaml (domyślnie config/core.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Opcjonalna ścieżka pliku JSON z raportem",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Formatuj JSON z wcięciami dla łatwiejszego czytania",
    )
    parser.add_argument(
        "--skip-license",
        action="store_true",
        help="Pomiń walidację licencji (np. w środowisku CI bez aktywnej licencji)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_promotion_report(
        args.environment,
        config_path=args.config,
        skip_license=args.skip_license,
    )

    json_kwargs = {"ensure_ascii": False}
    if args.pretty:
        json_kwargs["indent"] = 2
        json_kwargs["sort_keys"] = True

    payload = json.dumps(report, **json_kwargs)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - wywołanie CLI
    raise SystemExit(main())
