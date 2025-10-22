"""Procedura synchronizacji przed migracją środowiska paper → live."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from enum import Enum
from typing import Any, Iterable, Mapping, MutableSequence, Sequence

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


def _normalize_status(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, Enum):
        candidate = value.value
        text = str(candidate).strip().lower()
    else:
        candidate = getattr(value, "value", None)
        if isinstance(candidate, str):
            text = candidate.strip().lower()
        else:
            text = str(value).strip().lower()
    return text or "unknown"


def _summarize_live_readiness(
    checklist: Sequence[Mapping[str, Any]] | None,
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not checklist:
        if metadata:
            status = _normalize_status(metadata.get("status"))
        else:
            status = "not_configured"
        base = {"status": status}
        if metadata and metadata.get("reasons"):
            base["reasons"] = tuple(metadata["reasons"])
        return base

    blocked_items: list[str] = []
    blocked_documents: list[str] = []
    reasons: list[str] = []

    for entry in checklist:
        if not isinstance(entry, Mapping):
            continue
        status = _normalize_status(entry.get("status"))
        item = str(entry.get("item") or "unknown")
        if status != "ok":
            if item not in blocked_items:
                blocked_items.append(item)
            for reason in entry.get("reasons", ()) or ():
                reasons.append(str(reason))

        details = entry.get("details")
        if isinstance(details, Mapping):
            documents = details.get("documents")
            if isinstance(documents, Iterable):
                for document in documents:
                    if not isinstance(document, Mapping):
                        continue
                    doc_status = _normalize_status(document.get("status"))
                    if doc_status == "ok":
                        continue
                    document_name = str(document.get("name") or "unknown")
                    if document_name not in blocked_documents:
                        blocked_documents.append(document_name)
                    for reason in document.get("reasons", ()) or ():
                        reasons.append(str(reason))

    if metadata:
        meta_status = _normalize_status(metadata.get("status"))
        if meta_status != "ok" and "live_checklist" not in blocked_items:
            blocked_items.append("live_checklist")
        for reason in metadata.get("reasons", ()) or ():
            reasons.append(str(reason))
        checklist_meta = metadata.get("checklist")
        if isinstance(checklist_meta, Mapping):
            meta_check_status = _normalize_status(checklist_meta.get("status"))
            if meta_check_status != "ok" and "live_checklist" not in blocked_items:
                blocked_items.append("live_checklist")
            for reason in checklist_meta.get("reasons", ()) or ():
                reasons.append(str(reason))
        documents = metadata.get("documents")
        if isinstance(documents, Iterable):
            for document in documents:
                if not isinstance(document, Mapping):
                    continue
                doc_status = _normalize_status(document.get("status"))
                if doc_status == "ok":
                    continue
                document_name = str(document.get("name") or "unknown")
                if document_name not in blocked_documents:
                    blocked_documents.append(document_name)
                for reason in document.get("reasons", ()) or ():
                    reasons.append(str(reason))

    final_status = "ok" if not blocked_items and not blocked_documents else "blocked"

    summary: dict[str, Any] = {"status": final_status}
    if blocked_items:
        summary["blocked_items"] = tuple(blocked_items)
    if blocked_documents:
        summary["blocked_documents"] = tuple(blocked_documents)
    if reasons:
        summary["reasons"] = tuple(dict.fromkeys(reasons))
    return summary


def build_promotion_report(
    environment_name: str,
    *,
    config_path: str | Path,
    skip_license: bool = False,
    core_config: CoreConfig | None = None,
) -> Mapping[str, Any]:
    """Buduje raport synchronizacji przed uruchomieniem środowiska live."""

    config_path_obj = Path(config_path)
    if core_config is None:
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

    report["live_readiness_summary"] = _summarize_live_readiness(checklist, metadata)

    return report


def _aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    config_path: Path,
) -> Mapping[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary_status = "ok"
    ok_envs: list[str] = []
    blocked_details: list[Mapping[str, Any]] = []
    blocked_documents_flat: list[str] = []
    blocked_items_flat: list[str] = []
    reasons: MutableSequence[str] = []
    license_issues: list[Mapping[str, Any]] = []

    for report in reports:
        environment_name = str(report.get("environment", "unknown"))
        summary = report.get("live_readiness_summary") or {}
        status = _normalize_status(summary.get("status"))
        if status == "ok":
            ok_envs.append(environment_name)
        else:
            summary_status = "blocked"
            blocked_entry = {
                "environment": environment_name,
                "status": status,
                "blocked_items": tuple(summary.get("blocked_items", ()) or ()),
                "blocked_documents": tuple(
                    summary.get("blocked_documents", ()) or ()
                ),
                "reasons": tuple(summary.get("reasons", ()) or ()),
            }
            blocked_details.append(blocked_entry)
            reasons.extend(blocked_entry["reasons"])
            blocked_documents_flat.extend(blocked_entry["blocked_documents"])
            blocked_items_flat.extend(blocked_entry["blocked_items"])

        license_section = report.get("license") or {}
        license_status = _normalize_status(license_section.get("status"))
        if license_status not in {"ok", "valid", "active"}:
            license_issue = {
                "environment": environment_name,
                "status": license_status,
                "errors": tuple(license_section.get("errors", ()) or ()),
                "warnings": tuple(license_section.get("warnings", ()) or ()),
            }
            license_issues.append(license_issue)

    summary: dict[str, Any] = {
        "status": summary_status,
        "ok_environments": tuple(ok_envs),
        "total_environments": len(reports),
    }
    if blocked_details:
        blocked_environments = tuple(
            entry["environment"] for entry in blocked_details
        )
        summary["blocked_environments"] = blocked_environments
        if blocked_items_flat:
            summary["blocked_items"] = tuple(
                dict.fromkeys(blocked_items_flat)
            )
        if blocked_documents_flat:
            summary["blocked_documents"] = tuple(
                dict.fromkeys(blocked_documents_flat)
            )
        summary["blocked"] = tuple(blocked_details)
    if license_issues:
        summary["license_issues"] = tuple(license_issues)
    if reasons:
        summary["reasons"] = tuple(dict.fromkeys(reasons))

    aggregate_report: dict[str, Any] = {
        "generated_at": generated_at,
        "source_config": str(config_path),
        "environments": tuple(
            str(report.get("environment", "unknown")) for report in reports
        ),
        "summary": summary,
        "live_readiness_summary": summary,
        "reports": tuple(reports),
    }
    return aggregate_report


def _render_json_report(report: Mapping[str, Any], *, pretty: bool) -> str:
    json_kwargs = {"ensure_ascii": False}
    if pretty:
        json_kwargs["indent"] = 2
        json_kwargs["sort_keys"] = True
    return json.dumps(report, **json_kwargs)


def _render_markdown_report(report: Mapping[str, Any]) -> str:
    if "reports" in report:
        return "\n".join(_render_markdown_multi_lines(report)) + "\n"
    return "\n".join(_render_markdown_single_lines(report)) + "\n"


def _render_markdown_multi_lines(report: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    generated_at = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    environments = tuple(report.get("environments", ()))
    lines.append("# Raport promotion-to-live — zbiorczy")
    lines.append("")
    lines.append(f"- **Data wygenerowania:** {generated_at}")
    lines.append(f"- **Konfiguracja:** {report.get('source_config', 'unknown')}")
    lines.append(
        f"- **Środowiska:** {', '.join(map(str, environments)) or 'brak środowisk'}"
    )

    summary = report.get("summary") or {}
    lines.append("")
    lines.append("## Podsumowanie")
    lines.append("")
    lines.append(f"- **Status:** {summary.get('status', 'unknown')}")
    ok_envs = summary.get("ok_environments", ()) or ()
    if ok_envs:
        lines.append(f"- **Środowiska OK:** {', '.join(map(str, ok_envs))}")
    blocked_envs = summary.get("blocked_environments", ()) or ()
    if blocked_envs:
        lines.append(
            f"- **Środowiska z blokadami:** {', '.join(map(str, blocked_envs))}"
        )
    reasons = summary.get("reasons", ()) or ()
    if reasons:
        lines.append("- **Powody:**")
        for reason in reasons:
            lines.append(f"  - {reason}")

    license_issues = summary.get("license_issues", ()) or ()
    if license_issues:
        lines.append("")
        lines.append("### Problemy licencyjne")
        lines.append("")
        for issue in license_issues:
            if not isinstance(issue, Mapping):
                continue
            env = issue.get("environment", "unknown")
            status = issue.get("status", "unknown")
            lines.append(f"- **{env}:** {status}")
            if issue.get("errors"):
                lines.append("  - Błędy:")
                for error in issue["errors"]:
                    lines.append(f"    - {error}")
            if issue.get("warnings"):
                lines.append("  - Ostrzeżenia:")
                for warning in issue["warnings"]:
                    lines.append(f"    - {warning}")

    for single_report in report.get("reports", ()):
        if not isinstance(single_report, Mapping):
            continue
        lines.append("")
        lines.extend(_render_markdown_single_lines(single_report, heading_level=2))

    return lines


def _render_markdown_single_lines(
    report: Mapping[str, Any], *, heading_level: int = 1
) -> list[str]:
    heading_prefix = "#" * max(heading_level, 1)
    lines: list[str] = []
    generated_at = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    lines.append(
        f"{heading_prefix} Raport promotion-to-live — {report.get('environment', 'unknown')}"
    )
    lines.append("")
    lines.append(f"- **Data wygenerowania:** {generated_at}")
    lines.append(f"- **Środowisko:** {report.get('environment', 'unknown')}")
    lines.append(f"- **Giełda:** {report.get('exchange', 'unknown')}")
    lines.append(f"- **Profil ryzyka:** {report.get('risk_profile', 'unknown')}")

    summary = report.get("live_readiness_summary") or {}
    lines.append("")
    lines.append(f"{heading_prefix}# Podsumowanie gotowości LIVE")
    lines.append("")
    lines.append(f"- **Status:** {summary.get('status', 'unknown')}")
    blocked_items = summary.get("blocked_items", ()) or ()
    blocked_documents = summary.get("blocked_documents", ()) or ()
    reasons = summary.get("reasons", ()) or ()
    if blocked_items:
        blocked = ", ".join(str(item) for item in blocked_items)
        lines.append(f"- **Zablokowane pozycje:** {blocked}")
    if blocked_documents:
        blocked_docs = ", ".join(str(doc) for doc in blocked_documents)
        lines.append(f"- **Brakujące dokumenty:** {blocked_docs}")
    if reasons:
        lines.append("- **Powody:**")
        for reason in reasons:
            lines.append(f"  - {reason}")

    metadata = report.get("live_readiness_metadata") or {}
    if metadata:
        lines.append("")
        lines.append(f"{heading_prefix}## Metadane checklisty")
        lines.append("")
        if metadata.get("checklist_id"):
            lines.append(f"- **Checklist ID:** {metadata['checklist_id']}")
        if metadata.get("status"):
            lines.append(f"- **Status checklisty:** {metadata['status']}")
        if metadata.get("checklist_status"):
            lines.append(
                f"- **Status podpisanej checklisty:** {metadata['checklist_status']}"
            )
        if metadata.get("signed_by"):
            signed_by = ", ".join(str(member) for member in metadata["signed_by"])
            lines.append(f"- **Podpisy:** {signed_by}")
        if metadata.get("signed_at"):
            lines.append(f"- **Podpisano:** {metadata['signed_at']}")
        if metadata.get("signature_path"):
            lines.append(f"- **Ścieżka podpisu:** {metadata['signature_path']}")
        documents = metadata.get("documents") or ()
        if documents:
            lines.append("")
            lines.append(f"{heading_prefix}### Dokumenty")
            lines.append("")
            for document in documents:
                if not isinstance(document, Mapping):
                    continue
                doc_name = document.get("name", "unknown")
                lines.append(f"- **{doc_name}:**")
                doc_status = document.get("status")
                if doc_status is not None:
                    lines.append(f"  - Status: {doc_status}")
                if document.get("resolved_path"):
                    lines.append(
                        f"  - Plik: {document['resolved_path']}"
                    )
                if document.get("computed_sha256"):
                    lines.append(
                        f"  - SHA-256: {document['computed_sha256']}"
                    )
                if document.get("signed_by"):
                    signed_by_doc = ", ".join(
                        str(member) for member in document["signed_by"]
                    )
                    lines.append(f"  - Podpisy: {signed_by_doc}")
                if document.get("reasons"):
                    lines.append("  - Powody:")
                    for reason in document["reasons"]:
                        lines.append(f"    - {reason}")

    license_section = report.get("license") or {}
    lines.append("")
    lines.append(f"{heading_prefix}## Licencja")
    lines.append("")
    lines.append(f"- **Status:** {license_section.get('status', 'unknown')}")
    if license_section.get("errors"):
        lines.append("- **Błędy:**")
        for error in license_section["errors"]:
            lines.append(f"  - {error}")
    if license_section.get("warnings"):
        lines.append("- **Ostrzeżenia:**")
        for warning in license_section["warnings"]:
            lines.append(f"  - {warning}")

    risk_profile_details = report.get("risk_profile_details") or {}
    if risk_profile_details:
        lines.append("")
        lines.append(f"{heading_prefix}## Profil ryzyka")
        lines.append("")
        for key, value in risk_profile_details.items():
            lines.append(f"- **{key}:** {value}")

    alerting = report.get("alerting") or {}
    lines.append("")
    lines.append(f"{heading_prefix}## Alerting")
    lines.append("")
    lines.append(
        f"- **Kanały:** {', '.join(alerting.get('channels', ())) or 'brak'}"
    )
    lines.append(
        "- **Throttling:** "
        + ("skonfigurowany" if alerting.get("throttle_configured") else "brak")
    )
    lines.append(
        f"- **Backend audytu:** {alerting.get('audit_backend') or 'brak'}"
    )

    return lines


def _ensure_trailing_newline(payload: str) -> str:
    return payload if payload.endswith("\n") else payload + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "environment",
        nargs="*",
        help="Nazwa środowiska live do synchronizacji (można podać wiele)",
    )
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
        "--output-dir",
        help="Katalog docelowy na indywidualne raporty środowisk",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Formatuj JSON z wcięciami dla łatwiejszego czytania",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Format raportu (json lub markdown)",
    )
    parser.add_argument(
        "--all-live",
        action="store_true",
        help="Wygeneruj raporty dla wszystkich środowisk oznaczonych jako live",
    )
    parser.add_argument(
        "--skip-license",
        action="store_true",
        help="Pomiń walidację licencji (np. w środowisku CI bez aktywnej licencji)",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Zakończ z kodem wyjścia 2, jeśli checklisty lub dokumenty mają status inny niż OK",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config)
    core_config = load_core_config(config_path)

    selected_environments: list[str] = []
    if args.environment:
        selected_environments.extend([str(name) for name in args.environment])

    if args.all_live:
        live_envs = [
            name
            for name, environment in core_config.environments.items()
            if _normalize_status(getattr(environment, "environment", None)) == "live"
        ]
        selected_environments.extend(live_envs)

    if not selected_environments:
        print(
            "Brak wskazanych środowisk. Podaj nazwę środowiska lub użyj --all-live.",
            file=sys.stderr,
        )
        return 1

    deduplicated_environments = list(dict.fromkeys(selected_environments))

    reports: list[Mapping[str, Any]] = []
    for environment_name in deduplicated_environments:
        try:
            report = build_promotion_report(
                environment_name,
                config_path=config_path,
                skip_license=args.skip_license,
                core_config=core_config,
            )
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        reports.append(report)

    if len(reports) == 1:
        final_report = reports[0]
    else:
        final_report = _aggregate_reports(reports, config_path=config_path)

    if args.format == "markdown" and args.pretty:
        print("Ignoruję --pretty dla formatu markdown", file=sys.stderr)

    if args.format == "markdown":
        payload = _render_markdown_report(final_report)
    else:
        payload = _render_json_report(final_report, pretty=args.pretty)

    summary = (
        final_report.get("live_readiness_summary")
        or final_report.get("summary")
        or {}
    )
    summary_status = str(summary.get("status", "unknown")).lower()
    should_fail = args.fail_on_blocked and summary_status not in {"ok"}

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for report in reports:
            suffix = "md" if args.format == "markdown" else "json"
            filename = f"{report.get('environment', 'unknown')}.{suffix}"
            report_payload = (
                _render_markdown_report(report)
                if args.format == "markdown"
                else _render_json_report(report, pretty=args.pretty)
            )
            (output_dir / filename).write_text(
                _ensure_trailing_newline(report_payload),
                encoding="utf-8",
            )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            _ensure_trailing_newline(payload),
            encoding="utf-8",
        )
    print(payload)
    if should_fail:
        blocked_items = ", ".join(summary.get("blocked_items", ())) or "brak szczegółów"
        reasons = "; ".join(summary.get("reasons", ())) or "brak powodów"
        blocked_envs = tuple(summary.get("blocked_environments", ()) or ())
        if not blocked_envs:
            env_name = final_report.get("environment")
            if env_name:
                blocked_envs = (str(env_name),)
        blocked_envs_text = ", ".join(blocked_envs) or "brak środowisk"
        print(
            "Raport promotion-to-live zawiera blokady "
            f"(środowiska: {blocked_envs_text}; pozycje: {blocked_items}; powody: {reasons}).",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - wywołanie CLI
    raise SystemExit(main())
