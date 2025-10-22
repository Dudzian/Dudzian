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
    if skip_license:
        return {"status": "skipped", "reason": "cli_skip_requested"}
    if not license_config:
        return {"status": "not_configured", "reason": "missing_config"}

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


def _normalize_sequence(value: Iterable[Any] | None) -> tuple[Any, ...]:
    if not value:
        return ()
    return tuple(value)


def _extract_alerting_snapshot(
    environment: EnvironmentConfig,
) -> Mapping[str, Any]:
    channels = tuple(str(name) for name in environment.alert_channels or ())
    throttle_config = getattr(environment, "alert_throttle", None)
    if throttle_config is not None:
        throttle_payload: Mapping[str, Any] = {
            "window_seconds": getattr(throttle_config, "window_seconds", None),
            "exclude_severities": _normalize_sequence(
                getattr(throttle_config, "exclude_severities", None)
            ),
            "exclude_categories": _normalize_sequence(
                getattr(throttle_config, "exclude_categories", None)
            ),
            "max_entries": getattr(throttle_config, "max_entries", None),
        }
    else:
        throttle_payload = {}

    audit_config = getattr(environment, "alert_audit", None)
    if audit_config is not None:
        audit_payload: Mapping[str, Any] = {
            "backend": getattr(audit_config, "backend", None),
            "directory": getattr(audit_config, "directory", None),
        }
    else:
        audit_payload = {}

    return {
        "channels": channels,
        "throttle": throttle_payload,
        "audit": audit_payload,
    }


def _compare_alerting(
    target: EnvironmentConfig, baseline: EnvironmentConfig
) -> Mapping[str, Any]:
    target_snapshot = _extract_alerting_snapshot(target)
    baseline_snapshot = _extract_alerting_snapshot(baseline)

    baseline_channels = set(baseline_snapshot["channels"])
    target_channels = set(target_snapshot["channels"])

    missing_channels = tuple(sorted(baseline_channels - target_channels))
    extra_channels = tuple(sorted(target_channels - baseline_channels))

    status = "match"
    if missing_channels or extra_channels:
        status = "mismatch"

    throttle_diff: dict[str, Any] = {
        "baseline": baseline_snapshot["throttle"],
        "target": target_snapshot["throttle"],
    }
    throttle_baseline = baseline_snapshot["throttle"]
    throttle_target = target_snapshot["throttle"]
    if throttle_baseline or throttle_target:
        throttle_differences: list[Mapping[str, Any]] = []
        comparable_keys = {
            "window_seconds",
            "exclude_severities",
            "exclude_categories",
            "max_entries",
        }
        for key in sorted(comparable_keys):
            base_value = throttle_baseline.get(key)
            target_value = throttle_target.get(key)
            if base_value != target_value:
                throttle_differences.append(
                    {"field": key, "baseline": base_value, "target": target_value}
                )
        if throttle_differences:
            throttle_diff["differences"] = tuple(throttle_differences)
            throttle_diff["status"] = "mismatch"
            status = "mismatch"
        else:
            throttle_diff["status"] = "match"
    if not throttle_diff.get("status"):
        throttle_diff["status"] = "match"

    audit_diff: dict[str, Any] = {
        "baseline": baseline_snapshot["audit"],
        "target": target_snapshot["audit"],
    }
    audit_baseline = baseline_snapshot["audit"]
    audit_target = target_snapshot["audit"]
    if audit_baseline or audit_target:
        audit_differences: list[Mapping[str, Any]] = []
        for key in sorted({"backend", "directory"}):
            base_value = audit_baseline.get(key)
            target_value = audit_target.get(key)
            if base_value != target_value:
                audit_differences.append(
                    {"field": key, "baseline": base_value, "target": target_value}
                )
        if audit_differences:
            audit_diff["differences"] = tuple(audit_differences)
            audit_diff["status"] = "mismatch"
            status = "mismatch"
        else:
            audit_diff["status"] = "match"
    if not audit_diff.get("status"):
        audit_diff["status"] = "match"

    comparison: dict[str, Any] = {
        "status": status,
        "baseline_channels": tuple(sorted(baseline_channels)),
        "target_channels": tuple(sorted(target_channels)),
        "missing_channels": missing_channels,
        "extra_channels": extra_channels,
        "throttle": throttle_diff,
        "audit": audit_diff,
    }

    return comparison


def _compare_risk_profiles(
    target_name: str | None,
    target_profile: RiskProfileConfig | None,
    baseline_name: str | None,
    baseline_profile: RiskProfileConfig | None,
) -> Mapping[str, Any]:
    comparison: dict[str, Any] = {
        "target_profile": target_name,
        "baseline_profile": baseline_name,
    }
    if baseline_profile is None:
        comparison["status"] = "baseline_missing"
        return comparison
    if target_profile is None:
        comparison["status"] = "target_missing"
        return comparison

    differences: list[Mapping[str, Any]] = []
    comparable_fields = (
        "max_daily_loss_pct",
        "max_position_pct",
        "target_volatility",
        "max_leverage",
        "stop_loss_atr_multiple",
        "max_open_positions",
        "hard_drawdown_pct",
        "strategy_allocations",
        "instrument_buckets",
    )
    for field_name in comparable_fields:
        baseline_value = getattr(baseline_profile, field_name, None)
        target_value = getattr(target_profile, field_name, None)
        if baseline_value != target_value:
            if isinstance(baseline_value, Mapping):
                baseline_value = dict(baseline_value)
            if isinstance(target_value, Mapping):
                target_value = dict(target_value)
            if isinstance(baseline_value, Iterable) and not isinstance(
                baseline_value, (str, bytes)
            ):
                baseline_value = tuple(baseline_value)
            if isinstance(target_value, Iterable) and not isinstance(
                target_value, (str, bytes)
            ):
                target_value = tuple(target_value)
            differences.append(
                {
                    "field": field_name,
                    "baseline": baseline_value,
                    "target": target_value,
                }
            )

    if differences:
        comparison["status"] = "mismatch"
        comparison["differences"] = tuple(differences)
    else:
        comparison["status"] = "match"

    return comparison


def _build_baseline_comparison(
    *,
    core_config: CoreConfig,
    target_environment: EnvironmentConfig,
    baseline_environment_name: str,
    baseline_config: CoreConfig | None = None,
    baseline_source: Path | None = None,
) -> Mapping[str, Any]:
    source_config = baseline_config or core_config
    try:
        baseline_environment = source_config.environments[baseline_environment_name]
    except KeyError as exc:
        raise KeyError(
            f"Środowisko bazowe '{baseline_environment_name}' nie istnieje w konfiguracji"
        ) from exc

    risk_profile_comparison = _compare_risk_profiles(
        target_environment.risk_profile,
        _extract_risk_profile(core_config, target_environment),
        baseline_environment.risk_profile,
        _extract_risk_profile(source_config, baseline_environment),
    )
    alerting_comparison = _compare_alerting(target_environment, baseline_environment)

    statuses = [
        status
        for status in (
            risk_profile_comparison.get("status"),
            alerting_comparison.get("status"),
        )
        if status
    ]
    if statuses and all(status == "match" for status in statuses):
        overall_status = "match"
    elif any(status == "mismatch" for status in statuses):
        overall_status = "mismatch"
    else:
        overall_status = statuses[0] if statuses else "unknown"

    comparison: dict[str, Any] = {
        "baseline_environment": baseline_environment.name,
        "requested_environment": baseline_environment_name,
        "status": overall_status,
        "risk_profile": risk_profile_comparison,
        "alerting": alerting_comparison,
    }

    if baseline_source is not None:
        comparison["baseline_source_config"] = str(baseline_source)

    return comparison


def build_promotion_report(
    environment_name: str,
    *,
    config_path: str | Path,
    skip_license: bool = False,
    core_config: CoreConfig | None = None,
    document_root: str | Path | None = None,
    baseline_environment: str | None = None,
    baseline_config_path: str | Path | None = None,
    baseline_core_config: CoreConfig | None = None,
) -> Mapping[str, Any]:
    """Buduje raport synchronizacji przed uruchomieniem środowiska live."""

    config_path_obj = Path(config_path)
    if document_root is None:
        document_root_path: Path | None = config_path_obj.parent
    else:
        try:
            document_root_path = Path(document_root).expanduser().resolve()
        except Exception:
            document_root_path = Path(document_root)
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
        document_root=document_root_path,
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

        if document_root_path is not None:
            base_metadata.setdefault("document_root", str(document_root_path))

        report["live_readiness_metadata"] = base_metadata

    report["live_readiness_summary"] = _summarize_live_readiness(checklist, metadata)

    if baseline_environment:
        baseline_source_config_path: Path | None = None
        baseline_config = baseline_core_config
        if baseline_config is None and baseline_config_path is not None:
            baseline_source_config_path = Path(baseline_config_path)
            baseline_config = load_core_config(baseline_source_config_path)
        if baseline_config is None:
            baseline_config = core_config
            baseline_source_config_path = config_path_obj
        elif baseline_source_config_path is None and baseline_config_path is not None:
            baseline_source_config_path = Path(baseline_config_path)
        elif baseline_source_config_path is None:
            baseline_source_config_path = config_path_obj
        report["baseline_comparison"] = _build_baseline_comparison(
            core_config=core_config,
            target_environment=environment,
            baseline_environment_name=baseline_environment,
            baseline_config=baseline_config,
            baseline_source=baseline_source_config_path,
        )

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
    license_skipped: list[Mapping[str, Any]] = []

    baseline_total = 0
    baseline_match_count = 0
    baseline_mismatch_count = 0
    baseline_missing_count = 0
    baseline_unknown_count = 0
    baseline_mismatch_envs: list[str] = []
    baseline_missing_envs: list[str] = []
    baseline_details: list[Mapping[str, Any]] = []

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
        if license_status in {"skipped"}:
            license_skipped.append(
                {
                    "environment": environment_name,
                    "status": license_status,
                    "reason": license_section.get("reason"),
                }
            )
        elif license_status not in {"ok", "valid", "active"}:
            license_issue = {
                "environment": environment_name,
                "status": license_status,
                "errors": tuple(license_section.get("errors", ()) or ()),
                "warnings": tuple(license_section.get("warnings", ()) or ()),
            }
            license_issues.append(license_issue)

        baseline_section = report.get("baseline_comparison")
        if isinstance(baseline_section, Mapping) and baseline_section:
            baseline_total += 1
            baseline_status = _normalize_status(baseline_section.get("status"))
            risk_diff = baseline_section.get("risk_profile") or {}
            alerting_diff = baseline_section.get("alerting") or {}

            entry: dict[str, Any] = {
                "environment": environment_name,
                "status": baseline_status,
                "baseline_environment": baseline_section.get(
                    "baseline_environment"
                ),
                "risk_profile_status": _normalize_status(
                    risk_diff.get("status")
                ),
                "alerting_status": _normalize_status(alerting_diff.get("status")),
            }

            risk_differences: list[Mapping[str, Any]] = []
            for diff in risk_diff.get("differences", ()) or ():
                if isinstance(diff, Mapping):
                    risk_differences.append(
                        {
                            "field": diff.get("field"),
                            "baseline": diff.get("baseline"),
                            "target": diff.get("target"),
                        }
                    )
            if risk_differences:
                entry["risk_profile_differences"] = tuple(risk_differences)

            missing_channels = tuple(alerting_diff.get("missing_channels", ()) or ())
            extra_channels = tuple(alerting_diff.get("extra_channels", ()) or ())
            if missing_channels:
                entry["missing_channels"] = missing_channels
            if extra_channels:
                entry["extra_channels"] = extra_channels

            throttle_diff = alerting_diff.get("throttle") or {}
            throttle_status = _normalize_status(throttle_diff.get("status"))
            throttle_differences: list[Mapping[str, Any]] = []
            for diff in throttle_diff.get("differences", ()) or ():
                if isinstance(diff, Mapping):
                    throttle_differences.append(
                        {
                            "field": diff.get("field"),
                            "baseline": diff.get("baseline"),
                            "target": diff.get("target"),
                        }
                    )
            if throttle_status and throttle_status != "match":
                entry["throttle_status"] = throttle_status
            if throttle_differences:
                entry["throttle_differences"] = tuple(throttle_differences)

            audit_diff = alerting_diff.get("audit") or {}
            audit_status = _normalize_status(audit_diff.get("status"))
            audit_differences: list[Mapping[str, Any]] = []
            for diff in audit_diff.get("differences", ()) or ():
                if isinstance(diff, Mapping):
                    audit_differences.append(
                        {
                            "field": diff.get("field"),
                            "baseline": diff.get("baseline"),
                            "target": diff.get("target"),
                        }
                    )
            if audit_status and audit_status != "match":
                entry["audit_status"] = audit_status
            if audit_differences:
                entry["audit_differences"] = tuple(audit_differences)

            baseline_details.append(entry)

            if baseline_status == "match":
                baseline_match_count += 1
            elif baseline_status in {"baseline_missing", "target_missing"}:
                baseline_missing_count += 1
                baseline_missing_envs.append(environment_name)
            elif baseline_status == "mismatch":
                baseline_mismatch_count += 1
                baseline_mismatch_envs.append(environment_name)
            else:
                baseline_unknown_count += 1

    summary: dict[str, Any] = {
        "status": summary_status,
        "ok_environments": tuple(ok_envs),
        "ok_count": len(ok_envs),
        "total_environments": len(reports),
    }
    if blocked_details:
        blocked_environments = tuple(
            entry["environment"] for entry in blocked_details
        )
        summary["blocked_environments"] = blocked_environments
        summary["blocked_count"] = len(blocked_environments)
        if blocked_items_flat:
            unique_blocked_items = tuple(
                dict.fromkeys(blocked_items_flat)
            )
            summary["blocked_items"] = unique_blocked_items
            summary["blocked_items_count"] = len(unique_blocked_items)
        if blocked_documents_flat:
            unique_blocked_documents = tuple(
                dict.fromkeys(blocked_documents_flat)
            )
            summary["blocked_documents"] = unique_blocked_documents
            summary["blocked_documents_count"] = len(unique_blocked_documents)
        summary["blocked"] = tuple(blocked_details)
    summary["license_issue_count"] = len(license_issues)
    if license_issues:
        summary["license_issues"] = tuple(license_issues)
    summary["license_skipped_count"] = len(license_skipped)
    if license_skipped:
        summary["license_skipped"] = tuple(license_skipped)
        summary["license_skipped_reasons"] = tuple(
            dict.fromkeys(
                str(entry.get("reason", "unknown")) for entry in license_skipped
            )
        )
    if reasons:
        summary["reasons"] = tuple(dict.fromkeys(reasons))

    if baseline_total:
        summary["baseline_total"] = baseline_total
        summary["baseline_match_count"] = baseline_match_count
        summary["baseline_mismatch_count"] = baseline_mismatch_count
        summary["baseline_missing_count"] = baseline_missing_count
        if baseline_unknown_count:
            summary["baseline_unknown_count"] = baseline_unknown_count
        if baseline_mismatch_envs:
            summary["baseline_mismatch_environments"] = tuple(baseline_mismatch_envs)
        if baseline_missing_envs:
            summary["baseline_missing_environments"] = tuple(baseline_missing_envs)
        summary["baseline_status"] = (
            "match"
            if baseline_mismatch_count == 0 and baseline_missing_count == 0
            else "mismatch"
        )
        baseline_details_tuple = tuple(baseline_details)
        summary["baseline_comparisons"] = baseline_details_tuple
        if baseline_mismatch_envs:
            summary["baseline_mismatches"] = tuple(
                entry
                for entry in baseline_details_tuple
                if entry.get("environment") in baseline_mismatch_envs
            )
        if baseline_missing_envs:
            summary["baseline_missing"] = tuple(
                entry
                for entry in baseline_details_tuple
                if entry.get("environment") in baseline_missing_envs
            )

    baseline_sources: list[str] = []
    for report in reports:
        baseline_section = report.get("baseline_comparison") or {}
        source_value = baseline_section.get("baseline_source_config")
        if source_value and str(source_value) not in baseline_sources:
            baseline_sources.append(str(source_value))

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
    if baseline_sources:
        aggregate_report["baseline_source_configs"] = tuple(baseline_sources)
    return aggregate_report


def _collect_license_outcomes(
    final_report: Mapping[str, Any], reports: Sequence[Mapping[str, Any]]
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    summary = final_report.get("summary")
    if isinstance(summary, Mapping):
        issues = tuple(
            entry
            for entry in summary.get("license_issues", ()) or ()
            if isinstance(entry, Mapping)
        )
        skipped = tuple(
            entry
            for entry in summary.get("license_skipped", ()) or ()
            if isinstance(entry, Mapping)
        )
        return issues, skipped

    issues: list[Mapping[str, Any]] = []
    skipped: list[Mapping[str, Any]] = []
    for report in reports:
        license_section = report.get("license") or {}
        status = _normalize_status(license_section.get("status"))
        entry_base: Mapping[str, Any] = {
            "environment": report.get("environment", "unknown"),
            "status": status,
        }
        if status in {"ok", "valid", "active"}:
            continue
        if status == "skipped":
            skipped.append(
                {
                    **entry_base,
                    "reason": report.get("license", {}).get("reason"),
                }
            )
            continue
        issues.append(
            {
                **entry_base,
                "errors": tuple(license_section.get("errors", ()) or ()),
                "warnings": tuple(license_section.get("warnings", ()) or ()),
            }
        )
    return tuple(issues), tuple(skipped)


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
    baseline_sources = tuple(report.get("baseline_source_configs", ()) or ())
    if baseline_sources:
        lines.append(
            "- **Konfiguracje bazowe:** "
            + ", ".join(map(str, baseline_sources))
        )
    lines.append(
        f"- **Środowiska:** {', '.join(map(str, environments)) or 'brak środowisk'}"
    )

    summary = report.get("summary") or {}
    lines.append("")
    lines.append("## Podsumowanie")
    lines.append("")
    lines.append(f"- **Status:** {summary.get('status', 'unknown')}")
    total_envs = summary.get("total_environments")
    ok_count = summary.get("ok_count")
    blocked_count = summary.get("blocked_count")
    if isinstance(total_envs, int) and isinstance(ok_count, int):
        lines.append(
            f"- **Liczba środowisk OK:** {ok_count}/{total_envs}"
        )
    ok_envs = summary.get("ok_environments", ()) or ()
    if ok_envs:
        lines.append(f"- **Środowiska OK:** {', '.join(map(str, ok_envs))}")
    blocked_envs = summary.get("blocked_environments", ()) or ()
    if blocked_envs:
        blocked_line = f"- **Środowiska z blokadami:** {', '.join(map(str, blocked_envs))}"
        if isinstance(blocked_count, int):
            blocked_line += f" (łącznie: {blocked_count})"
        lines.append(blocked_line)
    reasons = summary.get("reasons", ()) or ()
    if reasons:
        lines.append("- **Powody:**")
        for reason in reasons:
            lines.append(f"  - {reason}")
    license_issue_count = summary.get("license_issue_count")
    if isinstance(license_issue_count, int):
        lines.append(f"- **Problemy licencyjne:** {license_issue_count}")
    license_skipped_count = summary.get("license_skipped_count")
    if isinstance(license_skipped_count, int) and license_skipped_count:
        lines.append(
            f"- **Pominięte walidacje licencji:** {license_skipped_count}"
        )
    skipped_reasons = summary.get("license_skipped_reasons", ()) or ()
    if skipped_reasons:
        lines.append(
            "- **Powody pominięcia licencji:** "
            + ", ".join(map(str, skipped_reasons))
        )

    baseline_total = summary.get("baseline_total")
    if isinstance(baseline_total, int) and baseline_total:
        lines.append(f"- **Porównania bazowe:** {baseline_total}")
        mismatch_count = summary.get("baseline_mismatch_count")
        if isinstance(mismatch_count, int):
            lines.append(f"- **Różnice vs baza:** {mismatch_count}")
        missing_count = summary.get("baseline_missing_count")
        if isinstance(missing_count, int) and missing_count:
            lines.append(
                f"- **Brakujące porównania bazowe:** {missing_count}"
            )
        mismatch_envs = summary.get("baseline_mismatch_environments", ()) or ()
        if mismatch_envs:
            lines.append(
                "- **Środowiska z różnicami bazowymi:** "
                + ", ".join(map(str, mismatch_envs))
            )
        missing_envs = summary.get("baseline_missing_environments", ()) or ()
        if missing_envs:
            lines.append(
                "- **Środowiska bez pełnej bazy:** "
                + ", ".join(map(str, missing_envs))
            )

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
    license_skipped = summary.get("license_skipped", ()) or ()
    if license_skipped:
        lines.append("")
        lines.append("### Pomięte walidacje licencji")
        lines.append("")
        for entry in license_skipped:
            if not isinstance(entry, Mapping):
                continue
            env = entry.get("environment", "unknown")
            status = entry.get("status", "skipped")
            lines.append(f"- **{env}:** {status}")

    baseline_mismatches = summary.get("baseline_mismatches", ()) or ()
    if baseline_mismatches:
        lines.append("")
        lines.append("### Różnice względem środowiska bazowego")
        lines.append("")
        for entry in baseline_mismatches:
            if not isinstance(entry, Mapping):
                continue
            env = entry.get("environment", "unknown")
            baseline_env = entry.get("baseline_environment", "unknown")
            status = entry.get("status", "unknown")
            lines.append(
                f"- **{env}** (baza: {baseline_env}) — status: {status}"
            )
            missing_channels = entry.get("missing_channels", ()) or ()
            if missing_channels:
                lines.append(
                    "  - Brakujące kanały: "
                    + ", ".join(map(str, missing_channels))
                )
            extra_channels = entry.get("extra_channels", ()) or ()
            if extra_channels:
                lines.append(
                    "  - Dodatkowe kanały: "
                    + ", ".join(map(str, extra_channels))
                )
            risk_diffs = entry.get("risk_profile_differences", ()) or ()
            if risk_diffs:
                lines.append("  - Różnice w profilu ryzyka:")
                for diff in risk_diffs:
                    if not isinstance(diff, Mapping):
                        continue
                    field = diff.get("field", "unknown")
                    lines.append(
                        f"    - {field}: bazowe={diff.get('baseline')}, docelowe={diff.get('target')}"
                    )
            throttle_diffs = entry.get("throttle_differences", ()) or ()
            if throttle_diffs:
                lines.append("  - Throttling:")
                for diff in throttle_diffs:
                    if not isinstance(diff, Mapping):
                        continue
                    field = diff.get("field", "unknown")
                    lines.append(
                        f"    - {field}: bazowe={diff.get('baseline')}, docelowe={diff.get('target')}"
                    )
            audit_diffs = entry.get("audit_differences", ()) or ()
            if audit_diffs:
                lines.append("  - Audyt alertów:")
                for diff in audit_diffs:
                    if not isinstance(diff, Mapping):
                        continue
                    field = diff.get("field", "unknown")
                    lines.append(
                        f"    - {field}: bazowe={diff.get('baseline')}, docelowe={diff.get('target')}"
                    )

    baseline_missing = summary.get("baseline_missing", ()) or ()
    if baseline_missing:
        lines.append("")
        lines.append("### Niepełne porównania bazowe")
        lines.append("")
        for entry in baseline_missing:
            if not isinstance(entry, Mapping):
                continue
            env = entry.get("environment", "unknown")
            baseline_env = entry.get("baseline_environment", "unknown")
            status = entry.get("status", "unknown")
            lines.append(
                f"- **{env}** (baza: {baseline_env}) — status: {status}"
            )

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
        if metadata.get("document_root"):
            lines.append(
                f"- **Katalog dokumentów:** {metadata['document_root']}"
            )
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
    if license_section.get("reason"):
        lines.append(f"- **Powód pominięcia:** {license_section['reason']}")

    risk_profile_details = report.get("risk_profile_details") or {}
    if risk_profile_details:
        lines.append("")
        lines.append(f"{heading_prefix}## Profil ryzyka")
        lines.append("")
        for key, value in risk_profile_details.items():
            lines.append(f"- **{key}:** {value}")

    baseline = report.get("baseline_comparison") or {}
    if baseline:
        lines.append("")
        lines.append(f"{heading_prefix}## Porównanie z bazowym środowiskiem")
        lines.append("")
        lines.append(
            f"- **Środowisko bazowe:** {baseline.get('baseline_environment', 'unknown')}"
        )
        if baseline.get("baseline_source_config"):
            lines.append(
                "- **Konfiguracja bazowa:** "
                f"{baseline.get('baseline_source_config')}"
            )
        lines.append(f"- **Status:** {baseline.get('status', 'unknown')}")

        risk_diff = baseline.get("risk_profile") or {}
        if risk_diff:
            lines.append("")
            lines.append(f"{heading_prefix}### Profil ryzyka (porównanie)")
            lines.append("")
            if risk_diff.get("baseline_profile") or risk_diff.get("target_profile"):
                lines.append(
                    "- **Profil docelowy:** "
                    f"{risk_diff.get('target_profile', 'unknown')}"
                )
                lines.append(
                    "- **Profil bazowy:** "
                    f"{risk_diff.get('baseline_profile', 'unknown')}"
                )
            lines.append(f"- **Status:** {risk_diff.get('status', 'unknown')}")
            differences = risk_diff.get("differences", ()) or ()
            if differences:
                lines.append("- **Różnice:**")
                for diff in differences:
                    field = diff.get("field", "unknown")
                    baseline_value = diff.get("baseline")
                    target_value = diff.get("target")
                    lines.append(
                        f"  - {field}: bazowe={baseline_value}, docelowe={target_value}"
                    )

        alerting_diff = baseline.get("alerting") or {}
        if alerting_diff:
            lines.append("")
            lines.append(f"{heading_prefix}### Alerting (porównanie)")
            lines.append("")
            lines.append(
                "- **Kanały docelowe:** "
                + (", ".join(alerting_diff.get("target_channels", ())) or "brak")
            )
            lines.append(
                "- **Kanały bazowe:** "
                + (", ".join(alerting_diff.get("baseline_channels", ())) or "brak")
            )
            lines.append(f"- **Status:** {alerting_diff.get('status', 'unknown')}")
            missing = alerting_diff.get("missing_channels", ()) or ()
            if missing:
                lines.append(
                    "- **Kanały brakujące względem bazy:** "
                    + ", ".join(map(str, missing))
                )
            extra = alerting_diff.get("extra_channels", ()) or ()
            if extra:
                lines.append(
                    "- **Kanały dodatkowe vs baza:** "
                    + ", ".join(map(str, extra))
                )

            throttle_diff = alerting_diff.get("throttle") or {}
            if throttle_diff:
                lines.append("- **Throttling:**")
                lines.append(
                    "  - Status: " + throttle_diff.get("status", "unknown")
                )
                differences = throttle_diff.get("differences", ()) or ()
                if differences:
                    lines.append("  - Różnice:")
                    for diff in differences:
                        field = diff.get("field", "unknown")
                        lines.append(
                            f"    - {field}: bazowe={diff.get('baseline')}, docelowe={diff.get('target')}"
                        )

            audit_diff = alerting_diff.get("audit") or {}
            if audit_diff:
                lines.append("- **Audyt alertów:**")
                lines.append("  - Status: " + audit_diff.get("status", "unknown"))
                differences = audit_diff.get("differences", ()) or ()
                if differences:
                    lines.append("  - Różnice:")
                    for diff in differences:
                        field = diff.get("field", "unknown")
                        lines.append(
                            f"    - {field}: bazowe={diff.get('baseline')}, docelowe={diff.get('target')}"
                        )

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
        "--document-root",
        help=(
            "Katalog bazowy z artefaktami checklisty (domyślnie katalog "
            "konfiguracji)"
        ),
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
        "--baseline",
        help=(
            "Nazwa środowiska referencyjnego do porównania profilu ryzyka i alertingu"
        ),
    )
    parser.add_argument(
        "--baseline-config",
        help=(
            "Alternatywna ścieżka do konfiguracji zawierającej środowisko bazowe"
        ),
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Zakończ z kodem wyjścia 2, jeśli checklisty lub dokumenty mają status inny niż OK",
    )
    parser.add_argument(
        "--fail-on-license",
        action="store_true",
        help="Zakończ z kodem wyjścia 3, jeżeli walidacja licencji zakończy się statusem innym niż OK",
    )
    parser.add_argument(
        "--fail-on-skipped-license",
        action="store_true",
        help="Zakończ z kodem wyjścia 4, gdy walidacja licencji zostanie pominięta",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config)
    core_config = load_core_config(config_path)

    baseline_config_path: Path | None = None
    baseline_core_config: CoreConfig | None = None
    if args.baseline_config:
        baseline_config_path = Path(args.baseline_config)
        baseline_core_config = load_core_config(baseline_config_path)

    if args.document_root:
        try:
            document_root = Path(args.document_root).expanduser().resolve()
        except Exception:  # pragma: no cover - defensywne fallbacki ścieżek
            document_root = Path(args.document_root)
    else:
        document_root = None

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
                document_root=document_root,
                baseline_environment=args.baseline,
                baseline_config_path=baseline_config_path,
                baseline_core_config=baseline_core_config,
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
    should_fail_blocked = args.fail_on_blocked and summary_status not in {"ok"}
    license_issues, license_skipped = _collect_license_outcomes(final_report, reports)
    should_fail_license = args.fail_on_license and bool(license_issues)
    should_fail_skipped = args.fail_on_skipped_license and bool(license_skipped)

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

    exit_code = 0
    if should_fail_license:
        license_statuses = [
            f"{issue.get('environment', 'unknown')}:{issue.get('status', 'unknown')}"
            for issue in license_issues
        ]
        license_errors: list[str] = []
        for issue in license_issues:
            license_errors.extend(str(err) for err in issue.get("errors", ()))
        license_warnings: list[str] = []
        for issue in license_issues:
            license_warnings.extend(str(warn) for warn in issue.get("warnings", ()))
        status_text = ", ".join(license_statuses) or "brak statusów"
        errors_text = "; ".join(dict.fromkeys(license_errors)) or "brak błędów"
        warnings_text = "; ".join(dict.fromkeys(license_warnings)) or "brak ostrzeżeń"
        print(
            "Raport promotion-to-live zawiera problemy licencyjne "
            f"(statusy: {status_text}; błędy: {errors_text}; ostrzeżenia: {warnings_text}).",
            file=sys.stderr,
        )
        exit_code = max(exit_code, 3)
    if should_fail_skipped:
        skipped_descriptors = [
            f"{entry.get('environment', 'unknown')}:{entry.get('reason', 'unknown')}"
            for entry in license_skipped
        ]
        skipped_text = ", ".join(skipped_descriptors) or "brak środowisk"
        print(
            "Walidacja licencji została pominięta dla środowisk "
            f"({skipped_text}).",
            file=sys.stderr,
        )
        exit_code = max(exit_code, 4)
    if should_fail_blocked:
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
        exit_code = max(exit_code, 2)
    if (
        exit_code == 0
        and license_skipped
        and args.fail_on_license
        and not should_fail_skipped
    ):
        skipped_envs = ", ".join(
            str(entry.get("environment", "unknown")) for entry in license_skipped
        ) or "brak środowisk"
        skipped_reasons = ", ".join(
            dict.fromkeys(
                str(entry.get("reason", "unknown")) for entry in license_skipped
            )
        ) or "unknown"
        print(
            "Walidacja licencji została pominięta dla środowisk: "
            f"{skipped_envs} (status=skipped; powód: {skipped_reasons}).",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":  # pragma: no cover - wywołanie CLI
    raise SystemExit(main())
