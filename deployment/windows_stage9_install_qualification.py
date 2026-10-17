"""Separate passive clean-base and legally enrolled Stage 9 qualification.

All payload, SCM, ACL, PostgreSQL and runtime validators belong to the existing
host-inspection authorities. These entry points select the appropriate phase;
neither can install, enroll, start services or authorize Stage 10. Saved facts
are rehearsal inputs and cannot earn a live PASS.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

from deployment import windows_stage9_host_inspection as host
from deployment.windows_installer.contract import CONTRACT


def _base_services(facts: dict[str, Any]) -> None:
    for service in facts["services"]:
        if service["name"] == CONTRACT.postgresql_service:
            continue
        if (
            service.get("start_type") != 3
            or service.get("state") != "Stopped"
            or type(service.get("process_id")) is not int
            or service["process_id"] != 0
            or service.get("second_process_id") != 0
            or service.get("second_state") != 1
            or service.get("process_path") not in (None, "")
            or service.get("process_started") is not None
        ):
            raise host.HostInspectionError("clean-base backend/verifier must remain Demand/stopped")


def _base_enrollment(facts: dict[str, Any]) -> None:
    if facts.get("enrollment") != {"state": "absent", "files": []}:
        raise host.HostInspectionError("clean-base enrollment absence is unavailable")


def _confirmed_legal_enrollment(facts: dict[str, Any]) -> None:
    """Resolve the accepted canonical authority, never a saved PASS or boolean.

    The production handoff loader currently fails closed: ceremony trust alone
    cannot supply legal enrollment. Keep that boundary, rather than substitute
    local marker contents for the accepted external provisioning authority.
    """
    enrollment = facts.get("enrollment")
    if not isinstance(enrollment, dict) or enrollment.get("state") != "present":
        raise host.HostInspectionError("confirmed legal enrollment is unavailable")
    composition = importlib.import_module("deployment.windows_installer.corehost_composition")
    provisioning = composition.load_windows_external_provisioning_handoff()
    reference, claim, _ = composition.resolve_accepted_handoff(provisioning)
    machine = Path(facts["paths"]["program_data"])
    provision = importlib.import_module("deployment.windows_installer.provision")
    provision._committed_record(machine)
    path = machine / "State" / provision.ENROLLMENT_FINALIZATION
    digest = host._sha(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("phase") != "COMMITTED"
        or record.get("claim_reference") != reference
        or record.get("account_id") != claim.account_id
        or record.get("device_installation_id") != claim.device_installation_id
        or host._sha(path) != digest
    ):
        raise host.HostInspectionError("confirmed legal enrollment identity is unavailable")


def _operation(report: dict[str, Any], name: str, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except Exception as exc:
        missing = isinstance(exc, (KeyError, OSError)) or (
            isinstance(exc, host.HostInspectionError) and "unavailable" in str(exc)
        )
        host._check(
            report,
            name,
            "BLOCKED" if missing else "FAIL",
            f"Passive qualification failed ({type(exc).__name__}).",
        )
    else:
        host._check(
            report, name, "PASS", "Existing product authority accepted the collected facts."
        )


def _qualify(
    qualified_msi: Path,
    manifest_path: Path,
    clean_receipt_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_msi_sha256: str,
    expected_receipt_sha256: str,
    source_revision: str,
    ci_run_id: str,
    live: bool,
    observation: Any,
    enrolled: bool,
) -> dict[str, Any]:
    phase = "enrolled-runtime" if enrolled else "clean-base-installation"
    report = host._report("windows-stage9-" + phase, live)
    report["qualification_phase"] = phase
    if live and observation is not None:
        host._check(
            report, "live_collection", "FAIL", "Live qualification cannot consume saved facts."
        )
        return host._finish(report)
    digests = {
        "expected_manifest_sha256": expected_manifest_sha256,
        "expected_msi_sha256": expected_msi_sha256,
        "expected_receipt_sha256": expected_receipt_sha256,
    }
    try:
        manifest = host._qualified_install_inputs(
            qualified_msi, manifest_path, clean_receipt_path, **digests
        )
    except Exception as exc:
        host._check(
            report,
            "qualified_inputs",
            "FAIL",
            f"Qualified input read failed ({type(exc).__name__}).",
        )
        return host._finish(report)
    host._check(
        report, "qualified_inputs", "PASS", "Approved MSI, manifest and receipt were rehashed."
    )
    if enrolled:
        try:
            enrollment_facts = host.collect_host_observation() if live else observation
            if not isinstance(enrollment_facts, dict):
                raise host.HostInspectionError("confirmed legal enrollment is unavailable")
            _confirmed_legal_enrollment(enrollment_facts)
        except Exception as exc:
            host._check(
                report,
                "legal_enrollment_authority",
                "BLOCKED",
                "Confirmed accepted provisioning authority and committed legal enrollment "
                "are required "
                f"before runtime inspection ({type(exc).__name__}); no service was started.",
            )
            return host._finish(report)
        host._check(
            report,
            "legal_enrollment_authority",
            "PASS",
            "Canonical accepted enrollment was resolved.",
        )
        report["legal_production_enrollment"] = "CONFIRMED" if live else "UNVERIFIED"
    try:
        facts = (
            host._collect_post_install(
                qualified_msi, include_readiness=enrolled, include_configuration=True
            )
            if live
            else observation
        )
        if (
            not isinstance(facts, dict)
            or facts.get("os") != "Windows"
            or facts.get("native_x64") is not True
            or not host._identity(facts.get("host_identity"))
            or not host._paths(facts.get("paths"))
            or facts["paths"].get("program_files_exists") is not True
            or facts["paths"].get("program_data_exists") is not True
        ):
            raise host.HostInspectionError("native installed host identity/paths are unavailable")
        if enrolled and facts["host_identity"] != enrollment_facts["host_identity"]:
            raise host.HostInspectionError("host changed after enrollment authority resolution")
        if enrolled:
            _confirmed_legal_enrollment(facts)
        report["host_identity"] = facts["host_identity"]
        root = facts["paths"]["program_files"]
        payload = Path(root if live else facts.get("payload_root", root))
    except Exception as exc:
        host._check(
            report,
            "host_access",
            "BLOCKED",
            f"Native passive collection failed ({type(exc).__name__}).",
        )
        return host._finish(report)
    operations: list[tuple[str, Callable[[], Any]]] = [
        ("installed_files", lambda: host._installed_files(payload, manifest["installed_files"])),
        ("msi_registration", lambda: host._registration(facts, manifest)),
        (
            "runtime_provenance" if enrolled else "artifact_provenance",
            lambda: host.qualify_runtime_provenance(
                manifest_path,
                clean_receipt_path,
                payload / "CryptoHunterBackend.exe",
                facts["products"][0]["version"],
                source_revision=source_revision,
                ci_run_id=ci_run_id,
                ci_provider="https://github.com",
            ),
        ),
        (
            "scm_configuration",
            lambda: host._services(facts, root, enrolled=enrolled, clean_base=not enrolled),
        ),
        ("postgresql_static", lambda: host._postgresql_static(facts)),
        ("postgresql_static_configuration", lambda: host._postgresql_configuration(facts)),
        (
            "postgresql_effective_sql",
            lambda: host._postgresql_effective(facts, require_live=live),
        ),
        ("product_acl", lambda: host._product_acl(facts)),
    ]
    if enrolled:
        operations.extend(
            [
                ("runtime_mtls_readiness", lambda: host._readiness(facts, root)),
            ]
        )
    else:
        operations.extend(
            [
                ("base_services_stopped", lambda: _base_services(facts)),
                ("enrollment_absent", lambda: _base_enrollment(facts)),
            ]
        )
    for name, operation in operations:
        _operation(report, name, operation)
    host._report_postgresql_evidence(report, facts)
    if report.get("missing_authorities"):
        for check in report["checks"]:
            if check["name"] == "postgresql_effective_sql" and check["status"] == "BLOCKED":
                check["details"] = (
                    "Complete service-token SQL authority is unavailable; see the separate "
                    "passive SQL capabilities and missing authorities. Static files, saved "
                    "reports and readiness markers cannot supply missing authentication proof."
                )
    _operation(
        report,
        "qualified_input_stability",
        lambda: host._qualified_install_inputs(
            qualified_msi, manifest_path, clean_receipt_path, **digests
        ),
    )
    _operation(
        report,
        "final_scm_process_identity",
        lambda: host._stable_service_instances(
            facts.get("services"),
            host._powershell(host._FINAL_SERVICE_SCRIPT) if live else facts.get("final_services"),
        ),
    )
    return host._finish(report)


def qualify_clean_base_installation(
    qualified_msi: Path,
    manifest_path: Path,
    clean_receipt_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_msi_sha256: str,
    expected_receipt_sha256: str,
    source_revision: str,
    ci_run_id: str,
    live: bool = False,
    observation: Any = None,
) -> dict[str, Any]:
    """Qualify the clean MSI base, without backend/verifier runtime readiness."""
    return _qualify(
        qualified_msi,
        manifest_path,
        clean_receipt_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_msi_sha256=expected_msi_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        source_revision=source_revision,
        ci_run_id=ci_run_id,
        live=live,
        observation=observation,
        enrolled=False,
    )


def qualify_enrolled_runtime(
    qualified_msi: Path,
    manifest_path: Path,
    clean_receipt_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_msi_sha256: str,
    expected_receipt_sha256: str,
    source_revision: str,
    ci_run_id: str,
    live: bool = False,
    observation: Any = None,
) -> dict[str, Any]:
    """Require legal enrollment before any passive runtime qualification."""
    return _qualify(
        qualified_msi,
        manifest_path,
        clean_receipt_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_msi_sha256=expected_msi_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        source_revision=source_revision,
        ci_run_id=ci_run_id,
        live=live,
        observation=observation,
        enrolled=True,
    )
