"""Prepare a reviewable clean-MSI operation; deliberately contains no executor.

Requalify inputs on every invocation, including after native host/session reads.
Even a complete plan is not a permission to install, enroll or start services.
PRE-01B.2 must implement the separately reviewed materialization/consent path.
"""

from __future__ import annotations

import subprocess
from copy import deepcopy
from typing import Any

from bot_core.persistence.fingerprints import canonical_json_sha256
from deployment.windows_stage9_operator_approval import (
    OperatorApprovalError,
    OperatorAuthenticationUnavailable,
    authenticate_operator,
    validate_operator_approval,
)

HANDOFF_INPUTS = frozenset(
    {
        "archive",
        "artifact_metadata",
        "workflow_run_metadata",
        "production_trust_package",
        "allowed_root",
        "expected_artifact_id",
        "expected_archive_sha256",
        "expected_revision",
        "expected_run_id",
        "expected_run_attempt",
        "expected_production_trust_run_id",
        "expected_production_trust_manifest_sha256",
    }
)


def prepare_installation_plan(
    approval: Any, *, policy: dict[str, Any], handoff_inputs: dict[str, Any]
) -> dict[str, Any]:
    """Native checks plus immutable handoff, yielding only a future operation plan.

    The callable has no install/execute option and is not wired into any workflow.
    Inputs cannot replace authenticated GitHub or native host/session collectors.
    Tests substitute those code boundaries, never a supplied PASS report.
    """
    from deployment.windows_stage9_handoff import qualified_handoff
    from deployment.windows_stage9_host_inspection import run_host_inspection

    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "windows-stage9-installation-preparation",
        "status": "BLOCKED",
        "overall_status": "BLOCKED",
        "checks": [],
        "installation_execution_authorized": False,
        "stage_9": "IN_PROGRESS",
        "legal_production_enrollment": "NOT_PERFORMED",
        "run_stage10": False,
        "stage10_status": "BLOCKED",
        "stage10_qualification": "NOT_RUN",
    }
    stage = "handoff_inputs"
    try:
        if not isinstance(handoff_inputs, dict) or set(handoff_inputs) != HANDOFF_INPUTS:
            raise OperatorApprovalError("exact reviewed handoff inputs are required")
        inputs = dict(handoff_inputs)
        original_approval = canonical_json_sha256(approval)
        original_policy = canonical_json_sha256(policy)
        approval_copy, policy_copy = deepcopy(approval), deepcopy(policy)
        stage = "qualified_handoff"
        handoff = qualified_handoff(**inputs)
        if handoff.get("status") != "PASS":
            report["status"] = report["overall_status"] = (
                "FAIL" if handoff.get("status") == "FAIL" else "BLOCKED"
            )
            report["checks"].append({"name": stage, "status": report["status"]})
            return report
        details = deepcopy(handoff["details"])
        stage = "approval_data"
        approval_digest = validate_operator_approval(approval_copy, details, policy=policy_copy)
        stage = "clean_target_host"
        host = run_host_inspection(live=True)
        if host.get("status") != "PASS":
            report["checks"].append({"name": stage, "status": "BLOCKED"})
            return report
        if host.get("host_identity") != policy_copy["host_identity"]:
            raise OperatorApprovalError("inspected target differs from approved host")
        stage = "operator_authentication"
        session = authenticate_operator(approval_copy, details, policy=policy_copy)
        if session.get("approval_sha256") != approval_digest:
            raise OperatorApprovalError("session is not bound to this approval")
        stage = "final_integrity_and_expiry"
        # A stored PASS report cannot be reused after files or GitHub state change.
        fresh = qualified_handoff(**inputs)
        if fresh.get("status") != "PASS" or fresh.get("details") != details:
            raise OperatorApprovalError("qualification changed before plan completion")
        if (
            validate_operator_approval(approval_copy, details, policy=policy_copy)
            != approval_digest
            or canonical_json_sha256(approval) != original_approval
            or canonical_json_sha256(policy) != original_policy
        ):
            raise OperatorApprovalError("approval or independent policy changed")
        report.update(
            status="PREPARED_REVIEW_ONLY",
            overall_status="BLOCKED",
            approval_sha256=approval_digest,
            host_identity=deepcopy(policy_copy["host_identity"]),
            qualification=details,
            plan={
                "action": "install-exact-qualified-msi-on-clean-host",
                "msi_member": details["msi_file"],
                "msi_sha256": details["msi_sha256"],
                "manifest_sha256": details["manifest_sha256"],
                "future_msiexec_arguments": [
                    "/i",
                    "<protected-materialized-qualified-msi>",
                    "/qn",
                    "/norestart",
                    "REBOOT=ReallySuppress",
                ],
                "post_install": "deployment.windows_stage9_host_inspection.verify_post_install",
                "backend_start_mode": "Demand",
                "backend_start_authorized": False,
                "freshness_verifier_start_authorized": False,
            },
        )
        report["checks"].append({"name": "reviewable_plan", "status": "PASS"})
    except OperatorAuthenticationUnavailable:
        report["checks"].append(
            {
                "name": stage,
                "status": "BLOCKED",
                "reason": "native operator authentication unavailable",
            }
        )
    except (OperatorApprovalError, ValueError, TypeError, KeyError, AttributeError):
        report["status"] = report["overall_status"] = "FAIL"
        report["checks"].append(
            {
                "name": stage,
                "status": "FAIL",
                "reason": "required identity, approval or integrity check failed",
            }
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):
        report["checks"].append(
            {
                "name": stage,
                "status": "BLOCKED",
                "reason": "required host or GitHub read unavailable",
            }
        )
    return report
