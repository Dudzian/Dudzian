"""Review-only MSI planning with hermetic handoff and native observation seams."""

from __future__ import annotations

import importlib
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from deployment import (
    windows_stage9_install_preparation as preparation,
    windows_stage9_operator_approval as approval_module,
)

NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
HOST = {
    "hostname": "TEST-ONLY-WINDOWS",
    "machine_guid": "11111111-1111-4111-8111-111111111111",
    "hardware_uuid": "22222222-2222-4222-8222-222222222222",
}
SID = "S-1-5-21-111-222-333-1001"


@pytest.fixture
def boundaries(monkeypatch, tmp_path):
    handoff_module = importlib.import_module("deployment.windows_stage9_handoff")
    host_module = importlib.import_module("deployment.windows_stage9_host_inspection")

    class FixedDatetime(datetime):
        current = NOW

        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return cls.current

    monkeypatch.setattr(approval_module, "datetime", FixedDatetime)
    details = {
        "repository": "Dudzian/Dudzian",
        "source_revision": "a" * 40,
        "ci_run_id": "12345",
        "ci_run_attempt": 2,
        "artifact_id": "67890",
        "archive_sha256": "b" * 64,
        "msi_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "production_trust": {"run_id": "98765", "manifest_sha256": "e" * 64},
        "msi_file": "CryptoHunter.msi",
    }
    document = {
        **{key: deepcopy(value) for key, value in details.items() if key != "msi_file"},
        "schema_version": 1,
        "approval_id": "33333333-3333-4333-8333-333333333333",
        "host_identity": deepcopy(HOST),
        "operator_sid": SID,
        "issued_at": "2026-10-16T11:00:00Z",
        "expires_at": "2026-10-16T13:00:00Z",
        "scope": [
            "transfer-qualified-msi",
            "prepare-clean-install",
            "verify-installed-payload",
        ],
    }
    policy = {"host_identity": deepcopy(HOST), "operator_sids": [SID]}
    inputs = {
        "archive": tmp_path / "qualified.zip",
        "artifact_metadata": tmp_path / "artifact.json",
        "workflow_run_metadata": tmp_path / "workflow.json",
        "production_trust_package": tmp_path / "production-trust",
        "allowed_root": tmp_path,
        "expected_artifact_id": "67890",
        "expected_archive_sha256": "b" * 64,
        "expected_revision": "a" * 40,
        "expected_run_id": "12345",
        "expected_run_attempt": 2,
        "expected_production_trust_run_id": "98765",
        "expected_production_trust_manifest_sha256": "e" * 64,
    }
    events = []

    def qualified(**_kwargs):
        events.append("handoff")
        return {"status": "PASS", "details": deepcopy(details)}

    def host(**kwargs):
        assert kwargs == {"live": True}
        events.append("host")
        return {"status": "PASS", "host_identity": deepcopy(HOST)}

    def operator():
        events.append("operator")
        return {
            "operator_sid": SID,
            "host_identity": deepcopy(HOST),
            "authenticated": True,
            "elevated": True,
            "interactive": True,
            "session_id": 7,
            "impersonation": "None",
        }

    handoff = Mock(side_effect=qualified)
    inspection = Mock(side_effect=host)
    native = Mock(side_effect=operator)
    monkeypatch.setattr(handoff_module, "qualified_handoff", handoff)
    monkeypatch.setattr(host_module, "run_host_inspection", inspection)
    monkeypatch.setattr(approval_module, "_collect_operator_session", native)

    def forbidden_process(*args, **kwargs):
        pytest.fail("preparation must never launch an installer, service, or native process")

    monkeypatch.setattr(subprocess, "run", forbidden_process)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    return SimpleNamespace(
        approval=document,
        policy=policy,
        inputs=inputs,
        details=details,
        handoff=handoff,
        host=inspection,
        operator=native,
        clock=FixedDatetime,
        events=events,
    )


def prepare(boundaries):
    return preparation.prepare_installation_plan(
        boundaries.approval, policy=boundaries.policy, handoff_inputs=boundaries.inputs
    )


def assert_execution_blocked(report):
    assert report["installation_execution_authorized"] is False
    assert report["stage_9"] == "IN_PROGRESS"
    assert report["legal_production_enrollment"] == "NOT_PERFORMED"
    assert report["run_stage10"] is False
    assert report["stage10_status"] == "BLOCKED"
    assert report["stage10_qualification"] == "NOT_RUN"


def assert_failed_at(report, name, status="FAIL"):
    assert_execution_blocked(report)
    assert report["overall_status"] == status
    assert report["status"] == status
    assert report["checks"][-1]["name"] == name
    assert report["checks"][-1]["status"] == status
    assert "plan" not in report


def test_complete_preparation_is_only_reviewable_and_requalifies_after_native_reads(boundaries):
    original_approval, original_policy = deepcopy(boundaries.approval), deepcopy(boundaries.policy)
    report = prepare(boundaries)
    assert report["status"] == "PREPARED_REVIEW_ONLY"
    assert report["overall_status"] == "BLOCKED"
    assert_execution_blocked(report)
    assert report["approval_sha256"] == canonical_json_sha256(original_approval)
    assert report["qualification"] == boundaries.details
    assert report["host_identity"] == HOST
    assert boundaries.events == ["handoff", "host", "operator", "handoff"]
    assert boundaries.handoff.call_count == 2
    assert all(call.kwargs == boundaries.inputs for call in boundaries.handoff.call_args_list)
    assert boundaries.host.call_count == boundaries.operator.call_count == 1
    plan = report["plan"]
    assert plan["msi_sha256"] == boundaries.details["msi_sha256"]
    assert plan["manifest_sha256"] == boundaries.details["manifest_sha256"]
    assert plan["msi_member"] == boundaries.details["msi_file"]
    assert plan["future_msiexec_arguments"] == [
        "/i",
        "<protected-materialized-qualified-msi>",
        "/qn",
        "/norestart",
        "REBOOT=ReallySuppress",
    ]
    assert plan["backend_start_mode"] == "Demand"
    assert plan["backend_start_authorized"] is False
    assert plan["freshness_verifier_start_authorized"] is False
    assert boundaries.approval == original_approval
    assert boundaries.policy == original_policy


@pytest.mark.parametrize("missing", sorted(preparation.HANDOFF_INPUTS))
def test_each_pinned_handoff_input_is_required_before_any_observation(boundaries, missing):
    del boundaries.inputs[missing]
    assert_failed_at(prepare(boundaries), "handoff_inputs")
    boundaries.handoff.assert_not_called()
    boundaries.host.assert_not_called()
    boundaries.operator.assert_not_called()


@pytest.mark.parametrize("supplied", [None, [], {"status": "PASS"}, {"approved": True}])
def test_saved_pass_report_cannot_replace_handoff_inputs(boundaries, supplied):
    boundaries.inputs = supplied
    assert_failed_at(prepare(boundaries), "handoff_inputs")
    boundaries.handoff.assert_not_called()


def test_extra_handoff_flags_are_rejected(boundaries):
    boundaries.inputs["approved"] = True
    assert_failed_at(prepare(boundaries), "handoff_inputs")
    boundaries.handoff.assert_not_called()


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "UNVERIFIED", None])
def test_first_handoff_failure_stops_before_host_or_operator(boundaries, status):
    boundaries.handoff.side_effect = None
    boundaries.handoff.return_value = {"status": status}
    assert_failed_at(
        prepare(boundaries), "qualified_handoff", "FAIL" if status == "FAIL" else "BLOCKED"
    )
    boundaries.host.assert_not_called()
    boundaries.operator.assert_not_called()


@pytest.mark.parametrize(
    "error", [OSError("test"), RuntimeError("test"), subprocess.TimeoutExpired("test", 1)]
)
def test_github_or_artifact_read_failure_blocks_preparation(boundaries, error):
    boundaries.handoff.side_effect = error
    assert_failed_at(prepare(boundaries), "qualified_handoff", "BLOCKED")
    boundaries.host.assert_not_called()


@pytest.mark.parametrize("key", ["ci_run_attempt", "msi_sha256", "production_trust"])
def test_stale_approval_fails_before_native_host_read(boundaries, key):
    boundaries.approval[key] = "stale-or-unqualified"
    assert_failed_at(prepare(boundaries), "approval_data")
    boundaries.host.assert_not_called()
    boundaries.operator.assert_not_called()


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "UNVERIFIED", None])
def test_unclean_or_unverified_target_never_collects_operator(boundaries, status):
    boundaries.host.side_effect = None
    boundaries.host.return_value = {"status": status}
    assert_failed_at(prepare(boundaries), "clean_target_host", "BLOCKED")
    boundaries.operator.assert_not_called()
    assert boundaries.handoff.call_count == 1


def test_target_host_must_equal_independent_approved_inventory(boundaries):
    boundaries.host.side_effect = None
    boundaries.host.return_value = {
        "status": "PASS",
        "host_identity": {**HOST, "hostname": "OTHER"},
    }
    assert_failed_at(prepare(boundaries), "clean_target_host")
    boundaries.operator.assert_not_called()


def test_native_host_read_failure_remains_blocked(boundaries):
    boundaries.host.side_effect = OSError("TEST_ONLY_SENSITIVE_HOST_MESSAGE")
    report = prepare(boundaries)
    assert_failed_at(report, "clean_target_host", "BLOCKED")
    assert "TEST_ONLY_SENSITIVE_HOST_MESSAGE" not in str(report)


def test_native_authentication_unavailable_is_blocked_not_approved(boundaries):
    boundaries.operator.side_effect = approval_module.OperatorAuthenticationUnavailable("test")
    assert_failed_at(prepare(boundaries), "operator_authentication", "BLOCKED")
    assert boundaries.handoff.call_count == 1


def test_authenticated_session_still_requires_elevation(boundaries):
    boundaries.operator.side_effect = None
    boundaries.operator.return_value = {
        "operator_sid": SID,
        "host_identity": deepcopy(HOST),
        "authenticated": True,
        "elevated": False,
        "interactive": True,
        "session_id": 7,
        "impersonation": "None",
    }
    assert_failed_at(prepare(boundaries), "operator_authentication")
    assert boundaries.handoff.call_count == 1


def test_session_must_bind_the_approval_digest(boundaries, monkeypatch):
    monkeypatch.setattr(
        preparation,
        "authenticate_operator",
        lambda *args, **kwargs: {
            "approval_sha256": "f" * 64,
            "installation_authorized": False,
        },
    )
    assert_failed_at(prepare(boundaries), "operator_authentication")


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "UNVERIFIED"])
def test_second_handoff_failure_invalidates_preparation(boundaries, status):
    boundaries.handoff.side_effect = [
        {"status": "PASS", "details": deepcopy(boundaries.details)},
        {"status": status},
    ]
    assert_failed_at(prepare(boundaries), "final_integrity_and_expiry")
    assert boundaries.host.call_count == boundaries.operator.call_count == 1


@pytest.mark.parametrize(
    "key", ["archive_sha256", "ci_run_attempt", "production_trust", "msi_file"]
)
def test_handoff_change_after_native_reads_invalidates_even_a_second_pass(boundaries, key):
    changed = deepcopy(boundaries.details)
    changed[key] = "TEST_ONLY_CHANGED_DURING_NATIVE_OBSERVATION"
    boundaries.handoff.side_effect = [
        {"status": "PASS", "details": deepcopy(boundaries.details)},
        {"status": "PASS", "details": changed},
    ]
    assert_failed_at(prepare(boundaries), "final_integrity_and_expiry")


def test_second_github_read_unavailable_is_blocked(boundaries):
    boundaries.handoff.side_effect = [
        {"status": "PASS", "details": deepcopy(boundaries.details)},
        OSError("test"),
    ]
    assert_failed_at(prepare(boundaries), "final_integrity_and_expiry", "BLOCKED")


@pytest.mark.parametrize("changed", ["approval", "policy"])
def test_caller_mutation_during_preparation_fails_final_integrity(boundaries, changed):
    original_handoff = boundaries.handoff.side_effect

    def handoff(**kwargs):
        result = original_handoff(**kwargs)
        if boundaries.handoff.call_count == 2:
            if changed == "approval":
                boundaries.approval["approval_id"] = "44444444-4444-4444-8444-444444444444"
            else:
                boundaries.policy["operator_sids"].append("S-1-5-21-111-222-333-1002")
        return result

    boundaries.handoff.side_effect = handoff
    assert_failed_at(prepare(boundaries), "final_integrity_and_expiry")


def test_expiry_after_native_authentication_before_completion_fails(boundaries):
    original_handoff = boundaries.handoff.side_effect

    def handoff(**kwargs):
        result = original_handoff(**kwargs)
        if boundaries.handoff.call_count == 2:
            boundaries.clock.current = NOW + timedelta(hours=1)
        return result

    boundaries.handoff.side_effect = handoff
    assert_failed_at(prepare(boundaries), "final_integrity_and_expiry")


def test_second_invocation_cannot_reuse_a_stored_prepared_plan(boundaries):
    first = prepare(boundaries)
    assert first["status"] == "PREPARED_REVIEW_ONLY"
    boundaries.handoff.side_effect = OSError("fresh GitHub observation unavailable")
    assert_failed_at(prepare(boundaries), "qualified_handoff", "BLOCKED")
    assert boundaries.handoff.call_count == 3


@pytest.mark.parametrize("option", ["execute", "install", "run_stage10", "snapshot"])
def test_public_preparation_api_has_no_execution_or_snapshot_switch(boundaries, option):
    with pytest.raises(TypeError):
        preparation.prepare_installation_plan(
            boundaries.approval,
            policy=boundaries.policy,
            handoff_inputs=boundaries.inputs,
            **{option: True},
        )
    boundaries.handoff.assert_not_called()
