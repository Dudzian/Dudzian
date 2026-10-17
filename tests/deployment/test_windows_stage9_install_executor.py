"""Hermetic executor security checks; the real installer boundary is never called."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from deployment import (
    windows_stage9_install_executor as executor,
    windows_stage9_operator_approval as approval_module,
)

NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
HOST = {
    "hostname": "TEST-ONLY-WINDOWS",
    "machine_guid": "11111111-1111-4111-8111-111111111111",
    "hardware_uuid": "22222222-2222-4222-8222-222222222222",
}
SID = "S-1-5-21-111-222-333-1001"
OTHER_SID = "S-1-5-21-111-222-333-1002"


@pytest.fixture(autouse=True)
def no_native_processes(monkeypatch):
    def forbidden_process(*args, **kwargs):
        pytest.fail("tests must never launch msiexec or any native process")

    monkeypatch.setattr(subprocess, "run", forbidden_process)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)


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
    artifact_root = tmp_path / "transferred-artifacts"
    artifact_root.mkdir()
    staging_root = tmp_path / "protected-staging"
    staging_root.mkdir()
    msi_bytes = b"TEST ONLY; never a runnable MSI"
    archive = artifact_root / "qualified.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("CryptoHunter.msi", msi_bytes)
    details = {
        "repository": "Dudzian/Dudzian",
        "source_revision": "a" * 40,
        "ci_run_id": "12345",
        "ci_run_attempt": 2,
        "artifact_id": "67890",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "msi_sha256": hashlib.sha256(msi_bytes).hexdigest(),
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
        "scope": list(approval_module.APPROVAL_SCOPE),
    }
    inputs = {
        "archive": archive,
        "artifact_metadata": artifact_root / "artifact.json",
        "workflow_run_metadata": artifact_root / "workflow.json",
        "production_trust_package": artifact_root / "production-trust",
        "allowed_root": artifact_root,
        "expected_artifact_id": "67890",
        "expected_archive_sha256": details["archive_sha256"],
        "expected_revision": "a" * 40,
        "expected_run_id": "12345",
        "expected_run_attempt": 2,
        "expected_production_trust_run_id": "98765",
        "expected_production_trust_manifest_sha256": "e" * 64,
    }
    inputs["artifact_metadata"].write_text("{}", encoding="utf-8")
    inputs["workflow_run_metadata"].write_text("{}", encoding="utf-8")
    inputs["production_trust_package"].mkdir()
    policy = {
        "schema_version": 1,
        "host_identity": deepcopy(HOST),
        "operator_sids": [SID],
        "artifact_root": str(artifact_root),
        "staging_root": str(staging_root),
        "executor_root": str(Path(executor.__file__).resolve().parent.parent),
        "selection": {key: value for key, value in inputs.items() if key.startswith("expected_")},
    }
    policy_path = tmp_path / "independent-host-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    state = SimpleNamespace(
        approval=document,
        policy=policy,
        policy_path=policy_path,
        inputs=inputs,
        details=details,
        clock=FixedDatetime,
        staging_root=staging_root,
        events=[],
        guards=[],
        materialized=[],
        reports=[],
        pin_error=None,
        lock_active=False,
        write_hook=None,
        close_error=None,
    )

    def qualified(**kwargs):
        state.events.append("handoff")
        assert set(kwargs) == set(inputs)
        return {"status": "PASS", "details": deepcopy(details)}

    def host(**kwargs):
        assert kwargs == {"live": True}
        state.events.append("host")
        return {"status": "PASS", "host_identity": deepcopy(HOST)}

    def operator():
        state.events.append("operator")
        return {
            "operator_sid": SID,
            "host_identity": deepcopy(HOST),
            "authenticated": True,
            "elevated": True,
            "interactive": True,
            "session_id": 7,
            "impersonation": "None",
        }

    class LockedInstallation:
        def __init__(self, source, member, digest):
            state.events.append("materialize")
            self.root = staging_root / f"installation-{len(state.materialized)}"
            self.root.mkdir()
            self.msi_path = self.root / "qualified.msi"
            with zipfile.ZipFile(source) as package:
                self.msi_path.write_bytes(package.read(member))
            self.log_path = self.root / "installer.log"
            self.report_path = self.root / "installation-report.json"
            self.journal_path = self.root / "installation-intent.json"
            self.msiexec_path = Path("C:/Windows/System32/msiexec.exe")
            self.digest = digest
            self.argv = [str(self.msiexec_path), *executor.INSTALL_ARGUMENTS]
            self.argv[2] = str(self.msi_path)
            self.argv[-1] = str(self.log_path)
            state.materialized.append(self)

        def verify_pins(self):
            state.events.append("verify-msi")
            if hashlib.sha256(self.msi_path.read_bytes()).hexdigest() != self.digest:
                raise RuntimeError("materialized MSI changed")

        def write_report(self, report):
            assert state.lock_active, "durable reports must be written before releasing custody"
            if not self.journal_path.exists():
                self.journal_path.write_text(json.dumps(report), encoding="utf-8")
            self.report_path.write_text(json.dumps(report), encoding="utf-8")
            state.reports.append(deepcopy(report))
            if state.write_hook is not None:
                state.write_hook(report)

    class LockedInputs:
        def __init__(self):
            self.files = {
                path: path.read_bytes()
                for path in (
                    policy_path,
                    archive,
                    inputs["artifact_metadata"],
                    inputs["workflow_run_metadata"],
                )
            }

        def verify_pins(self):
            state.events.append("verify-inputs")
            if state.pin_error:
                raise state.pin_error
            if any(path.read_bytes() != content for path, content in self.files.items()):
                raise RuntimeError("pinned input changed")

        def materialize_msi(self, *, archive, msi_member, expected_msi_sha256):
            return LockedInstallation(archive, msi_member, expected_msi_sha256)

    @contextmanager
    def locked_inputs(actual_policy_path, actual_inputs, actual_staging_root):
        assert Path(actual_policy_path) == policy_path
        assert Path(actual_staging_root) == staging_root
        assert actual_inputs == state.inputs
        guard = LockedInputs()
        state.guards.append(guard)
        guard.verify_pins()
        state.lock_active = True
        try:
            yield guard
        finally:
            state.lock_active = False
            state.events.append("close-pins")
            if state.close_error is not None:
                raise state.close_error

    state.handoff = Mock(side_effect=qualified)
    state.host = Mock(side_effect=host)
    state.operator = Mock(side_effect=operator)
    state.confirm = Mock(side_effect=lambda _digest: state.events.append("confirm") or True)
    state.run = Mock(side_effect=lambda _argv: state.events.append("installer") or 0)
    monkeypatch.setattr(handoff_module, "qualified_handoff", state.handoff)
    monkeypatch.setattr(host_module, "run_host_inspection", state.host)
    monkeypatch.setattr(approval_module, "_collect_operator_session", state.operator)
    monkeypatch.setattr(executor, "_locked_inputs", locked_inputs)
    monkeypatch.setattr(executor, "_confirm_plan", state.confirm)
    monkeypatch.setattr(executor, "_run_installer", state.run)
    return state


def review(state):
    return executor.review_installation_plan(
        state.approval,
        policy_path=state.policy_path,
        handoff_inputs=state.inputs,
        staging_root=state.staging_root,
    )


def authorize(state):
    report = review(state)
    assert report["status"] == "PREPARED_REVIEW_ONLY", report
    document = deepcopy(state.approval)
    document.update(
        approval_id="44444444-4444-4444-8444-444444444444",
        issued_at="2026-10-16T11:59:00Z",
        expires_at="2026-10-16T12:09:00Z",
        scope=["install-exact-qualified-msi-on-clean-host"],
        plan_sha256=report["plan_sha256"],
        preparation_approval_sha256=report["preparation_approval_sha256"],
        policy_sha256=report["policy_sha256"],
    )
    state.handoff.reset_mock()
    state.host.reset_mock()
    state.operator.reset_mock()
    state.events.clear()
    return document


def execute(state, authorization):
    return executor.execute_installation(
        state.approval,
        authorization,
        policy_path=state.policy_path,
        handoff_inputs=state.inputs,
        staging_root=state.staging_root,
    )


def assert_safe_report(report):
    assert report["installation_execution_authorized"] is False
    assert report["restart_authorized"] is False
    assert report["automatic_retry_authorized"] is False
    assert report["stage_9"] == "IN_PROGRESS"
    assert report["legal_production_enrollment"] == "NOT_PERFORMED"
    assert report["run_stage10"] is False
    assert report["stage10_status"] == "BLOCKED"
    assert report["stage10_qualification"] == "NOT_RUN"


def assert_no_install(state, report):
    state.run.assert_not_called()
    assert report["installer_invoked"] is False
    assert_safe_report(report)
    assert report["overall_status"] in {"FAIL", "BLOCKED"}


def test_review_binds_complete_plan_without_granting_execution(boundaries):
    report = review(boundaries)
    assert report["status"] == "PREPARED_REVIEW_ONLY"
    assert report["overall_status"] == "BLOCKED"
    assert report["plan_sha256"] == canonical_json_sha256(report["plan"])
    assert report["preparation_approval_sha256"] == canonical_json_sha256(boundaries.approval)
    assert report["policy_sha256"] == canonical_json_sha256(boundaries.policy)
    assert_no_install(boundaries, report)
    assert boundaries.host.call_count == boundaries.operator.call_count == 1
    assert boundaries.handoff.call_count == 2
    boundaries.confirm.assert_not_called()
    assert not boundaries.materialized
    assert review(boundaries)["plan_sha256"] == report["plan_sha256"]


@pytest.mark.parametrize("authorization", [None, {}, [], {"approved": True}, {"status": "PASS"}])
def test_missing_or_flag_only_execution_authorization_never_launches(boundaries, authorization):
    assert_no_install(boundaries, execute(boundaries, authorization))
    boundaries.confirm.assert_not_called()


def test_preparation_consent_is_not_installation_authorization(boundaries):
    assert_no_install(boundaries, execute(boundaries, deepcopy(boundaries.approval)))
    boundaries.confirm.assert_not_called()


def test_environment_flag_cannot_grant_execution(boundaries, monkeypatch):
    monkeypatch.setenv("CRYPTOHUNTER_STAGE9_INSTALL_APPROVED", "true")
    monkeypatch.setenv("APPROVED", "true")
    assert_no_install(boundaries, execute(boundaries, None))


@pytest.mark.parametrize("missing", sorted(executor.EXECUTION_KEYS))
def test_each_execution_binding_is_mandatory_before_consent(boundaries, missing):
    authorization = authorize(boundaries)
    del authorization[missing]
    assert_no_install(boundaries, execute(boundaries, authorization))
    boundaries.confirm.assert_not_called()


def test_extra_execution_fields_cannot_grant_additional_authority(boundaries):
    authorization = authorize(boundaries)
    authorization["approved"] = True
    assert_no_install(boundaries, execute(boundaries, authorization))
    boundaries.confirm.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operator_sid", OTHER_SID),
        ("host_identity", {**HOST, "hostname": "OTHER"}),
        ("msi_sha256", "f" * 64),
        ("manifest_sha256", "f" * 64),
        ("archive_sha256", "f" * 64),
        ("artifact_id", "67891"),
        ("ci_run_id", "12346"),
        ("ci_run_attempt", 3),
        ("production_trust", {"run_id": "99999", "manifest_sha256": "e" * 64}),
        ("scope", list(approval_module.APPROVAL_SCOPE)),
        ("plan_sha256", "f" * 64),
        ("preparation_approval_sha256", "f" * 64),
        ("policy_sha256", "f" * 64),
        ("approval_id", "33333333-3333-4333-8333-333333333333"),
        ("expires_at", "2026-10-16T12:00:00Z"),
        ("issued_at", "2026-10-16T12:01:00Z"),
        ("issued_at", "2026-10-16T11:58:59Z"),
    ],
)
def test_wrong_execution_binding_and_lifetime_never_launches(boundaries, field, value):
    authorization = authorize(boundaries)
    authorization[field] = value
    assert_no_install(boundaries, execute(boundaries, authorization))
    boundaries.confirm.assert_not_called()


@pytest.mark.parametrize("option", ["approved", "snapshot", "session", "run_installer", "execute"])
def test_public_api_has_no_injected_approval_observation_or_process_switch(boundaries, option):
    with pytest.raises(TypeError):
        executor.execute_installation(
            boundaries.approval,
            None,
            policy_path=boundaries.policy_path,
            handoff_inputs=boundaries.inputs,
            staging_root=boundaries.staging_root,
            **{option: True},
        )
    boundaries.run.assert_not_called()


@pytest.mark.parametrize("changed", ["artifact", "run", "attempt", "archive_digest"])
def test_policy_pins_original_artifact_run_attempt_and_sha(boundaries, changed):
    authorization = authorize(boundaries)
    key = {
        "artifact": "expected_artifact_id",
        "run": "expected_run_id",
        "attempt": "expected_run_attempt",
        "archive_digest": "expected_archive_sha256",
    }[changed]
    boundaries.inputs[key] = "99999"
    assert_no_install(boundaries, execute(boundaries, authorization))


@pytest.mark.parametrize("missing", sorted(executor.HANDOFF_INPUTS))
def test_each_handoff_input_is_required_before_entering_native_custody(boundaries, missing):
    del boundaries.inputs[missing]
    report = execute(boundaries, None)
    assert_no_install(boundaries, report)
    boundaries.handoff.assert_not_called()
    assert not boundaries.guards


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("artifact_root", "relative-root"),
        ("staging_root", "/some-other-protected-directory"),
        ("executor_root", "/downloaded-artifact-with-executable-code"),
        ("selection", {"expected_run_id": "12345"}),
        ("operator_sids", [OTHER_SID]),
        ("host_identity", {**HOST, "hostname": "OTHER"}),
    ],
)
def test_independent_policy_cannot_be_replaced_by_approval_values(boundaries, field, value):
    authorization = authorize(boundaries)
    boundaries.policy[field] = value
    boundaries.policy_path.write_text(json.dumps(boundaries.policy), encoding="utf-8")
    assert_no_install(boundaries, execute(boundaries, authorization))


@pytest.mark.parametrize("document", ["{", "[]", '{"approved":true}'])
def test_unreadable_or_snapshot_only_host_policy_never_launches(boundaries, document):
    boundaries.policy_path.write_text(document, encoding="utf-8")
    assert_no_install(boundaries, execute(boundaries, None))


def test_prepared_plan_cannot_be_passed_as_handoff_inputs(boundaries):
    prepared = review(boundaries)
    boundaries.inputs = prepared
    assert_no_install(boundaries, execute(boundaries, None))


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "UNVERIFIED", None])
def test_failed_fresh_handoff_cannot_use_a_saved_plan(boundaries, status):
    authorization = authorize(boundaries)
    boundaries.handoff.side_effect = None
    boundaries.handoff.return_value = {"status": status}
    assert_no_install(boundaries, execute(boundaries, authorization))


@pytest.mark.parametrize("reason", ["existing-installation", "enrolled-host", "residue"])
def test_unclean_existing_enrolled_or_residual_host_cannot_install(boundaries, reason):
    authorization = authorize(boundaries)
    boundaries.host.side_effect = None
    boundaries.host.return_value = {
        "status": "BLOCKED",
        "host_identity": deepcopy(HOST),
        "checks": [{"name": reason, "status": "FAIL"}],
    }
    assert_no_install(boundaries, execute(boundaries, authorization))
    boundaries.confirm.assert_not_called()


def test_native_operator_sid_must_match_even_with_valid_documents(boundaries):
    authorization = authorize(boundaries)
    observed = boundaries.operator.side_effect()
    observed["operator_sid"] = OTHER_SID
    boundaries.operator.side_effect = None
    boundaries.operator.return_value = observed
    assert_no_install(boundaries, execute(boundaries, authorization))


@pytest.mark.parametrize("field", ["authenticated", "elevated", "interactive"])
def test_native_operator_authentication_requires_live_elevated_interactive_token(boundaries, field):
    authorization = authorize(boundaries)
    observed = boundaries.operator.side_effect()
    observed[field] = False
    boundaries.operator.side_effect = None
    boundaries.operator.return_value = observed
    assert_no_install(boundaries, execute(boundaries, authorization))


def test_operator_is_authenticated_again_immediately_before_process(boundaries):
    authorization = authorize(boundaries)
    observed = boundaries.operator.side_effect()
    boundaries.operator.side_effect = [
        deepcopy(observed),
        deepcopy(observed),
        {**observed, "operator_sid": OTHER_SID},
    ]
    assert_no_install(boundaries, execute(boundaries, authorization))
    assert boundaries.operator.call_count == 3
    boundaries.confirm.assert_called_once()


def test_wrong_native_host_is_rejected_even_when_operator_is_allowed(boundaries):
    authorization = authorize(boundaries)
    boundaries.host.side_effect = None
    boundaries.host.return_value = {
        "status": "PASS",
        "host_identity": {**HOST, "hostname": "OTHER"},
    }
    assert_no_install(boundaries, execute(boundaries, authorization))


def test_cancellation_never_launches_an_installer(boundaries):
    authorization = authorize(boundaries)
    boundaries.confirm.side_effect = None
    boundaries.confirm.return_value = False
    report = execute(boundaries, authorization)
    assert_no_install(boundaries, report)
    assert report["status"] == "CANCELLED"
    boundaries.confirm.assert_called_once_with(authorization["plan_sha256"])


@pytest.mark.parametrize("changed", ["policy", "archive", "expiry", "host", "operator"])
def test_confirmation_wait_cannot_stale_integrity_identity_or_approval(boundaries, changed):
    authorization = authorize(boundaries)

    def confirm(_digest):
        if changed == "policy":
            boundaries.policy["operator_sids"].append(OTHER_SID)
            boundaries.policy_path.write_text(json.dumps(boundaries.policy), encoding="utf-8")
        elif changed == "archive":
            boundaries.inputs["archive"].write_bytes(b"REPLACED ZIP")
        elif changed == "expiry":
            boundaries.clock.current = NOW + timedelta(minutes=9)
        elif changed == "host":
            boundaries.host.side_effect = None
            boundaries.host.return_value = {"status": "BLOCKED"}
        else:
            boundaries.operator.side_effect = approval_module.OperatorAuthenticationUnavailable(
                "native session lost"
            )
        return True

    boundaries.confirm.side_effect = confirm
    assert_no_install(boundaries, execute(boundaries, authorization))


def test_authorization_must_remain_valid_after_durable_journal_write(boundaries):
    authorization = authorize(boundaries)

    def write_hook(report):
        if report.get("installer_invocation_prepared"):
            boundaries.clock.current = NOW + timedelta(minutes=9)

    boundaries.write_hook = write_hook
    assert_no_install(boundaries, execute(boundaries, authorization))
    assert boundaries.reports


def test_native_materialization_cannot_add_unreviewed_installer_arguments(boundaries):
    authorization = authorize(boundaries)

    def confirm(_digest):
        boundaries.materialized[-1].argv.append("UNREVIEWED_PROPERTY=1")
        return True

    boundaries.confirm.side_effect = confirm
    assert_no_install(boundaries, execute(boundaries, authorization))


def test_caller_mutation_does_not_change_frozen_qualified_inputs(boundaries):
    authorization = authorize(boundaries)

    def confirm(_digest):
        boundaries.inputs["expected_run_attempt"] = 3
        authorization["msi_sha256"] = "f" * 64
        boundaries.approval["operator_sid"] = OTHER_SID
        return True

    boundaries.confirm.side_effect = confirm
    report = execute(boundaries, authorization)
    assert report["status"] == "INSTALLED"
    boundaries.run.assert_called_once()
    assert report["plan"]["qualification"]["ci_run_attempt"] == 2
    assert report["plan"]["operator_sid"] == SID
    assert all(
        call.kwargs["expected_run_attempt"] == 2 for call in boundaries.handoff.call_args_list
    )


@pytest.mark.parametrize("error", [OSError("native unavailable"), RuntimeError("unsafe ACL")])
def test_native_pin_or_acl_failure_never_launches(boundaries, error):
    authorization = authorize(boundaries)
    boundaries.pin_error = error
    assert_no_install(boundaries, execute(boundaries, authorization))


def test_changed_materialized_msi_is_rejected_before_process(boundaries, monkeypatch):
    authorization = authorize(boundaries)
    original_prepare = executor.prepare_installation_plan
    preparations = 0

    def prepare(*args, **kwargs):
        nonlocal preparations
        preparations += 1
        report = original_prepare(*args, **kwargs)
        if preparations >= 2 and boundaries.materialized:
            boundaries.materialized[-1].msi_path.write_bytes(b"SUBSTITUTED MSI")
        return report

    monkeypatch.setattr(executor, "prepare_installation_plan", prepare)
    # The second preparation runs after consent; its final read is a useful
    # simulation of a substitution just before the process hand-off.
    report = execute(boundaries, authorization)
    assert_no_install(boundaries, report)


@pytest.mark.parametrize(
    ("returncode", "status"),
    [
        (0, "INSTALLED"),
        (3010, "INSTALLED_REBOOT_REQUIRED"),
        (1641, "UNEXPECTED_REBOOT_INITIATED"),
        (1602, "CANCELLED"),
        (1603, "INSTALLER_ERROR"),
    ],
)
def test_installer_codes_have_explicit_nonrestarting_classification(boundaries, returncode, status):
    authorization = authorize(boundaries)

    def installer(_argv):
        assert boundaries.lock_active
        journal = json.loads(boundaries.materialized[-1].journal_path.read_text(encoding="utf-8"))
        assert journal["installer_invocation_prepared"] is True
        assert journal["installer_invoked"] is False
        return returncode

    boundaries.run.side_effect = installer
    report = execute(boundaries, authorization)
    assert report["status"] == status
    assert report["installer_invoked"] is True
    assert_safe_report(report)
    boundaries.run.assert_called_once()
    argv = boundaries.run.call_args.args[0]
    installation = boundaries.materialized[-1]
    assert isinstance(argv, list)
    assert argv[0] == str(installation.msiexec_path)
    assert argv[argv.index("/i") + 1] == str(installation.msi_path)
    assert "/norestart" in argv and "REBOOT=ReallySuppress" in argv
    assert "ALLUSERS=1" in argv
    assert argv[argv.index("/L*V") + 1] == str(installation.log_path)
    assert not any(option in argv for option in ("/x", "/f", "/forcerestart", "/promptrestart"))
    assert boundaries.host.call_count >= 2
    assert boundaries.operator.call_count >= 3
    assert boundaries.reports and boundaries.reports[-1]["status"] == status
    assert report["intent_journal"] == str(installation.journal_path)
    assert json.loads(installation.report_path.read_text(encoding="utf-8"))["status"] == status
    assert json.loads(installation.journal_path.read_text(encoding="utf-8"))["status"] == (
        "INSTALLATION_OUTCOME_UNKNOWN"
    )


@pytest.mark.parametrize("error", [OSError("TEST_ONLY_SENSITIVE"), KeyboardInterrupt()])
def test_process_error_or_interruption_is_reported_without_automatic_retry(boundaries, error):
    authorization = authorize(boundaries)
    boundaries.run.side_effect = error
    report = execute(boundaries, authorization)
    assert report["installer_invoked"] is True
    assert report["status"] in {"INSTALLATION_OUTCOME_UNKNOWN", "INTERRUPTED_OUTCOME_UNKNOWN"}
    assert_safe_report(report)
    boundaries.run.assert_called_once()
    assert boundaries.reports
    assert "TEST_ONLY_SENSITIVE" not in str(report)


@pytest.mark.parametrize("returncode", [None, True, "0"])
def test_missing_or_malformed_native_exit_status_cannot_claim_installed(boundaries, returncode):
    authorization = authorize(boundaries)
    boundaries.run.side_effect = None
    boundaries.run.return_value = returncode
    report = execute(boundaries, authorization)
    assert_safe_report(report)
    assert report["status"] == "FAIL"
    assert report["installer_invoked"] is True
    assert report["installation_executed"] is False
    boundaries.run.assert_called_once()


def test_integrity_loss_after_installer_returns_cannot_claim_success(boundaries):
    authorization = authorize(boundaries)

    def installer(_argv):
        boundaries.materialized[-1].msi_path.write_bytes(b"UNEXPECTED POST-INSTALL CHANGE")
        return 0

    boundaries.run.side_effect = installer
    report = execute(boundaries, authorization)
    assert_safe_report(report)
    assert report["status"] == "INSTALLATION_OUTCOME_UNKNOWN"
    assert report["installer_return_code"] == 0
    boundaries.run.assert_called_once()
    assert boundaries.reports[-1]["status"] == "INSTALLATION_OUTCOME_UNKNOWN"


def test_unwritable_durable_journal_prevents_process_invocation(boundaries):
    authorization = authorize(boundaries)

    def write_hook(_report):
        raise OSError("TEST_ONLY_SENSITIVE")

    boundaries.write_hook = write_hook
    report = execute(boundaries, authorization)
    assert_no_install(boundaries, report)
    assert report["status"] == "REPORT_WRITE_FAILED_OUTCOME_UNKNOWN"
    assert "TEST_ONLY_SENSITIVE" not in str(report)


def test_interruption_during_consent_never_calls_installer(boundaries):
    authorization = authorize(boundaries)
    boundaries.confirm.side_effect = KeyboardInterrupt()
    report = execute(boundaries, authorization)
    assert_no_install(boundaries, report)
    assert report["status"] == "CANCELLED"
    assert boundaries.reports


def test_default_process_boundary_passes_argv_without_shell(monkeypatch):
    process = Mock()
    process.wait.return_value = 3010
    launch = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", launch)
    argv = ["C:/Windows/System32/msiexec.exe", "/i", "C:/Protected/qualified.msi"]
    assert executor._run_installer(argv) == 3010
    launch.assert_called_once_with(
        argv,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        cwd=str(Path(argv[0]).parent),
    )
    process.wait.assert_called_once_with()
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_interrupted_wait_retains_custody_until_single_installer_finishes(monkeypatch):
    process = Mock()
    process.wait.side_effect = [KeyboardInterrupt(), KeyboardInterrupt(), 3010]
    launch = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", launch)
    argv = ["C:/Windows/System32/msiexec.exe", "/i", "C:/Protected/qualified.msi"]
    with pytest.raises(executor.InstallerWaitInterrupted) as interrupted:
        executor._run_installer(argv)
    assert interrupted.value.returncode == 3010
    assert process.wait.call_count == 3
    launch.assert_called_once()
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


@pytest.mark.parametrize(
    ("returncode", "status"),
    [(0, "INSTALLED"), (3010, "INSTALLED_REBOOT_REQUIRED"), (1641, "UNEXPECTED_REBOOT_INITIATED")],
)
def test_installer_completion_after_interruption_is_classified_without_retry(
    boundaries, returncode, status
):
    authorization = authorize(boundaries)
    boundaries.run.side_effect = executor.InstallerWaitInterrupted(returncode)
    report = execute(boundaries, authorization)
    assert_safe_report(report)
    assert report["status"] == status
    assert report["operator_wait_interrupted"] is True
    assert report["installer_invoked"] is True
    assert report["installer_return_code"] == returncode
    assert report["checks"][-1] == {"name": "post_process_integrity", "status": "PASS"}
    boundaries.run.assert_called_once()
    assert boundaries.reports[-1]["status"] == status


@pytest.mark.parametrize("after_launch", [False, True])
def test_custody_handle_release_error_remains_a_readable_failure(boundaries, after_launch):
    authorization = authorize(boundaries) if after_launch else None
    boundaries.close_error = OSError("TEST_ONLY_SENSITIVE")
    report = execute(boundaries, authorization)
    assert_safe_report(report)
    assert report["status"] == "HANDLE_RELEASE_FAILED"
    assert report["overall_status"] == "FAIL"
    assert "TEST_ONLY_SENSITIVE" not in str(report)
    if after_launch:
        boundaries.run.assert_called_once()
    else:
        assert_no_install(boundaries, report)


@pytest.fixture
def native_console(monkeypatch):
    kernel = SimpleNamespace(
        GetStdHandle=Mock(side_effect=[10, 11]),
        GetConsoleMode=Mock(return_value=1),
        WriteConsoleW=Mock(return_value=1),
        ReadConsoleW=Mock(return_value=1),
    )
    loader = Mock(return_value=kernel)
    monkeypatch.setattr(executor, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(executor.ctypes, "WinDLL", loader, raising=False)
    return SimpleNamespace(kernel=kernel, loader=loader)


@pytest.mark.parametrize(
    ("response", "accepted"),
    [
        ("INSTALL " + "d" * 64, True),
        ("INSTALL " + "f" * 64, False),
        ("approved", False),
        ("", False),
        (" INSTALL " + "d" * 64, False),
    ],
)
def test_native_console_requires_exact_digest_phrase(native_console, response, accepted):
    def read(_handle, target, _maximum, _count, _reserved):
        target.value = response + "\r\n"
        return 1

    native_console.kernel.ReadConsoleW.side_effect = read
    assert executor._confirm_plan("d" * 64) is accepted
    prompt = native_console.kernel.WriteConsoleW.call_args.args[1].value
    assert "INSTALL " + "d" * 64 in prompt
    native_console.loader.assert_called_once_with(
        "kernel32.dll", use_last_error=True, winmode=0x00000800
    )


@pytest.mark.parametrize("failure", ["GetConsoleMode", "WriteConsoleW", "ReadConsoleW"])
def test_piped_missing_or_failed_console_cannot_authorize(native_console, failure):
    getattr(native_console.kernel, failure).return_value = 0
    with pytest.raises(approval_module.OperatorAuthenticationUnavailable):
        executor._confirm_plan("d" * 64)


def test_no_native_windows_console_cannot_be_replaced_by_stdin_or_environment(monkeypatch):
    monkeypatch.setattr(executor, "os", SimpleNamespace(name="posix"))
    monkeypatch.setenv("CRYPTOHUNTER_STAGE9_INSTALL_APPROVED", "true")
    with pytest.raises(approval_module.OperatorAuthenticationUnavailable):
        executor._confirm_plan("d" * 64)
