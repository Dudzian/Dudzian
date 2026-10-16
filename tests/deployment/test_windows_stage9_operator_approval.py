"""Hermetic approval data and native identity boundary tests; no host changes."""

from __future__ import annotations

import base64
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from deployment import windows_stage9_operator_approval as approval_module

NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
HOST = {
    "hostname": "TEST-ONLY-WINDOWS",
    "machine_guid": "11111111-1111-4111-8111-111111111111",
    "hardware_uuid": "22222222-2222-4222-8222-222222222222",
}
SID = "S-1-5-21-111-222-333-1001"
TRUSTED_POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def qualified_details() -> dict:
    return {
        "repository": "Dudzian/Dudzian",
        "source_revision": "a" * 40,
        "ci_run_id": "12345",
        "ci_run_attempt": 2,
        "artifact_id": "67890",
        "archive_sha256": "b" * 64,
        "msi_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "production_trust": {"run_id": "98765", "manifest_sha256": "e" * 64},
    }


def host_policy() -> dict:
    return {"host_identity": deepcopy(HOST), "operator_sids": [SID]}


def approval_document() -> dict:
    return {
        **qualified_details(),
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


def session_observation() -> dict:
    return {
        "operator_sid": SID,
        "host_identity": deepcopy(HOST),
        "authenticated": True,
        "elevated": True,
        "interactive": True,
        "session_id": 7,
        "impersonation": "None",
    }


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return NOW

    monkeypatch.setattr(approval_module, "datetime", FixedDatetime)


def validate(document=None, *, qualification=None, policy=None, now=NOW):
    return approval_module.validate_operator_approval(
        approval_document() if document is None else document,
        qualified_details() if qualification is None else qualification,
        policy=host_policy() if policy is None else policy,
        now=now,
    )


def test_valid_bound_document_returns_only_its_review_digest():
    document = approval_document()
    original = deepcopy(document)
    assert validate(document) == canonical_json_sha256(original)
    assert document == original
    assert "approved" not in document


@pytest.mark.parametrize("document", [None, [], {}, {"approved": True}])
def test_non_document_or_boolean_approval_is_rejected(document):
    with pytest.raises(approval_module.OperatorApprovalError):
        approval_module.validate_operator_approval(
            document, qualified_details(), policy=host_policy(), now=NOW
        )


@pytest.mark.parametrize("missing", sorted(approval_module.APPROVAL_KEYS))
def test_every_approval_field_is_required(missing):
    document = approval_document()
    del document[missing]
    with pytest.raises(approval_module.OperatorApprovalError, match="schema"):
        validate(document)


@pytest.mark.parametrize("extra", ["approved", "authenticated", "status", "run_stage10"])
def test_caller_supplied_authorization_flags_do_not_extend_the_schema(extra):
    document = approval_document()
    document[extra] = True
    with pytest.raises(approval_module.OperatorApprovalError, match="schema"):
        validate(document)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("approval_id", None),
        ("approval_id", "invalid"),
        ("approval_id", "00000000-0000-0000-0000-000000000000"),
        ("approval_id", "33333333333343338333333333333333"),
        ("repository", "untrusted/repository"),
        ("source_revision", "a" * 39),
        ("source_revision", "A" * 40),
        ("ci_run_id", 12345),
        ("ci_run_id", "0"),
        ("ci_run_id", "012345"),
        ("ci_run_attempt", True),
        ("ci_run_attempt", "2"),
        ("ci_run_attempt", 0),
        ("artifact_id", ""),
        ("artifact_id", "67890/other"),
        ("archive_sha256", "b" * 63),
        ("msi_sha256", "C" * 64),
        ("manifest_sha256", None),
        ("production_trust", {}),
        ("production_trust", True),
        ("operator_sid", "S-1-5-18"),
        ("operator_sid", "S-1-5-21-111-222-333-1002"),
        ("scope", ["prepare-clean-install"]),
        ("scope", ["transfer-qualified-msi", "prepare-clean-install", "install"]),
    ],
)
def test_malformed_or_expanded_approval_data_fails_closed(key, value):
    document = approval_document()
    document[key] = value
    with pytest.raises(approval_module.OperatorApprovalError):
        validate(document)


@pytest.mark.parametrize("key", approval_module.BINDING_KEYS)
def test_every_pin_must_equal_the_fresh_handoff(key):
    qualification = qualified_details()
    qualification[key] = "other-qualified-value"
    with pytest.raises(approval_module.OperatorApprovalError, match="freshly qualified"):
        validate(qualification=qualification)


@pytest.mark.parametrize("key", approval_module.BINDING_KEYS)
def test_missing_fresh_handoff_pin_cannot_be_replaced_by_approval(key):
    qualification = qualified_details()
    del qualification[key]
    with pytest.raises(approval_module.OperatorApprovalError, match="freshly qualified"):
        validate(qualification=qualification)


def test_rerun_attempt_and_production_trust_have_exact_binding():
    qualification = qualified_details()
    qualification["ci_run_attempt"] += 1
    with pytest.raises(approval_module.OperatorApprovalError):
        validate(qualification=qualification)
    qualification = qualified_details()
    qualification["production_trust"]["manifest_sha256"] = "f" * 64
    with pytest.raises(approval_module.OperatorApprovalError):
        validate(qualification=qualification)


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"approved": True},
        {"host_identity": HOST, "operator_sids": []},
        {"host_identity": HOST, "operator_sids": ["S-1-5-18"]},
        {"host_identity": HOST, "operator_sids": SID},
        {"host_identity": HOST, "operator_sids": ["S-1-5-21-111-222-333-1002"]},
        {"host_identity": HOST, "operator_sids": [SID], "approved": True},
    ],
)
def test_independent_host_policy_is_required(policy):
    with pytest.raises(approval_module.OperatorApprovalError):
        validate(policy=policy)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("hostname", "OTHER-HOST"),
        ("hostname", ""),
        ("hostname", " TEST-ONLY-WINDOWS"),
        ("machine_guid", "invalid"),
        ("machine_guid", "00000000-0000-0000-0000-000000000000"),
        ("hardware_uuid", "44444444-4444-4444-8444-444444444444"),
    ],
)
def test_approval_cannot_choose_a_host_outside_independent_inventory(key, value):
    document = approval_document()
    document["host_identity"][key] = value
    with pytest.raises(approval_module.OperatorApprovalError):
        validate(document)


@pytest.mark.parametrize("host", [None, {}, {**HOST, "approved": True}])
def test_host_identity_has_exact_stable_schema(host):
    document = approval_document()
    document["host_identity"] = host
    with pytest.raises(approval_module.OperatorApprovalError):
        validate(document)


@pytest.mark.parametrize(
    ("issued", "expires"),
    [
        ("2026-10-16T11:00:00Z", "2026-10-16T12:00:00Z"),
        ("2026-10-16T12:00:01Z", "2026-10-16T13:00:00Z"),
        ("2026-10-16T11:00:00Z", "2026-10-17T11:00:00.000001Z"),
        ("2026-10-16T11:00:00Z", "2026-10-16T11:00:00Z"),
        ("2026-10-16T11:00:00Z", "2026-10-16T10:00:00Z"),
        ("2026-10-16T11:00:00+00:00", "2026-10-16T13:00:00Z"),
        ("2026-10-16T11:00:00Z", "2026-10-16T13:00:00+01:00"),
        ("invalidZ", "2026-10-16T13:00:00Z"),
        (None, "2026-10-16T13:00:00Z"),
    ],
)
def test_utc_lifetime_and_expiry_are_fail_closed(issued, expires):
    document = approval_document()
    document.update(issued_at=issued, expires_at=expires)
    with pytest.raises(approval_module.OperatorApprovalError):
        validate(document)


def test_issue_boundary_and_exactly_twenty_four_hours_are_valid():
    document = approval_document()
    document.update(issued_at="2026-10-16T12:00:00Z", expires_at="2026-10-17T12:00:00Z")
    assert validate(document) == canonical_json_sha256(document)


@pytest.mark.parametrize(
    "now", [NOW.replace(tzinfo=None), NOW.astimezone(timezone(timedelta(hours=1)))]
)
def test_validation_requires_an_aware_utc_clock(now):
    with pytest.raises(approval_module.OperatorApprovalError, match="UTC aware"):
        validate(now=now)


def test_native_matching_session_is_still_not_installation_authorization(monkeypatch):
    document = approval_document()
    observed = session_observation()
    monkeypatch.setattr(approval_module, "_collect_operator_session", lambda: observed)
    result = approval_module.authenticate_operator(
        document, qualified_details(), policy=host_policy()
    )
    assert result["approval_sha256"] == canonical_json_sha256(document)
    assert result["host_identity"] == HOST
    assert result["operator_sid"] == SID
    assert result["installation_authorized"] is False
    assert result["host_identity"] is not observed["host_identity"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("operator_sid", "S-1-5-21-111-222-333-1002"),
        ("host_identity", {**HOST, "hostname": "OTHER-HOST"}),
        ("host_identity", None),
        ("authenticated", False),
        ("authenticated", 1),
        ("elevated", False),
        ("elevated", "true"),
        ("interactive", False),
        ("session_id", 0),
        ("session_id", -1),
        ("session_id", True),
        ("session_id", "7"),
        ("impersonation", "Impersonation"),
        ("impersonation", "Delegation"),
        ("impersonation", None),
    ],
)
def test_native_session_mismatch_or_unsafe_token_is_rejected(monkeypatch, key, value):
    observed = session_observation()
    observed[key] = value
    monkeypatch.setattr(approval_module, "_collect_operator_session", lambda: observed)
    with pytest.raises(approval_module.OperatorApprovalError):
        approval_module.authenticate_operator(
            approval_document(), qualified_details(), policy=host_policy()
        )


def test_invalid_approval_cannot_trigger_native_collection(monkeypatch):
    def unexpected():
        pytest.fail("invalid approval must fail before native reads")

    monkeypatch.setattr(approval_module, "_collect_operator_session", unexpected)
    document = approval_document()
    document["approved"] = True
    with pytest.raises(approval_module.OperatorApprovalError):
        approval_module.authenticate_operator(document, qualified_details(), policy=host_policy())


def test_approval_change_during_native_read_is_rejected(monkeypatch):
    document = approval_document()

    def collect():
        document["approval_id"] = "44444444-4444-4444-8444-444444444444"
        return session_observation()

    monkeypatch.setattr(approval_module, "_collect_operator_session", collect)
    with pytest.raises(approval_module.OperatorApprovalError, match="changed"):
        approval_module.authenticate_operator(document, qualified_details(), policy=host_policy())


def test_expiry_is_checked_again_after_native_collection(monkeypatch):
    class AdvancingDatetime(datetime):
        current = NOW

        @classmethod
        def now(cls, tz=None):
            return cls.current

    def collect():
        AdvancingDatetime.current = NOW + timedelta(hours=1)
        return session_observation()

    monkeypatch.setattr(approval_module, "datetime", AdvancingDatetime)
    monkeypatch.setattr(approval_module, "_collect_operator_session", collect)
    with pytest.raises(approval_module.OperatorApprovalError, match="expired"):
        approval_module.authenticate_operator(
            approval_document(), qualified_details(), policy=host_policy()
        )


def test_non_windows_native_authentication_is_unavailable_without_a_process(monkeypatch):
    monkeypatch.setattr(approval_module, "os", SimpleNamespace(name="posix"))

    def unexpected(*args, **kwargs):
        pytest.fail("non-Windows hosts must not run PowerShell")

    monkeypatch.setattr(approval_module, "_read_json_command", unexpected)
    with pytest.raises(approval_module.OperatorAuthenticationUnavailable):
        approval_module._collect_operator_session()


def test_native_collector_uses_fixed_script_and_no_approval_values(monkeypatch):
    calls = []
    monkeypatch.setattr(approval_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(approval_module, "native_powershell_path", lambda: TRUSTED_POWERSHELL)

    def read(command):
        calls.append(command)
        return session_observation()

    monkeypatch.setattr(approval_module, "_read_json_command", read)
    assert approval_module._collect_operator_session() == session_observation()
    assert len(calls) == 1
    command = calls[0]
    assert command[:4] == [
        TRUSTED_POWERSHELL,
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
    ]
    script = base64.b64decode(command[4]).decode("utf-16-le")
    assert "WindowsIdentity]::GetCurrent()" in script
    assert "UserInteractive" in script
    assert SID not in script
    assert HOST["hostname"] not in script


@pytest.mark.parametrize(
    "error",
    [
        OSError("test"),
        ValueError("test"),
        RuntimeError("test"),
        subprocess.TimeoutExpired("test", 1),
    ],
)
def test_native_read_errors_never_authenticate(monkeypatch, error):
    monkeypatch.setattr(approval_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(approval_module, "native_powershell_path", lambda: TRUSTED_POWERSHELL)

    def unavailable(_command):
        raise error

    monkeypatch.setattr(approval_module, "_read_json_command", unavailable)
    with pytest.raises(
        approval_module.OperatorAuthenticationUnavailable, match="observation failed"
    ):
        approval_module._collect_operator_session()


@pytest.mark.parametrize("observed", [None, [], "PASS", True])
def test_native_collector_rejects_non_identity_results(monkeypatch, observed):
    monkeypatch.setattr(approval_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(approval_module, "native_powershell_path", lambda: TRUSTED_POWERSHELL)
    monkeypatch.setattr(approval_module, "_read_json_command", lambda _command: observed)
    with pytest.raises(approval_module.OperatorAuthenticationUnavailable):
        approval_module._collect_operator_session()
