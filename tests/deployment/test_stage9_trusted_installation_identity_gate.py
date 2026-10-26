"""Synthetic gate success proves comparison behavior, never host isolation."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json

import pytest

from deployment import stage9_trusted_installation_contract as contract
from deployment import stage9_trusted_installation_identity_gate as gate


_ROLES = ("root_source", "history_source", "cca")
_PRIVATE = "/private/custody/DO_NOT_REPORT"


def _manifest() -> dict:
    roles = []
    for index, role in enumerate(_ROLES, 1):
        unit = f"synthetic-{role.replace('_', '-')}.service"
        roles.append(
            {
                "role": role,
                "installation_identity": f"synthetic-{role}",
                "unit": unit,
                "principal": f"synthetic_{role}",
                "uid": 2000 + index,
                "gid": 3000 + index,
                "supplementary_groups": [],
                "unit_file": {
                    "path": f"/etc/systemd/system/{unit}",
                    "sha256": "ab" * 32,
                    "uid": 0,
                    "gid": 0,
                    "mode": 0o644,
                },
                "executable": {
                    "path": f"/opt/synthetic/{role}",
                    "sha256": "cd" * 32,
                    "uid": 0,
                    "gid": 0,
                    "mode": 0o755,
                },
                "control_group": f"/system.slice/{unit}",
            }
        )
    return {
        "schema": contract.SCHEMA,
        "version": contract.VERSION,
        "installation_id": "synthetic-test-installation",
        "roles": roles,
    }


def _raw(document: dict) -> bytes:
    return json.dumps(document, allow_nan=False).encode("utf-8")


def _pin(raw: bytes) -> dict:
    return {
        "schema": "CryptoHunter.SourceCCAInstallationApprovalPinV1",
        "version": 1,
        "installation_id": "synthetic-test-installation",
        "approval_id": "independent-synthetic-review",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


class _SyntheticFile(AbstractContextManager):
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.fd = 17
        self.identity = (1, int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big"))
        self.sha256 = ("ab" if path.endswith(".service") else "cd") * 32
        self.rechecks = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def recheck(self):
        self.rechecks += 1


def _service(role, _descriptor):
    index = _ROLES.index(role.role) + 1
    return {
        "pid": 5000 + index,
        "start_time_ticks": 6000 + index,
        "uid": role.uid,
        "gid": role.gid,
        "groups": [],
        "namespaces": {"mnt": 7000 + index, "ipc": 8000 + index, "net": 9000 + index},
    }


def _inspect(*, document=None, pin_document=None, pin_missing=False, **kwargs):
    raw = _raw(_manifest() if document is None else document)
    pin_raw = None if pin_missing else _raw(_pin(raw) if pin_document is None else pin_document)
    return gate._inspect(
        raw,
        pin_raw,
        file_factory=kwargs.get("file_factory", _SyntheticFile),
        service_observer=kwargs.get("service_observer", _service),
        synthetic=True,
    )


def _blocked(report):
    assert report["status"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["production_emission"] == "BLOCKED"
    assert report["blockers"]
    encoded = json.dumps(report, allow_nan=False)
    assert _PRIVATE not in encoded
    assert "/etc/systemd/" not in encoded
    assert "/opt/synthetic/" not in encoded
    assert "PRODUCTION_READY" not in encoded


def test_synthetic_manifest_and_protected_pin_match_cannot_qualify_production():
    report = _inspect()
    _blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert report["installation_checks"] == "MATCHED"
    assert report["trust"] == "PROTECTED_PIN_MATCHED_PROVENANCE_UNQUALIFIED"
    assert not report["observed_failures"]
    assert set(report["role_checks"]) == set(_ROLES)
    assert all(item["checks"] == "MATCHED" for item in report["role_checks"].values())


def test_missing_manifest_fails_closed_without_touching_installed_services():
    report = gate.inspect_installation(None)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert "MANIFEST_MISSING" in report["observed_failures"]


def test_absent_manifest_file_is_sanitized(tmp_path):
    report = gate.inspect_installation(str(tmp_path / "missing-private-manifest"))
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]
    assert "missing-private-manifest" not in json.dumps(report)


def test_unprotected_caller_manifest_cannot_self_authorize_without_pin():
    touched = []

    def forbidden_file(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("observation must not proceed without expected pin")

    report = _inspect(pin_missing=True, file_factory=forbidden_file)
    _blocked(report)
    assert report["trust"] == "NOT_QUALIFIED"
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert "INDEPENDENT_APPROVAL_PIN_MISSING" in report["observed_failures"]
    assert not touched


def test_manifest_and_approval_pin_cannot_share_one_input_path(tmp_path):
    path = str(tmp_path / "self-approved.json")
    report = gate.inspect_installation(path, approval_pin_path=path)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unexpected"),
        ("version", 2),
        ("version", True),
        ("installation_id", "other-installation"),
        ("approval_id", ""),
        ("approval_id", "../self-approved"),
        ("manifest_sha256", "ef" * 32),
        ("manifest_sha256", "AA" * 32),
        ("manifest_sha256", None),
    ],
)
def test_pin_requires_exact_reviewed_manifest_binding(field, value):
    raw = _raw(_manifest())
    pin = _pin(raw)
    pin[field] = value
    report = _inspect(pin_document=pin)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["trust"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


def test_pin_rejects_unknown_self_qualification_flag():
    raw = _raw(_manifest())
    pin = _pin(raw)
    pin["independently_qualified"] = True
    report = _inspect(pin_document=pin)
    _blocked(report)
    assert report["trust"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


def test_duplicate_pin_fields_are_not_independent_review():
    raw = _raw(_manifest())
    pin_raw = _raw(_pin(raw)).replace(b'"version": 1', b'"version": 1, "version": 1')
    report = gate._inspect(
        raw,
        pin_raw,
        file_factory=_SyntheticFile,
        service_observer=_service,
        synthetic=True,
    )
    _blocked(report)
    assert report["trust"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


def test_manifest_self_qualification_flag_is_rejected_before_observation():
    document = _manifest()
    document["independently_qualified"] = True
    report = _inspect(document=document)
    _blocked(report)
    assert "MANIFEST_FIELDS_INVALID" in report["observed_failures"]


def test_exact_manifest_bytes_are_pinned_not_equivalent_json():
    document = _manifest()
    raw = _raw(document)
    changed_raw = json.dumps(document, indent=2).encode("utf-8")
    assert contract.parse_manifest(changed_raw).installation_id == document["installation_id"]
    report = gate._inspect(
        changed_raw,
        _raw(_pin(raw)),
        file_factory=_SyntheticFile,
        service_observer=_service,
        synthetic=True,
    )
    _blocked(report)
    assert report["trust"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


@pytest.mark.parametrize("kind", ["unit", "executable"])
def test_substituted_file_digest_refuses_installation(kind):
    class ChangedFile(_SyntheticFile):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            if (kind == "unit") == path.endswith(".service"):
                self.sha256 = "ef" * 32

    report = _inspect(file_factory=ChangedFile)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert "INSTALLATION_DIGEST_MISMATCH" in report["observed_failures"]


def test_file_replacement_during_gate_rechecks_refuses_installation():
    class ChangedFile(_SyntheticFile):
        def recheck(self):
            raise OSError(_PRIVATE)

    report = _inspect(file_factory=ChangedFile)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


@pytest.mark.parametrize("error", [OSError(_PRIVATE), TimeoutError(_PRIVATE), ValueError(_PRIVATE)])
def test_backend_failures_and_timeouts_are_sanitized(error):
    def unavailable(_role, _descriptor):
        raise error

    report = _inspect(service_observer=unavailable)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


def test_shared_pid_is_not_three_independent_service_identities():
    def shared_process(role, descriptor):
        snapshot = _service(role, descriptor)
        snapshot["pid"] = 5001
        return snapshot

    report = _inspect(service_observer=shared_process)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert "SHARED_PROCESS_IDENTITY" in report["observed_failures"]


@pytest.mark.parametrize("field,value", [("uid", 1), ("gid", 1), ("groups", [9999])])
def test_wrong_effective_process_principal_or_groups_refuses_installation(field, value):
    def wrong_identity(role, descriptor):
        snapshot = _service(role, descriptor)
        snapshot[field] = value
        return snapshot

    report = _inspect(service_observer=wrong_identity)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"pid": 5001},
        {"pid": True, "start_time_ticks": 1},
        {"pid": 5001, "start_time_ticks": float("nan")},
        {"pid": 5001, "start_time_ticks": 1, "private_path": _PRIVATE},
    ],
)
def test_malformed_backend_records_cannot_be_reported_as_matched(snapshot):
    report = _inspect(service_observer=lambda _role, _descriptor: snapshot)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert "INVALID_SERVICE_OBSERVATION" in report["observed_failures"]


def test_installed_objects_remain_pinned_through_process_observation():
    opened = []

    class TrackedFile(_SyntheticFile):
        def __enter__(self):
            opened.append(self)
            return self

    def observing(role, descriptor):
        assert any(item.path == role.unit_file.path for item in opened)
        assert any(item.path == role.executable.path for item in opened)
        return _service(role, descriptor)

    report = _inspect(file_factory=TrackedFile, service_observer=observing)
    _blocked(report)
    assert report["installation_checks"] == "MATCHED"
    assert len(opened) == 6
    assert all(item.rechecks > 0 for item in opened)


def test_process_change_between_three_role_passes_is_refused():
    calls = {role: 0 for role in _ROLES}

    def changing_process(role, descriptor):
        calls[role.role] += 1
        snapshot = _service(role, descriptor)
        if calls[role.role] > 1:
            snapshot["start_time_ticks"] += 1
        return snapshot

    report = _inspect(service_observer=changing_process)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert "SERVICE_CHANGED_DURING_CHECK" in report["observed_failures"]


@pytest.mark.parametrize("namespace", ["mnt", "ipc", "net"])
def test_shared_cross_role_namespace_refuses_installation(namespace):
    def shared_namespace(role, descriptor):
        snapshot = _service(role, descriptor)
        snapshot["namespaces"][namespace] = 7001
        return snapshot

    report = _inspect(service_observer=shared_namespace)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


def test_distinct_paths_cannot_alias_one_installed_file_object():
    class AliasedFile(_SyntheticFile):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.identity = (1, 12345)

    report = _inspect(file_factory=AliasedFile)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--manifest"],
        ["--unknown", _PRIVATE],
        ["--approval-pin", _PRIVATE],
        ["--manifest", _PRIVATE, "--manifest", _PRIVATE],
    ],
)
def test_cli_invalid_arguments_emit_only_sanitized_refusal(arguments, capsys):
    assert gate.main(arguments) == 1
    report = json.loads(capsys.readouterr().out)
    _blocked(report)
    assert report["observed_failures"] == ["CLI_ARGUMENT_INVALID"]


def test_cli_matching_comparisons_keep_nonzero_unqualified_exit(monkeypatch, capsys):
    monkeypatch.setattr(gate, "inspect_installation", lambda *args, **kwargs: _inspect())
    assert gate.main(["--manifest", _PRIVATE, "--approval-pin", "/independent/pin.json"]) == 1
    report = json.loads(capsys.readouterr().out)
    _blocked(report)
    assert report["installation_checks"] == "MATCHED"


def test_public_loader_retains_independent_input_descriptors_and_never_qualifies(monkeypatch):
    raw = _raw(_manifest())
    pin_raw = _raw(_pin(raw))
    opened = []

    class ProtectedSyntheticFile(_SyntheticFile):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.data = raw if path == "/independent/manifest.json" else pin_raw

        def __enter__(self):
            opened.append(self)
            return self

    monkeypatch.setattr(gate, "PinnedFile", ProtectedSyntheticFile)
    monkeypatch.setattr(gate, "observe_service", _service)
    report = gate.inspect_installation(
        "/independent/manifest.json", approval_pin_path="/independent/approval.json"
    )
    _blocked(report)
    assert report["snapshot"] == "UNTRUSTED_LIVE_OBSERVATION"
    assert report["installation_checks"] == "MATCHED"
    assert report["trust"] == "PROTECTED_PIN_MATCHED_PROVENANCE_UNQUALIFIED"
    assert len(opened) == 8
    assert all(item.rechecks > 0 for item in opened)


@pytest.mark.parametrize("stage", ["recheck", "close"])
def test_protected_input_failure_after_matching_revokes_every_matched_role(monkeypatch, stage):
    raw = _raw(_manifest())
    pin_raw = _raw(_pin(raw))

    class ChangedInput(_SyntheticFile):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.data = raw if path == "/independent/manifest.json" else pin_raw

        def recheck(self):
            if stage == "recheck" and self.path == "/independent/approval.json":
                raise OSError(_PRIVATE)
            super().recheck()

        def __exit__(self, *args):
            if stage == "close" and self.path == "/independent/approval.json":
                raise OSError(_PRIVATE)
            return False

    monkeypatch.setattr(gate, "PinnedFile", ChangedInput)
    monkeypatch.setattr(gate, "observe_service", _service)
    report = gate.inspect_installation(
        "/independent/manifest.json", approval_pin_path="/independent/approval.json"
    )
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["trust"] == "NOT_QUALIFIED"
    assert all(item["checks"] == "NOT_QUALIFIED" for item in report["role_checks"].values())


def test_manifest_input_path_cannot_be_an_installed_executable(monkeypatch):
    raw = _raw(_manifest())
    manifest_path = _manifest()["roles"][0]["executable"]["path"]

    class SelfSuppliedFile(_SyntheticFile):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.data = raw if path == manifest_path else _raw(_pin(raw))

    monkeypatch.setattr(gate, "PinnedFile", SelfSuppliedFile)
    report = gate.inspect_installation(
        manifest_path, approval_pin_path="/independent/approval.json"
    )
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert "INPUT_PATHS_NOT_INDEPENDENT" in report["observed_failures"]


@pytest.mark.parametrize("raw", [b"", b"[]", b"NaN", b"\xff", b"x" * 4097])
def test_pin_parser_refuses_malformed_nonfinite_or_oversized_bytes(raw):
    report = gate._inspect(
        _raw(_manifest()),
        raw,
        file_factory=_SyntheticFile,
        service_observer=_service,
        synthetic=True,
    )
    _blocked(report)
    assert report["trust"] == "NOT_QUALIFIED"
    assert "APPROVAL_PIN_INVALID" in report["observed_failures"]


@pytest.mark.parametrize("source", ["installation", "file", "runtime"])
def test_mutated_known_refusal_codes_are_sanitized_again_at_report_boundary(source):
    refusal_type = {
        "installation": contract.InstallationRefusal,
        "file": gate.FileRefusal,
        "runtime": gate.RuntimeRefusal,
    }[source]
    refusal = refusal_type("BACKEND_OBSERVATION_FAILED")
    refusal.code = _PRIVATE

    def malicious_backend(*args, **kwargs):
        raise refusal

    report = _inspect(file_factory=malicious_backend)
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"]


def test_distinct_manifest_and_pin_paths_cannot_alias_one_protected_input(monkeypatch):
    service_calls = []

    class AliasedProtectedInput(_SyntheticFile):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.identity = (19, 23)
            self.data = _raw(_manifest())

    def forbidden_service(*args):
        service_calls.append(args)
        raise AssertionError("aliased approval inputs must be refused before observing services")

    monkeypatch.setattr(gate, "PinnedFile", AliasedProtectedInput)
    monkeypatch.setattr(gate, "observe_service", forbidden_service)
    report = gate.inspect_installation(
        "/independent/manifest.json", approval_pin_path="/independent/approval.json"
    )
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["trust"] == "NOT_QUALIFIED"
    assert report["observed_failures"] == ["INPUT_PATHS_NOT_INDEPENDENT"]
    assert not service_calls


@pytest.mark.parametrize(
    "reserved_identity", [(19, 23), (29, 31)], ids=["manifest", "approval-pin"]
)
def test_installed_executable_cannot_alias_a_reserved_protected_input(reserved_identity):
    raw = _raw(_manifest())
    service_calls = []

    class AliasedExecutable(_SyntheticFile):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            if path == "/opt/synthetic/root_source":
                self.identity = reserved_identity

    def forbidden_service(*args):
        service_calls.append(args)
        raise AssertionError("an executable alias of protected input must be refused first")

    report = gate._inspect(
        raw,
        _raw(_pin(raw)),
        file_factory=AliasedExecutable,
        service_observer=forbidden_service,
        synthetic=True,
        reserved_identities=((19, 23), (29, 31)),
    )
    _blocked(report)
    assert report["installation_checks"] == "NOT_QUALIFIED"
    assert report["observed_failures"] == ["INPUT_PATHS_NOT_INDEPENDENT"]
    assert not service_calls
