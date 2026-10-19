"""Real Linux TEST log writes, flock, crash/restart and fail-closed recovery."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

from cryptography.hazmat.primitives import serialization
import pytest

from bot_core import local_signing_lifecycle_evidence as lifecycle
from bot_core import test_only_cca_decision_log as log_module
from bot_core.independent_cca_decision import parse_independent_cca_decision
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import ProviderRole
from tests.security._cca_decision_fixtures import build_cca_case, sign_cca_payload
from tests.security._cca_log_process import start_log_process, start_rotation
from tests.security._custody_source_demo import _before, _operation

pytestmark = pytest.mark.integration


def _case(tmp_path, operation="test-operation", target="VERIFY_ONLY"):
    record = _operation(
        tmp_path,
        _before(ProviderRole.ROOT_PROOF_SIGNING.value),
        {"operation_id": operation, "target_state": target},
    )
    return build_cca_case(record.descriptor, lifecycle._document(replace(record, state="COMPLETE")))


@pytest.fixture
def retained(tmp_path):
    log, floor, leases = (tmp_path / name for name in ("log", "external-test-floor", "leases"))
    for directory in (log, floor, leases):
        directory.mkdir(mode=0o700)
    for name, raw in (
        ("source-lease", b""),
        ("credential-lease", b""),
        ("credential-state", b"ACTIVE:1:1"),
    ):
        path = leases / name
        path.write_bytes(raw)
        path.chmod(0o600)
    case = _case(tmp_path)
    pin = log_module.provision_test_only_cca_log(
        log,
        floor,
        installation_id=case.payload["installation"]["installation_id"],
        namespace=case.payload["completion_authority"]["provider_namespace"],
    )
    return pin, case, leases


def _constructor(case):
    def create(position):
        payload = deepcopy(case.payload)
        payload["decision_id"] = position.entry_id
        payload["decision_log_position"] = position.document
        return sign_cca_payload(case, payload)

    return create


def _append(pin, case, **kwargs):
    return log_module.TestOnlyCCADecisionLog(pin).decide(
        case.payload["close"]["operation_id"],
        case.immutable_binding,
        create_decision=_constructor(case),
        **kwargs,
    )


def _forbidden(_position):
    pytest.fail("exact retry/recovery must not construct or sign a new decision")


def _lookup(pin, case):
    return log_module.TestOnlyCCADecisionLog(pin).exact_lookup(
        case.payload["close"]["operation_id"], case.immutable_binding
    )


def _fixture(case, leases):
    return {
        "operation_id": case.payload["close"]["operation_id"],
        "binding": case.immutable_binding,
        "payload": case.payload,
        "test_key": case.private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        ),
        "lease_directory": str(leases),
    }


def _receive(process, connection):
    try:
        assert connection.poll(20), "TEST CCA process did not respond"
        response = connection.recv()
        process.join(15)
        assert not process.is_alive() and process.exitcode == 0
        return response
    finally:
        connection.close()
        if process.is_alive():
            process.kill()
            process.join(5)


def test_original_retry_restart_sequence_and_prior_original_digest(retained, tmp_path):
    pin, case, _ = retained
    original = _append(pin, case)
    path = Path(pin.log_directory) / "00000000000000000001.decision"
    before = path.stat()
    assert original == path.read_bytes() == _lookup(pin, case)
    assert (
        log_module.TestOnlyCCADecisionLog(pin).decide(
            "test-operation", case.immutable_binding, create_decision=_forbidden
        )
        == original
    )
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    other = _case(tmp_path, "other-operation")
    appended = _append(pin, other)
    position = parse_independent_cca_decision(appended).payload["decision_log_position"]
    assert position["sequence"] == 2
    assert position["previous_entry_sha256"] == hashlib.sha256(original).hexdigest()
    assert log_module.TestOnlyCCADecisionLog.status == "NOT_QUALIFIED"
    assert log_module.TestOnlyCCADecisionLog.production_emission == "BLOCKED"


def test_same_key_different_descriptor_is_durable_conflict(retained, tmp_path):
    pin, case, _ = retained
    original = _append(pin, case)
    conflict = _case(tmp_path, target="REVOKED")
    for _ in range(2):
        with pytest.raises(log_module.TestOnlyCCALogConflict):
            _append(pin, conflict)
    conflicts = list(Path(pin.log_directory).glob("conflict.*"))
    assert len(conflicts) == 1
    before = conflicts[0].read_bytes()
    assert (
        json.loads(before)["accepted_binding_sha256"]
        == hashlib.sha256(case.immutable_binding).hexdigest()
    )
    assert _lookup(pin, case) == original
    assert conflicts[0].read_bytes() == before


@pytest.mark.parametrize(
    "phase",
    [
        "before_intent",
        "after_intent_write",
        "after_intent_publication",
        "after_floor_reservation",
        "after_decision_write",
        "after_decision_publication",
        "after_durable_decision",
    ],
)
def test_real_crash_and_restart_exact_recovery_never_resigns(retained, phase):
    pin, case, leases = retained
    process, connection = start_log_process(pin, _fixture(case, leases), crash_at=phase)
    process.join(20)
    if process.is_alive():
        process.kill()
        process.join(5)
    connection.close()
    assert process.exitcode == 73
    recoverable = phase in {
        "after_decision_write",
        "after_decision_publication",
        "after_durable_decision",
    }
    if phase == "before_intent":
        assert list(Path(pin.log_directory).iterdir())
        response = _receive(*start_log_process(pin, _fixture(case, leases)))
        assert response[0] == "ok"
        return
    response = _receive(*start_log_process(pin, _fixture(case, leases), forbid_signing=True))
    if recoverable:
        assert response[0] == "ok"
        assert response[1] == _lookup(pin, case)
    else:
        assert response[0] == "refused" and response[1] == "TestOnlyCCALogIndeterminate"
        assert not list(Path(pin.log_directory).glob("*.decision"))


@pytest.mark.parametrize("phase", ["after_intent_publication", "after_floor_reservation"])
def test_pending_intent_also_retains_conflicting_binding(retained, tmp_path, phase):
    pin, case, leases = retained
    process, connection = start_log_process(pin, _fixture(case, leases), crash_at=phase)
    process.join(20)
    connection.close()
    assert process.exitcode == 73
    conflict = _case(tmp_path, target="REVOKED")
    with pytest.raises(log_module.TestOnlyCCALogConflict):
        _append(pin, conflict)
    assert len(list(Path(pin.log_directory).glob("conflict.*"))) == 1
    with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
        _lookup(pin, case)
    with pytest.raises(log_module.TestOnlyCCALogConflict):
        _lookup(pin, conflict)


@pytest.mark.parametrize("conflicting", [False, True])
def test_separate_process_competitors_serialize_same_and_conflicting_bindings(
    retained, tmp_path, conflicting
):
    pin, case, leases = retained
    first, first_pipe = start_log_process(
        pin, _fixture(case, leases), pause_at="after_floor_reservation"
    )
    second = None
    try:
        assert first_pipe.poll(20) and first_pipe.recv() == ("barrier", "after_floor_reservation")
        competitor = _case(tmp_path, target="REVOKED") if conflicting else case
        # Distinct upstream leases ensure this contention reaches the CCA log
        # flock, rather than blocking at the source/lifecycle fixture locks.
        competitor_leases = tmp_path / "competitor-leases"
        shutil.copytree(leases, competitor_leases)
        second, second_pipe = start_log_process(
            pin, _fixture(competitor, competitor_leases), announce_entry=True
        )
        assert second_pipe.poll(20) and second_pipe.recv() == ("ready", "before_log")
        assert not second_pipe.poll(0.2)
        first_pipe.send("resume")
        original = _receive(first, first_pipe)
        result = _receive(second, second_pipe)
        assert original[0] == "ok"
        if conflicting:
            assert result[0:2] == ("refused", "TestOnlyCCALogConflict")
        else:
            assert result == original
        assert len(list(Path(pin.log_directory).glob("*.decision"))) == 1
    finally:
        for process in (first, second):
            if process is not None and process.is_alive():
                process.kill()
                process.join(5)


@pytest.mark.parametrize("state", [b"VERIFY_ONLY:2:1", b"REVOKED:2:1"])
def test_real_lifecycle_lease_serializes_rotation_until_decision_retained(retained, state):
    pin, case, leases = retained
    process, connection = start_log_process(
        pin, _fixture(case, leases), pause_at="after_floor_reservation"
    )
    assert connection.poll(20) and connection.recv() == ("barrier", "after_floor_reservation")
    rotation, rotation_pipe = start_rotation(str(leases), state)
    try:
        assert rotation_pipe.poll(20) and rotation_pipe.recv() == (
            "ready",
            "before_credential_lock",
        )
        assert not rotation_pipe.poll(0.2)
        assert (leases / "credential-state").read_bytes() == b"ACTIVE:1:1"
        connection.send("resume")
        assert _receive(process, connection)[0] == "ok"
        assert _receive(rotation, rotation_pipe) == "rotated"
        response = _receive(*start_log_process(pin, _fixture(case, leases), forbid_signing=True))
        assert response[0] == "refused" and "rotated/revoked" in response[2]
        # Structural exact lookup is not historical credential eligibility.
        assert _lookup(pin, case)
    finally:
        for child in (process, rotation):
            if child.is_alive():
                child.kill()
                child.join(5)


@pytest.mark.parametrize(
    "attack",
    [
        "symlink",
        "hardlink",
        "mode",
        "fifo",
        "log_directory",
        "floor_directory",
        "floor_missing",
        "log_missing",
        "decision_missing",
        "floor_rollback",
        "floor_ahead",
        "entry_replay",
        "previous_digest",
    ],
)
def test_missing_rollback_replay_and_unsafe_files_fail_closed(retained, tmp_path, attack):
    pin, case, _ = retained
    floor = Path(pin.floor_directory) / "floor"
    genesis = floor.read_bytes()
    original = _append(pin, case)
    decision = Path(pin.log_directory) / "00000000000000000001.decision"
    if attack in {"symlink", "hardlink", "fifo"}:
        saved = tmp_path / "saved"
        decision.rename(saved)
        if attack == "symlink":
            decision.symlink_to(saved)
        elif attack == "hardlink":
            os.link(saved, decision)
        else:
            os.mkfifo(decision, 0o400)
    elif attack == "mode":
        decision.chmod(0o440)
    elif attack in {"log_directory", "floor_directory"}:
        directory = Path(getattr(pin, attack))
        saved = tmp_path / "saved-directory"
        directory.rename(saved)
        shutil.copytree(saved, directory)
    elif attack in {"floor_missing", "log_missing", "decision_missing"}:
        target = {
            "floor_missing": floor,
            "log_missing": Path(pin.log_directory) / "manifest",
            "decision_missing": decision,
        }[attack]
        target.unlink()
    elif attack in {"floor_rollback", "floor_ahead"}:
        floor.write_bytes(
            genesis
            if attack == "floor_rollback"
            else canonical_json_bytes({**json.loads(floor.read_bytes()), "sequence": 2})
        )
    elif attack == "entry_replay":
        _append(pin, _case(tmp_path, "other-operation"))
        target = decision.with_name("00000000000000000002.decision")
        target.chmod(0o600)
        target.write_bytes(original)
        target.chmod(0o400)
    else:
        intent = decision.with_suffix(".intent")
        value = json.loads(intent.read_bytes())
        value["position"]["previous_entry_sha256"] = "a" * 64
        intent.chmod(0o600)
        intent.write_bytes(canonical_json_bytes(value))
        intent.chmod(0o400)
    with pytest.raises(log_module.TestOnlyCCALogError):
        _lookup(pin, case)


@pytest.mark.parametrize("target", ["decision", "lock", "directory", "stage"])
def test_live_same_byte_inode_and_directory_swaps_are_detected(retained, target):
    pin, case, _ = retained
    original = _append(pin, case) if target == "decision" else None
    root = Path(pin.log_directory)

    def attack(phase):
        expected = "after_decision_write" if target == "stage" else "before_intent"
        if phase != expected:
            return
        if target == "directory":
            root.rename(root.with_name("old-log"))
            shutil.copytree(root.with_name("old-log"), root)
            return
        path = (
            root
            / {
                "decision": "00000000000000000001.decision",
                "lock": "lock",
                "stage": "00000000000000000001.decision.stage",
            }[target]
        )
        raw, mode = path.read_bytes(), path.stat().st_mode & 0o777
        path.rename(path.with_name("replacement-old"))
        path.write_bytes(raw)
        path.chmod(mode)

    case_to_append = _case(root.parent, "another-operation") if original else case
    with pytest.raises(log_module.TestOnlyCCALogError):
        _append(pin, case_to_append, fault_injector=attack)


@pytest.mark.parametrize("target", ["log", "floor", "decision", "ancestor"])
def test_real_linux_acl_is_rejected(retained, target):
    pin, case, _ = retained
    _append(pin, case)
    root = Path(pin.log_directory)
    path = {
        "log": root,
        "floor": Path(pin.floor_directory),
        "decision": root / "00000000000000000001.decision",
        "ancestor": root.parent,
    }[target]
    # Native Linux POSIX ACL xattr with an additional named user and a mask.
    permissions = 7 if path.is_dir() else 4
    entries = [
        (1, permissions, 0xFFFFFFFF),
        # The current UID is mapped even in one-UID user namespaces. A named
        # entry still creates a real extended ACL and is forbidden by the log.
        (2, permissions, os.geteuid()),
        (4, 0, 0xFFFFFFFF),
        (16, 0, 0xFFFFFFFF),
        (32, 0, 0xFFFFFFFF),
    ]
    raw = struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)
    original_mode = path.stat().st_mode
    os.setxattr(path, "system.posix_acl_access", raw)
    try:
        assert path.stat().st_mode == original_mode
        with pytest.raises(log_module.TestOnlyCCALogError, match="ACL"):
            _lookup(pin, case)
    finally:
        os.removexattr(path, "system.posix_acl_access")


def test_fsync_failure_never_acknowledges_and_restart_resyncs_floor(retained, monkeypatch):
    pin, case, _ = retained
    original_fsync, original_replace = os.fsync, os.replace
    published = []

    def replace_floor(*args, **kwargs):
        original_replace(*args, **kwargs)
        published.append(True)

    def fail_final_directory(descriptor):
        if len(published) == 2:
            raise OSError("injected final floor directory fsync failure")
        original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace_floor)
        patch.setattr(os, "fsync", fail_final_directory)
        with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
            _append(pin, case)
    calls = []

    def resync(descriptor):
        calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", resync)
    assert _lookup(pin, case) and len(calls) >= 2
    monkeypatch.setattr(
        os, "fsync", lambda descriptor: (_ for _ in ()).throw(OSError("still unavailable"))
    )
    with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
        _lookup(pin, case)


def test_atomic_publication_never_overwrites_and_rename_failure_preserves_stage(
    retained, monkeypatch
):
    pin, case, _ = retained
    original_publish = log_module._Directory.publish

    def fail_decision(directory, stage, target):
        if target.endswith(".decision"):
            raise OSError("injected rename failure")
        return original_publish(directory, stage, target)

    with monkeypatch.context() as patch:
        patch.setattr(log_module._Directory, "publish", fail_decision)
        with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
            _append(pin, case)
    staged = Path(pin.log_directory) / "00000000000000000001.decision.stage"
    raw = staged.read_bytes()
    assert _lookup(pin, case) == raw and not staged.exists()
    directory = log_module._Directory(Path(pin.log_directory))
    try:
        directory.stage("duplicate.stage", b"replacement")
        with pytest.raises(FileExistsError):
            directory.publish("duplicate.stage", "00000000000000000001.decision")
        assert (Path(pin.log_directory) / "00000000000000000001.decision").read_bytes() == raw
    finally:
        directory.close()


def test_absence_and_missing_floor_cannot_provision_or_sign(retained):
    pin, case, _ = retained
    with pytest.raises(log_module.TestOnlyCCALogIndeterminate, match="NOT_FOUND"):
        _lookup(pin, case)
    (Path(pin.floor_directory) / "floor").unlink()
    with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
        log_module.TestOnlyCCADecisionLog(pin).decide(
            "test-operation", case.immutable_binding, create_decision=_forbidden
        )
    assert not list(Path(pin.log_directory).glob("*.intent"))


@pytest.mark.parametrize("attack", ["bytes", "filename", "accepted_binding"])
def test_retained_conflict_records_have_closed_schema_and_original_digest(
    retained, tmp_path, attack
):
    pin, case, _ = retained
    _append(pin, case)
    with pytest.raises(log_module.TestOnlyCCALogConflict):
        _append(pin, _case(tmp_path, target="REVOKED"))
    path = next(Path(pin.log_directory).glob("conflict.*"))
    if attack == "filename":
        path.rename(path.with_name("conflict." + "0" * 64))
    else:
        value = (
            {}
            if attack == "bytes"
            else {**json.loads(path.read_bytes()), "accepted_binding_sha256": "0" * 64}
        )
        raw = canonical_json_bytes(value)
        path.chmod(0o600)
        path.write_bytes(raw)
        path.chmod(0o400)
        if attack == "accepted_binding":
            path.rename(path.with_name("conflict." + hashlib.sha256(raw).hexdigest()))
    with pytest.raises(log_module.TestOnlyCCALogError):
        _lookup(pin, case)


def test_lock_second_open_is_bound_to_validated_inode(retained, monkeypatch):
    pin, case, _ = retained
    original_open = os.open
    calls = []

    def replace_before_lock_open(name, flags, *args, **kwargs):
        if name == "lock":
            calls.append(True)
            if len(calls) == 2:
                path = Path(pin.log_directory) / "lock"
                path.rename(path.with_name("old-lock"))
                path.write_bytes(b"")
                path.chmod(0o600)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_lock_open)
    with pytest.raises(log_module.TestOnlyCCALogError, match="LOCK_REPLACED"):
        _append(pin, case)
    assert not list(Path(pin.log_directory).glob("*.intent"))


@pytest.mark.parametrize("fault", ["write", "file_fsync", "floor_replace"])
def test_real_io_failure_never_acknowledges_or_reinvokes_callback(retained, monkeypatch, fault):
    pin, case, _ = retained
    calls = []

    def forbidden_io(*args, **kwargs):
        raise OSError("injected TEST I/O failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            os,
            {"write": "write", "file_fsync": "fsync", "floor_replace": "replace"}[fault],
            forbidden_io,
        )
        with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
            log_module.TestOnlyCCADecisionLog(pin).decide(
                "test-operation",
                case.immutable_binding,
                create_decision=lambda position: calls.append(position),
            )
    assert calls == []
    with pytest.raises(log_module.TestOnlyCCALogError):
        _lookup(pin, case)


def test_external_floor_detects_restoring_old_log_snapshot(retained, tmp_path):
    pin, case, _ = retained
    original = _append(pin, case)
    root = Path(pin.log_directory)
    snapshot = tmp_path / "old-log-snapshot"
    shutil.copytree(root, snapshot)
    _append(pin, _case(tmp_path, "next-operation"))
    # Restore contents within the same pinned directory inode; the separate
    # TEST floor remains at sequence two, and must detect the restored history.
    for path in root.iterdir():
        path.unlink()
    for path in snapshot.iterdir():
        shutil.copy2(path, root / path.name)
    assert (root / "00000000000000000001.decision").read_bytes() == original
    with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
        _lookup(pin, case)


def test_decision_file_fsync_failure_can_only_recover_original_bytes(retained, monkeypatch):
    pin, case, _ = retained
    original_fsync = os.fsync
    armed = []

    def arm(phase):
        if phase == "after_floor_reservation":
            armed.append(True)

    def fail_once(descriptor):
        if armed:
            armed.clear()
            raise OSError("injected decision file fsync failure")
        original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_once)
        with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
            _append(pin, case, fault_injector=arm)
    staged = Path(pin.log_directory) / "00000000000000000001.decision.stage"
    original = staged.read_bytes()
    assert (
        log_module.TestOnlyCCADecisionLog(pin).decide(
            "test-operation", case.immutable_binding, create_decision=_forbidden
        )
        == original
    )


def test_partial_original_decision_stage_stays_unknown_without_resigning(retained, monkeypatch):
    pin, case, _ = retained
    original_write = os.write
    armed = []

    def arm(phase):
        if phase == "after_floor_reservation":
            armed.append(0)

    def partial_write(descriptor, raw):
        if armed:
            if armed[0] == 0:
                armed[0] = 1
                return original_write(descriptor, raw[:10])
            raise OSError("injected partial decision write")
        return original_write(descriptor, raw)

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", partial_write)
        with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
            _append(pin, case, fault_injector=arm)
    with pytest.raises(log_module.TestOnlyCCALogError):
        log_module.TestOnlyCCADecisionLog(pin).decide(
            "test-operation", case.immutable_binding, create_decision=_forbidden
        )
    assert not list(Path(pin.log_directory).glob("*.decision"))


@pytest.mark.parametrize(
    "kind", ["wrong_installation", "wrong_namespace", "noncanonical", "not_bytes"]
)
def test_malformed_bindings_never_publish_intent(retained, kind):
    pin, case, _ = retained
    value = json.loads(case.immutable_binding)
    if kind == "wrong_installation":
        value["installation"]["installation_id"] = "other-installation"
    elif kind == "wrong_namespace":
        value["completion_authority"]["provider_namespace"] = "other-namespace"
    raw = canonical_json_bytes(value)
    if kind == "noncanonical":
        raw = json.dumps(value).encode()
    elif kind == "not_bytes":
        raw = None
    with pytest.raises((log_module.TestOnlyCCALogError, ValueError)):
        log_module.TestOnlyCCADecisionLog(pin).decide(
            "test-operation", raw, create_decision=_forbidden
        )
    assert not list(Path(pin.log_directory).glob("*.intent"))
