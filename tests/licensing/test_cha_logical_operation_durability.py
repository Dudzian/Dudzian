"""Hosted ABI and physical persistence checks, never Windows qualification.

The inherited #3071 fixture simulates NCrypt/TBS while retaining real package,
custody, ACTIVE and LPPI signature verification. No authority guard is bypassed.
"""

from __future__ import annotations

import ctypes
import os
import stat
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.lppi_authenticated_operation import authenticated_operation_signed_bytes
from bot_core.persistence import physical_durability
from deployment import (
    windows_production_cha_operation as operation,
    windows_production_lppi_operation as upstream_installed,
)
from tests.licensing import test_lppi_authenticated_operation as upstream_tests

challenge_harness = upstream_tests.challenge_harness
integration = upstream_tests.integration
package = upstream_tests.package
lppi = upstream_tests.lppi
active = upstream_tests.active
committed = upstream_tests.committed


@pytest.fixture
def reserved(committed, monkeypatch):
    monkeypatch.setattr(operation, "_utc_now", lambda: upstream_tests.RESERVATION_NOW)
    upstream_raw = upstream_installed._state_path().read_bytes()
    with operation._locked_state() as path:
        state = operation._reserve(path, upstream_raw)
    return SimpleNamespace(state=state, upstream_raw=upstream_raw, source=committed, path=path)


@pytest.fixture
def cha_committed(reserved):
    with operation._locked_state() as path:
        state = operation._commit(path, canonical_json_bytes(reserved.state))
    return SimpleNamespace(state=state, reservation=reserved, path=path)


def test_directory_creation_never_recreates_existing_ancestor(monkeypatch, tmp_path):
    target = tmp_path / "nested" / "state"
    original_mkdir = Path.mkdir
    attempted = []

    def guarded_mkdir(path, *args, **kwargs):
        if path.exists():
            attempted.append(path)
            raise PermissionError("existing ancestor must never be recreated")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    operation._create_directory(target)
    assert target.is_dir()
    assert attempted == []


def test_single_fixed_lifecycle_retains_complete_authenticated_upstream(reserved):
    state = reserved.state
    assert reserved.path == upstream_installed._state_path().with_name(
        "initial-cha-logical-operation.json"
    )
    assert set(state) == {
        "schema_version",
        "status",
        "history",
        "mapping_generation",
        "pdsa_trust_domain",
        "provisioning_operation_id",
        "logical_operation_id",
        "assigned_at_utc",
        "lppi_operation_state_raw_hex",
    }
    assert state["schema_version"] == "CHAInitialLogicalOperationV1"
    assert state["status"] == "AGO_RESERVED" and state["history"] == ["AGO_RESERVED"]
    assert state["mapping_generation"] == 1
    assert state["lppi_operation_state_raw_hex"] == reserved.upstream_raw.hex()
    assert state["provisioning_operation_id"] == reserved.source.value.provisioning_operation_id
    assert state["pdsa_trust_domain"] == reserved.source.value.binding.document["pdsa_trust_domain"]
    assert operation._read(reserved.path) == state
    changed_upstream = parse_canonical(reserved.upstream_raw)
    changed_upstream["signature_hex"] = upstream_tests._low_s(
        reserved.source.lppi.successor.private,
        authenticated_operation_signed_bytes(reserved.source.value.binding.canonical_bytes),
    ).hex()
    assert changed_upstream["signature_hex"] != reserved.source.value.signature.hex()
    before = reserved.path.read_bytes()
    with operation._locked_state() as path:
        with pytest.raises(operation.CHAOperationError, match="CONFLICT"):
            operation._reserve(path, canonical_json_bytes(changed_upstream))
    assert reserved.path.read_bytes() == before


def test_commit_advances_only_progression_and_keeps_exact_snapshot(cha_committed):
    state, reservation = cha_committed.state, cha_committed.reservation
    assert state["status"] == "PRVOP_AGO_BIJECTION_COMMITTED"
    assert state["history"] == ["AGO_RESERVED", "PRVOP_AGO_BIJECTION_COMMITTED"]
    for field in set(state) - {"status", "history"}:
        assert state[field] == reservation.state[field]
    assert operation._read(cha_committed.path) == state
    assert cha_committed.path.read_bytes() == canonical_json_bytes(state)
    assert operation._reserve(cha_committed.path, reservation.upstream_raw) == state


def test_retained_record_rejects_malformed_or_oversized_input(reserved):
    path = reserved.path
    canonical = canonical_json_bytes(reserved.state)
    path.unlink()
    assert operation._read(path) is None
    for raw in (
        b"{",
        canonical + b"\n",
        b'{"schema_version":"CHAInitialLogicalOperationV1",' + canonical[1:],
        b"[]",
        b"\xff",
        b" " * 4_194_305,
    ):
        path.write_bytes(raw)
        with pytest.raises(operation.CHAOperationError):
            operation._read(path)


def test_every_missing_record_field_is_rejected(reserved):
    for field in reserved.state:
        changed = dict(reserved.state)
        del changed[field]
        reserved.path.write_bytes(canonical_json_bytes(changed))
        with pytest.raises(operation.CHAOperationError):
            operation._read(reserved.path)


def test_unknown_record_fields_are_rejected(reserved):
    for field in ("account_id", "signature_hex", "caller_extension"):
        reserved.path.write_bytes(canonical_json_bytes(reserved.state | {field: "caller"}))
        with pytest.raises(operation.CHAOperationError):
            operation._read(reserved.path)


def test_record_identity_progression_and_snapshot_validation_fail_closed(reserved):
    for field, value in (
        ("schema_version", "CHAInitialLogicalOperationV2"),
        ("status", "ACCOUNT_GENESIS_COMMITTED"),
        ("status", "PRVOP_RESERVED"),
        ("history", []),
        ("history", ["AGO_RESERVED", "PRVOP_AGO_BIJECTION_COMMITTED"]),
        ("mapping_generation", True),
        ("mapping_generation", "1"),
        ("mapping_generation", 0),
        ("mapping_generation", 2),
        ("pdsa_trust_domain", "TEST_ONLY_2_OF_3_ED25519"),
        ("provisioning_operation_id", "prvop_caller"),
        ("logical_operation_id", "ago_caller"),
        ("assigned_at_utc", "2026-10-06T12:00:00Z"),
        ("assigned_at_utc", "2026-10-06T12:00:00.123+00:00"),
        ("assigned_at_utc", "2026-02-30T12:00:00.123Z"),
        ("lppi_operation_state_raw_hex", ""),
        ("lppi_operation_state_raw_hex", "7B7D"),
        ("lppi_operation_state_raw_hex", "7b7d"),
        ("lppi_operation_state_raw_hex", None),
    ):
        reserved.path.write_bytes(canonical_json_bytes(reserved.state | {field: value}))
        with pytest.raises(operation.CHAOperationError):
            operation._read(reserved.path)


def test_retained_state_open_failure_has_stable_domain_error(monkeypatch, tmp_path):
    refused = Mock(side_effect=PermissionError("state is inaccessible"))
    monkeypatch.setattr(
        operation,
        "os",
        SimpleNamespace(open=refused, O_RDONLY=os.O_RDONLY, O_NOFOLLOW=0),
    )
    with pytest.raises(operation.CHAOperationError):
        operation._read(tmp_path / "initial-cha-logical-operation.json")


def test_oversized_retained_file_fails_before_parsing(reserved, monkeypatch):
    reserved.path.write_bytes(b" " * 4_194_305)
    parser = Mock(side_effect=AssertionError("oversized state reached canonical parser"))
    monkeypatch.setattr(operation, "parse_canonical", parser)
    with pytest.raises(operation.CHAOperationError, match="INVALID_RETAINED"):
        operation._read(reserved.path)
    parser.assert_not_called()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_state_file_physical_boundary_rejects_alias_or_wrong_kind(reserved, tmp_path, kind):
    path = reserved.path
    if kind == "hardlink":
        os.link(path, tmp_path / "TEST_ONLY-retained-state-alias")
    elif kind == "symlink":
        target = tmp_path / "TEST_ONLY-retained-state-target"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        path.unlink()
        path.mkdir()
    with pytest.raises((ValueError, RuntimeError)):
        operation._read(path)


@pytest.mark.parametrize("kind", ["symlink", "file", "hardlinked_file"])
def test_state_directory_physical_boundary_rejects_alias_or_wrong_kind(monkeypatch, tmp_path, kind):
    directory = tmp_path / "TEST_ONLY-cha-state"
    if kind == "symlink":
        target = tmp_path / "TEST_ONLY-cha-real-state"
        target.mkdir()
        directory.symlink_to(target, target_is_directory=True)
    else:
        directory.write_bytes(b"not a directory")
        if kind == "hardlinked_file":
            os.link(directory, tmp_path / "TEST_ONLY-directory-file-alias")
    monkeypatch.setattr(
        operation, "_state_path", lambda: directory / "initial-cha-logical-operation.json"
    )
    with pytest.raises((ValueError, RuntimeError)):
        with operation._locked_state():
            pytest.fail("unsafe directory granted the operation lock")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_lock_file_physical_boundary_rejects_alias_or_wrong_kind(monkeypatch, tmp_path, kind):
    lock = tmp_path / "initial-cha-logical-operation.lock"
    if kind == "directory":
        lock.mkdir()
    else:
        target = tmp_path / "TEST_ONLY-lock-target"
        target.write_bytes(b"lock")
        if kind == "symlink":
            lock.symlink_to(target)
        else:
            os.link(target, lock)
    monkeypatch.setattr(
        operation, "_state_path", lambda: tmp_path / "initial-cha-logical-operation.json"
    )
    with pytest.raises((ValueError, RuntimeError)):
        with operation._locked_state():
            pytest.fail("unsafe file granted the operation lock")


@pytest.mark.parametrize("boundary", ["state", "lock"])
def test_hardlink_created_after_path_check_fails_descriptor_validation(
    reserved, monkeypatch, tmp_path, boundary
):
    target = reserved.path if boundary == "state" else reserved.path.with_suffix(".lock")
    original = os.open
    linked = []

    def open_then_link(path, flags, *arguments):
        descriptor = original(path, flags, *arguments)
        if Path(path) == target and not linked:
            os.link(target, tmp_path / "TEST_ONLY-racing-hardlink")
            linked.append(True)
        return descriptor

    monkeypatch.setattr(operation, "os", SimpleNamespace(**(vars(os) | {"open": open_then_link})))
    with pytest.raises(operation.CHAOperationError):
        if boundary == "state":
            operation._read(reserved.path)
        else:
            with operation._locked_state():
                pytest.fail("racing hardlink granted lock ownership")
    assert linked == [True]


@pytest.mark.parametrize("component", ["directory", "state", "lock"])
def test_hosted_windows_reparse_attributes_are_rejected(reserved, monkeypatch, component):
    """Simulate lstat attributes only; this is not a physical reparse-point claim."""
    target = {
        "directory": reserved.path.parent,
        "state": reserved.path,
        "lock": reserved.path.with_name("initial-cha-logical-operation.lock"),
    }[component]
    original = Path.lstat

    def reparse_lstat(path):
        information = original(path)
        if path == target:
            return SimpleNamespace(
                st_mode=information.st_mode,
                st_nlink=information.st_nlink,
                st_file_attributes=0x400,
            )
        return information

    monkeypatch.setattr(Path, "lstat", reparse_lstat)
    with pytest.raises((ValueError, RuntimeError)):
        if component == "state":
            operation._read(reserved.path)
        else:
            with operation._locked_state():
                pytest.fail("reparse point granted the operation lock")


@pytest.mark.skipif(os.name == "nt", reason="real POSIX nonblocking flock boundary")
def test_nonblocking_state_lock_reports_busy_and_releases_after_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(
        operation, "_state_path", lambda: tmp_path / "initial-cha-logical-operation.json"
    )
    with pytest.raises(RuntimeError, match="TEST_ONLY-body-failure"):
        with operation._locked_state():
            with pytest.raises(operation.CHAOperationError, match="BUSY"):
                with operation._locked_state():
                    pytest.fail("a second owner acquired the held state lock")
            raise RuntimeError("TEST_ONLY-body-failure")
    with operation._locked_state() as path:
        assert path == tmp_path / "initial-cha-logical-operation.json"


@pytest.mark.parametrize("busy", [True, False])
def test_hosted_windows_lock_uses_nonblocking_one_byte_lock_and_unlock(monkeypatch, tmp_path, busy):
    locking = Mock(side_effect=OSError("held") if busy else None)
    abi = SimpleNamespace(locking=locking, LK_NBLCK=2, LK_UNLCK=0)
    monkeypatch.setitem(sys.modules, "msvcrt", abi)
    monkeypatch.setattr(operation, "os", SimpleNamespace(**(vars(os) | {"name": "nt"})))
    monkeypatch.setattr(
        operation, "_state_path", lambda: tmp_path / "initial-cha-logical-operation.json"
    )
    if busy:
        with pytest.raises(operation.CHAOperationError, match="BUSY"):
            with operation._locked_state():
                pytest.fail("busy Windows lock entered its body")
        assert locking.call_count == 1
    else:
        with pytest.raises(RuntimeError, match="TEST_ONLY-body-failure"):
            with operation._locked_state():
                assert locking.call_count == 1
                raise RuntimeError("TEST_ONLY-body-failure")
        assert locking.call_count == 2
    first = locking.call_args_list[0].args
    assert first[1:] == (abi.LK_NBLCK, 1)
    assert (tmp_path / "initial-cha-logical-operation.lock").read_bytes() == b"\0"
    if not busy:
        assert locking.call_args_list[1].args == (first[0], abi.LK_UNLCK, 1)


@pytest.mark.skipif(os.name != "posix", reason="real POSIX physical publication failure boundaries")
@pytest.mark.parametrize("fence", ["temporary_file", "published_file", "directory", "rename"])
def test_physical_publication_failure_never_reports_success_and_keeps_safe_retry(
    reserved, monkeypatch, fence
):
    before = reserved.path.read_bytes()
    original_fsync = os.fsync
    original_replace = os.replace
    calls = []

    def fsync(descriptor):
        info = os.fstat(descriptor)
        kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
        calls.append(kind)
        if (
            (fence == "directory" and kind == "directory")
            or (fence == "temporary_file" and calls == ["file"])
            or (fence == "published_file" and calls == ["file", "file"])
        ):
            raise OSError("TEST_ONLY-durability-fence-failure")
        return original_fsync(descriptor)

    def replace(source, destination):
        if fence == "rename":
            raise OSError("TEST_ONLY-durability-fence-failure")
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(physical_durability.os, "fsync", fsync)
        patch.setattr(physical_durability.os, "replace", replace)
        with pytest.raises((operation.CHAOperationError, OSError)):
            operation._commit(reserved.path, before)
    retained = operation._read(reserved.path)
    if fence in {"temporary_file", "rename"}:
        assert retained == reserved.state and reserved.path.read_bytes() == before
    else:
        assert retained["status"] == "PRVOP_AGO_BIJECTION_COMMITTED"
    assert not list(reserved.path.parent.glob(f".{reserved.path.name}.*"))
    with operation._locked_state() as path:
        retried = operation._reserve(path, reserved.upstream_raw)
        committed = operation._commit(path, canonical_json_bytes(retried))
    assert committed["logical_operation_id"] == reserved.state["logical_operation_id"]
    assert committed["status"] == "PRVOP_AGO_BIJECTION_COMMITTED"


@pytest.mark.skipif(os.name != "posix", reason="real POSIX atomic publication boundary")
def test_commit_uses_one_atomic_same_directory_replacement(reserved, monkeypatch):
    before = reserved.path.read_bytes()
    replacements = []
    original = os.replace

    def replace(source, destination):
        assert Path(source).parent == reserved.path.parent
        assert Path(destination) == reserved.path
        assert reserved.path.read_bytes() == before
        assert operation._read(Path(source))["status"] == "PRVOP_AGO_BIJECTION_COMMITTED"
        replacements.append((Path(source), Path(destination)))
        return original(source, destination)

    monkeypatch.setattr(physical_durability.os, "replace", replace)
    operation._commit(reserved.path, before)
    assert len(replacements) == 1
    assert not replacements[0][0].exists()


@pytest.mark.skipif(os.name != "posix", reason="real POSIX physical publication failure boundaries")
def test_public_establish_fails_if_commit_file_flush_fails_then_repairs_exact_winner(
    reserved, monkeypatch
):
    original_fsync = os.fsync
    published_before_failure = []

    def fail_commit_flush(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            published_before_failure.append(reserved.path.read_bytes())
            raise OSError("TEST_ONLY-commit-file-flush-failure")
        return original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(physical_durability.os, "fsync", fail_commit_flush)
        with pytest.raises(operation.CHAOperationError, match="PERSISTENCE_FAILED"):
            operation.establish_installed_cha_logical_operation(reserved.source.value)
    assert published_before_failure == [canonical_json_bytes(reserved.state)]
    assert operation._read(reserved.path) == reserved.state
    established = operation.establish_installed_cha_logical_operation(reserved.source.value)
    assert established.logical_operation_id == reserved.state["logical_operation_id"]


@pytest.mark.skipif(os.name != "posix", reason="real POSIX restart physical publication fence")
def test_loader_reasserts_durability_before_issuing_retained_capability(cha_committed, monkeypatch):
    before = cha_committed.path.read_bytes()
    original = os.fsync
    file_flushes = []

    def fsync(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            file_flushes.append(descriptor)
            raise OSError("TEST_ONLY-restart-file-flush-failure")
        return original(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(physical_durability.os, "fsync", fsync)
        with pytest.raises(operation.CHAOperationError, match="PERSISTENCE_FAILED"):
            operation.load_installed_cha_logical_operation(cha_committed.reservation.source.value)
    assert len(file_flushes) == 1 and cha_committed.path.read_bytes() == before
    loaded = operation.load_installed_cha_logical_operation(cha_committed.reservation.source.value)
    assert loaded.logical_operation_id == cha_committed.state["logical_operation_id"]


def test_committed_state_rejects_mutation_including_valid_second_identity(cha_committed):
    before = cha_committed.path.read_bytes()
    for mutation in ("logical_operation_id", "rollback", "assigned_at_utc"):
        changed = dict(cha_committed.state)
        if mutation == "logical_operation_id":
            identity = uuid.UUID(changed[mutation].removeprefix("ago_"))
            changed[mutation] = "ago_" + str(uuid.UUID(int=identity.int ^ 1))
        elif mutation == "rollback":
            changed.update(status="AGO_RESERVED", history=["AGO_RESERVED"])
        else:
            changed[mutation] = "2026-10-06T12:00:00.124Z"
        with pytest.raises(operation.CHAOperationError):
            operation._write(cha_committed.path, changed)
        assert cha_committed.path.read_bytes() == before
    with pytest.raises(operation.CHAOperationError, match="CONFLICT"):
        operation._commit(cha_committed.path, canonical_json_bytes(cha_committed.reservation.state))
    assert cha_committed.path.read_bytes() == before


@pytest.mark.parametrize("success", [True, False])
def test_hosted_windows_publication_calls_native_write_through_abi(monkeypatch, tmp_path, success):
    """Exercise the shared durable publisher used by CHA; no native qualification."""
    source, destination = tmp_path / "reserved.tmp", tmp_path / "reserved.json"
    source.write_bytes(b"state")

    def native_move(source_name, destination_name, flags):
        if success:
            os.replace(source_name, destination_name)
        return int(success)

    move = Mock(side_effect=native_move)
    library = Mock(return_value=SimpleNamespace(MoveFileExW=move))
    native_error = Mock(return_value=OSError(5, "TEST_ONLY-Windows-denied"))
    monkeypatch.setattr(ctypes, "WinDLL", library, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", Mock(return_value=5), raising=False)
    monkeypatch.setattr(ctypes, "WinError", native_error, raising=False)
    if success:
        physical_durability.publish_file_atomically_durably(source, destination, _platform="nt")
        assert destination.read_bytes() == b"state" and not source.exists()
        native_error.assert_not_called()
    else:
        with pytest.raises(physical_durability.PhysicalDurabilityError):
            physical_durability.publish_file_atomically_durably(source, destination, _platform="nt")
        assert source.read_bytes() == b"state" and not destination.exists()
        native_error.assert_called_once_with(5)
    library.assert_called_once_with("kernel32", use_last_error=True)
    move.assert_called_once_with(str(source), str(destination), 0x1 | 0x8)
    assert move.argtypes == (
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.DWORD,
    )
    assert move.restype is ctypes.wintypes.BOOL


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent-directory durability fences")
def test_new_nested_state_directory_fences_every_created_ancestor(monkeypatch, tmp_path):
    directory = tmp_path / "TEST_ONLY-new-parent" / "new-child"
    monkeypatch.setattr(
        operation, "_state_path", lambda: directory / "initial-cha-logical-operation.json"
    )
    original = operation.flush_created_directory_metadata
    fenced = []

    def flush(path):
        fenced.append(Path(path))
        original(path)

    monkeypatch.setattr(operation, "flush_created_directory_metadata", flush)
    with operation._locked_state():
        assert tmp_path in fenced
        assert directory.parent in fenced
    assert directory.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent-directory durability fences")
def test_directory_creation_flush_failure_prevents_lock_ownership(monkeypatch, tmp_path):
    directory = tmp_path / "TEST_ONLY-new-state"
    monkeypatch.setattr(
        operation, "_state_path", lambda: directory / "initial-cha-logical-operation.json"
    )
    flush = Mock(side_effect=OSError("TEST_ONLY-directory-creation-flush-failure"))
    monkeypatch.setattr(operation, "flush_created_directory_metadata", flush)
    with pytest.raises((operation.CHAOperationError, OSError)):
        with operation._locked_state():
            pytest.fail("directory creation became authority without its durability fence")
    assert flush.called
    assert not (directory / "initial-cha-logical-operation.lock").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent-directory durability fences")
def test_retry_repairs_directory_creation_fence_after_failed_first_flush(monkeypatch, tmp_path):
    directory = tmp_path / "TEST_ONLY-new-parent" / "new-child"
    monkeypatch.setattr(
        operation, "_state_path", lambda: directory / "initial-cha-logical-operation.json"
    )
    original = operation.flush_created_directory_metadata
    fenced = []

    def fail_created_parent(path):
        if Path(path) == tmp_path:
            raise OSError("TEST_ONLY-created-parent-flush-failure")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(operation, "flush_created_directory_metadata", fail_created_parent)
        with pytest.raises(operation.CHAOperationError, match="PERSISTENCE_FAILED"):
            with operation._locked_state():
                pytest.fail("first failed parent fence granted state ownership")
    assert directory.parent.exists()

    def observe_repair(path):
        fenced.append(Path(path))
        return original(path)

    monkeypatch.setattr(operation, "flush_created_directory_metadata", observe_repair)
    with operation._locked_state():
        assert tmp_path in fenced
        assert directory.parent in fenced
