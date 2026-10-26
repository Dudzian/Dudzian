"""Unprivileged package boundaries; these are not physical qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from deployment import stage9_source_cca_test_package as package
from deployment.stage9_trusted_installation_contract import parse_manifest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-only descriptor, UID and native ELF package boundaries"
)

_RUN = "123456abcdef"
_COMMIT = "a" * 40


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def compiler(_source, target, role):
        target.write_bytes(b"\x7fELF_SYNTHETIC_" + role.encode())
        target.chmod(0o700)

    monkeypatch.setattr(package, "_compile", compiler)
    path = tmp_path / "package"
    result = package.build_package(path, _RUN, 62000, _COMMIT)
    return path, result


def _rehash(path):
    inventory = json.loads((path / "inventory.json").read_bytes())
    for name in inventory["files"]:
        inventory["files"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "inventory.json").write_bytes(package._json(inventory))


def test_plan_isolated_and_deterministic():
    plan = package.plan(_RUN)
    assert plan == package.plan(_RUN)
    assert set(plan) == {"run_id", "uid_base", "root", "roles"}
    assert plan["root"] == f"/run/ch-s9-test-{_RUN}"
    assert tuple(role["role"] for role in plan["roles"]) == package.ROLES
    for field in ("uid", "gid", "principal", "executable", "unit", "control_group"):
        assert len({role[field] for role in plan["roles"]}) == 3
    assert all(len(role["principal"]) < 32 for role in plan["roles"])


@pytest.mark.parametrize("value", ["", "../escape", "a" * 11, "a" * 13, "A" * 12, None, 123])
def test_bad_run_identity_refused(value):
    with pytest.raises(package.HarnessRefusal, match="RUN_ID_INVALID"):
        package.plan(value)


@pytest.mark.parametrize("value", [True, 0, 59999, 64998, 70000, "62000", None])
def test_bad_numeric_identity_refused(value):
    with pytest.raises(package.HarnessRefusal, match="UID_RANGE_INVALID"):
        package.plan(_RUN, value)


def test_prepare_retains_closed_test_only_snapshot(prepared):
    path, result = prepared
    assert result["commit"] == _COMMIT
    assert result["plan"] == package.plan(_RUN)
    assert result == package.validate_package(path, result["package_sha256"])
    manifest = parse_manifest(result["files"]["manifest.json"])
    assert manifest.installation_id.startswith("TEST_ONLY-SYNTHETIC-")
    pin = json.loads(result["files"]["TEST_ONLY-pin.json"])
    assert pin["manifest_sha256"] == manifest.sha256
    assert pin["approval_id"].startswith("TEST_ONLY-NOT-INDEPENDENT-APPROVAL-")
    assert path.stat().st_mode & 0o777 == 0o700
    assert b"PRODUCTION_NO_GO" in result["files"]["README.txt"]
    assert b"custody" not in result["files"]["root_source.service"].lower()


def test_package_accepts_lexically_normalized_relative_path(prepared, monkeypatch):
    path, expected = prepared
    subdirectory = path.parent / "current"
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)
    assert package.validate_package(Path("../package"), expected["package_sha256"]) == expected


def test_prepare_root_refused_before_creating(tmp_path, monkeypatch):
    monkeypatch.setattr(package.os, "geteuid", lambda: 0)
    with pytest.raises(package.HarnessRefusal, match="UNPRIVILEGED_PREPARATION_REQUIRED"):
        package.build_package(tmp_path / "not-created", _RUN, 62000, _COMMIT)
    assert not (tmp_path / "not-created").exists()


def test_prepare_existing_destination_not_taken_over(prepared):
    path, _result = prepared
    original = (path / "inventory.json").read_bytes()
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_DESTINATION_EXISTS"):
        package.build_package(path, _RUN, 62000, _COMMIT)
    assert (path / "inventory.json").read_bytes() == original


def test_prepare_invalid_commit_before_creating(tmp_path):
    path = tmp_path / "not-created"
    with pytest.raises(package.HarnessRefusal, match="COMMIT_INVALID"):
        package.build_package(path, _RUN, 62000, "main")
    assert not path.exists()


def test_prepare_compile_failure_retains_local_partial_package(tmp_path, monkeypatch):
    def failed(*_args):
        raise package.HarnessRefusal("OFFLINE_COMPILER_FAILED")

    monkeypatch.setattr(package, "_compile", failed)
    path = tmp_path / "partial"
    with pytest.raises(package.HarnessRefusal, match="OFFLINE_COMPILER_FAILED"):
        package.build_package(path, _RUN, 62000, _COMMIT)
    assert (path / "worker.c").is_file()
    assert not (path / "inventory.json").exists()


def test_wrong_inventory_digest_refused(prepared):
    path, _result = prepared
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_DIGEST_MISMATCH"):
        package.validate_package(path, "0" * 64)


@pytest.mark.parametrize("value", ["", "A" * 64, "ab", True, 2])
def test_invalid_digest_refused(prepared, value):
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_DIGEST_INVALID"):
        package.validate_package(prepared[0], value)


def test_extra_file_refused(prepared):
    path, _result = prepared
    (path / "foreign").write_text("must not adopt")
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_CONTENTS_INVALID"):
        package.validate_package(path)


def test_payload_digest_mismatch_refused(prepared):
    path, _result = prepared
    (path / "cca").write_bytes(b"\x7fELF_REPLACED")
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_FILE_DIGEST_MISMATCH"):
        package.validate_package(path)


@pytest.mark.parametrize(
    "filename",
    [
        "manifest.json",
        "root_source.service",
        "resources.json",
        "TEST_ONLY-pin.json",
        "README.txt",
        "worker.c",
    ],
)
def test_rehashed_noncanonical_payload_refused(prepared, filename):
    path, _result = prepared
    file = path / filename
    file.write_bytes(file.read_bytes() + b"\n")
    _rehash(path)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_CANONICAL_CONTENT_MISMATCH"):
        package.validate_package(path)


@pytest.mark.parametrize(
    "field,value",
    [("extra", True), ("version", True), ("TEST_ONLY", False), ("schema", "other"), ("files", [])],
)
def test_closed_inventory_schema_refused(prepared, field, value):
    path, _result = prepared
    inventory = json.loads((path / "inventory.json").read_bytes())
    inventory[field] = value
    (path / "inventory.json").write_bytes(package._json(inventory))
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_(SCHEMA|CONTENTS)_INVALID"):
        package.validate_package(path)


def test_duplicate_inventory_json_refused(prepared):
    path, _result = prepared
    inventory = path / "inventory.json"
    inventory.write_bytes(b'{"schema":"one",' + inventory.read_bytes()[1:])
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_SCHEMA_INVALID"):
        package.validate_package(path)


def test_symlink_payload_refused(prepared, tmp_path):
    path, _result = prepared
    executable = path / "cca"
    outside = tmp_path / "foreign"
    executable.rename(outside)
    executable.symlink_to(outside)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_FILE_UNSAFE"):
        package.validate_package(path)
    assert outside.exists()


def test_hardlink_payload_refused(prepared, tmp_path):
    path, _result = prepared
    outside = tmp_path / "foreign-link"
    os.link(path / "cca", outside)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_FILE_UNSAFE"):
        package.validate_package(path)
    assert outside.exists()


def test_symlink_directory_refused(prepared, tmp_path):
    path, _result = prepared
    link = tmp_path / "alias"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_DIRECTORY_UNSAFE"):
        package.validate_package(link)


def test_symlink_ancestor_refused(prepared, tmp_path):
    path, _result = prepared
    link = tmp_path / "alias"
    link.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_VALIDATION_FAILED"):
        package.validate_package(link / path.name)


@pytest.mark.parametrize("target,mode", [("cca", 0o777), ("manifest.json", 0o644), (".", 0o755)])
def test_unsafe_permissions_refused(prepared, target, mode):
    path, _result = prepared
    (path / target).chmod(mode)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_(FILE|DIRECTORY)_UNSAFE"):
        package.validate_package(path)


def test_acl_refused(prepared, monkeypatch):
    monkeypatch.setattr(package.os, "listxattr", lambda _fd: ["system.posix_acl_access"])
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_ACL_FORBIDDEN"):
        package.validate_package(prepared[0])


def test_acl_observer_unavailable_refused(prepared, monkeypatch):
    monkeypatch.setattr(package.os, "listxattr", None)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_METADATA_UNAVAILABLE"):
        package.validate_package(prepared[0])


def test_changed_object_during_descriptor_read_refused(prepared, monkeypatch):
    path, _result = prepared
    original = package.os.read
    changed = False

    def mutate(fd, count):
        nonlocal changed
        raw = original(fd, count)
        if not changed:
            changed = True
            inventory = path / "inventory.json"
            inventory.write_bytes(inventory.read_bytes() + b"\n")
        return raw

    monkeypatch.setattr(package.os, "read", mutate)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_CHANGED_DURING_READ"):
        package.validate_package(path)


def test_foreign_package_owner_refused(prepared, monkeypatch):
    monkeypatch.setattr(package.os, "geteuid", lambda: os.getuid() + 100)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_DIRECTORY_UNSAFE"):
        package.validate_package(prepared[0])


def test_root_validation_uses_same_unprivileged_package(prepared, monkeypatch):
    path, expected = prepared
    monkeypatch.setattr(package.os, "geteuid", lambda: 0)
    assert package.validate_package(path, expected["package_sha256"]) == expected


def test_prepare_refuses_symlink_ancestor(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(package.HarnessRefusal, match="PACKAGE_CREATION_FAILED"):
        package.build_package(alias / "new", _RUN, 62000, _COMMIT)
    assert not (target / "new").exists()


def test_non_native_binary_refused_even_after_rehash(prepared):
    path, _result = prepared
    (path / "cca").write_bytes(b"#!/usr/bin/python3\n")
    _rehash(path)
    with pytest.raises(package.HarnessRefusal, match="NATIVE_EXECUTABLE_REQUIRED"):
        package.validate_package(path)


def test_shared_binary_bytes_refused_even_after_rehash(prepared):
    path, _result = prepared
    (path / "cca").write_bytes((path / "root_source").read_bytes())
    _rehash(path)
    with pytest.raises(package.HarnessRefusal, match="DISTINCT_EXECUTABLES_REQUIRED"):
        package.validate_package(path)


def test_canonical_units_preserve_gate_requirements(prepared):
    _path, result = prepared
    for role in package.ROLES:
        raw = result["files"][f"{role}.service"]
        for directive in (
            b"NoNewPrivileges=yes\n",
            b"CapabilityBoundingSet=\n",
            b"AmbientCapabilities=\n",
            b"PrivateNetwork=yes\n",
            b"PrivateIPC=yes\n",
            b"PrivateMounts=yes\n",
            b"ProtectSystem=strict\n",
            b"ProtectHome=yes\n",
            b"SupplementaryGroups=\n",
            b"DynamicUser=no\n",
            b"InaccessiblePaths=-/home -/root -/var/lib -/etc/cryptohunter -/run/cryptohunter -/opt/cryptohunter\n",
        ):
            assert directive in raw


def test_real_offline_compiler_creates_independent_signal_controlled_elf(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("optional local compiler unavailable")
    result = package.build_package(tmp_path / "native", _RUN, 62000, _COMMIT)
    assert len({hashlib.sha256(result["files"][role]).hexdigest() for role in package.ROLES}) == 3
    process = subprocess.Popen([str(tmp_path / "native" / "cca")], stdin=subprocess.DEVNULL)
    try:
        assert process.poll() is None
        deadline = time.monotonic() + 3
        mask = (1 << (signal.SIGTERM - 1)) | (1 << (signal.SIGINT - 1))
        while time.monotonic() < deadline:
            status = Path(f"/proc/{process.pid}/status").read_text()
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            if int(fields["SigCgt"].strip(), 16) & mask == mask:
                break
            time.sleep(0.01)
        else:
            pytest.fail("local worker did not install controlled signal handlers")
        process.terminate()
        assert process.wait(timeout=3) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_compiler_timeout_sanitized(tmp_path, monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("PRIVATE_DO_NOT_REPORT", 30)

    monkeypatch.setattr(package.subprocess, "run", timeout)
    with pytest.raises(package.HarnessRefusal, match="OFFLINE_COMPILER_UNAVAILABLE") as error:
        package._compile(tmp_path / "source", tmp_path / "target", "cca")
    assert "PRIVATE" not in str(error.value)
