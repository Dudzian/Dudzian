"""Modele bramki VM nie są dowodem wykonania natywnej próby."""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment import stage9_vm_environment as guard

REVISION = "a" * 40
ENVIRONMENT = "noble-vm-test-only"
MACHINE = "b" * 32
BOOT = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def isolated_os_module(monkeypatch):
    monkeypatch.setattr(guard, "os", SimpleNamespace(**vars(os)))


def _approval():
    return {
        "schema": 1,
        "purpose": "TEST_ONLY",
        "approved": True,
        "disposable_vm": True,
        "isolation_no_host_shares": True,
        "network_disconnected": True,
        "no_production_material": True,
        "machine_id": MACHINE,
        "boot_id": BOOT,
        "revision": REVISION,
        "environment_id": ENVIRONMENT,
        "vm_image_sha256": "c" * 64,
        "vm_snapshot": "S3-approved-prerequisites",
        "packages": {name: "1.0-test" for name in guard.REQUIRED_PACKAGES},
    }


def _facts():
    return {
        "os_id": "ubuntu",
        "os_version": "24.04",
        "architecture": "x86_64",
        "python_executable": "/usr/bin/python3.12",
        "python_version": "Python 3.12.3",
        "pid1": "systemd",
        "virtualization": "kvm",
        "container": "none",
        "container_markers": [],
        "shared_mounts": [],
        "network_disconnected": True,
        "packages": _approval()["packages"],
        "machine_id": MACHINE,
        "boot_id": BOOT,
        "observation_complete": True,
    }


@pytest.fixture
def approved(monkeypatch):
    document, facts = _approval(), _facts()
    monkeypatch.setattr(guard.sys, "platform", "linux")
    monkeypatch.setattr(guard.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(guard, "_read_approval", lambda _path: json.dumps(document).encode())
    monkeypatch.setattr(guard, "inspect_environment", lambda: facts)
    return document, facts


def _execute():
    return guard.require_approved_vm(revision=REVISION, environment_id=ENVIRONMENT)


def test_exact_approval_allows_only_matching_disposable_boot(approved):
    document, facts = approved
    assert _execute() == {"approval": document, "environment": facts}


@pytest.mark.parametrize(
    "field,value",
    [
        ("os_id", "debian"),
        ("os_version", "22.04"),
        ("architecture", "aarch64"),
        ("python_executable", "/usr/local/bin/python3.12"),
        ("python_version", "Python 3.11.9"),
        ("python_version", None),
        ("pid1", "tail"),
        ("virtualization", "none"),
        ("virtualization", "unknown-vm"),
        ("container", "docker"),
        ("container_markers", ["/.dockerenv"]),
        ("shared_mounts", ["virtiofs"]),
        ("network_disconnected", False),
        ("network_disconnected", None),
        ("observation_complete", False),
        ("observation_complete", 1),
    ],
)
def test_unsafe_or_unavailable_environment_refuses(approved, field, value):
    approved[1][field] = value
    with pytest.raises(guard.VMGuardError, match="ENVIRONMENT_UNSAFE_OR_UNAVAILABLE"):
        _execute()


@pytest.mark.parametrize("field", ["machine_id", "boot_id", "revision", "environment_id"])
def test_approval_must_bind_exact_machine_boot_commit_and_environment(approved, field):
    document, _facts = approved
    replacements = {
        "machine_id": "d" * 32,
        "boot_id": "99999999-2222-3333-4444-555555555555",
        "revision": "d" * 40,
        "environment_id": "different-environment",
    }
    document[field] = replacements[field]
    with pytest.raises(guard.VMGuardError, match="APPROVAL_BINDING_MISMATCH"):
        _execute()


def test_package_drift_does_not_update_approved_versions(approved):
    before = copy.deepcopy(approved[0])
    approved[1]["packages"]["openssl"] = "2.0-unapproved"
    with pytest.raises(guard.VMGuardError, match="PACKAGE_BASELINE_MISMATCH"):
        _execute()
    assert approved[0] == before


@pytest.mark.parametrize("flag", sorted(guard._APPROVAL_FLAGS))
@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_every_operator_assertion_must_be_literal_true(approved, flag, value):
    approved[0][flag] = value
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        _execute()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 2),
        ("purpose", "PRODUCTION"),
        ("machine_id", "not-a-machine-id"),
        ("boot_id", "not-a-boot-id"),
        ("revision", "a" * 39),
        ("environment_id", "../../physical"),
        ("vm_image_sha256", "latest"),
        ("vm_snapshot", ""),
        ("packages", {}),
        ("packages", []),
    ],
)
def test_malformed_approval_refuses(approved, field, value):
    approved[0][field] = value
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        _execute()


def test_extra_field_and_extra_package_are_not_implicitly_approved(approved):
    approved[0]["unexpected"] = "PRIVATE_SECRET"
    with pytest.raises(guard.VMGuardError) as error:
        _execute()
    assert error.value.code == "VM_ENVIRONMENT_REFUSED"
    assert "PRIVATE_SECRET" not in str(error.value)
    del approved[0]["unexpected"]
    approved[0]["packages"]["foreign"] = "1.0"
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        _execute()


@pytest.mark.parametrize("raw", [b"{", b"[]", b"null", b"\xff", b'{"schema":1,"schema":1}'])
def test_invalid_json_and_duplicate_fields_refuse(approved, monkeypatch, raw):
    monkeypatch.setattr(guard, "_read_approval", lambda _path: raw)
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        _execute()


def test_duplicate_nested_package_field_refuses(approved, monkeypatch):
    raw = json.dumps(approved[0]).replace('"openssl": "1.0-test"', '"openssl":"1","openssl":"1"')
    monkeypatch.setattr(guard, "_read_approval", lambda _path: raw.encode())
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        _execute()


def test_execute_requires_administrator_before_reading_approval(approved, monkeypatch):
    monkeypatch.setattr(guard.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(guard, "_approval", lambda _path: pytest.fail("read before root gate"))
    with pytest.raises(guard.VMGuardError, match="EXECUTION_REQUIRES_VM_ADMINISTRATOR"):
        _execute()


@pytest.mark.parametrize("path", [Path("relative"), Path("/"), Path("/etc/../approval")])
def test_approval_path_is_absolute_and_cannot_traverse_parents(path):
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        guard._read_approval(path)


def _protected_reader(monkeypatch, *, changed_index=None, mode=None, uid=0, nlink=1, data=b"{}"):
    # Model kontroli Linux działa też na POSIX hosta; nie jest próbą jego ACL.
    monkeypatch.setattr(guard, "sys", SimpleNamespace(platform="linux"))
    opened = []
    closed = []
    statuses = []
    chunks = [data, b""]

    def open_file(path, flags, *args, **kwargs):
        index = len(opened)
        opened.append((path, flags, kwargs))
        directory = index < 2
        actual_mode = (stat.S_IFDIR | 0o755) if directory else (stat.S_IFREG | 0o600)
        statuses.append(
            SimpleNamespace(
                st_mode=mode if index == changed_index and mode is not None else actual_mode,
                st_uid=uid if index == changed_index else 0,
                st_nlink=nlink if index == changed_index else 1,
                st_dev=1,
                st_ino=index,
                st_size=len(data),
                st_mtime_ns=1,
                st_ctime_ns=1,
                st_gid=0,
            )
        )
        return index + 100

    monkeypatch.setattr(guard.os, "open", open_file)
    monkeypatch.setattr(guard.os, "fstat", lambda fd: statuses[fd - 100])
    monkeypatch.setattr(guard.os, "close", closed.append)
    monkeypatch.setattr(guard.os, "read", lambda _fd, _length: chunks.pop(0))
    monkeypatch.setattr(guard.os, "listxattr", lambda _fd: [], raising=False)
    return opened, closed


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor boundary")
def test_approval_is_read_through_pinned_nofollow_descriptors(monkeypatch):
    opened, closed = _protected_reader(monkeypatch)
    assert guard._read_approval(Path("/etc/approval.json")) == b"{}"
    assert len(opened) == len(closed) == 3
    assert all(flags & os.O_NOFOLLOW for _path, flags, _kwargs in opened)
    assert opened[-1][1] & os.O_NONBLOCK
    assert all(not flags & (os.O_CREAT | os.O_TRUNC | os.O_WRONLY) for _, flags, _ in opened)
    assert opened[-1][2]["dir_fd"] == 101


@pytest.mark.skipif(os.name != "posix", reason="Model wymaga stałych deskryptorów POSIX")
def test_reader_acl_contract_covers_directories_not_existing_final_file(monkeypatch):
    _protected_reader(monkeypatch)
    observed = []

    def attributes(fd):
        observed.append(fd)
        return ["system.posix_acl_access"] if fd == 102 else []

    monkeypatch.setattr(guard.os, "listxattr", attributes)
    assert guard._read_approval(Path("/etc/approval.json")) == b"{}"
    assert observed == [100, 101]


@pytest.mark.skipif(os.name != "posix", reason="Model wymaga stałych deskryptorów POSIX")
@pytest.mark.parametrize("index", [0, 1], ids=["root", "przodek"])
@pytest.mark.parametrize("acl", ["system.posix_acl_access", "system.posix_acl_default"])
def test_directory_acl_refuses_before_reading_approval(monkeypatch, index, acl):
    opened, closed = _protected_reader(monkeypatch)

    def attributes(fd):
        return [acl] if fd == index + 100 else []

    def forbidden_read(*_args):
        pytest.fail("ACL katalogu musi zatrzymać odczyt danych zatwierdzenia")

    monkeypatch.setattr(guard.os, "listxattr", attributes)
    monkeypatch.setattr(guard.os, "read", forbidden_read)
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        guard._read_approval(Path("/etc/approval.json"))
    assert len(opened) == index + 1
    assert sorted(closed) == list(range(100, 101 + index))


@pytest.mark.skipif(os.name != "posix", reason="Model wymaga stałych deskryptorów POSIX")
@pytest.mark.parametrize("index", [0, 1], ids=["root", "przodek"])
def test_directory_xattr_denial_refuses_before_reading_approval(monkeypatch, index):
    opened, closed = _protected_reader(monkeypatch)

    def attributes(fd):
        if fd == index + 100:
            raise PermissionError("Model niedostępnej obserwacji ACL")
        return []

    def forbidden_read(*_args):
        pytest.fail("Brak obserwacji ACL musi zatrzymać odczyt danych zatwierdzenia")

    monkeypatch.setattr(guard.os, "listxattr", attributes)
    monkeypatch.setattr(guard.os, "read", forbidden_read)
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        guard._read_approval(Path("/etc/approval.json"))
    assert len(opened) == index + 1
    assert sorted(closed) == list(range(100, 101 + index))


@pytest.mark.parametrize(
    "index,mode,uid,nlink",
    [
        (0, stat.S_IFDIR | 0o777, 0, 1),
        (1, stat.S_IFDIR | 0o775, 0, 1),
        (1, stat.S_IFDIR | 0o755, 1000, 1),
        (2, stat.S_IFREG | 0o600, 1000, 1),
        (2, stat.S_IFREG | 0o666, 0, 1),
        (2, stat.S_IFREG | 0o600, 0, 2),
        (2, stat.S_IFLNK | 0o777, 0, 1),
        (2, stat.S_IFIFO | 0o600, 0, 1),
    ],
)
@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor boundary")
def test_unprotected_ancestor_or_foreign_approval_refuses(monkeypatch, index, mode, uid, nlink):
    _protected_reader(monkeypatch, changed_index=index, mode=mode, uid=uid, nlink=nlink)
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        guard._read_approval(Path("/etc/approval.json"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor boundary")
def test_approval_size_is_bounded(monkeypatch):
    _protected_reader(monkeypatch, data=b"x" * (guard._MAX_APPROVAL + 1))
    with pytest.raises(guard.VMGuardError, match="APPROVAL_INVALID"):
        guard._read_approval(Path("/etc/approval.json"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor boundary")
def test_missing_approval_and_symlink_os_errors_are_sanitized(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("PRIVATE_PATH_OR_SECRET")

    monkeypatch.setattr(guard.os, "open", fail)
    with pytest.raises(guard.VMGuardError) as error:
        guard._read_approval(Path("/etc/approval.json"))
    assert str(error.value) == "APPROVAL_INVALID"


def _ip(rows):
    return 0, json.dumps(rows)


def _offline_network():
    return (
        _ip([{"ifname": "lo", "flags": ["UP", "LOOPBACK"]}, {"ifname": "ens3", "flags": []}]),
        _ip([{"ifname": "lo", "addr_info": [{}]}, {"ifname": "ens3", "addr_info": []}]),
        _ip([{"dev": "lo"}]),
    )


def test_network_requires_no_active_links_addresses_or_routes():
    assert guard._network_disconnected(*_offline_network()) is True


@pytest.mark.parametrize("case", ["up", "address", "route", "unknown", "malformed", "empty"])
def test_active_or_unavailable_network_fails_closed(case):
    observations = list(_offline_network())
    if case == "up":
        observations[0] = _ip([{"ifname": "ens3", "flags": ["UP"]}])
    elif case == "address":
        observations[1] = _ip([{"ifname": "ens3", "addr_info": [{"local": "192.0.2.1"}]}])
    elif case == "route":
        observations[2] = _ip([{"dev": "ens3", "dst": "default"}])
    elif case == "unknown":
        observations[0] = None
    elif case == "malformed":
        observations[0] = 0, "not JSON"
    elif case == "empty":
        observations[0] = _ip([])
    assert guard._network_disconnected(*observations) is not True


@pytest.mark.parametrize(
    "filesystem", ["virtiofs", "9p", "vboxsf", "fuse.vmhgfs-fuse", "cifs", "nfs4"]
)
def test_host_share_filesystems_are_detected_without_reading_their_contents(filesystem):
    mount = f"42 1 0:1 / /mnt rw - {filesystem} host rw"
    assert guard._shared_mounts(mount) == [filesystem]


@pytest.mark.parametrize("raw", [None, "", "malformed mountinfo"])
def test_unreadable_or_malformed_mount_table_is_unavailable(raw):
    assert guard._shared_mounts(raw) is None


def test_package_observer_requires_complete_installed_unique_versions():
    valid = "\n".join(f"{name}\t1.0-test\tii " for name in sorted(guard.REQUIRED_PACKAGES))
    assert guard._packages((0, valid)) == _approval()["packages"]
    assert guard._packages((1, valid)) is None
    assert guard._packages((0, valid + "\n" + valid.splitlines()[0])) is None
    assert guard._packages((0, valid.replace("\tii ", "\trc ", 1))) is None
    assert guard._packages((0, "\n".join(valid.splitlines()[1:]))) is None
    assert guard._packages((0, "unexpected")) is None


def test_inspect_collects_public_facts_without_mutations(monkeypatch):
    reads, commands = [], []
    public = {
        "/etc/os-release": 'ID=ubuntu\nVERSION_ID="24.04"',
        "/proc/1/comm": "systemd",
        "/etc/machine-id": MACHINE,
        "/proc/sys/kernel/random/boot_id": BOOT,
        "/proc/self/mountinfo": "1 0 0:1 / / rw - ext4 /dev/vda rw",
    }

    def read(path, *_limits):
        reads.append(path)
        return public[path]

    def query(args):
        commands.append(args)
        if args[0] == "/usr/bin/systemd-detect-virt":
            return (0, "kvm") if args[-1] == "--vm" else (1, "none")
        if args[0] == "/usr/bin/python3.12":
            return 0, "Python 3.12.3"
        if args[0] == "/usr/bin/dpkg-query":
            return 0, "\n".join(f"{name}\t1.0-test\tii " for name in guard.REQUIRED_PACKAGES)
        index = {"link": 0, "address": 1, "route": 2}[args[2]]
        return _offline_network()[index]

    def absent(_path):
        raise FileNotFoundError

    monkeypatch.setattr(guard.sys, "platform", "linux")
    monkeypatch.setattr(guard.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(guard, "_public_text", read)
    monkeypatch.setattr(guard, "_query", query)
    monkeypatch.setattr(guard.os, "lstat", absent)
    result = guard.inspect_environment()
    assert result == _facts()
    assert set(reads) == set(public)
    assert len(commands) == 7
    assert all(
        not any(word in args for word in ("sudo", "install", "start", "add")) for args in commands
    )


@pytest.mark.parametrize("platform_name", ["win32", "darwin"])
def test_inspect_on_unsupported_platform_never_issues_linux_commands(monkeypatch, platform_name):
    monkeypatch.setattr(guard.sys, "platform", platform_name)
    monkeypatch.setattr(guard, "_query", lambda _args: pytest.fail("Linux query on unsupported OS"))
    facts = guard.inspect_environment()
    assert facts["observation_complete"] is False


@pytest.mark.parametrize("case", ["timeout", "os", "unicode"])
def test_query_failures_cannot_become_positive_evidence(monkeypatch, case):
    def failure(*_args, **_kwargs):
        if case == "timeout":
            raise subprocess.TimeoutExpired("fixed command", 15)
        if case == "unicode":
            raise UnicodeError("PRIVATE")
        raise OSError("PRIVATE")

    monkeypatch.setattr(guard.subprocess, "run", failure)
    assert guard._query(["/fixed-command"]) is None


def test_query_uses_fixed_sanitized_subprocess_boundary(monkeypatch):
    def run(arguments, **options):
        assert arguments == ["/fixed-command"]
        assert options["stdin"] == subprocess.DEVNULL
        assert options["stderr"] == subprocess.DEVNULL
        assert options["cwd"] == "/"
        assert options["timeout"] == 15
        assert options["env"] == {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        assert "shell" not in options
        return SimpleNamespace(returncode=0, stdout="observed\n")

    monkeypatch.setattr(guard.subprocess, "run", run)
    assert guard._query(["/fixed-command"]) == (0, "observed")


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor boundary")
def test_public_reader_only_reads_bounded_regular_file(tmp_path):
    path = tmp_path / "public-fact"
    path.write_bytes(b"public\n")
    assert guard._public_text(str(path)) == "public"
    assert guard._public_text(str(path), 3) is None
    assert guard._public_text(str(tmp_path)) is None
    assert guard._public_text(str(tmp_path / "missing")) is None
    path.write_bytes(b"\xff")
    assert guard._public_text(str(path)) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, (None, None)),
        ('ID=ubuntu\nVERSION_ID="24.04"', ("ubuntu", "24.04")),
        ("ID=ubuntu\nID=debian", (None, None)),
        ("ID=$(untrusted-command)", (None, None)),
        ("# comment\nNAME=Ubuntu\nID=ubuntu", ("ubuntu", None)),
    ],
)
def test_os_release_is_parsed_as_data_without_shell_expansion(raw, expected):
    assert guard._os_identity(raw) == expected


@pytest.mark.parametrize("case", ["missing-loopback", "invalid-flags", "empty-addresses"])
def test_incomplete_network_inventory_is_not_disconnected(case):
    observations = list(_offline_network())
    if case == "missing-loopback":
        observations[0] = _ip([{"ifname": "ens3", "flags": []}])
    elif case == "invalid-flags":
        observations[0] = _ip([{"ifname": "lo", "flags": [1]}])
    else:
        observations[1] = _ip([])
    assert guard._network_disconnected(*observations) is None


@pytest.mark.parametrize(
    "probe", ["machine_id", "boot_id", "pid1", "vm", "container", "architecture"]
)
def test_inspect_does_not_publish_arbitrary_probe_contents(monkeypatch, probe):
    marker = "PRIVATE_UNTRUSTED_PROBE_CONTENT"
    paths = {
        "machine_id": "/etc/machine-id",
        "boot_id": "/proc/sys/kernel/random/boot_id",
        "pid1": "/proc/1/comm",
    }
    monkeypatch.setattr(guard.sys, "platform", "linux")
    monkeypatch.setattr(
        guard.platform, "machine", lambda: marker if probe == "architecture" else "x86_64"
    )
    monkeypatch.setattr(
        guard, "_public_text", lambda path, *_limits: marker if path == paths.get(probe) else None
    )

    def query(arguments):
        if arguments[-1] == "--vm" and probe == "vm":
            return 0, marker
        if arguments[-1] == "--container" and probe == "container":
            return 0, marker
        return None

    def absent(_path):
        raise FileNotFoundError

    monkeypatch.setattr(guard, "_query", query)
    monkeypatch.setattr(guard.os, "lstat", absent)
    facts = guard.inspect_environment()
    assert marker not in json.dumps(facts)
    assert facts["observation_complete"] is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor boundary")
@pytest.mark.parametrize(
    "field",
    [
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
    ],
)
def test_approval_metadata_drift_during_read_refuses(monkeypatch, field):
    _protected_reader(monkeypatch)
    stable_fstat = guard.os.fstat
    file_reads = 0

    def changed(fd):
        nonlocal file_reads
        value = stable_fstat(fd)
        if fd == 102:
            file_reads += 1
            if file_reads > 1:
                fields = vars(value).copy()
                fields[field] += 1
                return SimpleNamespace(**fields)
        return value

    monkeypatch.setattr(guard.os, "fstat", changed)
    with pytest.raises(guard.VMGuardError, match="APPROVAL_CHANGED_DURING_READ"):
        guard._read_approval(Path("/etc/approval.json"))
