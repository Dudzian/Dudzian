"""Hermetic Win32 adapter and pin-lifetime tests; never invoke an installer."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import stat
import types
import zipfile
from pathlib import Path

import pytest

from deployment import windows_stage9_install_native as native


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *arguments):
        return self.callback(*arguments)


class Win32:
    """Portable Win32 syscall stand-in, never a native security authority.

    Handles are logical IDs, not host file descriptors. Keeping POSIX fds open
    made directory pins fail on Windows and /proc/self/fd fail on macOS; it
    also prevented the path-replacement negative tests on Windows.
    Identity is observed via the test filesystem at each operation so that a
    replaced original is detected, while production still uses real Win32 pins.
    Access/share modes, protected DACLs and reparse-point opens are not enforced;
    these tests prove control flow, not native Windows security semantics.
    """

    def __init__(self):
        self.handles = {}
        self.positions = {}
        self.next_handle = 100000
        self.calls = []
        self.fail = set()
        self.architecture = 9
        self.descriptors = []
        self.functions = {}
        self.stable_ancestors = set()

    def __getattr__(self, name):
        if name not in self.functions:
            self.functions[name] = Function(lambda *args: self.call(name, *args))
        return self.functions[name]

    def call(self, name, *args):
        self.calls.append((name, args))
        if name in self.fail:
            return ctypes.c_void_p(-1).value if name == "CreateFileW" else 0
        if name == "GetNativeSystemInfo":
            args[0]._obj[0] = self.architecture
        elif name == "ConvertStringSecurityDescriptorToSecurityDescriptorW":
            self.descriptors.append(args[0])
            args[2]._obj.value = 123
        elif name == "CreateFileW":
            path, access, share, security, creation, flags, template = args
            target = Path(path)
            if creation == native._CREATE_NEW:
                # No descriptor is held by the fake across the test: real
                # Windows file handles would prohibit unlink/replacement
                # operations intentionally exercised by these negative tests.
                with target.open("xb"):
                    pass
            elif not target.exists() and not target.is_symlink():
                raise FileNotFoundError(path)
            handle = self.next_handle
            self.next_handle += 1
            self.handles[handle] = (target, access, share)
            self.positions[handle] = 0
            return handle
        elif name in {"GetFileInformationByHandle", "GetFileInformationByHandleEx"}:
            metadata = self.handles[args[0]][0].stat()
            if name == "GetFileInformationByHandle":
                info = args[1]._obj
                info.attributes = 16 if stat.S_ISDIR(metadata.st_mode) else 0
                info.volume = metadata.st_dev & 0xFFFFFFFF
                info.index_high = metadata.st_ino >> 32
                info.index_low = metadata.st_ino & 0xFFFFFFFF
                info.links = 1 if stat.S_ISDIR(metadata.st_mode) else metadata.st_nlink
                info.size_high = metadata.st_size >> 32
                info.size_low = metadata.st_size & 0xFFFFFFFF
                if stat.S_ISDIR(metadata.st_mode):
                    info.size_high = info.size_low = 0
            else:
                basic = args[2]._obj
                basic.write, basic.change = metadata.st_mtime_ns, metadata.st_ctime_ns
                if self.handles[args[0]][0] in self.stable_ancestors:
                    basic.write = basic.change = 0
        elif name == "GetFinalPathNameByHandleW":
            # Path.resolve observes symlink targets on all three host OSes,
            # without assuming Linux's /proc/self/fd is present.
            target = str(self.handles[args[0]][0].resolve())
            args[1].value = "\\\\?\\" + target
            return len(args[1].value)
        elif name == "SetFilePointerEx":
            assert args[3] == 0  # Only FILE_BEGIN is used by the adapter.
            self.positions[args[0]] = args[1]
        elif name == "ReadFile":
            with self.handles[args[0]][0].open("rb") as source:
                source.seek(self.positions[args[0]])
                data = source.read(args[2])
            self.positions[args[0]] += len(data)
            ctypes.memmove(args[1], data, len(data))
            args[3]._obj.value = len(data)
        elif name == "WriteFile":
            data = ctypes.string_at(args[1], args[2])
            with self.handles[args[0]][0].open("r+b") as target:
                target.seek(self.positions[args[0]])
                written = target.write(data)
            self.positions[args[0]] += written
            args[3]._obj.value = written
        elif name == "SetEndOfFile":
            with self.handles[args[0]][0].open("r+b") as target:
                target.truncate(self.positions[args[0]])
        elif name == "FlushFileBuffers":
            pass  # Each simulated write already closes and flushes its file.
        elif name == "CreateDirectoryW":
            Path(args[0]).mkdir(mode=0o700)
        elif name == "CloseHandle":
            self.handles.pop(args[0])
            self.positions.pop(args[0])
        elif name != "LocalFree":
            raise AssertionError(f"unexpected native call {name}")
        return 1


@pytest.mark.parametrize("directory", [True, False])
def test_portable_fake_pins_directories_and_files_without_host_fds(
    tmp_path, monkeypatch, directory
):
    # The mocked kernel must be equally usable on Windows, Linux and macOS.
    # In particular, do not rely on os.open(directory) or /proc/self/fd.
    api = Win32()
    monkeypatch.setattr(native, "os", types.SimpleNamespace(name="nt", scandir=os.scandir))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: api, raising=False)
    target = tmp_path / "protected"
    if directory:
        target.mkdir()
    else:
        target.write_bytes(b"immutable-test-input")
    adapter = native._Native()
    handle = adapter.open(target)
    try:
        identity = adapter.identity(handle)
        assert identity.final_path == str(target.resolve())
        assert bool(identity.attributes & 0x10) is directory
        assert api.handles[handle][0] == target
        if not directory:
            assert b"".join(adapter.chunks(handle)) == b"immutable-test-input"
    finally:
        adapter.close(handle)
    assert api.handles == {}
    assert api.positions == {}


@pytest.fixture
def win32(monkeypatch):
    api = Win32()
    monkeypatch.setattr(native, "os", types.SimpleNamespace(name="nt", scandir=os.scandir))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: api, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    return api


@pytest.fixture
def protected(tmp_path, monkeypatch, win32):
    # Other concurrently running pytest sessions may mutate /tmp. The fake
    # models protected host ancestry, not the live shared Linux scratch root.
    win32.stable_ancestors = set(tmp_path.parents)
    roots = {name: tmp_path / name for name in ("artifacts", "policy", "checkout", "staging", "os")}
    for root in roots.values():
        root.mkdir()
    policy = roots["policy"] / "host.json"
    policy.write_text('{"host_identity":{}}')
    (roots["checkout"] / "executor.py").write_text("# protected source\n")
    executable = roots["os"] / "msiexec.exe"
    executable.write_bytes(b"native trusted OS executable stand-in")
    payload = b"qualified test MSI bytes; never executable"
    archive = roots["artifacts"] / "release.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("release/exact.msi", payload)
    monkeypatch.setattr(native, "_checked_path", lambda path: path)
    monkeypatch.setattr(native, "_native_msiexec_path", lambda: executable)
    observations = []
    monkeypatch.setattr(native, "_collect_acl_tree", lambda path: observations.append(path) or {})
    monkeypatch.setattr(native, "validate_acl_tree", lambda observed, **kwargs: [])
    arguments = {
        "artifact_root": roots["artifacts"],
        "policy_path": policy,
        "executor_root": roots["checkout"],
        "staging_root": roots["staging"],
    }
    material = {
        "archive": archive,
        "msi_member": "release/exact.msi",
        "expected_msi_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return roots, arguments, material, observations


def test_windows_native_adapter_layout_and_x64_boundary(win32):
    assert ctypes.sizeof(native._FileInfo) == 52
    assert ctypes.sizeof(native._BasicInfo) == 40
    native._Native()
    win32.architecture = 12
    with pytest.raises(native.NativeInstallationUnavailable, match="AMD64"):
        native._Native()


@pytest.mark.parametrize("failed_handle", [None, ctypes.c_void_p(-1).value])
def test_createfile_error_is_captured_before_security_descriptor_cleanup(
    tmp_path, monkeypatch, win32, failed_handle
):
    last_error = [0]
    freed = []

    def create_file(*_arguments):
        last_error[0] = 32  # ERROR_SHARING_VIOLATION
        return failed_handle

    def local_free(descriptor):
        freed.append(descriptor.value)
        last_error[0] = 5  # Cleanup must not replace CreateFileW's error.
        return None

    monkeypatch.setattr(win32, "CreateFileW", Function(create_file))
    monkeypatch.setattr(win32, "LocalFree", Function(local_free))
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error[0])
    with pytest.raises(OSError) as error:
        native._Native().open(tmp_path / "protected")
    assert error.value.errno == 32
    assert freed == [123]
    assert last_error[0] == 5
    assert win32.handles == {}


def test_off_windows_has_no_native_or_process_fallback(monkeypatch):
    monkeypatch.setattr(native, "os", types.SimpleNamespace(name="posix"))
    with pytest.raises(native.NativeInstallationUnavailable):
        native._Native()
    monkeypatch.setattr(native, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(ctypes, "WinDLL", None, raising=False)
    with pytest.raises(native.NativeInstallationUnavailable, match="DLL"):
        native._Native()


@pytest.mark.parametrize("value", ["relative/file", "C:\\x\\..\\y", "\\\\host\\share\\x", "C:\\"])
def test_noncanonical_or_nonlocal_paths_rejected(value):
    with pytest.raises(native.InstallationIntegrityError):
        native._checked_path(Path(value))


def test_success_keeps_originals_and_msi_pinned_until_context_exit(protected, win32):
    roots, arguments, material, observations = protected
    with native.locked_installation_inputs(**arguments) as locked:
        assert set(observations) == {
            arguments["artifact_root"],
            arguments["policy_path"].parent,
            arguments["executor_root"],
            arguments["staging_root"],
        }
        assert arguments["policy_path"] in {entry[0] for entry in win32.handles.values()}
        installed = locked.materialize_msi(**material)
        assert installed.msi_path.read_bytes() == b"qualified test MSI bytes; never executable"
        assert installed.argv == [
            str(roots["os"] / "msiexec.exe"),
            "/i",
            str(installed.msi_path),
            "/qn",
            "/norestart",
            "REBOOT=ReallySuppress",
            "ALLUSERS=1",
            "MSIINSTALLPERUSER=0",
            "/L*V",
            str(installed.log_path),
        ]
        installed.write_report({"status": "RUNNING", "installation_authorized": True})
        installed.verify_pins()
        installed.write_report({"status": "FAIL"})
        assert json.loads(installed.report_path.read_text()) == {"status": "FAIL"}
        installed.verify_pins()
        pinned = {entry[0]: entry for entry in win32.handles.values()}
        assert pinned[installed.msi_path][1:] == (native._READ, native._SHARE_READ)
        assert pinned[material["archive"]][1:] == (native._READ, native._SHARE_READ)
        assert pinned[arguments["policy_path"]][1:] == (native._READ, native._SHARE_READ)
        assert pinned[installed.log_path][2] == native._SHARE_READ | native._SHARE_WRITE
        # Data-read access is needed for real Windows delete-sharing denial;
        # the portable fake only checks this request, not OS enforcement.
        assert pinned[installed.log_path][1] == native._READ_DATA | native._READ_ATTRIBUTES
        assert pinned[installed.log_path][1] & native._WRITE == 0
        assert pinned[installed.report_path][2] == native._SHARE_READ
        assert all("(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)" in item for item in win32.descriptors)
    assert win32.handles == {}
    assert installed.msi_path.exists() and installed.report_path.exists()
    with pytest.raises(native.InstallationIntegrityError, match="closed"):
        installed.verify_pins()
    with pytest.raises(native.InstallationIntegrityError, match="releasing"):
        installed.write_report({})


def test_inputs_reject_unsafe_acl_and_always_release_handles(protected, monkeypatch, win32):
    roots, arguments, material, observations = protected
    monkeypatch.setattr(native, "validate_acl_tree", lambda observed, **kwargs: ["unsafe ACE"])
    with pytest.raises(native.InstallationIntegrityError, match="ACL"):
        with native.locked_installation_inputs(**arguments):
            pytest.fail("unsafe ACL yielded a live protected capability")
    assert win32.handles == {}
    assert list(roots["staging"].iterdir()) == []


def test_acl_uses_existing_authority_without_policy_trust(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "_collect_acl_tree", lambda path: {"live": str(path)})
    monkeypatch.setattr(
        native,
        "validate_acl_tree",
        lambda observation, **kwargs: calls.append((observation, kwargs)) or [],
    )
    native._require_safe_acl(Path("C:\\protected"))
    assert calls == [
        (
            {"live": "C:\\protected"},
            {
                "runner_root": "C:\\protected",
                "service_sid": native.SYSTEM_SID,
                "trusted_write_sids": frozenset(),
            },
        )
    ]


def test_acl_collection_encodes_path_as_data_and_reuses_shared_script(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "native_powershell_path", lambda: "C:\\Windows\\trusted.exe")
    monkeypatch.setattr(native, "_read_json_command", lambda command: calls.append(command) or {})
    native._collect_acl_tree(Path("C:\\protected\\untrusted';throw 'injection"))
    import base64

    script = base64.b64decode(calls[0][-1]).decode("utf-16-le")
    assert script.startswith(native.WINDOWS_ACL_TREE_SCRIPT)
    assert "untrusted';throw 'injection" not in script
    assert "Collect-RunnerAclTree -Root $root" in script
    assert calls[0][0] == "C:\\Windows\\trusted.exe"


@pytest.mark.parametrize("changed", ["artifact_root", "executor_root", "policy_path"])
def test_original_bytes_cannot_change_after_qualification(protected, changed, win32):
    roots, arguments, material, observations = protected
    with pytest.raises(native.InstallationIntegrityError, match="changed"):
        with native.locked_installation_inputs(**arguments) as locked:
            target = arguments[changed]
            if target.is_dir():
                target = next(target.iterdir())
            target.write_bytes(b"substituted bytes")
            locked.materialize_msi(**material)
    assert win32.handles == {}
    assert list(roots["staging"].iterdir()) == []


def test_replaced_original_path_fails_even_when_same_length(protected):
    roots, arguments, material, observations = protected
    with pytest.raises(native.InstallationIntegrityError):
        with native.locked_installation_inputs(**arguments) as locked:
            archive = material["archive"]
            original = archive.read_bytes()
            archive.unlink()
            archive.write_bytes(original)
            locked.verify_pins()


@pytest.mark.parametrize("kind", ["symlink", "junction", "hardlink"])
def test_aliases_never_yield_a_protected_context(protected, monkeypatch, kind, win32):
    roots, arguments, material, observations = protected
    target = arguments["artifact_root"] / "alias"
    if kind == "hardlink":
        os.link(material["archive"], target)
    elif kind == "symlink":
        target.symlink_to(material["archive"])
    else:
        # Reparse-tagged directory identity is the native junction boundary.
        target.mkdir()
        original = native._Native.identity

        def junction_identity(self, handle):
            identity = original(self, handle)
            if Path(identity.final_path) == target:
                return native._Identity(
                    identity.final_path,
                    identity.attributes | 0x400,
                    identity.file_id,
                    identity.links,
                    identity.size,
                    identity.write,
                    identity.change,
                )
            return identity

        monkeypatch.setattr(native._Native, "identity", junction_identity)
    with pytest.raises(native.InstallationIntegrityError, match="reparse|hardlink"):
        with native.locked_installation_inputs(**arguments):
            pytest.fail("aliased original input was accepted")
    assert win32.handles == {}


def test_materialized_msi_is_hashed_from_open_handle_and_rechecked(protected, monkeypatch):
    roots, arguments, material, observations = protected
    with native.locked_installation_inputs(**arguments) as locked:
        installed = locked.materialize_msi(**material)
        original_chunks = locked._native.chunks

        def changed_chunks(handle):
            if handle == installed._msi.handle:
                yield b"different bytes from actual pinned handle"
            else:
                yield from original_chunks(handle)

        monkeypatch.setattr(locked._native, "chunks", changed_chunks)
        with pytest.raises(native.InstallationIntegrityError, match="content changed"):
            installed.verify_pins()


def test_materialized_sha_mismatch_never_returns_installer_capability(protected):
    roots, arguments, material, observations = protected
    with pytest.raises(native.InstallationIntegrityError, match="SHA-256"):
        with native.locked_installation_inputs(**arguments) as locked:
            locked.materialize_msi(**{**material, "expected_msi_sha256": "0" * 64})
    reports = list(roots["staging"].glob("*/installation-report.json"))
    assert len(reports) == 1
    failure = json.loads(reports[0].read_text())
    assert failure["status"] == "MATERIALIZATION_FAILED"
    assert failure["installer_invoked"] is False


def test_immutable_invocation_journal_survives_partial_final_report(protected, monkeypatch):
    roots, arguments, material, observations = protected
    with native.locked_installation_inputs(**arguments) as locked:
        installed = locked.materialize_msi(**material)
        intent = {"status": "INSTALLATION_OUTCOME_UNKNOWN", "installer_invoked": False}
        installed.write_report(intent)
        journal_bytes = installed.journal_path.read_bytes()
        assert json.loads(journal_bytes) == intent
        original_write = locked._native.write

        def partial_write(handle, chunks):
            if handle == installed._report.handle:
                original_write(handle, iter((b'{"status":',)))
                raise OSError("interrupted final snapshot")
            original_write(handle, chunks)

        monkeypatch.setattr(locked._native, "write", partial_write)
        with pytest.raises(OSError, match="interrupted"):
            installed.write_report({"status": "INSTALLED"})
        assert installed.journal_path.read_bytes() == journal_bytes
        assert json.loads(installed.journal_path.read_text()) == intent


@pytest.mark.parametrize(
    "member",
    [
        "../escape.msi",
        "/absolute.msi",
        "x\\y.msi",
        "x//y.msi",
        "x/./y.msi",
        "x/../y.msi",
        "x.exe",
        "",
        42,
    ],
)
def test_unsafe_zip_member_name_rejected_before_materialization(protected, member):
    roots, arguments, material, observations = protected
    with native.locked_installation_inputs(**arguments) as locked:
        with pytest.raises(native.InstallationIntegrityError):
            locked.materialize_msi(**{**material, "msi_member": member})
    assert list(roots["staging"].iterdir()) == []


@pytest.mark.parametrize("member_type", ["missing", "duplicate", "symlink", "empty"])
def test_zip_member_identity_type_and_size_are_fail_closed(protected, member_type):
    roots, arguments, material, observations = protected
    with zipfile.ZipFile(material["archive"], "w") as package:
        if member_type == "missing":
            package.writestr("other.msi", b"other")
        elif member_type == "duplicate":
            package.writestr(material["msi_member"], b"first")
            with pytest.warns(UserWarning, match="Duplicate"):
                package.writestr(material["msi_member"], b"second")
        elif member_type == "symlink":
            item = zipfile.ZipInfo(material["msi_member"])
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(item, b"target")
        else:
            package.writestr(material["msi_member"], b"")
    with pytest.raises(native.InstallationIntegrityError, match="ZIP member"):
        with native.locked_installation_inputs(**arguments) as locked:
            locked.materialize_msi(**material)


def test_only_native_system_binary_allows_windows_servicing_hardlinks(protected):
    roots, arguments, material, observations = protected
    os.link(roots["os"] / "msiexec.exe", roots["os"] / "winsxs-msiexec.exe")
    with native.locked_installation_inputs(**arguments) as locked:
        installed = locked.materialize_msi(**material)
        installed.verify_pins()
        os.link(installed.msi_path, installed.msi_path.parent / "payload-alias.msi")
        with pytest.raises(native.InstallationIntegrityError):
            installed.verify_pins()


def test_log_can_grow_but_cannot_be_replaced(protected):
    roots, arguments, material, observations = protected
    with native.locked_installation_inputs(**arguments) as locked:
        installed = locked.materialize_msi(**material)
        installed.log_path.write_bytes(b"full MSI verbose diagnostic log")
        installed.verify_pins()
        installed.log_path.unlink()
        installed.log_path.write_bytes(b"replacement")
        with pytest.raises(native.InstallationIntegrityError):
            installed.verify_pins()


def test_second_materialization_is_not_an_automatic_retry(protected):
    roots, arguments, material, observations = protected
    with native.locked_installation_inputs(**arguments) as locked:
        installed = locked.materialize_msi(**material)
        installed.write_report({"status": "INSTALLED", "installer_invoked": True})
        with pytest.raises(native.InstallationIntegrityError, match="one installation"):
            locked.materialize_msi(**material)
        assert json.loads(installed.report_path.read_text())["status"] == "INSTALLED"


def test_separate_trust_roots_required(protected, win32):
    roots, arguments, material, observations = protected
    arguments["staging_root"] = arguments["artifact_root"] / "mutable-staging"
    with pytest.raises(native.InstallationIntegrityError, match="separate"):
        with native.locked_installation_inputs(**arguments):
            pytest.fail("overlapping trust domains accepted")
    assert win32.handles == {}


@pytest.mark.parametrize(
    "function",
    [
        "CreateFileW",
        "GetFileInformationByHandle",
        "GetFileInformationByHandleEx",
        "GetFinalPathNameByHandleW",
        "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        "ReadFile",
        "WriteFile",
        "SetFilePointerEx",
        "SetEndOfFile",
        "FlushFileBuffers",
        "CreateDirectoryW",
    ],
)
def test_native_failures_do_not_yield_installation(protected, win32, function):
    roots, arguments, material, observations = protected
    win32.fail.add(function)
    with pytest.raises((OSError, native.InstallationIntegrityError)):
        with native.locked_installation_inputs(**arguments) as locked:
            locked.materialize_msi(**material)
    assert win32.handles == {}


def test_unexpected_exception_closes_every_pin_without_removing_evidence(protected, win32):
    roots, arguments, material, observations = protected
    with pytest.raises(KeyboardInterrupt):
        with native.locked_installation_inputs(**arguments) as locked:
            installed = locked.materialize_msi(**material)
            installed.write_report({"status": "RUNNING"})
            raise KeyboardInterrupt
    assert win32.handles == {}
    assert json.loads(installed.report_path.read_text()) == {"status": "RUNNING"}


def test_bounded_stream_rejects_truncated_or_excess_bytes():
    assert b"".join(native._stream_chunks(io.BytesIO(b"abc"), 3)) == b"abc"
    with pytest.raises(native.InstallationIntegrityError, match="incomplete"):
        list(native._stream_chunks(io.BytesIO(b"ab"), 3))
    with pytest.raises(native.InstallationIntegrityError, match="exceeded"):
        list(native._stream_chunks(io.BytesIO(b"abcd"), 3))


def test_native_msiexec_resolution_uses_os_api_and_rejects_non_system32(monkeypatch):
    monkeypatch.setattr(native, "_native_system_directory", lambda: "C:\\Windows\\System32")
    assert str(native._native_msiexec_path()) == "C:\\Windows\\System32\\msiexec.exe"
    monkeypatch.setattr(native, "_native_system_directory", lambda: "C:\\untrusted")
    with pytest.raises(native.NativeInstallationUnavailable):
        native._native_msiexec_path()
