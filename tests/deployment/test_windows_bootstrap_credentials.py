"""Native creation models; opt-in Windows proof uses only synthetic temp files."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import codecs
import ctypes
import os
import re
import subprocess
import sys

import pytest

from deployment.windows_installer import provision


class _RelativeCredentialRead:
    """Read fixture children by pinned handles without trusting host ancestry."""

    def __init__(self):
        from deployment import windows_stage10_diagnostic_windows as layout

        # Python imports this codec lazily. Resolve it while still SYSTEM;
        # Anonymous must not need access to the Python installation to read a
        # fixture child whose directory handle has already been pinned.
        codecs.lookup("utf-16-le")
        self.layout = layout
        self.boundary = layout._Native()

    @staticmethod
    def _check_leaf(name: str) -> None:
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", name) is None
            or name in {".", ".."}
        ):
            raise ValueError("relative fixture access requires a bounded leaf")

    def pin_directory(self, path: Path):
        handle = self.boundary.files.CreateFile(
            str(path),
            0x80000000 | 0x00100000 | 0x20,  # read, synchronize, traverse
            0x7,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        try:
            attributes = self.boundary.files.GetFileInformationByHandle(int(handle))[0]
            if not attributes & 0x10 or attributes & 0x400:
                raise RuntimeError("fixture parent is not an ordinary directory")
            return handle
        except BaseException:
            handle.Close()
            raise

    def close(self, handle: int) -> None:
        try:
            self.boundary.close(handle)
        except OSError:
            # A failed close must never satisfy a private-read PermissionError
            # assertion after an unexpectedly successful open/read.
            raise RuntimeError("relative fixture handle close failed") from None

    def open(self, parent: int, name: str, *, directory: bool = False) -> int:
        self._check_leaf(name)
        encoded = name.encode("utf-16-le")
        buffer = ctypes.create_string_buffer(encoded)
        string = self.layout._UnicodeString(len(encoded), len(encoded), ctypes.addressof(buffer))
        attributes = self.layout._ObjectAttributes(
            ctypes.sizeof(self.layout._ObjectAttributes),
            parent,
            ctypes.pointer(string),
            self.layout._OBJ_CASE_INSENSITIVE | self.layout._OBJ_DONT_REPARSE,
            None,
            None,
        )
        handle, status = ctypes.c_void_p(), self.layout._IoStatus()
        result = self.boundary.nt.NtCreateFile(
            ctypes.byref(handle),
            0x80000000 | 0x00100000,
            ctypes.byref(attributes),
            ctypes.byref(status),
            None,
            0,
            0x7,
            self.layout._FILE_OPEN,
            self.layout._OPEN_REPARSE_POINT
            | self.layout._SYNCHRONOUS_IO_NONALERT
            | (0x1 if directory else self.layout._NON_DIRECTORY_FILE),
            None,
            0,
        )
        result = ctypes.c_int32(result).value
        completed = ctypes.c_int32(int(status.status_or_pointer or 0) & 0xFFFFFFFF).value
        valid_handle = handle.value not in {None, 0, ctypes.c_void_p(-1).value}
        if result != 0 or completed != 0:
            if valid_handle:
                self.close(int(handle.value))
            code = self.boundary.nt.RtlNtStatusToDosError(result if result else completed)
            raise getattr(ctypes, "WinError")(int(code))
        if not valid_handle:
            raise RuntimeError("relative fixture open returned no handle")
        return int(handle.value)

    def read(self, parent: int, name: str) -> bytes:
        handle = self.open(parent, name)
        try:
            error, payload = self.boundary.files.ReadFile(handle, 4096)
            if error:
                raise getattr(ctypes, "WinError")(int(error))
            return payload
        finally:
            self.close(handle)


class NativeFailure(OSError):
    def __init__(self, code: int):
        super().__init__(code, "controlled native refusal")
        self.winerror = code


class Acl:
    def __init__(self):
        self.aces = []

    def AddAccessAllowedAceEx(self, _revision, flags, mask, sid):
        self.aces.append(((0, flags), mask, sid))

    def GetAceCount(self):
        return len(self.aces)

    def GetAce(self, index):
        return self.aces[index]


class Descriptor:
    def __init__(self):
        self.owner = None
        self.dacl = None
        self.control = 0

    def SetSecurityDescriptorOwner(self, owner, _defaulted):
        self.owner = owner

    def SetSecurityDescriptorDacl(self, present, dacl, _defaulted):
        assert present
        self.dacl = dacl

    def SetSecurityDescriptorControl(self, mask, value):
        self.control = (self.control & ~mask) | value

    def GetSecurityDescriptorControl(self):
        return self.control, 1

    def GetSecurityDescriptorOwner(self):
        return self.owner

    def GetSecurityDescriptorDacl(self):
        return self.dacl


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch):
    """Model OS boundaries while keeping actual files for partial-write cleanup."""
    state = SimpleNamespace(
        descriptors={},
        handles=[],
        events=[],
        fail=None,
        token_sid=provision.SYSTEM_SID,
        thread_error=1008,
        process_tokens=0,
        reparse=False,
    )
    constants = SimpleNamespace(
        TOKEN_QUERY=8,
        READ_CONTROL=0x20000,
        GENERIC_WRITE=0x40000000,
        CREATE_NEW=1,
        OPEN_EXISTING=3,
        FILE_ATTRIBUTE_TEMPORARY=0x100,
        FILE_FLAG_BACKUP_SEMANTICS=0x02000000,
        FILE_ATTRIBUTE_REPARSE_POINT=0x400,
    )
    # Pinned pywin32 312 exports file access rights from ntsecuritycon; its
    # win32con module deliberately has no FILE_READ_ATTRIBUTES attribute.
    file_access = SimpleNamespace(FILE_READ_ATTRIBUTES=0x80)

    class Handle:
        def __init__(self, path=None, descriptor=None, fd=None):
            self.path, self.descriptor, self.fd = path, descriptor, fd
            self.closed = False
            state.handles.append(self)

        def __int__(self):
            return id(self)

        def Close(self):
            assert not self.closed
            if self.fd is not None:
                os.close(self.fd)
            self.closed = True
            state.events.append(("close", self.path))

    def thread_token(*_args):
        if state.thread_error is not None:
            raise NativeFailure(state.thread_error)
        return Handle()

    def process_token(*_args):
        state.process_tokens += 1
        return Handle()

    def get_security(handle, *_args):
        state.events.append(("qualify", handle.path))
        if state.fail == "qualify-directory" and handle.path.is_dir():
            raise NativeFailure(5)
        if state.fail == "qualify-file" and not handle.path.is_dir():
            handle.descriptor.control = 0
        return handle.descriptor

    def assert_creation(attributes, *, inherit):
        assert attributes.bInheritHandle is False
        descriptor = attributes.SECURITY_DESCRIPTOR
        assert descriptor.owner == provision.SYSTEM_SID
        assert descriptor.control == 0x1000
        assert descriptor.dacl.aces == [
            ((0, 3 if inherit else 0), provision.FILE_ALL_ACCESS, provision.SYSTEM_SID)
        ]

    def mkdir(name, attributes):
        path = Path(name)
        assert_creation(attributes, inherit=True)
        state.events.append(("create-directory", path))
        if state.fail == "directory-create":
            raise NativeFailure(5)
        try:
            path.mkdir()
        except FileExistsError as exc:
            raise NativeFailure(183) from exc
        state.descriptors[path] = attributes.SECURITY_DESCRIPTOR

    def create(name, access, share, attributes, disposition, flags, template):
        path = Path(name)
        assert share == 0 and template is None
        if disposition == constants.CREATE_NEW:
            assert_creation(attributes, inherit=False)
            assert access == constants.GENERIC_WRITE | constants.READ_CONTROL
            assert flags == constants.FILE_ATTRIBUTE_TEMPORARY
            state.events.append(("create-file", path))
            if state.fail == "create-" + path.name:
                raise NativeFailure(5)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                raise NativeFailure(183) from exc
            state.descriptors[path] = attributes.SECURITY_DESCRIPTOR
            return Handle(path, attributes.SECURITY_DESCRIPTOR, fd)
        assert disposition == constants.OPEN_EXISTING
        assert attributes is None
        # GetSecurityInfo needs READ_CONTROL; GetFileInformationByHandle also
        # needs FILE_READ_ATTRIBUTES. Opening with READ_CONTROL alone is invalid.
        assert access == constants.READ_CONTROL | 0x80
        assert flags == constants.FILE_FLAG_BACKUP_SEMANTICS | 0x00200000
        state.events.append(("open-directory", path))
        if state.fail == "directory-open":
            raise NativeFailure(5)
        return Handle(path, state.descriptors[path])

    def resolve_handle(handle_id):
        assert isinstance(handle_id, int)
        handle = next(handle for handle in state.handles if int(handle) == handle_id)
        assert not handle.closed
        return handle

    def information(handle_id):
        resolve_handle(handle_id)
        if state.fail == "directory-attributes":
            raise NativeFailure(5)
        return (0x400 if state.reparse else 0,)

    def write(handle_id, payload):
        handle = resolve_handle(handle_id)
        assert ("qualify", handle.path) in state.events
        assert handle.descriptor.control == 0x1000
        state.events.append(("write", handle.path))
        if state.fail in ("write-" + handle.path.name, "short-write"):
            os.write(handle.fd, payload[:2])
            if state.fail == "short-write":
                return 0, 2
            raise NativeFailure(5)
        return 0, os.write(handle.fd, payload)

    security = SimpleNamespace(
        ACL=Acl,
        ACL_REVISION_DS=4,
        ACCESS_ALLOWED_ACE_TYPE=0,
        SECURITY_DESCRIPTOR=Descriptor,
        SE_DACL_PROTECTED=0x1000,
        SE_FILE_OBJECT=1,
        OWNER_SECURITY_INFORMATION=1,
        DACL_SECURITY_INFORMATION=4,
        ConvertStringSidToSid=lambda sid: sid,
        ConvertSidToStringSid=lambda sid: sid,
        GetSecurityInfo=get_security,
        OpenThreadToken=thread_token,
        OpenProcessToken=process_token,
        TokenUser=1,
        GetTokenInformation=lambda *_args: (state.token_sid, 0),
    )
    modules = {
        "pywintypes": SimpleNamespace(error=NativeFailure, SECURITY_ATTRIBUTES=SimpleNamespace),
        "win32con": constants,
        "ntsecuritycon": file_access,
        "win32api": SimpleNamespace(GetCurrentThread=lambda: 2, GetCurrentProcess=lambda: 1),
        "win32security": security,
        "win32file": SimpleNamespace(
            CreateDirectory=mkdir,
            CreateFile=create,
            WriteFile=write,
            GetFileInformationByHandle=information,
            FILE_FLAG_OPEN_REPARSE_POINT=0x00200000,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    state.security, state.constants = security, constants
    return state


def test_directory_attribute_right_uses_pinned_pywin32_module(tmp_path: Path, native) -> None:
    assert not hasattr(sys.modules["win32con"], "FILE_READ_ATTRIBUTES")
    assert sys.modules["ntsecuritycon"].FILE_READ_ATTRIBUTES == 0x80
    with provision._bootstrap_password_files(tmp_path):
        assert any(event[0] == "open-directory" for event in native.events)
    assert list(tmp_path.iterdir()) == []


def test_native_creates_final_system_acl_before_each_write_and_cleans_up(
    tmp_path: Path, native, capsys, caplog
) -> None:
    environment = dict(os.environ)
    with provision._bootstrap_password_files(tmp_path) as (password_file, passfile):
        password = password_file.read_text().strip()
        assert len(password) >= 48
        assert passfile.read_text() == (
            f"127.0.0.1:{provision.CONTRACT.postgresql_port}:*:postgres:{password}\n"
        )
        for path in (password_file, passfile):
            descriptor = native.descriptors[path]
            assert descriptor.owner == provision.SYSTEM_SID
            assert descriptor.dacl.aces == [
                ((0, 0), provision.FILE_ALL_ACCESS, provision.SYSTEM_SID)
            ]
            assert native.events.index(("create-file", path)) < native.events.index(
                ("qualify", path)
            )
            assert native.events.index(("qualify", path)) < native.events.index(("write", path))
    assert list(tmp_path.iterdir()) == []
    assert all(handle.closed for handle in native.handles)
    assert dict(os.environ) == environment
    captured = capsys.readouterr()
    assert password not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize(
    "failure",
    (
        "directory-create",
        "directory-open",
        "directory-attributes",
        "qualify-directory",
        "create-initdb.password",
        "create-pgpass",
        "qualify-file",
        "write-initdb.password",
        "write-pgpass",
        "short-write",
    ),
)
def test_creation_qualification_and_partial_write_failures_remove_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native, failure: str
) -> None:
    native.fail = failure
    generated = []
    original = provision.secrets.token_urlsafe

    def password(length):
        generated.append(True)
        return original(length)

    monkeypatch.setattr(provision.secrets, "token_urlsafe", password)
    with pytest.raises((OSError, provision.ProvisionError)):
        with provision._bootstrap_password_files(tmp_path):
            pytest.fail("failed credential must never be handed to a consumer")
    assert list(tmp_path.iterdir()) == []
    assert all(handle.closed for handle in native.handles)
    if failure in {
        "directory-create",
        "directory-open",
        "directory-attributes",
        "qualify-directory",
    }:
        assert generated == []
    if failure == "qualify-file":
        assert not any(event[0] == "write" for event in native.events)


@pytest.mark.parametrize("thread_error", (None, 1008, 5))
@pytest.mark.parametrize("identity", (provision.SYSTEM_SID, provision.ADMINISTRATORS_SID))
def test_effective_identity_is_required_before_creating_or_generating_secret(
    tmp_path: Path, native, thread_error: int | None, identity: str
) -> None:
    native.thread_error, native.token_sid = thread_error, identity
    if thread_error == 5 or identity != provision.SYSTEM_SID:
        with pytest.raises((NativeFailure, provision.ProvisionError)):
            with provision._bootstrap_password_files(tmp_path):
                pytest.fail("unqualified token accepted")
        assert list(tmp_path.iterdir()) == []
        assert not any(event[0].startswith("create") for event in native.events)
    else:
        with provision._bootstrap_password_files(tmp_path):
            pass
    assert native.process_tokens == int(thread_error == 1008)
    assert all(handle.closed for handle in native.handles)


@pytest.mark.parametrize(
    "change",
    (
        "owner",
        "null-owner",
        "unprotected",
        "null-dacl",
        "extra-ace",
        "inherited-ace",
        "wrong-flags",
        "wrong-mask",
        "deny",
        "service-sid",
    ),
)
def test_native_qualifier_rejects_every_broadened_descriptor(native, change: str) -> None:
    attributes = provision._bootstrap_security_attributes(inherit=False)
    descriptor = attributes.SECURITY_DESCRIPTOR
    if change == "owner":
        descriptor.owner = provision.ADMINISTRATORS_SID
    elif change == "null-owner":
        descriptor.owner = None
    elif change == "unprotected":
        descriptor.control = 0
    elif change == "null-dacl":
        descriptor.dacl = None
    elif change == "extra-ace":
        descriptor.dacl.aces.append(((0, 0), provision.FILE_GENERIC_READ, "S-1-1-0"))
    else:
        descriptor.dacl.aces = [
            (
                (
                    1 if change == "deny" else 0,
                    16 if change == "inherited-ace" else 3 if change == "wrong-flags" else 0,
                ),
                provision.FILE_GENERIC_READ
                if change == "wrong-mask"
                else provision.FILE_ALL_ACCESS,
                "S-1-5-80-123" if change == "service-sid" else provision.SYSTEM_SID,
            )
        ]
    handle = SimpleNamespace(path=Path("private-file"), descriptor=descriptor)
    with pytest.raises(provision.ProvisionError, match="differs"):
        provision._qualify_bootstrap_handle(handle, inherit=False)


def test_exclusive_create_never_truncates_existing_file(tmp_path: Path, native) -> None:
    path = tmp_path / "pgpass"
    path.write_text("existing synthetic fixture")
    with pytest.raises(NativeFailure) as captured:
        provision._write_bootstrap_secret(path, "new synthetic fixture")
    assert captured.value.winerror == 183
    assert path.read_text() == "existing synthetic fixture"
    assert not any(event[0] == "write" for event in native.events)


def test_directory_collision_is_bounded_and_never_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native
) -> None:
    monkeypatch.setattr(provision.secrets, "token_hex", lambda _size: "collision")
    existing = tmp_path / ".postgres-bootstrap-collision"
    existing.mkdir()
    (existing / "sentinel").write_text("preserve")
    with pytest.raises(provision.ProvisionError, match="unique"):
        with provision._bootstrap_password_files(tmp_path):
            pytest.fail("existing directory adopted")
    assert (existing / "sentinel").read_text() == "preserve"
    assert len([event for event in native.events if event[0] == "create-directory"]) == 10


def test_reparse_directory_refusal_closes_handle_before_cleanup(tmp_path: Path, native) -> None:
    native.reparse = True
    with pytest.raises(provision.ProvisionError, match="reparse"):
        with provision._bootstrap_password_files(tmp_path):
            pytest.fail("reparse directory accepted")
    assert list(tmp_path.iterdir()) == []
    assert all(handle.closed for handle in native.handles)


def test_baseexception_also_removes_secret_files(tmp_path: Path, native) -> None:
    with pytest.raises(KeyboardInterrupt):
        with provision._bootstrap_password_files(tmp_path):
            raise KeyboardInterrupt
    assert list(tmp_path.iterdir()) == []
    assert all(handle.closed for handle in native.handles)


@pytest.mark.parametrize(
    "name", ("", ".", "..", "../outside", "a/b", "a\\b", "C:child", "a\0b", "a" * 129)
)
def test_relative_fixture_read_refuses_paths_outside_one_bounded_leaf(name):
    with pytest.raises(ValueError):
        _RelativeCredentialRead._check_leaf(name)


def test_relative_fixture_codec_is_ready_before_restricted_token_boundary():
    # A fresh interpreter must not inherit this pytest process's codec cache.
    # Model Anonymous being unable to import Python files after construction.
    script = """
import builtins
import codecs
import ctypes
import encodings
import importlib.util
import sys
from types import SimpleNamespace
from deployment import windows_stage10_diagnostic_windows as layout

spec = importlib.util.spec_from_file_location("credential_fixture", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
codecs.unregister(encodings.search_function)
encodings._cache.clear()
sys.modules.pop("encodings.utf_16_le", None)
codecs.register(encodings.search_function)

def create(output, _access, _attributes, status, *_args):
    ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 123
    ctypes.cast(status, ctypes.POINTER(layout._IoStatus)).contents.status_or_pointer = 0
    return 0

layout._Native = lambda: SimpleNamespace(nt=SimpleNamespace(NtCreateFile=create))
probe = module._RelativeCredentialRead()
original_import = builtins.__import__

def restricted_import(name, *args, **kwargs):
    if name == "encodings.utf_16_le":
        raise ModuleNotFoundError("controlled unavailable codec")
    return original_import(name, *args, **kwargs)

builtins.__import__ = restricted_import
try:
    handle = probe.open(456, "anonymous-readable-control")
except LookupError:
    sys.exit(52)
if handle != 123:
    sys.exit(53)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 0, f"codec boundary proof rejected ({result.returncode})"


@pytest.mark.parametrize(
    ("result", "completed", "returned", "accepted"),
    ((0x103, 0, 123, False), (0, 0x103, 123, False), (0, 0, None, False), (0, 0, 123, True)),
)
def test_relative_fixture_read_never_accepts_pending_or_missing_handle(
    monkeypatch, result, completed, returned, accepted
):
    from deployment import windows_stage10_diagnostic_windows as layout

    closed = []

    def create(output, _access, _attributes, status, *_args):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = returned
        ctypes.cast(status, ctypes.POINTER(layout._IoStatus)).contents.status_or_pointer = completed
        return result

    probe = object.__new__(_RelativeCredentialRead)
    probe.layout = layout
    probe.boundary = SimpleNamespace(
        nt=SimpleNamespace(NtCreateFile=create, RtlNtStatusToDosError=lambda _status: 997),
        close=closed.append,
    )
    monkeypatch.setattr(
        ctypes, "WinError", lambda code: OSError(code, "controlled native status"), raising=False
    )
    if accepted:
        handle = probe.open(456, ".postgres-bootstrap-fixture")
        assert handle == returned
        probe.close(handle)
    else:
        with pytest.raises((OSError, RuntimeError)):
            probe.open(456, ".postgres-bootstrap-fixture")
    assert closed == ([returned] if returned is not None else [])


@pytest.mark.skipif(
    sys.platform != "win32"
    or os.environ.get("CRYPTOHUNTER_RUN_WINDOWS_BOOTSTRAP_ACL_INTEGRATION") != "1",
    reason="requires an explicitly selected isolated LocalSystem Windows test process",
)
@pytest.mark.parametrize("crash", (False, True))
def test_native_windows_private_credentials_deny_anonymous_and_survive_crash_privately(
    tmp_path: Path, crash: bool
) -> None:
    """No PostgreSQL, SCM, MSI or enrollment; only disposable synthetic files."""
    import win32api
    import win32security

    provision._require_bootstrap_system()
    # Deliberately broad inheritance must not enter any bootstrap descriptor.
    provision._protect(
        tmp_path,
        {
            provision.SYSTEM_SID: provision.FILE_ALL_ACCESS,
            "S-1-1-0": provision.FILE_ALL_ACCESS,
            "S-1-5-7": provision.FILE_ALL_ACCESS,  # ANONYMOUS LOGON is not always in Everyone.
        },
        inherit=True,
    )
    control = tmp_path / "anonymous-readable-control"
    control.write_bytes(b"synthetic readable control")
    script = (
        "import os,sys; from pathlib import Path; "
        "from deployment.windows_installer import provision; "
        "context=provision._bootstrap_password_files(Path(sys.argv[1])); "
        "paths=context.__enter__(); "
        "os._exit(23)"
        if crash
        else None
    )
    if crash:
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=30)
        assert result.returncode == 23
        directory = next(tmp_path.glob(".postgres-bootstrap-*"))
        paths = directory / "initdb.password", directory / "pgpass"
    else:
        context = provision._bootstrap_password_files(tmp_path)
        paths = context.__enter__()
        directory = paths[0].parent
    try:
        provision._qualify_bootstrap_directory(directory)
        for path in paths:
            assert path.is_file()
            # Inspect each native file descriptor; do not repair its ACL.
            import win32con
            import win32file

            handle = win32file.CreateFile(
                str(path), win32con.READ_CONTROL, 0, None, win32con.OPEN_EXISTING, 0, None
            )
            try:
                provision._qualify_bootstrap_handle(handle, inherit=False)
            finally:
                handle.Close()
        probe = _RelativeCredentialRead()
        public_parent = probe.pin_directory(tmp_path)
        try:
            private_parent = probe.pin_directory(directory)
            try:
                win32security.ImpersonateAnonymousToken(win32api.GetCurrentThread())
                try:
                    # Host ancestor traversal is irrelevant: both opens are
                    # relative to already pinned, fixture-only directory handles.
                    control_matches = (
                        probe.read(int(public_parent), control.name)
                        == b"synthetic readable control"
                    )
                    assert control_matches
                    unexpected = None
                    try:
                        with pytest.raises(PermissionError) as denied_directory:
                            unexpected = probe.open(
                                int(public_parent), directory.name, directory=True
                            )
                        assert denied_directory.value.winerror == 5
                    finally:
                        if unexpected is not None:
                            probe.close(unexpected)
                    # Even an attacker given the private-directory capability
                    # must still be denied each file by its own SYSTEM-only DACL.
                    for path in paths:
                        with pytest.raises(PermissionError) as denied_file:
                            probe.read(int(private_parent), path.name)
                        assert denied_file.value.winerror == 5
                finally:
                    win32security.RevertToSelf()
            finally:
                private_parent.Close()
        finally:
            public_parent.Close()
    finally:
        if crash:
            provision.shutil.rmtree(directory)
        else:
            context.__exit__(None, None, None)
    control.unlink()
    assert not list(tmp_path.glob(".postgres-bootstrap-*"))


def test_cli_diagnostic_failure_never_renders_original_credential_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, caplog
) -> None:
    """Exercise the CLI's error boundary with every privileged operation replaced."""
    credential = provision.secrets.token_urlsafe(48)

    def fail_install(_program_files, _program_data, stage):
        stage("POSTGRES_INITDB")
        raise RuntimeError("private provider diagnostic: " + credential)

    def fail_diagnostic(*_args):
        raise PermissionError("private diagnostic publication failure: " + credential)

    monkeypatch.setattr(provision, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr(provision, "clear_safe_failure", lambda: None)
    monkeypatch.setattr(provision, "install", fail_install)
    monkeypatch.setattr(provision, "write_safe_failure", fail_diagnostic)
    assert (
        provision.main(
            [
                "install",
                "--program-files",
                str(tmp_path / "files"),
                "--program-data",
                str(tmp_path / "data"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert credential not in captured.out + captured.err + caplog.text
    assert "first_exception=RuntimeError" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "files").exists() and not (tmp_path / "data").exists()
