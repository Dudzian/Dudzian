"""Syntetyczne modele granicy Windows i odrębne izolowane próby plików NTFS."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import importlib
import os
from pathlib import Path, PureWindowsPath
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import windows_stage10_diagnostic_windows as publisher
from deployment.windows_stage10_diagnostic_output import DiagnosticOutputError


class ModelNative:
    def __init__(self) -> None:
        self.entries: dict[str, bytes] = {"unrelated.json": b"chronione dane"}
        self.handles: dict[int, str] = {}
        self.calls: list[tuple[str, Any]] = []
        self.failure: str | None = None
        self.before_publish = None
        self.after_write = None
        self.directory_open = False

    @contextmanager
    def directory(self, root: Path):
        self.directory_open = True
        self.calls.append(("directory", root))
        try:
            yield {"handle": 7, "path": root}
        finally:
            self.directory_open = False

    def exists(self, context, name):
        assert self.directory_open and context["handle"] == 7
        self.calls.append(("exists", name))
        return name in self.entries

    def create(self, context, name):
        assert self.directory_open and context["handle"] == 7
        self.calls.append(("create", name))
        if name in self.entries:
            raise FileExistsError(name)
        if self.failure == "create":
            raise OSError("syntetyczna odmowa utworzenia")
        self.entries[name] = b""
        self.handles[42] = name
        return 42

    def verify_file(self, context, name, handle):
        assert context["handle"] == 7 and self.handles[handle] == name
        self.calls.append(("verify_file", name))
        if self.failure == "verify_file":
            raise OSError("syntetyczna odmowa tożsamości pliku")

    def verify_directory(self, context):
        assert context["handle"] == 7
        self.calls.append(("verify_directory", context["path"]))
        if self.failure == "verify_directory":
            raise OSError("syntetyczna podmiana katalogu lub ACL")

    def write(self, handle, raw):
        assert self.directory_open
        self.calls.append(("write", handle))
        self.entries[self.handles[handle]] = raw[:3]
        if self.failure == "interrupt":
            raise KeyboardInterrupt("syntetyczne przerwanie zapisu")
        if self.failure == "write":
            raise OSError("syntetyczny błąd zapisu")
        self.entries[self.handles[handle]] = raw
        if self.after_write is not None:
            self.after_write()

    def publish(self, context, handle, name):
        assert self.directory_open and context["handle"] == 7
        self.calls.append(("publish", name))
        if self.before_publish is not None:
            self.before_publish()
        if self.failure == "publish":
            raise OSError("syntetyczny błąd publikacji")
        if name in self.entries:
            raise FileExistsError(name)
        source = self.handles[handle]
        self.entries[name] = self.entries.pop(source)

    def close(self, handle):
        assert self.directory_open
        self.calls.append(("close", handle))
        self.handles.pop(handle)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ModelNative:
    native = ModelNative()
    monkeypatch.setattr(publisher, "_Native", lambda: native)
    return native


def test_new_complete_report_uses_pinned_handle_until_commit_and_close(tmp_path, model):
    raw = b'{"status":"BLOCKED"}\n'
    publisher.publish_windows(tmp_path, "new-report.json", raw)
    assert model.entries == {"unrelated.json": b"chronione dane", "new-report.json": raw}
    assert not model.directory_open and not model.handles
    operations = [name for name, _ in model.calls]
    assert operations == [
        "directory",
        "exists",
        "create",
        "verify_file",
        "write",
        "verify_file",
        "verify_directory",
        "publish",
        "close",
    ]


@pytest.mark.parametrize(
    "name", ["receipt.json", "contract.json", "manifest.json", "old-report.json"]
)
def test_existing_artifact_is_refused_without_any_mutation(tmp_path, model, name):
    original = b'{"status":"PASS","history":"old"}'
    model.entries[name] = original
    before = dict(model.entries)
    with pytest.raises(DiagnosticOutputError, match="istnieje"):
        publisher.publish_windows(tmp_path, name, b'{"status":"BLOCKED"}')
    assert model.entries == before
    assert not model.handles
    assert all(operation not in {"create", "write", "publish"} for operation, _ in model.calls)


@pytest.mark.parametrize(
    "failure", ["create", "verify_file", "write", "publish", "verify_directory"]
)
def test_failure_keeps_unrelated_data_and_never_publishes_final_name(tmp_path, model, failure):
    model.failure = failure
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(tmp_path, "new-report.json", b'{"status":"BLOCKED"}')
    assert model.entries["unrelated.json"] == b"chronione dane"
    assert "new-report.json" not in model.entries
    assert not model.handles and not model.directory_open
    leftovers = set(model.entries) - {"unrelated.json"}
    assert all(name.startswith(".stage10-") and name.endswith(".partial") for name in leftovers)


def test_interruption_preserves_owned_partial_without_path_cleanup(tmp_path, model):
    model.failure = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        publisher.publish_windows(tmp_path, "new-report.json", b'{"status":"BLOCKED"}')
    assert "new-report.json" not in model.entries
    assert model.entries["unrelated.json"] == b"chronione dane"
    assert len(model.entries) == 2
    assert not model.handles and not model.directory_open


def test_racing_destination_is_not_overwritten(tmp_path, model):
    original = b'{"status":"PASS","history":"race"}'
    model.before_publish = lambda: model.entries.update({"new-report.json": original})
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(tmp_path, "new-report.json", b'{"status":"BLOCKED"}')
    assert model.entries["new-report.json"] == original
    assert any(name.endswith(".partial") for name in model.entries)
    assert model.entries["unrelated.json"] == b"chronione dane"


def test_root_change_during_write_is_refused_before_commit(tmp_path, model):
    model.after_write = lambda: setattr(model, "failure", "verify_directory")
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(tmp_path, "new-report.json", b'{"status":"BLOCKED"}')
    assert "new-report.json" not in model.entries
    assert not any(name == "publish" for name, _ in model.calls)
    assert model.entries["unrelated.json"] == b"chronione dane"


@pytest.mark.parametrize(
    "name",
    [
        "../input.json",
        "/absolute.json",
        "a\\b.json",
        "a:b.json",
        ".",
        "CON.json",
        "report.",
        "file\x00.json",
        "żółć.json",
    ],
)
def test_unsafe_basename_refuses_before_native_access(tmp_path, model, name):
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(tmp_path, name, b"{}")
    assert not model.calls


def test_rename_record_has_64_bit_abi_and_does_not_enable_replacement():
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("Model ABI 64-bit Windows")
    record = publisher._rename_record(1234, "report.json")
    assert record.replace_if_exists == 0
    assert record.root_directory == 1234
    assert record.file_name_length == len("report.json".encode("utf-16-le"))
    assert type(record).root_directory.offset == 8
    assert type(record).file_name_length.offset == 16
    assert type(record).file_name.offset == 20
    assert ctypes.string_at(
        ctypes.addressof(record) + 20, record.file_name_length
    ) == "report.json".encode("utf-16-le")


@pytest.mark.parametrize("create", [False, True])
def test_native_open_is_relative_to_verified_root_and_never_uses_open_always(create):
    native = publisher._Native.__new__(publisher._Native)
    observed = {}

    def create_file(
        handle,
        access,
        attributes,
        status,
        allocation,
        file_attributes,
        share,
        disposition,
        options,
        ea,
        ea_size,
    ):
        attrs = ctypes.cast(attributes, ctypes.POINTER(publisher._ObjectAttributes)).contents
        string = attrs.name.contents
        observed.update(
            root=attrs.root,
            attributes=attrs.attributes,
            name=ctypes.string_at(string.buffer, string.length).decode("utf-16-le"),
            share=share,
            disposition=disposition,
            options=options,
        )
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p)).contents.value = 42
        return 0

    native.nt = SimpleNamespace(NtCreateFile=create_file)
    assert native._relative({"handle": 123}, "temporary.partial", create=create) == 42
    assert observed["root"] == 123
    assert observed["name"] == "temporary.partial"
    assert observed["attributes"] & publisher._OBJ_DONT_REPARSE
    assert observed["disposition"] == (publisher._FILE_CREATE if create else publisher._FILE_OPEN)
    assert observed["share"] == (0 if create else 7)
    assert observed["options"] & publisher._OPEN_REPARSE_POINT


@pytest.mark.parametrize("status", [0xC0000034, 0xC000003A, 0xC000000F])
def test_native_probe_only_treats_specific_missing_statuses_as_absence(status):
    native = publisher._Native.__new__(publisher._Native)
    native.nt = SimpleNamespace(NtCreateFile=lambda *_args: ctypes.c_int32(status).value)
    assert native._relative({"handle": 123}, "report.json", create=False) is None


def test_native_probe_does_not_treat_access_denied_as_absence():
    native = publisher._Native.__new__(publisher._Native)
    native.nt = SimpleNamespace(
        NtCreateFile=lambda *_args: ctypes.c_int32(0xC0000022).value,
        RtlNtStatusToDosError=lambda _status: 5,
    )
    with pytest.raises(OSError):
        native._relative({"handle": 123}, "report.json", create=False)


def test_native_publication_uses_same_handle_and_relative_nonreplacing_rename():
    native = publisher._Native.__new__(publisher._Native)
    observed = {}

    def rename(handle, io_status, record, size, information_class):
        assert io_status._obj.information == 0
        raw = ctypes.string_at(record, size)
        observed.update(handle=handle, information_class=information_class, raw=raw)
        return 0  # STATUS_SUCCESS

    native.nt = SimpleNamespace(NtSetInformationFile=rename)
    native.publish({"handle": 7}, 42, "report.json")
    assert observed["handle"] == 42
    assert observed["information_class"] == publisher._FILE_RENAME_INFORMATION == 10
    assert observed["raw"][:4] == bytes(4)
    assert int.from_bytes(observed["raw"][8:16], "little") == 7
    assert observed["raw"][20:42] == "report.json".encode("utf-16-le")


@pytest.mark.parametrize(
    "status,winerror",
    [(0xC0000035, 183), (0xC0000022, 5), (0x00000103, None)],
)
def test_native_rename_refuses_collisions_errors_and_pending_status(status, winerror):
    native = publisher._Native.__new__(publisher._Native)
    seen = []

    def rename(handle, io_status, record, size, information_class):
        seen.append((handle, information_class))
        return ctypes.c_int32(status).value

    native.nt = SimpleNamespace(
        NtSetInformationFile=rename,
        RtlNtStatusToDosError=lambda _status: winerror,
    )
    if winerror is None:
        with pytest.raises(DiagnosticOutputError, match="Nieukończona"):
            native.publish({"handle": 7}, 42, "report.json")
    else:
        with pytest.raises(OSError) as caught:
            native.publish({"handle": 7}, 42, "report.json")
        assert caught.value.errno == winerror
    assert seen == [(42, publisher._FILE_RENAME_INFORMATION)]


def test_native_writer_uses_complete_write_loop_then_flushes_once():
    native = publisher._Native.__new__(publisher._Native)
    output = bytearray()
    flushes = []

    def write(handle, buffer, size, written, overlapped):
        assert handle == 42 and overlapped is None
        count = min(size, 3)
        output.extend(ctypes.string_at(buffer, count))
        ctypes.cast(written, ctypes.POINTER(ctypes.c_uint32)).contents.value = count
        return 1

    native.kernel = SimpleNamespace(
        WriteFile=write,
        FlushFileBuffers=lambda handle: flushes.append(handle) or 1,
    )
    raw = b'{"status":"BLOCKED"}\n'
    native.write(42, raw)
    assert bytes(output) == raw
    assert flushes == [42]


@pytest.mark.parametrize("written_count", [0, 4])
def test_native_writer_rejects_zero_or_oversized_byte_count(written_count):
    native = publisher._Native.__new__(publisher._Native)

    def write(handle, buffer, size, written, overlapped):
        ctypes.cast(written, ctypes.POINTER(ctypes.c_uint32)).contents.value = written_count
        return 1

    native.kernel = SimpleNamespace(
        WriteFile=write,
        FlushFileBuffers=lambda _handle: pytest.fail("Nie flushuj niepełnego zapisu"),
    )
    with pytest.raises(DiagnosticOutputError, match="Niepełny"):
        native.write(42, b"abc")


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_native_io_failure_prevents_success(failure, monkeypatch):
    native = publisher._Native.__new__(publisher._Native)

    def write(handle, buffer, size, written, overlapped):
        ctypes.cast(written, ctypes.POINTER(ctypes.c_uint32)).contents.value = size
        return 0 if failure == "write" else 1

    native.kernel = SimpleNamespace(
        WriteFile=write,
        FlushFileBuffers=lambda _handle: 0,
    )
    monkeypatch.setattr(native, "_failure", lambda: OSError("syntetyczna odmowa I/O"))
    with pytest.raises(OSError, match="I/O"):
        native.write(42, b"abc")


def test_directory_chain_remains_pinned_without_delete_sharing_and_narrow_root_trust():
    native = publisher._Native.__new__(publisher._Native)
    opened = []
    closed = []
    qualified = []

    class Handle:
        def __init__(self, number):
            self.number = number

        def __int__(self):
            return self.number

        def Close(self):
            closed.append(self.number)

    def open_directory(path, access, share, security, disposition, options, template):
        handle = Handle(len(opened) + 1)
        opened.append((handle, path, access, share, disposition, options))
        return handle

    def qualify(handle, path, _native, trusted, *, directory, custody):
        assert not closed
        qualified.append((int(handle), str(path), trusted, directory, custody))

    native.files = SimpleNamespace(CreateFile=open_directory)
    native.fs = SimpleNamespace(
        _qualify_local_path=lambda _root, _files: None,
        _qualify=qualify,
    )
    native.native = (native.files, None, None)
    trusted = frozenset({"current-token", publisher._SYSTEM_SID, publisher._ADMINISTRATORS_SID})
    native._trusted = lambda: trusted
    root = PureWindowsPath("C:\\Users\\diagnostics")
    with native.directory(root) as context:
        assert len(opened) == 3
        assert context["handle"] == 3
        native.verify_directory(context)
        assert not closed
    assert closed == [3, 2, 1]
    for handle, path, access, share, disposition, options in opened:
        assert access & publisher._LIST_DIRECTORY
        assert access & publisher._READ_CONTROL
        assert not share & 4  # FILE_SHARE_DELETE jest zabronione.
        assert disposition == 3  # OPEN_EXISTING; katalogi nie są tworzone.
        assert options & publisher._OPEN_REPARSE_POINT
    for _, path, allowed, directory, custody in qualified:
        assert directory is True
        if path == str(root):
            assert custody is True and allowed == trusted
            assert publisher._TRUSTED_INSTALLER_SID not in allowed
        else:
            assert custody is False
            assert allowed == trusted | {publisher._TRUSTED_INSTALLER_SID}


@pytest.fixture
def native_root(tmp_path: Path) -> Path:
    if os.name != "nt":
        pytest.skip("Izolowana próba NTFS wymaga natywnego Windows")
    security = importlib.import_module("win32security")
    api = importlib.import_module("win32api")
    token = security.OpenProcessToken(api.GetCurrentProcess(), 0x0008)
    try:
        user = security.GetTokenInformation(token, security.TokenUser)
        current = user[0] if isinstance(user, tuple) else user
    finally:
        token.Close()
    root = tmp_path / "approved"
    root.mkdir()
    dacl = security.ACL()
    for sid in (
        current,
        security.ConvertStringSidToSid(publisher._SYSTEM_SID),
        security.ConvertStringSidToSid(publisher._ADMINISTRATORS_SID),
    ):
        dacl.AddAccessAllowedAceEx(security.ACL_REVISION, 3, 0x1F01FF, sid)
    security.SetNamedSecurityInfo(
        str(root),
        security.SE_FILE_OBJECT,
        security.OWNER_SECURITY_INFORMATION
        | security.DACL_SECURITY_INFORMATION
        | security.PROTECTED_DACL_SECURITY_INFORMATION,
        current,
        None,
        dacl,
        None,
    )
    return root


def test_native_windows_publication_is_complete_and_collision_preserves_bytes(native_root):
    raw = b'{"status":"BLOCKED"}\n'
    publisher.publish_windows(native_root, "report.json", raw)
    assert (native_root / "report.json").read_bytes() == raw
    assert list(native_root.iterdir()) == [native_root / "report.json"]
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(native_root, "report.json", b'{"status":"PASS"}')
    assert (native_root / "report.json").read_bytes() == raw


def test_native_windows_missing_directory_is_not_created(native_root):
    missing = native_root / "missing"
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(missing, "report.json", b"{}")
    assert not missing.exists()


def test_native_windows_junction_root_is_rejected_without_changing_target(native_root, tmp_path):
    junction = tmp_path / "junction"
    command = Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"
    subprocess.run(
        [str(command), "/d", "/c", "mklink", "/J", str(junction), str(native_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(junction, "report.json", b"{}")
    assert list(native_root.iterdir()) == []


def test_native_windows_symlink_preserves_referenced_input(native_root, tmp_path):
    original = tmp_path / "input.json"
    original.write_bytes(b"chroniony input")
    output = native_root / "report.json"
    try:
        output.symlink_to(original)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Token testowy nie ma uprawnienia tworzenia symlinków")
        raise
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(native_root, "report.json", b"{}")
    assert output.is_symlink()
    assert original.read_bytes() == b"chroniony input"


def test_native_windows_unsafe_output_acl_is_refused(native_root):
    security = importlib.import_module("win32security")
    dacl = security.ACL()
    dacl.AddAccessAllowedAceEx(
        security.ACL_REVISION,
        3,
        0x1F01FF,
        security.ConvertStringSidToSid("S-1-1-0"),
    )
    security.SetNamedSecurityInfo(
        str(native_root),
        security.SE_FILE_OBJECT,
        security.DACL_SECURITY_INFORMATION | security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        dacl,
        None,
    )
    with pytest.raises(DiagnosticOutputError):
        publisher.publish_windows(native_root, "report.json", b"{}")
    assert list(native_root.iterdir()) == []
