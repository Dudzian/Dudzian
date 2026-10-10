"""Native cross-process custody exclusion on POSIX and Windows."""

from __future__ import annotations

import errno
from contextlib import contextmanager
import importlib
import multiprocessing
import os
from pathlib import Path, PureWindowsPath
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from bot_core import local_signing_custody as custody
from tests.security._local_signing_platform import requires_native_custody_locking


@pytest.fixture
def descriptor_only_adapter(monkeypatch):
    """Isolate byte-lock unit tests; native NTFS tests below use the real API."""

    @contextmanager
    def opened(path, *, write=False, create=False):
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    monkeypatch.setattr(custody, "_open_custody_descriptor", opened)


def _lock_worker(path, exclusive, entered, release, outcome):
    try:
        with custody._custody_lock(Path(path), exclusive=exclusive):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test release barrier timed out")
        outcome.put("released")
    except Exception as exc:
        outcome.put(type(exc).__name__)
        raise


@requires_native_custody_locking
@pytest.mark.parametrize("reader", [False, True])
@pytest.mark.parametrize("termination", ["normal", "abrupt"])
def test_native_two_process_exclusion_and_os_release(tmp_path, reader, termination):
    context = multiprocessing.get_context("spawn")
    first_entered, second_entered = context.Event(), context.Event()
    first_release, second_release = context.Event(), context.Event()
    outcomes = context.Queue()
    first = context.Process(
        target=_lock_worker, args=(str(tmp_path), True, first_entered, first_release, outcomes)
    )
    second = context.Process(
        target=_lock_worker,
        args=(str(tmp_path), not reader, second_entered, second_release, outcomes),
    )
    try:
        first.start()
        assert first_entered.wait(20)
        second.start()
        assert not second_entered.wait(0.3)
        if termination == "normal":
            first_release.set()
            assert outcomes.get(timeout=10) == "released"
        else:
            first.terminate()
        first.join(10)
        assert not first.is_alive()
        assert second_entered.wait(20)
        second_release.set()
        assert outcomes.get(timeout=10) == "released"
        second.join(10)
        assert second.exitcode == 0
    finally:
        for process in (first, second):
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(10)
        outcomes.close()


@requires_native_custody_locking
def test_exception_releases_native_custody_lock(tmp_path):
    with pytest.raises(RuntimeError, match="injected"):
        with custody._custody_lock(tmp_path, exclusive=True):
            raise RuntimeError("injected")
    # A different process must acquire it; a same-process probe is insufficient.
    context = multiprocessing.get_context("spawn")
    entered, release, outcome = context.Event(), context.Event(), context.Queue()
    release.set()
    process = context.Process(
        target=_lock_worker, args=(str(tmp_path), True, entered, release, outcome)
    )
    try:
        process.start()
        assert entered.wait(20)
        assert outcome.get(timeout=10) == "released"
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(10)
        outcome.close()


@pytest.mark.parametrize("exclusive", [False, True])
def test_windows_adapter_uses_one_byte_exclusive_ownership_and_explicit_unlock(
    tmp_path, monkeypatch, exclusive, descriptor_only_adapter
):
    calls = []

    def locking(descriptor, mode, length):
        calls.append((mode, length, os.lseek(descriptor, 0, os.SEEK_CUR)))

    windows = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", windows)
    monkeypatch.setattr(
        custody,
        "os",
        SimpleNamespace(
            **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}, **{}
        ),
    )
    monkeypatch.setattr(custody.os, "name", "nt")
    with pytest.raises(RuntimeError):
        with custody._custody_lock(tmp_path, exclusive=exclusive):
            raise RuntimeError("injected")
    assert calls == [(2, 1, 0), (0, 1, 0)]
    assert (tmp_path / custody._LOCK_FILENAME).read_bytes() == b"\0"


@pytest.mark.parametrize("error", [errno.EIO, errno.EACCES])
def test_windows_adapter_lock_error_never_yields_unlocked(
    tmp_path, monkeypatch, error, descriptor_only_adapter
):
    def locking(descriptor, mode, length):
        raise OSError(error, "injected lock failure")

    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    )
    platform_os = SimpleNamespace(
        **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}
    )
    platform_os.name = "nt"
    monkeypatch.setattr(custody, "os", platform_os)
    clock = iter([0.0, 6.0])
    monkeypatch.setattr(custody, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(custody.LocalSigningCustodyError):
        with custody._custody_lock(tmp_path, exclusive=True):
            pytest.fail("unlocked fallback")


@pytest.mark.skipif(os.name != "posix", reason="Unix mode bits are POSIX-specific")
def test_posix_custody_lock_rejects_world_readable_file(tmp_path):
    path = tmp_path / custody._LOCK_FILENAME
    path.write_bytes(b"\0")
    path.chmod(0o644)
    with pytest.raises(custody.LocalSigningCustodyError, match="permissions"):
        with custody._custody_lock(tmp_path, exclusive=True):
            pytest.fail("unsafe permissions")


def test_unsupported_platform_has_no_unlocked_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(custody, "os", SimpleNamespace(name="unsupported"))
    with pytest.raises(custody.LocalSigningCustodyError, match="unavailable"):
        with custody._custody_lock(tmp_path, exclusive=True):
            pytest.fail("unsupported platform")


@requires_native_custody_locking
def test_delayed_holder_allows_waiter_after_release(tmp_path):
    context = multiprocessing.get_context("spawn")
    entered, release, outcomes = context.Event(), context.Event(), context.Queue()
    holder = context.Process(
        target=_lock_worker, args=(str(tmp_path), True, entered, release, outcomes)
    )
    try:
        holder.start()
        assert entered.wait(20)
        # Delay a genuine holder, rather than substituting the native lock/clock.
        import threading

        timer = threading.Timer(1.0, release.set)
        timer.start()
        started = time.monotonic()
        with custody._custody_lock(tmp_path, exclusive=True):
            assert release.is_set()
            assert time.monotonic() - started >= 0.8
        timer.join(5)
        assert outcomes.get(timeout=10) == "released"
        holder.join(10)
        assert holder.exitcode == 0
    finally:
        release.set()
        if holder.is_alive():
            holder.terminate()
        holder.join(10)
        outcomes.close()


@pytest.mark.skipif(os.name != "nt", reason="actual Windows five-second acquisition budget")
def test_native_windows_timeout_is_retryable_and_never_enters_unlocked(tmp_path):
    context = multiprocessing.get_context("spawn")
    entered, release, outcomes = context.Event(), context.Event(), context.Queue()
    holder = context.Process(
        target=_lock_worker, args=(str(tmp_path), True, entered, release, outcomes)
    )
    try:
        holder.start()
        assert entered.wait(20)
        started = time.monotonic()
        with pytest.raises(custody.LocalSigningCustodyBusy, match="timed out"):
            with custody._custody_lock(tmp_path, exclusive=True):
                pytest.fail("timeout granted ownership")
        assert 4.8 <= time.monotonic() - started < 8.0
        assert not release.is_set() and holder.is_alive()
        release.set()
        assert outcomes.get(timeout=10) == "released"
        holder.join(10)
        with custody._custody_lock(tmp_path, exclusive=True):
            pass  # Busy left no latch or surviving ownership.
    finally:
        release.set()
        if holder.is_alive():
            holder.terminate()
        holder.join(10)
        outcomes.close()


@pytest.mark.skipif(os.name != "nt", reason="native NTFS DACL qualification")
@pytest.mark.parametrize("target", ["directory", "lock", "metadata"])
def test_native_windows_rejects_untrusted_write_dacl(tmp_path, target):
    from bot_core.windows_custody_filesystem import open_custody_file

    security = importlib.import_module("win32security")
    security_flags = (
        security.PROTECTED_DACL_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION
    )
    path = (
        tmp_path
        if target == "directory"
        else tmp_path / (custody._LOCK_FILENAME if target == "lock" else "custody.json")
    )
    if target != "directory":
        path.write_bytes(b"\0")
    original = security.GetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT, 0x4)
    dacl = security.ACL()
    api = importlib.import_module("win32api")
    token = security.OpenProcessToken(api.GetCurrentProcess(), 0x8)
    try:
        user = security.GetTokenInformation(token, security.TokenUser)
        sid = user[0] if isinstance(user, tuple) else user
    finally:
        token.Close()
    dacl.AddAccessAllowedAce(2, 0x1F01FF, sid)
    dacl.AddAccessAllowedAce(2, 0x1F01FF, security.ConvertStringSidToSid("S-1-1-0"))
    security.SetNamedSecurityInfo(
        str(path), security.SE_FILE_OBJECT, security_flags, None, None, dacl, None
    )
    try:
        with pytest.raises(custody.LocalSigningCustodyError, match="DACL"):
            if target == "metadata":
                with open_custody_file(path):
                    pytest.fail("untrusted metadata opened")
            else:
                with custody._custody_lock(tmp_path, exclusive=True):
                    pytest.fail("untrusted object locked")
    finally:
        security.SetNamedSecurityInfo(
            str(path),
            security.SE_FILE_OBJECT,
            security_flags,
            None,
            None,
            original.GetSecurityDescriptorDacl(),
            None,
        )


@pytest.mark.skipif(os.name != "nt", reason="native NTFS junction and hardlink qualification")
@pytest.mark.parametrize("redirection", ["junction", "hardlink"])
def test_native_windows_rejects_redirected_custody_objects(tmp_path, redirection):
    directory = tmp_path / "custody"
    directory.mkdir()
    if redirection == "junction":
        link = tmp_path / "junction"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(directory)],
            capture_output=True,
            timeout=10,
            check=True,
        )
        assert result.returncode == 0
        path = link / "child"
        (directory / "child").mkdir()
    else:
        file = directory / custody._LOCK_FILENAME
        file.write_bytes(b"\0")
        os.link(file, tmp_path / "alias")
        path = directory
    try:
        with pytest.raises(custody.LocalSigningCustodyError):
            with custody._custody_lock(path, exclusive=True):
                pytest.fail("redirected object locked")
    finally:
        if redirection == "junction":
            link.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="native NTFS opened object identity")
def test_native_windows_detects_swap_between_path_check_and_open(tmp_path, monkeypatch):
    files = importlib.import_module("win32file")
    original = files.CreateFile
    path = tmp_path / custody._LOCK_FILENAME
    path.write_bytes(b"old")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"new")

    def swapped(name, *args):
        if name == str(path):
            os.replace(replacement, path)
        return original(name, *args)

    monkeypatch.setattr(files, "CreateFile", swapped)
    with pytest.raises(custody.LocalSigningCustodyError, match="changed before native open"):
        with custody._custody_lock(tmp_path, exclusive=True):
            pytest.fail("substituted object locked")


@pytest.mark.skipif(os.name != "nt", reason="native NTFS share-delete exclusion")
def test_native_windows_pins_directory_and_lock_against_replacement(tmp_path):
    directory = tmp_path / "custody"
    directory.mkdir()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"untrusted")
    with custody._custody_lock(directory, exclusive=True):
        with pytest.raises(OSError):
            os.replace(replacement, directory / custody._LOCK_FILENAME)
        with pytest.raises(OSError):
            directory.rename(tmp_path / "moved")
    assert replacement.read_bytes() == b"untrusted"


def _acl_model(*, owner="owner", aces=(), null=False):
    dacl = None if null else SimpleNamespace(GetAceCount=lambda: len(aces), GetAce=aces.__getitem__)
    descriptor = SimpleNamespace(
        GetSecurityDescriptorOwner=lambda: owner, GetSecurityDescriptorDacl=lambda: dacl
    )
    return SimpleNamespace(
        SE_FILE_OBJECT=1, GetSecurityInfo=lambda *args: descriptor, ConvertSidToStringSid=str
    )


@pytest.mark.parametrize("custody_object", [False, True])
@pytest.mark.parametrize("right", [0x10000, 0x40000, 0x80000, 0x10000000, 0x40000000, 0x2, 0x100])
def test_ntfs_acl_policy_rejects_untrusted_mutation(right, custody_object):
    from bot_core.windows_custody_filesystem import _qualify_acl

    security = _acl_model(aces=(((0, 0), right, "untrusted"),))
    with pytest.raises(custody.LocalSigningCustodyError, match="untrusted modification"):
        _qualify_acl(object(), security, frozenset({"owner"}), custody=custody_object)


@pytest.mark.parametrize("defect", ["owner", "null", "unsupported", "object_write"])
def test_ntfs_acl_policy_rejects_missing_or_unqualified_authority(defect):
    from bot_core.windows_custody_filesystem import _qualify_acl

    security = _acl_model(
        owner="other" if defect == "owner" else "owner",
        null=defect == "null",
        aces=(((9, 0), 0, "owner"),)
        if defect == "unsupported"
        else ((((5, 0), 0x2, None, None, "untrusted"),) if defect == "object_write" else ()),
    )
    with pytest.raises(custody.LocalSigningCustodyError):
        _qualify_acl(object(), security, frozenset({"owner"}), custody=True)


def test_ntfs_acl_policy_accepts_trusted_writers_and_read_only_untrusted_aces():
    from bot_core.windows_custody_filesystem import _qualify_acl

    security = _acl_model(
        aces=(
            ((0, 0), 0x1F01FF, "owner"),
            ((0, 0), 0x120089, "reader"),
            ((1, 0), 0x1F01FF, "denied"),
            ((0, 8), 0x1F01FF, "inherit_only"),
        )
    )
    _qualify_acl(object(), security, frozenset({"owner"}), custody=True)
    # Ancestor creation rights cannot replace an existing pinned directory.
    _qualify_acl(
        object(), _acl_model(aces=(((0, 0), 4, "creator"),)), frozenset({"owner"}), custody=False
    )
    with pytest.raises(custody.LocalSigningCustodyError):
        _qualify_acl(
            object(), _acl_model(aces=(((0, 0), 4, "creator"),)), frozenset({"owner"}), custody=True
        )


@pytest.mark.parametrize("defect", ["reparse", "directory", "device", "hardlink", "redirected"])
def test_ntfs_opened_object_qualification_rejects_unsafe_object(tmp_path, defect):
    from bot_core.windows_custody_filesystem import _qualify

    info = (
        0x400 if defect == "reparse" else 0x10 if defect == "directory" else 0,
        None,
        None,
        None,
        1,
        0,
        0,
        2 if defect == "hardlink" else 1,
        0,
        1,
    )
    files = SimpleNamespace(
        GetFileInformationByHandle=lambda handle: info,
        GetFileType=lambda handle: 2 if defect == "device" else 1,
        GetFinalPathNameByHandle=lambda handle, flags: str(tmp_path / "other")
        if defect == "redirected"
        else str(tmp_path),
    )
    with pytest.raises(custody.LocalSigningCustodyError):
        _qualify(
            object(),
            tmp_path,
            (files, _acl_model(), None),
            frozenset({"owner"}),
            directory=False,
            custody=True,
        )


def test_ntfs_path_normalization_preserves_unc_and_drive_identity():
    from bot_core.windows_custody_filesystem import _normalized

    assert _normalized(r"\\?\C:\State\Custody") == _normalized(r"c:\state\custody")
    assert _normalized(r"\\?\UNC\server\share\custody") == _normalized(r"\\server\share\custody")


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\custody",
        r"\\?\UNC\server\share\custody",
        r"\\?\unc\server\share\custody",
        r"\\.\C:\custody",
        r"C:custody",
    ],
)
def test_ntfs_path_policy_rejects_unc_and_device_paths_before_native_access(path):
    from bot_core.windows_custody_filesystem import _qualify_local_path

    files = SimpleNamespace(GetDriveType=lambda root: pytest.fail("remote path was queried"))
    with pytest.raises(custody.LocalSigningCustodyError, match="absolute local fixed-drive"):
        _qualify_local_path(PureWindowsPath(path), files)


@pytest.mark.parametrize("drive_type", [0, 1, 2, 4, 5, 6])
def test_ntfs_path_policy_rejects_mapped_network_and_other_nonfixed_drives(drive_type):
    from bot_core.windows_custody_filesystem import _qualify_local_path

    queried = []

    def drive(root):
        queried.append(root)
        return drive_type

    with pytest.raises(custody.LocalSigningCustodyError, match="local fixed-drive"):
        _qualify_local_path(PureWindowsPath(r"Z:\custody"), SimpleNamespace(GetDriveType=drive))
    assert queried == ["z:\\"]


@pytest.mark.parametrize("path", [r"C:\custody", r"\\?\C:\custody"])
def test_ntfs_path_policy_accepts_local_absolute_drive_syntax(path):
    from bot_core.windows_custody_filesystem import _qualify_local_path

    _qualify_local_path(PureWindowsPath(path), SimpleNamespace(GetDriveType=lambda root: 3))


def test_mapped_network_directory_is_rejected_before_create_or_open(monkeypatch):
    from bot_core import windows_custody_filesystem as filesystem

    files = SimpleNamespace(
        GetDriveType=lambda root: 4,
        CreateFile=lambda *args: pytest.fail("mapped network object opened"),
    )
    monkeypatch.setattr(filesystem, "_native", lambda: (files, None, None))
    monkeypatch.setattr(
        filesystem, "_trusted_sids", lambda *args: pytest.fail("path validation was deferred")
    )
    with pytest.raises(custody.LocalSigningCustodyError, match="local fixed-drive"):
        with filesystem.pinned_directory(PureWindowsPath(r"Z:\custody"), create=True):
            pytest.fail("mapped network directory yielded")


@pytest.mark.parametrize(
    "defect", ["UNC", "mapped", "FAT32", "exFAT", "ReFS", "unknown", "acl", "serial"]
)
def test_file_type_disk_does_not_qualify_remote_or_unsupported_volume(monkeypatch, defect):
    from bot_core import windows_custody_filesystem as filesystem

    path = PureWindowsPath(r"C:\custody\record")
    info = (0, None, None, None, 123, 0, 0, 1, 0, 1)
    files = SimpleNamespace(
        GetFileInformationByHandle=lambda handle: info,
        GetFileType=lambda handle: 1,  # SMB also reports FILE_TYPE_DISK.
        GetFinalPathNameByHandle=lambda handle, flags: (
            r"\\server\share\record"
            if defect == "UNC"
            else r"\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\custody\record"
        )
        if flags == 1
        else str(path),
        GetDriveType=lambda root: 4 if defect == "mapped" else 3,
    )
    monkeypatch.setattr(
        filesystem,
        "_volume_information",
        lambda handle: (
            124 if defect == "serial" else 123,
            0 if defect == "acl" else 8,
            ""
            if defect == "unknown"
            else defect
            if defect in {"FAT32", "exFAT", "ReFS"}
            else "NTFS",
        ),
    )
    with pytest.raises(custody.LocalSigningCustodyError, match="volume"):
        filesystem._qualify(
            object(),
            path,
            (files, _acl_model(), None),
            frozenset({"owner"}),
            directory=False,
            custody=True,
        )


def test_ntfs_handle_volume_query_uses_native_handle_and_preserves_dword_serial(monkeypatch):
    from bot_core import windows_custody_filesystem as filesystem

    class Query:
        def __call__(self, handle, volume, volume_size, serial, maximum, flags, name, name_size):
            assert handle == 123 and volume is None and volume_size == 0
            serial._obj.value, maximum._obj.value, flags._obj.value = 0xF0123456, 255, 8
            name.value = "NTFS"
            assert name_size == 32
            return 1

    query = Query()
    monkeypatch.setattr(
        filesystem.ctypes,
        "WinDLL",
        lambda name, **kwargs: SimpleNamespace(GetVolumeInformationByHandleW=query),
        raising=False,
    )
    assert filesystem._volume_information(123) == (0xF0123456, 8, "NTFS")
    assert len(query.argtypes) == 8 and query.restype is filesystem.wintypes.BOOL


@pytest.mark.skipif(os.name != "nt", reason="native opened-handle volume identity and NTFS")
def test_native_windows_volume_qualification_uses_real_handle(tmp_path):
    from bot_core import windows_custody_filesystem as filesystem

    path = tmp_path / "volume-record"
    path.write_bytes(b"local NTFS")
    files = importlib.import_module("win32file")
    handle = files.CreateFile(str(path), 0x80020000, 3, None, 3, 0x200000, None)
    try:
        info = files.GetFileInformationByHandle(handle)
        serial, flags, name = filesystem._volume_information(handle)
        assert serial == info[4] & 0xFFFFFFFF and flags & 8 and name == "NTFS"
        filesystem._qualify_local_path(path, files)
        filesystem._qualify_volume(handle, files, info[4])
    finally:
        handle.Close()
    with filesystem.open_custody_file(path) as descriptor:
        assert os.read(descriptor, 32) == b"local NTFS"


def test_ntfs_directory_handles_pin_all_ancestors_without_share_delete(tmp_path, monkeypatch):
    from bot_core import windows_custody_filesystem as filesystem

    opened, closed = [], []

    def create(name, access, share, attributes, disposition, flags, template):
        assert access & 0x20000 and share == 3 and disposition == 3
        assert flags & 0x200000 and flags & 0x2000000
        opened.append(name)
        return SimpleNamespace(Close=lambda: closed.append(name))

    monkeypatch.setattr(
        filesystem, "_native", lambda: (SimpleNamespace(CreateFile=create), None, None)
    )
    monkeypatch.setattr(filesystem, "_trusted_sids", lambda *args: frozenset({"owner"}))
    monkeypatch.setattr(filesystem, "_qualify_local_path", lambda *args: None)
    monkeypatch.setattr(filesystem, "_qualify", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="injected"):
        with filesystem.pinned_directory(tmp_path):
            assert not closed
            raise RuntimeError("injected")
    assert opened == [str(p) for p in (*reversed(tmp_path.parents), tmp_path)]
    assert closed == list(reversed(opened))


@pytest.fixture
def modeled_windows_handles(monkeypatch):
    """Exercise handle ownership/error paths on Linux; native tests are separate."""
    from bot_core import windows_custody_filesystem as filesystem

    class Handle:
        def __init__(self, path, descriptor):
            self.path, self.descriptor = path, descriptor
            self.closed = False

        def __int__(self):
            assert self.descriptor is not None
            return self.descriptor

        def Detach(self):
            self.closed = True  # transferred to CRT

        def Close(self):
            if not self.closed:
                if self.descriptor is not None:
                    os.close(self.descriptor)
                self.closed = True

    handles = []

    def create(name, access, share, attributes, disposition, flags, template):
        assert share == 3 and flags & 0x200000
        opening = (os.O_RDWR if access & 0x40000000 else os.O_RDONLY) | (
            os.O_CREAT if disposition == 4 else 0
        )
        # The modeled directory handle has no CRT descriptor: Windows cannot
        # open directories through os.open, which is why production uses Win32.
        handle = Handle(name, None if flags & 0x2000000 else os.open(name, opening, 0o600))
        handles.append(handle)
        return handle

    def info(handle):
        value = os.stat(handle.path) if handle.descriptor is None else os.fstat(handle.descriptor)
        return (
            0x10 if stat.S_ISDIR(value.st_mode) else 0,
            None,
            None,
            None,
            value.st_dev,
            0,
            0,
            value.st_nlink,
            value.st_ino >> 32,
            value.st_ino & 0xFFFFFFFF,
        )

    files = SimpleNamespace(
        CreateFile=create,
        GetFileInformationByHandle=info,
        GetFileType=lambda handle: 1,
        GetFinalPathNameByHandle=lambda handle, flags: (
            r"\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\modeled"
            if flags == 1
            else handle.path
        ),
        GetDriveType=lambda root: 3,
    )
    token = SimpleNamespace(Close=lambda: None)
    security = _acl_model()
    security.OpenProcessToken = lambda *args: token
    security.TokenUser = 1
    security.GetTokenInformation = lambda *args: ("owner", 0)
    security.LookupAccountName = lambda *args: ("service", None, None)
    api = SimpleNamespace(GetCurrentProcess=lambda: 1, error=OSError)
    monkeypatch.setattr(filesystem, "_native", lambda: (files, security, api))
    # POSIX paths and volume replies are explicit models. Dedicated Windows
    # tests exercise real path syntax and the opened-handle native query.
    monkeypatch.setattr(filesystem, "_qualify_local_path", lambda *args: None)
    monkeypatch.setattr(
        filesystem, "_volume_information", lambda handle: (info(handle)[4] & 0xFFFFFFFF, 8, "NTFS")
    )
    platform_os = SimpleNamespace(**{name: getattr(os, name) for name in dir(os)})
    platform_os.O_BINARY = 0
    monkeypatch.setattr(filesystem, "os", platform_os)
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(open_osfhandle=lambda handle, flags: handle)
    )
    return filesystem, files, handles


@pytest.mark.parametrize("error", [OSError, AttributeError])
@pytest.mark.parametrize("target", ["parent", "file"])
def test_windows_adapter_closes_handles_when_native_volume_query_fails(
    tmp_path, modeled_windows_handles, monkeypatch, error, target
):
    filesystem, _, handles = modeled_windows_handles
    original = filesystem._volume_information
    path = tmp_path / "record"
    path.write_bytes(b"unchanged")

    def failed(handle):
        if target == "parent" or handle.path == str(path):
            raise error("native volume query unavailable")
        return original(handle)

    monkeypatch.setattr(filesystem, "_volume_information", failed)
    with pytest.raises(custody.LocalSigningCustodyError, match="qualification failed"):
        with filesystem.open_custody_file(path, write=True):
            pytest.fail("unqualified volume yielded")
    assert handles and all(handle.closed for handle in handles)
    assert path.read_bytes() == b"unchanged"


@pytest.mark.parametrize("create", [False, True])
def test_windows_descriptor_adapter_transfers_verified_handle_and_closes_on_exception(
    tmp_path, modeled_windows_handles, create
):
    filesystem, _, handles = modeled_windows_handles
    path = tmp_path / "record"
    if not create:
        path.write_bytes(b"data")
    with pytest.raises(RuntimeError, match="injected"):
        with filesystem.open_custody_file(path, write=True, create=create) as descriptor:
            os.write(descriptor, b"safe")
            assert path.read_bytes() == b"safe"
            raise RuntimeError("injected")
    assert all(handle.closed for handle in handles)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_windows_descriptor_adapter_closes_all_handles_on_substitution(
    tmp_path, modeled_windows_handles
):
    filesystem, files, handles = modeled_windows_handles
    path, replacement = tmp_path / "record", tmp_path / "replacement"
    path.write_bytes(b"old")
    replacement.write_bytes(b"new")
    original = files.CreateFile

    def swapped(name, *args):
        if name == str(path):
            os.replace(replacement, path)
        return original(name, *args)

    files.CreateFile = swapped
    with pytest.raises(custody.LocalSigningCustodyError, match="changed before native open"):
        with filesystem.open_custody_file(path):
            pytest.fail("substitution accepted")
    assert all(handle.closed for handle in handles)


def test_windows_directory_adapter_creates_under_pinned_qualified_parent(
    tmp_path, modeled_windows_handles
):
    filesystem, _, handles = modeled_windows_handles
    directory = tmp_path / "new" / "custody"
    with filesystem.pinned_directory(directory, create=True):
        assert directory.is_dir()
    assert all(handle.closed for handle in handles)
