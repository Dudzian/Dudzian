"""Modele publikacji i POSIX DAC; nie zastępują próby rzeczywistych UID w VM.

Modelowane są tylko autoryzacja administratora oraz operacje FS. Nie zmieniamy
UID, procesu ani executable; żaden test nie emituje natywnego PASS.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import pytest

if sys.platform != "linux":
    pytest.skip("Publikacja dotyczy Linux.", allow_module_level=True)

from scripts.diagnostics import stage9_cross_uid_identity as harness
from scripts.diagnostics import stage9_cross_uid_trial as trial

PATH = Path("/var/lib/stage9-cross-uid/model/dane.json")
GIDS = {trial.IPC_GROUP: 42, trial.RUNTIME_GROUP: 43}
ROLE_GROUPS = {"runtime": {42, 43}, "verifier": {42}, "obcy": {44}}


class ModelFS:
    """Osobny model deskryptorów; nie podszywa się pod tożsamość procesu testów."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.data = bytearray()
        self.mode = 0o600
        self.gid = 777  # Grupa odziedziczona z hipotetycznego katalogu SGID.
        self.directory = False
        self.acls: set[str] = set()
        self.replaced = False
        self.hardlink = False
        self.wrong_chown = False
        self.write_error = False
        self.final_sync_error = False

    def info(self, *, linked: bool = False) -> Any:
        return SimpleNamespace(
            st_mode=(stat.S_IFDIR if self.directory else stat.S_IFREG) | self.mode,
            st_uid=0,
            st_gid=self.gid,
            st_nlink=2 if self.hardlink else 1,
            st_dev=1,
            st_ino=11 + int(linked and self.replaced),
            st_size=len(self.data),
        )

    def open(self, name: str, flags: int, mode: int | None = None, **kwargs: Any) -> int:
        self.calls.append(("open", name, flags, mode, kwargs))
        if mode is not None:
            self.mode = mode
        return 4

    def close(self, descriptor: int) -> None:
        self.calls.append(("close", descriptor))

    def chmod(self, descriptor: int, mode: int) -> None:
        self.calls.append(("chmod", mode, len(self.data)))
        self.mode = mode

    def chown(self, descriptor: int, uid: int, gid: int) -> None:
        self.calls.append(("chown", uid, gid))
        self.gid = 888 if self.wrong_chown else gid

    def write(self, descriptor: int, raw: memoryview) -> int:
        if self.write_error:
            raise OSError("Model błędu zapisu")
        chunk = bytes(raw[:3])
        self.calls.append(("write", len(chunk)))
        self.data.extend(chunk)
        return len(chunk)

    def sync(self, descriptor: int) -> None:
        self.calls.append(("fsync", descriptor))
        if self.final_sync_error and self.mode == 0o640:
            raise OSError("Model błędu fsync po publikacji")

    def mkdir(self, name: str, mode: int, **kwargs: Any) -> None:
        self.calls.append(("mkdir", name, mode, kwargs))
        self.directory = True
        self.mode = mode | stat.S_ISGID

    def boundary(self) -> SimpleNamespace:
        return SimpleNamespace(
            O_RDONLY=os.O_RDONLY,
            O_WRONLY=os.O_WRONLY,
            O_CREAT=os.O_CREAT,
            O_EXCL=os.O_EXCL,
            O_NOFOLLOW=os.O_NOFOLLOW,
            O_DIRECTORY=os.O_DIRECTORY,
            open=self.open,
            close=self.close,
            fstat=lambda _fd: self.info(),
            stat=lambda *_args, **_kw: self.info(linked=True),
            fchmod=self.chmod,
            fchown=self.chown,
            write=self.write,
            fsync=self.sync,
            listxattr=lambda _fd: sorted(self.acls),
            mkdir=self.mkdir,
        )


@pytest.fixture
def fs(monkeypatch: pytest.MonkeyPatch) -> ModelFS:
    model = ModelFS()
    monkeypatch.setattr(trial, "require_administrator", lambda: None)
    monkeypatch.setattr(trial, "group_gid", lambda group: GIDS[group])
    monkeypatch.setattr(trial, "_open_directory", lambda _path: 3)
    monkeypatch.setattr(trial, "os", model.boundary())
    return model


def dac(mode: int, gid: int, role: str, bits: int) -> bool:
    """Niezależne oczekiwanie DAC dla nie-właściciela, bez capabilities i ACL."""
    allowed = (mode >> 3) & 7 if gid in ROLE_GROUPS[role] else mode & 7
    return allowed & bits == bits


@pytest.mark.parametrize(
    ("group", "mode", "readers"),
    [
        (None, 0o600, set()),
        (trial.IPC_GROUP, 0o640, {"runtime", "verifier"}),
        (trial.RUNTIME_GROUP, 0o640, {"runtime"}),
    ],
)
def test_file_publication_has_only_required_readers_after_complete_write(
    fs: ModelFS, group: str | None, mode: int, readers: set[str]
) -> None:
    raw = b"pelny-niesekretny-raport"
    trial.write_new(PATH, raw, mode, group=group)
    opened = next(call for call in fs.calls if call[0] == "open")
    assert opened[3] == 0o600
    assert opened[2] & (os.O_EXCL | os.O_NOFOLLOW) == os.O_EXCL | os.O_NOFOLLOW
    assert bytes(fs.data) == raw
    assert fs.mode == mode and fs.gid == (GIDS[group] if group else 0)
    chown_index = next(i for i, call in enumerate(fs.calls) if call[0] == "chown")
    assert chown_index > max(i for i, call in enumerate(fs.calls) if call[0] == "write")
    assert ("fsync", 4) in fs.calls[:chown_index]
    for i, call in enumerate(fs.calls):
        if call[:2] == ("chmod", 0o640):
            assert i > chown_index and call[2] == len(raw)
    assert {role for role in ROLE_GROUPS if dac(fs.mode, fs.gid, role, 4)} == readers
    assert all(not dac(fs.mode, fs.gid, role, 2) for role in ROLE_GROUPS)
    assert ("close", 4) in fs.calls and ("close", 3) in fs.calls


@pytest.mark.parametrize("acl", ["system.posix_acl_access", "system.posix_acl_default"])
def test_inherited_file_acl_is_refused_before_group_publication(fs: ModelFS, acl: str) -> None:
    fs.acls.add(acl)
    with pytest.raises(trial.TrialError, match="ACL"):
        trial.write_new(PATH, b"dane", 0o640, group=trial.IPC_GROUP)
    assert fs.mode == 0o600
    assert not any(call[:2] == ("chmod", 0o640) for call in fs.calls)
    assert not fs.data


@pytest.mark.parametrize("failure", ["hardlink", "replaced", "wrong_chown", "write_error"])
def test_failed_file_cannot_publish_group_access(fs: ModelFS, failure: str) -> None:
    setattr(fs, failure, True)
    with pytest.raises((trial.TrialError, OSError)):
        trial.write_new(PATH, b"dane", 0o640, group=trial.IPC_GROUP)
    assert fs.mode == 0o600
    assert not any(call[:2] == ("chmod", 0o640) for call in fs.calls)


def test_final_sync_failure_withdraws_published_access(fs: ModelFS) -> None:
    fs.final_sync_error = True
    with pytest.raises(OSError, match="fsync"):
        trial.write_new(PATH, b"pelne-dane", 0o640, group=trial.IPC_GROUP)
    assert any(call[:2] == ("chmod", 0o640) for call in fs.calls)
    assert fs.mode == 0o600
    assert fs.calls[-3:] == [("chmod", 0o600, len(fs.data)), ("close", 4), ("close", 3)]


@pytest.mark.parametrize(
    ("mode", "group"), [(0o644, None), (0o640, None), (0o600, trial.IPC_GROUP)]
)
def test_unsupported_sharing_is_refused_before_creation(
    fs: ModelFS, mode: int, group: str | None
) -> None:
    with pytest.raises(trial.TrialError, match="tryb"):
        trial.write_new(PATH, b"dane", mode, group=group)
    assert not fs.calls


def test_snapshot_directory_stays_private_until_explicit_publication(fs: ModelFS) -> None:
    directory = PATH.parent
    descriptor = trial.new_snapshot_directory(directory)
    assert descriptor == 4 and fs.mode == 0o700 and fs.gid == GIDS[trial.RUNTIME_GROUP]
    assert next(call for call in fs.calls if call[0] == "mkdir")[2] == 0o700
    flags = next(call for call in fs.calls if call[0] == "open")[2]
    assert flags & (os.O_DIRECTORY | os.O_NOFOLLOW) == os.O_DIRECTORY | os.O_NOFOLLOW
    assert all(not dac(fs.mode, fs.gid, role, 5) for role in ROLE_GROUPS)
    assert not any(call[:2] == ("chmod", 0o750) for call in fs.calls)
    trial.publish_snapshot_directory(descriptor, directory)
    assert fs.mode == 0o750
    assert {role for role in ROLE_GROUPS if dac(fs.mode, fs.gid, role, 5)} == {"runtime"}
    assert all(not dac(fs.mode, fs.gid, role, 2) for role in ROLE_GROUPS)


@pytest.mark.parametrize("acl", ["system.posix_acl_access", "system.posix_acl_default"])
def test_inherited_directory_acl_is_refused_before_publication(fs: ModelFS, acl: str) -> None:
    fs.acls.add(acl)
    with pytest.raises(trial.TrialError, match="ACL"):
        trial.new_snapshot_directory(PATH.parent)
    assert fs.mode & 0o077 == 0
    assert not any(call[:2] == ("chmod", 0o750) for call in fs.calls)
    assert ("close", 4) in fs.calls


@pytest.mark.parametrize("failure", ["acl", "replaced", "gid"])
def test_changed_directory_is_not_published(fs: ModelFS, failure: str) -> None:
    descriptor = trial.new_snapshot_directory(PATH.parent)
    if failure == "acl":
        fs.acls.add("system.posix_acl_default")
    elif failure == "replaced":
        fs.replaced = True
    else:
        fs.gid = 888
    with pytest.raises(trial.TrialError):
        trial.publish_snapshot_directory(descriptor, PATH.parent)
    assert fs.mode == 0o700
    assert not any(call[:2] == ("chmod", 0o750) for call in fs.calls)


def test_equal_role_groups_refuse_before_directory_creation(
    fs: ModelFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trial, "group_gid", lambda _group: 42)
    with pytest.raises(trial.TrialError, match="odrębne"):
        trial.new_snapshot_directory(PATH.parent)
    assert not fs.calls


@pytest.mark.parametrize("failure", ["write", "publish", "probe", "after_probe"])
def test_controller_seals_directory_before_other_evidence_even_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    calls: list[tuple[Any, ...]] = []
    args = argparse.Namespace(manifest=Path("/etc/cryptohunter/model.json"))
    monkeypatch.setattr(harness, "require_trial", lambda *_a, **_kw: {"trial_id": "model"})
    monkeypatch.setattr(harness, "new_snapshot_directory", lambda _path: 4)
    monkeypatch.setattr(
        harness, "capture_snapshot", lambda _trial: {"process": {"uid": 20, "gid": 21}}
    )
    monkeypatch.setattr(harness, "probe_command", lambda *_a: ["MODEL_PRZEBIEGU"])

    def write(path: Path, raw: bytes, mode: int, **kwargs: Any) -> None:
        calls.append(("write", path.name, mode, kwargs))
        if failure == "write":
            raise OSError("MODELOWA_ODMOWA")

    def publish(*_args: Any) -> None:
        calls.append(("publish",))
        if failure == "publish":
            raise OSError("MODELOWA_ODMOWA")

    def collect(*_args: Any) -> bytes:
        calls.append(("probe",))
        if failure == "probe":
            raise TimeoutError("MODELOWA_ODMOWA")
        return b"{}"

    def listener(*_args: Any) -> bytes:
        assert calls[-2:] == [("seal", 4, 0o700), ("close", 4)]
        raise OSError("MODELOWA_ODMOWA")

    monkeypatch.setattr(harness, "write_new", write)
    monkeypatch.setattr(harness, "publish_snapshot_directory", publish)
    monkeypatch.setattr(harness, "collect_probe", collect)
    monkeypatch.setattr(harness, "collect_listener", listener)
    monkeypatch.setattr(
        harness,
        "os",
        SimpleNamespace(
            fchmod=lambda fd, mode: calls.append(("seal", fd, mode)),
            close=lambda fd: calls.append(("close", fd)),
        ),
    )
    with pytest.raises(OSError, match="MODELOWA_ODMOWA"):
        harness.run_probe(args)
    assert calls[-2:] == [("seal", 4, 0o700), ("close", 4)]
    assert calls[0][1:] == ("before.json", 0o640, {"group": trial.RUNTIME_GROUP})
    assert capsys.readouterr().out == ""
