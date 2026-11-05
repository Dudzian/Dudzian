"""Kontrakt katalogów: modele odmów oraz jedna sonda DAC własnego UID, bez VM."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import stage9_vm_environment as guard
from scripts.diagnostics import stage9_cross_uid_trial as trial


pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Modele wymagają stałych deskryptorów POSIX; brak kwalifikacji VM"
)


def model_walkers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    denied_open: bool = False,
    acl: str | None = None,
    unavailable: bool = False,
    stage: int = 1,
) -> SimpleNamespace:
    """Jedna modelowana granica OS dla obu walkerów i wspólnego predykatu."""
    opened: list[tuple[str, int]] = []
    descriptors: list[int] = []
    closed: list[int] = []
    fake_os = SimpleNamespace(**vars(os))

    def open_directory(path: str, flags: int, **kwargs: Any) -> int:
        index = len(opened)
        opened.append((str(path), flags))
        assert flags & os.O_DIRECTORY
        assert flags & os.O_NOFOLLOW
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        if hasattr(os, "O_PATH"):
            assert not flags & os.O_PATH
        if index:
            assert kwargs["dir_fd"] == descriptors[-1]
        if denied_open and index == stage:
            raise PermissionError(errno.EACCES, "Model katalogu bez prawa odczytu")
        descriptor = 100 + index
        descriptors.append(descriptor)
        return descriptor

    def status(_fd: int) -> SimpleNamespace:
        return SimpleNamespace(st_mode=stat.S_IFDIR | 0o750, st_uid=0, st_nlink=1)

    def attributes(fd: int) -> list[str]:
        if fd == 100 + stage:
            if unavailable:
                raise OSError(errno.EIO, "Model niedostępnej obserwacji xattr")
            if acl is not None:
                return [acl]
        return []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Walker nie może czytać danych pliku ani zmieniać filesystemu")

    fake_os.open = open_directory
    fake_os.fstat = status
    fake_os.listxattr = attributes
    fake_os.close = closed.append
    for name in (
        "read",
        "write",
        "chmod",
        "fchmod",
        "chown",
        "fchown",
        "mkdir",
        "unlink",
        "rename",
    ):
        setattr(fake_os, name, forbidden)
    monkeypatch.setattr(guard, "os", fake_os)
    monkeypatch.setattr(trial, "os", fake_os)
    monkeypatch.setattr(guard, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(trial, "sys", SimpleNamespace(platform="linux"))
    return SimpleNamespace(opened=opened, descriptors=descriptors, closed=closed)


def walk(reader: str) -> None:
    if reader == "approval":
        guard._read_approval(Path("/approved/approval.json"))
    else:
        descriptor = trial._open_directory(Path("/approved"))
        trial.os.close(descriptor)


@pytest.mark.parametrize("reader", ["approval", "trial"])
def test_search_only_ancestor_refuses_before_file_read_and_closes_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
) -> None:
    """Model EACCES przy O_RDONLY nie jest obchodzony samym prawem lookup pliku."""
    observations = model_walkers(monkeypatch, denied_open=True)
    expected_error = guard.VMGuardError if reader == "approval" else PermissionError
    with pytest.raises(expected_error):
        walk(reader)
    assert [path for path, _flags in observations.opened] == ["/", "approved"]
    assert observations.closed == observations.descriptors == [100]


@pytest.mark.parametrize("reader", ["approval", "trial"])
@pytest.mark.parametrize("stage", [0, 1], ids=["root", "przodek"])
@pytest.mark.parametrize(
    "observation", ["system.posix_acl_access", "system.posix_acl_default", "unavailable"]
)
def test_directory_acl_or_unavailable_observation_blocks_both_walkers(
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
    stage: int,
    observation: str,
) -> None:
    observations = model_walkers(
        monkeypatch,
        stage=stage,
        acl=None if observation == "unavailable" else observation,
        unavailable=observation == "unavailable",
    )
    expected_error = (
        guard.VMGuardError
        if reader == "approval"
        else OSError
        if observation == "unavailable"
        else trial.TrialError
    )
    with pytest.raises(expected_error):
        walk(reader)
    assert len(observations.opened) == stage + 1
    assert sorted(observations.closed) == observations.descriptors
    assert len(set(observations.closed)) == len(observations.closed)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_directory_predicate_does_not_claim_posix_acl_support_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    def forbidden(_fd: int) -> Any:
        raise AssertionError("Nieobsługiwana platforma nie może udawać obserwacji ACL Linux")

    monkeypatch.setattr(guard, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(guard, "os", SimpleNamespace(fstat=forbidden, listxattr=forbidden))
    assert guard._directory_without_acl(100) is False


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() == 0,
    reason="NOT_RUN: sonda wymaga zwykłego UID na Linux; nie kwalifikuje VM ani dwóch kont",
)
def test_own_uid_can_lookup_file_but_cannot_open_search_only_directory(tmp_path: Path) -> None:
    """Rzeczywisty DAC jednego UID; nie zmienia UID i nie wywołuje kwalifikatora VM."""
    directory = tmp_path / "search-only"
    directory.mkdir(mode=0o700)
    document = directory / "marker.json"
    document.write_bytes(b'{"scope":"TEST_ONLY"}\n')
    directory.chmod(0o100)
    try:
        assert document.read_bytes() == b'{"scope":"TEST_ONLY"}\n'
        try:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except PermissionError as exc:
            assert exc.errno == errno.EACCES
        else:
            os.close(descriptor)
            pytest.fail("O_RDONLY katalogu bez prawa odczytu powinno otrzymać EACCES")
    finally:
        directory.chmod(0o700)
