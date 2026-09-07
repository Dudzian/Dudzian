from __future__ import annotations

import os
from pathlib import Path

import pytest

from bot_core.persistence import physical_durability
from bot_core.persistence.physical_durability import (
    MOVEFILE_REPLACE_EXISTING,
    MOVEFILE_WRITE_THROUGH,
    PhysicalDurabilityError,
    atomic_write_bytes_durably,
    flush_created_directory_metadata,
    publish_file_atomically_durably,
)


def test_posix_directory_and_atomic_file_fences(tmp_path: Path) -> None:
    flush_created_directory_metadata(tmp_path, _platform="posix")
    target = tmp_path / "manifest.json"
    atomic_write_bytes_durably(target, b'{"durable":true}')
    assert target.read_bytes() == b'{"durable":true}'


def test_windows_publication_uses_exact_write_through_move_flags(
    tmp_path: Path,
) -> None:
    temporary, target = tmp_path / "temp", tmp_path / "manifest.json"
    temporary.write_bytes(b"manifest")
    calls: list[tuple[Path, Path, int]] = []

    def move(source: str, destination: str, flags: int) -> bool:
        calls.append((Path(source), Path(destination), flags))
        os.replace(source, destination)
        return True

    physical_durability._move_file_ex_windows(temporary, target, _move=move)
    assert calls == [
        (
            temporary.resolve(),
            target.resolve(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    ]
    assert target.read_bytes() == b"manifest"


def test_windows_move_failure_propagates_and_leaves_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary, target = tmp_path / "temp", tmp_path / "manifest.json"
    temporary.write_bytes(b"manifest")
    windows_move = physical_durability._move_file_ex_windows

    def fail(source: Path, target: Path) -> None:
        windows_move(source, target, _move=lambda _source, _target, _flags: False)

    monkeypatch.setattr(physical_durability, "_move_file_ex_windows", fail)
    with pytest.raises(PhysicalDurabilityError, match="publish durable file"):
        publish_file_atomically_durably(temporary, target, _platform="nt")
    assert temporary.read_bytes() == b"manifest"
    assert not target.exists()


def test_windows_directory_creation_does_not_open_or_flush_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        physical_durability,
        "_flush_directory_posix",
        lambda _path: pytest.fail("Windows used POSIX directory flush"),
    )
    flush_created_directory_metadata(tmp_path, _platform="nt")


@pytest.mark.skipif(os.name != "nt", reason="real Win32 publication runs only on Windows CI")
def test_real_windows_atomic_write(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    atomic_write_bytes_durably(target, b"windows-write-through")
    assert target.read_bytes() == b"windows-write-through"
