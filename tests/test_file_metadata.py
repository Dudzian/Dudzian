"""Testy modułu bot_core.runtime.file_metadata."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bot_core.runtime import file_metadata
from bot_core.runtime.file_metadata import file_reference_metadata


class _WindowsOsProxy:
    def __init__(self, real_os: object) -> None:
        self._real_os = real_os
        self.name = "nt"

    def __getattr__(self, item: str):
        return getattr(self._real_os, item)


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only chmod semantics")
def test_file_reference_metadata_warns_on_world_writable_parent(tmp_path: Path) -> None:
    parent_dir = tmp_path / "insecure"
    parent_dir.mkdir()
    parent_dir.chmod(0o777)

    target = parent_dir / "key.pem"
    target.write_text("dummy", encoding="utf-8")

    metadata = file_reference_metadata(target, role="tls_key")

    warnings = metadata.get("security_warnings", [])
    assert any(
        "Katalog nadrzędny jest zapisywalny dla wszystkich użytkowników" in message
        for message in warnings
    ), warnings
    flags = metadata.get("parent_security_flags", {})
    assert flags.get("world_writable") is True
    assert flags.get("permissions_supported") is True


def test_file_reference_metadata_skips_world_writable_warning_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Symulujemy środowisko Windows, w którym bity chmod nie reprezentują realnych uprawnień NTFS.
    monkeypatch.setattr(file_metadata, "os", _WindowsOsProxy(os))

    parent_dir = tmp_path / "insecure"
    parent_dir.mkdir()
    parent_dir.chmod(0o777)

    target = parent_dir / "key.pem"
    target.write_text("dummy", encoding="utf-8")

    metadata = file_reference_metadata(target, role="tls_key")

    warnings = metadata.get("security_warnings", [])
    assert all(
        "Katalog nadrzędny jest zapisywalny dla wszystkich użytkowników" not in message
        for message in warnings
    ), warnings
    flags = metadata.get("parent_security_flags", {})
    assert flags.get("world_writable") is False
    assert flags.get("group_writable") is False
    assert flags.get("permissions_supported") is False
