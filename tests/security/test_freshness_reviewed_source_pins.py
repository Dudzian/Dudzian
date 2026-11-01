"""Odmowy dokładnych bajtów zainstalowanego kodu, bez kwalifikacji hosta."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core import freshness_deployment_qualification as qualification


@pytest.fixture
def installed_source(tmp_path, monkeypatch):
    """Modeluj poprzedzające kontrole, zachowując rzeczywisty odczyt pliku."""
    filename, expected_digest = next(iter(qualification.REVIEWED_CODE_DIGESTS.items()))
    reviewed = Path(__file__).resolve().parents[2] / "bot_core" / filename
    raw = reviewed.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_digest
    directory = tmp_path / "installed-code"
    directory.mkdir()
    source = directory / filename
    source.write_bytes(raw)
    metadata = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o644)
    verifier = SimpleNamespace(pw_uid=11001, pw_gid=11001)
    runtime = SimpleNamespace(pw_uid=11002, pw_gid=11002)
    group = SimpleNamespace(
        gr_gid=11003,
        gr_mem=[qualification.VERIFIER_USER, qualification.RUNTIME_USER],
    )
    monkeypatch.setitem(
        sys.modules,
        "pwd",
        SimpleNamespace(
            getpwnam=lambda name: verifier if name == qualification.VERIFIER_USER else runtime
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "grp",
        SimpleNamespace(getgrnam=lambda _name: group, getgrall=lambda: [group]),
    )
    # Nie zmieniaj globalnego os.name: test modelu musi zachować natywne Path
    # także podczas uruchamiania na Windows.
    monkeypatch.setattr(qualification, "os", SimpleNamespace(name="posix"))

    def installed_path(value):
        if str(value) == "/usr/lib/cryptohunter/bot_core":
            return directory
        return Path(value)

    def installed_metadata(path, *, follow=False):
        if path == source:
            return metadata
        return SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o644)

    monkeypatch.setattr(qualification, "Path", installed_path)
    monkeypatch.setattr(qualification, "_mode", installed_metadata)
    monkeypatch.setattr(qualification, "_immutable_file", Mock())
    systemctl = Mock(side_effect=AssertionError("Odmowa musi poprzedzać systemctl"))
    connect = Mock(side_effect=AssertionError("Odmowa musi poprzedzać połączenie bazy"))
    monkeypatch.setattr(qualification, "_systemctl", systemctl)
    monkeypatch.setattr(qualification.psycopg, "connect", connect)
    return SimpleNamespace(
        source=source,
        raw=raw,
        metadata=metadata,
        systemctl=systemctl,
        connect=connect,
    )


def assert_installed_code_refused(installed, message):
    with pytest.raises(qualification.DeploymentQualificationError, match=message):
        qualification.qualify_production_local_deployment()
    installed.systemctl.assert_not_called()
    installed.connect.assert_not_called()


@pytest.mark.parametrize("mutation", ["single-byte", "crlf"])
def test_actual_installed_bytes_refuse_drift_before_systemd_or_database(installed_source, mutation):
    installed = installed_source
    if mutation == "single-byte":
        changed = bytes([installed.raw[0] ^ 1]) + installed.raw[1:]
        assert sum(left != right for left, right in zip(changed, installed.raw)) == 1
    else:
        assert b"\r\n" not in installed.raw and b"\n" in installed.raw
        changed = installed.raw.replace(b"\n", b"\r\n")
        assert changed != installed.raw
    installed.source.write_bytes(changed)
    assert_installed_code_refused(installed, "reviewed source digest mismatch")


@pytest.mark.parametrize(
    "uid,mode,message",
    [
        (11001, 0o644, "unsafe source owner"),
        (0, 0o664, "source is group/world writable"),
        (0, 0o646, "source is group/world writable"),
    ],
)
def test_installed_source_metadata_refuses_before_later_host_checks(
    installed_source, uid, mode, message
):
    installed = installed_source
    installed.metadata.st_uid = uid
    installed.metadata.st_mode = stat.S_IFREG | mode
    assert_installed_code_refused(installed, message)


@pytest.mark.skipif(os.name != "posix", reason="Rzeczywisty symlink instalacji jest granicą POSIX")
def test_actual_installed_symlink_refuses_even_with_reviewed_target_bytes(installed_source):
    installed = installed_source
    target = installed.source.with_suffix(".reviewed-target")
    installed.source.rename(target)
    installed.source.symlink_to(target.name)
    assert installed.source.read_bytes() == installed.raw
    assert_installed_code_refused(installed, "unsafe source owner")
