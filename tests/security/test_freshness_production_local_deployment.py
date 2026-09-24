"""Executable checks for reviewed service and filesystem boundary artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import textwrap

import pytest

from bot_core.freshness_deployment_qualification import EXPECTED_HBA, EXPECTED_IDENT
from bot_core.freshness_semantic_verifier import serve_local_unix_socket

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "deployment/systemd/cryptohunter-freshness-verifier.service"


class _NeverCalled:
    def verify_exact_candidate(self, *_values: bytes):
        raise AssertionError("no request expected")


def test_systemd_unit_has_reviewed_static_sandbox_contract():
    """Keep the portable unit-file contract independent of systemd tooling."""
    content = UNIT.read_text(encoding="ascii")
    for exact in (
        "User=os_freshness_crypto_verifier",
        "Group=freshness_verifier_ipc",
        "RuntimeDirectory=cryptohunter/freshness",
        "RuntimeDirectoryMode=0750",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "ProtectKernelTunables=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelLogs=yes",
        "ProtectControlGroups=yes",
        "RestrictSUIDSGID=yes",
        "LockPersonality=yes",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "MemoryDenyWriteExecute=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "RestrictAddressFamilies=AF_UNIX",
        "Environment=PYTHONNOUSERSITE=1",
    ):
        assert exact in content
    assert "Restart=always" not in content


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="wykonywalna walidacja unit file wymaga natywnego Linux systemd-analyze",
)
def test_systemd_unit_is_valid_with_native_systemd_analyze(tmp_path: Path):
    systemd_analyze = shutil.which("systemd-analyze")
    assert systemd_analyze is not None, "Linux CI musi zapewniać natywny systemd-analyze"
    postgres = tmp_path / "postgresql.service"
    postgres.write_text("[Service]\nType=oneshot\nExecStart=/bin/true\n")
    result = subprocess.run(
        [systemd_analyze, "verify", str(UNIT), str(postgres)], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_reviewed_authentication_templates_are_exact():
    hba = (ROOT / "deployment/postgresql/pg_hba.conf").read_text().splitlines()
    ident = (ROOT / "deployment/postgresql/pg_ident.conf").read_text().splitlines()
    assert len(hba) == len(EXPECTED_HBA) == 5
    assert len(ident) == len(EXPECTED_IDENT) == 3
    forbidden = ("trust", "md5", "scram", "host ", "sameuser")
    assert not any(item in line for item in forbidden for line in hba)


def test_qualification_imports_without_posix_account_modules_and_fails_closed():
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        sys.modules.pop("grp", None)
        sys.modules.pop("pwd", None)

        class NativeAccountBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname in {"grp", "pwd"}:
                    raise ModuleNotFoundError(f"blocked {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, NativeAccountBlocker())
        import bot_core.freshness_deployment_qualification as qualification
        assert qualification.EXPECTED_HBA
        assert qualification.EXPECTED_IDENT
        os.name = "nt"
        try:
            qualification.qualify_production_local_deployment()
        except qualification.DeploymentQualificationError:
            pass
        else:
            raise AssertionError("non-POSIX qualification did not fail closed")
        assert "grp" not in sys.modules
        assert "pwd" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("attack", ["symlink-directory", "symlink-path", "regular-path"])
@pytest.mark.skipif(
    os.name != "posix" or not hasattr(socket, "AF_UNIX"),
    reason="ochrona socketu wymaga POSIX uid/gid i AF_UNIX",
)
def test_socket_substitution_fails_closed(tmp_path: Path, attack: str):
    real = tmp_path / "real"
    real.mkdir(mode=0o750)
    directory = real
    if attack == "symlink-directory":
        directory = tmp_path / "link"
        directory.symlink_to(real, target_is_directory=True)
    path = directory / "verifier.sock"
    if attack == "symlink-path":
        outside = tmp_path / "outside"
        outside.touch()
        path.symlink_to(outside)
    elif attack == "regular-path":
        path.touch()
    with pytest.raises((ValueError, PermissionError)):
        serve_local_unix_socket(
            _NeverCalled(),
            str(path),
            allowed_peer_uid=os.getuid(),
            stop_after=0,
            socket_gid=os.getgid(),
            parent_mode=0o750,
        )


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(socket, "AF_UNIX"),
    reason="ochrona socketu wymaga POSIX uid/gid i AF_UNIX",
)
def test_only_exact_owned_stale_socket_is_recreated(tmp_path: Path):
    directory = tmp_path / "runtime"
    directory.mkdir(mode=0o750)
    path = directory / "verifier.sock"
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(str(path))
    stale.close()
    serve_local_unix_socket(
        _NeverCalled(),
        str(path),
        allowed_peer_uid=os.getuid(),
        stop_after=0,
        socket_gid=os.getgid(),
        parent_mode=0o750,
    )
    assert not path.exists()
