"""Executable checks for reviewed service and filesystem boundary artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import socket

import pytest

from bot_core.freshness_deployment_qualification import EXPECTED_HBA, EXPECTED_IDENT
from bot_core.freshness_semantic_verifier import serve_local_unix_socket

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "deployment/systemd/cryptohunter-freshness-verifier.service"


class _NeverCalled:
    def verify_exact_candidate(self, *_values: bytes):
        raise AssertionError("no request expected")


def test_systemd_unit_is_valid_and_has_reviewed_effective_sandbox(tmp_path: Path):
    import subprocess

    postgres = tmp_path / "postgresql.service"
    postgres.write_text("[Service]\nType=oneshot\nExecStart=/bin/true\n")
    result = subprocess.run(
        ["systemd-analyze", "verify", str(UNIT), str(postgres)],
        text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    content = UNIT.read_text(encoding="ascii")
    for exact in (
        "User=os_freshness_crypto_verifier", "Group=freshness_verifier_ipc",
        "RuntimeDirectory=cryptohunter/freshness", "RuntimeDirectoryMode=0750",
        "NoNewPrivileges=yes", "PrivateTmp=yes", "ProtectSystem=strict",
        "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectKernelModules=yes",
        "ProtectKernelLogs=yes", "ProtectControlGroups=yes", "RestrictSUIDSGID=yes",
        "LockPersonality=yes", "RestrictNamespaces=yes", "RestrictRealtime=yes",
        "MemoryDenyWriteExecute=yes", "CapabilityBoundingSet=", "AmbientCapabilities=",
        "RestrictAddressFamilies=AF_UNIX", "Environment=PYTHONNOUSERSITE=1",
    ):
        assert exact in content
    assert "Restart=always" not in content


def test_reviewed_authentication_templates_are_exact():
    hba = (ROOT / "deployment/postgresql/pg_hba.conf").read_text().splitlines()
    ident = (ROOT / "deployment/postgresql/pg_ident.conf").read_text().splitlines()
    assert len(hba) == len(EXPECTED_HBA) == 5
    assert len(ident) == len(EXPECTED_IDENT) == 3
    forbidden = ("trust", "md5", "scram", "host ", "sameuser")
    assert not any(item in line for item in forbidden for line in hba)


@pytest.mark.parametrize("attack", ["symlink-directory", "symlink-path", "regular-path"])
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
        serve_local_unix_socket(_NeverCalled(), str(path), allowed_peer_uid=os.getuid(),
                                stop_after=0, socket_gid=os.getgid(), parent_mode=0o750)


def test_only_exact_owned_stale_socket_is_recreated(tmp_path: Path):
    directory = tmp_path / "runtime"
    directory.mkdir(mode=0o750)
    path = directory / "verifier.sock"
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(str(path))
    stale.close()
    serve_local_unix_socket(_NeverCalled(), str(path), allowed_peer_uid=os.getuid(),
                            stop_after=0, socket_gid=os.getgid(), parent_mode=0o750)
    assert not path.exists()
