"""Process checks with synthetic hardware/credentials and child-only seccomp."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux custody identity and seccomp"),
]
_HELPER = Path(__file__).with_name("_custody_hwid_process.py")


@pytest.fixture(autouse=True)
def require_seccomp() -> None:
    try:
        ctypes.CDLL("libseccomp.so.2")
    except OSError:
        pytest.skip("libseccomp is required for the child-only AF_UNIX restriction")


def _child(action: str, fixture_path: Path, *, prefix: tuple[str, ...] = ()) -> dict:
    completed = subprocess.run(
        [*prefix, sys.executable, str(_HELPER), action, str(fixture_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(completed.stdout)


def test_hwid_is_stable_across_independent_af_unix_only_processes(tmp_path: Path) -> None:
    # Each child forbids uuid.getnode and supplies no MAC addresses, while a
    # real seccomp filter verifies non-AF_UNIX sockets fail in that process.
    observations = [_child("identity", tmp_path / "unused.json") for _ in range(4)]
    assert all(item["af_unix_only"] for item in observations)
    identities = {item["identity_digest"] for item in observations}
    assert len(identities) == 1
    assert all(item["versioned"] for item in observations)


def test_synthetic_secret_survives_process_restart_and_changed_hwid_is_denied(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "synthetic-keyring.json"
    written = _child("encrypt", fixture_path)
    index_path = fixture_path.with_suffix(".index.json")
    before = (fixture_path.read_bytes(), index_path.read_bytes())
    reopened = _child("decrypt", fixture_path)
    assert reopened == written
    assert _child("decrypt-history", fixture_path) == written
    assert (fixture_path.read_bytes(), index_path.read_bytes()) == before
    for change in ("changed-installation", "changed-hardware"):
        assert _child(change, fixture_path) == {"changed_identity_denied": True}
        assert (fixture_path.read_bytes(), index_path.read_bytes()) == before


def test_hwid_in_private_network_namespace(tmp_path: Path) -> None:
    unshare = shutil.which("unshare")
    if unshare is None:
        pytest.skip(
            "unshare is unavailable; installed PrivateNetwork qualification remains required"
        )
    prefix = (unshare, "--user", "--map-root-user", "--net")
    probe = subprocess.run(
        [*prefix, sys.executable, "-c", "pass"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode:
        pytest.skip("Disposable user/network namespaces unavailable: " + probe.stderr.strip()[:240])
    outside = _child("identity", tmp_path / "unused.json")
    inside = [_child("private-network", tmp_path / "unused.json", prefix=prefix) for _ in range(4)]
    assert all(observation == outside for observation in inside)
