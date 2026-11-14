"""Real gpgv checks using exclusively disposable or frozen SYNTHETIC TEST keys."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

from deployment.stage9_observer_gpgv_v4b import (
    GpgvRefusalV4b,
    GpgvToolV4b,
    verify_detached_v4b,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "stage9_observer" / "apt_v4b"
_TEST_TIME = 1767225600


def _gpg(home: Path, *arguments: str, raw: bytes | None = None) -> bytes:
    executable = shutil.which("gpg")
    if executable is None:
        pytest.skip("A preprovisioned local gpg is required for disposable TEST key generation")
    return subprocess.run(
        [
            executable,
            "--no-options",
            "--homedir",
            str(home),
            "--batch",
            "--no-tty",
            "--pinentry-mode",
            "loopback",
            "--passphrase",
            "",
            "--faked-system-time",
            "20260101T000000",
            *arguments,
        ],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GNUPGHOME": str(home)},
    ).stdout


def _fingerprints(home: Path) -> list[str]:
    return [
        line.split(":")[9]
        for line in _gpg(home, "--with-colons", "--list-keys").decode("ascii").splitlines()
        if line.startswith("fpr:")
    ]


@dataclass(frozen=True)
class TestSignerV4b:
    """A disposable TEST signing capability; never an Observer trust authority."""

    __test__ = False
    home: Path
    primary_fingerprint: str
    signing_fingerprint: str
    wrong_signing_fingerprint: str
    keyring: bytes
    signature_timestamp: int = _TEST_TIME

    def sign(self, release: bytes, *, signing_fingerprint: str | None = None) -> bytes:
        return _gpg(
            self.home,
            "--armor",
            "--digest-algo",
            "SHA256",
            "--local-user",
            (signing_fingerprint or self.signing_fingerprint) + "!",
            "--detach-sign",
            raw=release,
        )


@contextmanager
def runtime_test_signer(directory: Path) -> Iterator[TestSignerV4b]:
    """Generate only under a pytest temp directory and remove all private state."""
    home = directory / "synthetic-test-key-home"
    home.mkdir(mode=0o700)
    try:
        _gpg(home, "--quick-generate-key", "SYNTHETIC TEST ONLY V4b", "ed25519", "cert", "0")
        primary = _fingerprints(home)[0]
        _gpg(home, "--quick-add-key", primary, "ed25519", "sign", "0")
        _gpg(home, "--quick-add-key", primary, "ed25519", "sign", "0")
        fingerprints = _fingerprints(home)
        yield TestSignerV4b(
            home,
            primary,
            fingerprints[1],
            fingerprints[2],
            _gpg(home, "--export", primary),
        )
    finally:
        executable = shutil.which("gpgconf")
        if executable is not None:
            subprocess.run(
                [executable, "--homedir", str(home), "--kill", "gpg-agent"],
                check=True,
                timeout=10,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GNUPGHOME": str(home)},
            )
        shutil.rmtree(home)


def local_gpgv_tool() -> GpgvToolV4b:
    """Select a local TEST tool; its digest does not establish independent approval."""
    if sys.platform != "linux":
        pytest.skip("The detached gpgv adapter supports Linux only")
    path = os.environ.get("STAGE9_TEST_GPGV") or shutil.which("gpgv")
    if path is None:
        pytest.skip("A preprovisioned local gpgv is required; tests never download a tool")
    executable = Path(path).absolute()
    return GpgvToolV4b(str(executable), hashlib.sha256(executable.read_bytes()).hexdigest())


@pytest.fixture
def tool() -> GpgvToolV4b:
    return local_gpgv_tool()


def test_frozen_test_material_hashes() -> None:
    manifest = json.loads((_FIXTURES / "manifest.json").read_text())
    assert manifest["scope"] == "SYNTHETIC_TEST_ONLY_NO_PRODUCTION_TRUST"
    for record in manifest["files"]:
        assert (
            hashlib.sha256((_FIXTURES / record["name"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
    assert not any(_FIXTURES.glob("**/private-keys-v1.d"))


def test_real_frozen_test_signature(tool: GpgvToolV4b) -> None:
    manifest = json.loads((_FIXTURES / "manifest.json").read_text())
    evidence = verify_detached_v4b(
        release=(_FIXTURES / "Release").read_bytes(),
        signature=(_FIXTURES / "allowed.asc").read_bytes(),
        keyring=(_FIXTURES / "allowed-keyring.gpg").read_bytes(),
        tool=tool,
    )
    assert evidence.signing_fingerprint == manifest["fingerprints"]["allowed_signing"]
    assert evidence.primary_fingerprint == manifest["fingerprints"]["allowed_primary"]
    assert evidence.public_key_algorithm == 22
    assert evidence.hash_algorithm == 8
    assert evidence.signature_timestamp == _TEST_TIME


@pytest.mark.parametrize(
    "mutation",
    [
        "release",
        "signature",
        "unknown-signer",
        "weak-hash",
        "expired",
        "revoked-key",
        "two-signatures",
    ],
)
def test_real_cryptographic_refusals(tool: GpgvToolV4b, mutation: str) -> None:
    release = (_FIXTURES / "Release").read_bytes()
    signature = (_FIXTURES / "allowed.asc").read_bytes()
    keyring = (_FIXTURES / "allowed-keyring.gpg").read_bytes()
    if mutation == "release":
        release = release.replace(b"TEST ONLY", b"TAMPERED!")
    elif mutation == "signature":
        encoded = b"".join(
            line
            for line in signature.split(b"\n\n", 1)[1].splitlines()
            if line and not line.startswith((b"=", b"-"))
        )
        packet = base64.b64decode(encoded, validate=True)
        signature = packet[:-1] + bytes([packet[-1] ^ 1])
    elif mutation == "unknown-signer":
        signature = (_FIXTURES / "unknown.asc").read_bytes()
    elif mutation == "weak-hash":
        signature = (_FIXTURES / "weak-hash.asc").read_bytes()
    elif mutation == "expired":
        signature = (_FIXTURES / "expired.asc").read_bytes()
    elif mutation == "revoked-key":
        keyring = (_FIXTURES / "revoked-keyring.gpg").read_bytes()
    else:
        signature += signature
    with pytest.raises(GpgvRefusalV4b):
        verify_detached_v4b(release=release, signature=signature, keyring=keyring, tool=tool)


def test_real_other_subkey_is_distinct_crypto_evidence(tool: GpgvToolV4b) -> None:
    manifest = json.loads((_FIXTURES / "manifest.json").read_text())
    evidence = verify_detached_v4b(
        release=(_FIXTURES / "Release").read_bytes(),
        signature=(_FIXTURES / "wrong-subkey.asc").read_bytes(),
        keyring=(_FIXTURES / "allowed-keyring.gpg").read_bytes(),
        tool=tool,
    )
    assert (
        evidence.signing_fingerprint == manifest["fingerprints"]["disallowed_signing_fingerprint"]
    )
    assert evidence.signing_fingerprint != manifest["fingerprints"]["allowed_signing"]
    assert evidence.primary_fingerprint == manifest["fingerprints"]["allowed_primary"]


def test_runtime_test_signer_cleans_private_state(tmp_path: Path, tool: GpgvToolV4b) -> None:
    release = b"SYNTHETIC TEST ONLY disposable Release\n"
    with runtime_test_signer(tmp_path) as signer:
        evidence = verify_detached_v4b(
            release=release,
            signature=signer.sign(release),
            keyring=signer.keyring,
            tool=tool,
        )
        assert evidence.signing_fingerprint == signer.signing_fingerprint
        private_home = signer.home
    assert not private_home.exists()
