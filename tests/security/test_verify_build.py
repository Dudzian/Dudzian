from __future__ import annotations

import base64
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from scripts.security import verify_build


@pytest.fixture()
def ed25519_artifact(tmp_path: Path):
    binary_path = tmp_path / "artifact.bin"
    binary_path.write_bytes(b"binary payload for verification")

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    signature = private_key.sign(binary_path.read_bytes())
    signature_path = tmp_path / "artifact.sig"
    signature_path.write_text(base64.b64encode(signature).decode("ascii"), encoding="utf-8")

    public_key_path = tmp_path / "artifact_pub.pem"
    public_key_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    digest = verify_build._compute_digest(binary_path, algorithm="sha256")
    return binary_path, signature_path, public_key_path, digest


def test_verify_build_success(ed25519_artifact) -> None:
    artifact, signature_path, public_key_path, digest = ed25519_artifact
    exit_code = verify_build.main(
        [
            str(artifact),
            "--signature",
            str(signature_path),
            "--public-key",
            str(public_key_path),
            "--expected-digest",
            digest,
            "--report",
            str(artifact.parent / "report.json"),
        ]
    )
    assert exit_code == 0


def test_verify_build_digest_mismatch(ed25519_artifact) -> None:
    artifact, signature_path, public_key_path, digest = ed25519_artifact
    with pytest.raises(ValueError):
        verify_build.main(
            [
                str(artifact),
                "--signature",
                str(signature_path),
                "--public-key",
                str(public_key_path),
                "--expected-digest",
                "deadbeef",
            ]
        )
