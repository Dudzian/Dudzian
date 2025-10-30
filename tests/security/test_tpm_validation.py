from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from bot_core.security.signing import canonical_json_bytes
from bot_core.security.tpm import TpmValidationError, validate_attestation


def _write_evidence(tmp_path: Path, payload: dict[str, object], signature: dict[str, object] | None) -> Path:
    document: dict[str, object] = {"payload": payload}
    if signature:
        document["signature"] = signature
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence_path


def test_validate_attestation_with_signature(tmp_path: Path) -> None:
    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key

    payload = {
        "fingerprint": "ABCDEF0123456789",
        "sealed_fingerprint": "ABCDEF0123456789-SEALED",
        "nonce": "12345",
        "attested_at": "2024-01-01T12:00:00Z",
        "expires_at": "2099-01-01T00:00:00Z",
        "secure_enclave": {"type": "tpm2"},
    }
    payload_bytes = canonical_json_bytes(payload)
    signature_value = base64.b64encode(signing_key.sign(payload_bytes).signature).decode("ascii")
    signature = {"algorithm": "ed25519", "value": signature_value, "key_id": "primary"}

    evidence_path = _write_evidence(tmp_path, payload, signature)
    keyring_path = tmp_path / "keys.json"
    keyring_path.write_text(json.dumps({"primary": verify_key.encode().hex()}), encoding="utf-8")

    result = validate_attestation(
        evidence_path=evidence_path,
        expected_fingerprint="ABCDEF0123456789",
        keyring=str(keyring_path),
    )

    assert result.is_valid
    assert result.signature_verified is True
    assert result.signature_key == "primary"
    assert result.fingerprint == "ABCDEF0123456789"
    assert result.sealed_fingerprint == "ABCDEF0123456789-SEALED"
    assert not result.errors


def test_validate_attestation_without_signature_warns(tmp_path: Path) -> None:
    payload = {
        "fingerprint": "HELLO-123",
        "attested_at": datetime.now(timezone.utc).isoformat(),
        "secure_enclave": "sgx",
    }
    evidence_path = _write_evidence(tmp_path, payload, signature=None)

    result = validate_attestation(
        evidence_path=evidence_path,
        expected_fingerprint="HELLO-123",
        keyring=None,
    )

    assert result.status == "ok"
    assert "Dowód TPM nie zawiera podpisu" in " ".join(msg.message for msg in result.warnings)
    assert result.signature_verified is False


def test_validate_attestation_mismatch(tmp_path: Path) -> None:
    payload = {
        "fingerprint": "GOOD-123",
        "secure_enclave": "tpm",
        "attested_at": "2024-04-01T00:00:00+00:00",
    }
    evidence_path = _write_evidence(tmp_path, payload, signature=None)

    result = validate_attestation(
        evidence_path=evidence_path,
        expected_fingerprint="BAD-999",
        keyring=None,
    )

    assert result.status == "mismatch"
    assert any("Dowód TPM nie jest powiązany" in error.message for error in result.errors)


def test_validate_attestation_requires_valid_json(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(TpmValidationError):
        validate_attestation(
            evidence_path=evidence_path,
            expected_fingerprint=None,
            keyring=None,
        )
