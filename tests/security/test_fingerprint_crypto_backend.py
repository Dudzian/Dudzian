from __future__ import annotations

from bot_core.security import fingerprint_crypto


def test_fingerprint_crypto_roundtrip(monkeypatch) -> None:
    fingerprint_crypto.reset_backend()  # ensure fallback Python implementation
    secret = b"A" * 48
    fingerprint = "DEVICE-XYZ-123"

    document = fingerprint_crypto.encrypt_license_secret(
        secret,
        fingerprint,
        file_version=2,
    )

    recovered = fingerprint_crypto.decrypt_license_secret(
        document,
        fingerprint,
        file_version=2,
    )

    assert recovered == secret
    digest = fingerprint_crypto.current_hwid_digest(fingerprint)
    assert isinstance(digest, str) and len(digest) == 64
    fingerprint_crypto.reset_backend()
