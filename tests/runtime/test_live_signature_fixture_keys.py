from __future__ import annotations

import hashlib
import hmac
import json

from tests._live_signature_fixtures import derive_fixture_hmac_key

HEX_BYTES = set(b"0123456789abcdef")


def test_derive_fixture_hmac_key_is_deterministic_ascii_without_newline() -> None:
    key_a = derive_fixture_hmac_key(
        "risk/live/binance/risk_profile_alignment.pdf",
        "risk-key",
    )
    key_b = derive_fixture_hmac_key(
        "risk/live/binance/risk_profile_alignment.pdf",
        "risk-key",
    )

    assert key_a == key_b
    assert len(key_a) == 64
    assert b"\n" not in key_a
    assert b"\r" not in key_a
    assert set(key_a).issubset(HEX_BYTES)


def test_derive_fixture_hmac_key_can_sign_and_verify_payload() -> None:
    key = derive_fixture_hmac_key("compliance/live/binance/kyc_packet.pdf", "compliance-key")
    payload = {
        "document": {
            "name": "kyc_packet.pdf",
            "path": "compliance/live/binance/kyc_packet.pdf",
        },
        "hashes": {"sha256": "abc123"},
    }

    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(key, payload_bytes, hashlib.sha256).hexdigest()
    recomputed = hmac.new(key, payload_bytes, hashlib.sha256).hexdigest()

    assert hmac.compare_digest(signature, recomputed)
