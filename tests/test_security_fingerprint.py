import base64
import hashlib
import hmac
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.security.fingerprint import (
    DeviceFingerprintGenerator,
    FingerprintError,
    build_fingerprint_document,
    verify_document,
)
from bot_core.security.rotation import RotationRegistry
from bot_core.security.signing import canonical_json_bytes


def test_device_fingerprint_is_deterministic():
    env = {
        "OEM_CPU_ID": "CPU123",
        "OEM_TPM_ID": "TPM999",
        "OEM_DONGLE_ID": "DONGLE42",
        "OEM_MAC_ADDRESSES": "aa:bb:cc:dd:ee:ff,11:22:33:44:55:66",
        "OEM_HOSTNAME": "oem-node",
    }
    generator = DeviceFingerprintGenerator(env=env)
    factors = generator.collect_factors()
    assert factors["cpu_id"] == "CPU123"
    assert factors["tpm"] == "TPM999"
    assert factors["dongle"] == "DONGLE42"
    assert factors["hostname"] == "oem-node"
    assert factors["mac_addresses"] == ["112233445566", "aabbccddeeff"]

    fingerprint_one = generator.generate_fingerprint(factors=factors)
    fingerprint_two = generator.generate_fingerprint(factors=factors)
    assert fingerprint_one == fingerprint_two
    assert "-" in fingerprint_one


def test_build_fingerprint_document_signs_payload(tmp_path):
    key = b"k" * 48
    env = {"OEM_CPU_ID": "CPU-SIGN"}

    document = build_fingerprint_document(signing_key=key, key_id="sign-key-1", env=env)

    assert document.signature["algorithm"] == "HMAC-SHA384"
    assert verify_document({"payload": document.payload, "signature": document.signature}, key=key)

    payload_bytes = canonical_json_bytes(document.payload)
    recomputed = base64.b64encode(hmac.new(key, payload_bytes, hashlib.sha384).digest()).decode("ascii")
    assert recomputed == document.signature["value"]


def test_rotation_registry_guard(tmp_path):
    key = b"s" * 48
    registry_path = tmp_path / "rotation.json"
    registry = RotationRegistry(registry_path)
    registry.mark_rotated(
        "rotation-key",
        "oem-fingerprint-signing",
        timestamp=datetime.now(timezone.utc) - timedelta(days=200),
    )

    with pytest.raises(FingerprintError):
        build_fingerprint_document(
            signing_key=key,
            key_id="rotation-key",
            registry=registry,
            rotation_interval_days=90.0,
        )


def test_mark_rotation_when_requested(tmp_path):
    key = b"p" * 48
    registry_path = tmp_path / "rotation.json"
    registry = RotationRegistry(registry_path)

    before_status = registry.status("key-abc", "oem-fingerprint-signing")
    assert before_status.last_rotated is None

    build_fingerprint_document(
        signing_key=key,
        key_id="key-abc",
        registry=registry,
        mark_rotation=True,
    )

    after_status = registry.status("key-abc", "oem-fingerprint-signing")
    assert after_status.last_rotated is not None
