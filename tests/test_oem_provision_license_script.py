import base64
import hashlib
import hmac
import json
from pathlib import Path

import pytest


from bot_core.security.signing import canonical_json_bytes
from scripts import oem_provision_license


ROOT = Path(__file__).resolve().parents[1]
def test_oem_provision_license_creates_signed_entry(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    key_path = tmp_path / "signing.key"
    rotation_log = tmp_path / "rotation.json"

    key_bytes = b"z" * 48
    key_path.write_bytes(key_bytes)

    exit_code = oem_provision_license.main(
        [
            "--fingerprint",
            "ABCD-EFGH",
            "--signing-key-path",
            str(key_path),
            "--key-id",
            "license-key-1",
            "--output",
            str(registry_path),
            "--issuer",
            "QA-DEPARTMENT",
            "--profile",
            "paper",
            "--bundle-version",
            "1.2.3",
            "--feature",
            "daemon",
            "--feature",
            "ui",
            "--valid-days",
            "30",
            "--rotation-log",
            str(rotation_log),
        ]
    )

    assert exit_code == 0
    contents = registry_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1

    document = json.loads(contents[0])
    payload = document["payload"]
    assert payload["fingerprint"] == "ABCD-EFGH"
    assert payload["features"] == ["daemon", "ui"]
    assert payload["issuer"] == "QA-DEPARTMENT"

    signature = document["signature"]["value"]
    recomputed = base64.b64encode(
        hmac.new(key_bytes, canonical_json_bytes(payload), hashlib.sha384).digest()
    ).decode("ascii")
    assert signature == recomputed

    rotation_status = oem_provision_license.RotationRegistry(rotation_log).status(
        "license-key-1", oem_provision_license.DEFAULT_PURPOSE
    )
    assert rotation_status.last_rotated is not None


def test_oem_provision_license_from_request_file(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    key_path = tmp_path / "signing.key"
    rotation_log = tmp_path / "rotation.json"

    key_bytes = b"q" * 48
    key_path.write_bytes(key_bytes)

    request_data = {
        "fingerprint": "ZXCV-ASDF",
        "issuer": "FILE-DEFINED",
        "profile": "paper",
        "bundle_version": "2.0.0",
        "features": ["daemon", "ui"],
        "valid_days": 45,
        "notes": "from-file",
        "signing_key_path": str(key_path),
        "key_id": "lic-2025",
        "output": str(registry_path),
        "rotation_log": str(rotation_log),
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request_data, ensure_ascii=False), encoding="utf-8")

    exit_code = oem_provision_license.main([str(request_path), "--issuer", "CLI-OVERRIDE"])

    assert exit_code == 0
    contents = registry_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1

    document = json.loads(contents[0])
    payload = document["payload"]
    assert payload["fingerprint"] == "ZXCV-ASDF"
    assert payload["issuer"] == "CLI-OVERRIDE"
    assert payload["features"] == ["daemon", "ui"]
    assert payload["notes"] == "from-file"
    assert payload["bundle_version"] == "2.0.0"

    signature = document["signature"]["value"]
    recomputed = base64.b64encode(
        hmac.new(key_bytes, canonical_json_bytes(payload), hashlib.sha384).digest()
    ).decode("ascii")
    assert signature == recomputed

    rotation_status = oem_provision_license.RotationRegistry(rotation_log).status(
        "lic-2025", oem_provision_license.DEFAULT_PURPOSE
    )
    assert rotation_status.last_rotated is not None


@pytest.mark.parametrize("fingerprint", ["", "bad$chars", "with space"])
def test_oem_provision_license_rejects_invalid_fingerprint(tmp_path, fingerprint):
    key_path = tmp_path / "key.bin"
    key_path.write_bytes(b"q" * 48)

    exit_code = oem_provision_license.main([
        "--fingerprint",
        fingerprint,
        "--signing-key-path",
        str(key_path),
    ])

    assert exit_code != 0


def test_oem_provision_license_verify_mode(tmp_path):
    fingerprint_key = b"f" * 48
    fingerprint_payload = {
        "fingerprint": "OEM-FP-1234567890",
        "generated_at": "2025-01-01T00:00:00Z",
        "components": {
            "cpu": {"normalized": "intel xeon"},
            "mac": {"addresses": ["aa:bb:cc:dd:ee:ff"]},
            "tpm": {"serial": "TPM-123"},
            "dongle": {"serial": "DONGLE-XYZ"},
        },
    }
    fingerprint_signature = base64.b64encode(
        hmac.new(fingerprint_key, canonical_json_bytes(fingerprint_payload), hashlib.sha384).digest()
    ).decode("ascii")
    fingerprint_path = tmp_path / "fingerprint.expected.json"
    fingerprint_path.write_text(
        json.dumps(
            {
                "payload": fingerprint_payload,
                "signature": {
                    "algorithm": "HMAC-SHA384",
                    "value": fingerprint_signature,
                    "key_id": "fp-key",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.jsonl"
    license_key = base64.b64encode(b"l" * 48).decode("ascii")
    payload = oem_provision_license.build_license_payload_simple(
        fingerprint_text="ABCD-1234",
        issuer="VERIFIER",
        profile="paper",
        bundle_version="9.9.9",
        features=["daemon"],
        notes=None,
        valid_days=30,
    )
    record = oem_provision_license.sign_with_single_key(
        payload,
        key_bytes=base64.b64decode(license_key),
        key_id="lic-key",
    )
    registry_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code = oem_provision_license.main(
        [
            "--verify",
            "--fingerprint",
            str(fingerprint_path),
            "--fingerprint-key",
            f"fp-key=base64:{base64.b64encode(fingerprint_key).decode('ascii')}",
            "--license-key",
            f"lic-key=base64:{license_key}",
            "--output",
            str(registry_path),
        ]
    )

    assert exit_code == 0
