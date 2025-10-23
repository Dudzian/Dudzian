import base64
import hashlib
import hmac
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.security.signing import canonical_json_bytes
from scripts import oem_provision_license


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
