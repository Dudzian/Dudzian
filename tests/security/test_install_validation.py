import json
from pathlib import Path

import pytest

from bot_core.security import build_fingerprint_document
from bot_core.security.install_validation import validate_fingerprint_document


@pytest.mark.parametrize("missing_keys", [None, {}])
def test_validate_fingerprint_missing_file(tmp_path: Path, missing_keys):
    missing = tmp_path / "fingerprint.json"

    result = validate_fingerprint_document(document_path=missing, keys=missing_keys)

    assert result.status == "missing"
    assert not result.is_valid
    assert any(str(missing) in error for error in result.errors)


def test_validate_fingerprint_invalid_json(tmp_path: Path):
    malformed = tmp_path / "fingerprint.json"
    malformed.write_text("{\"payload\": 123", encoding="utf-8")

    result = validate_fingerprint_document(document_path=malformed)

    assert result.status == "invalid"
    assert not result.is_valid
    assert result.errors


def test_validate_fingerprint_signature_success(tmp_path: Path):
    key = b"x" * 32
    document = build_fingerprint_document(signing_key=key, key_id="test-key", env={"HOSTNAME": "pytest-host"})
    payload = json.loads(document.to_json())
    path = tmp_path / "fingerprint.json"
    path.write_text(document.to_json(), encoding="utf-8")

    result = validate_fingerprint_document(document_path=path, keys={"test-key": key})

    assert result.status == "ok"
    assert result.is_valid
    assert result.key_id == "test-key"
    assert result.fingerprint == payload["payload"].get("fingerprint")
    assert not result.errors


def test_validate_fingerprint_signature_mismatch(tmp_path: Path):
    key = b"y" * 32
    document = build_fingerprint_document(signing_key=key, key_id="expected-key", env={"HOSTNAME": "pytest-host"})
    path = tmp_path / "fingerprint.json"
    path.write_text(document.to_json(), encoding="utf-8")

    result = validate_fingerprint_document(document_path=path, keys={"other-key": b"z" * 32})

    assert result.status == "invalid"
    assert not result.is_valid
    assert result.errors
