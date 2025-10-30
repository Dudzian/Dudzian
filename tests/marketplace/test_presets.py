from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.marketplace import PresetRepository, sign_preset_payload


@pytest.fixture()
def ed25519_keypair() -> tuple[ed25519.Ed25519PrivateKey, str]:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_key, base64.b64encode(public_key).decode("ascii")


def _build_payload(preset_id: str, version: str) -> dict[str, object]:
    return {
        "name": f"Preset {preset_id}",
        "strategies": [
            {"name": "grid", "engine": "grid_trading", "parameters": {"grid_size": 5}},
        ],
        "metadata": {
            "id": preset_id,
            "version": version,
            "summary": "Test preset",
            "tags": ["test"],
        },
    }


def test_import_signed_json_preset(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)
    payload = _build_payload("alpha", "1.0.0")
    signature = sign_preset_payload(
        payload,
        private_key=private_key,
        key_id="author",
        signed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    document = {"preset": payload, "signature": signature.as_dict()}
    blob = json.dumps(document, ensure_ascii=False).encode("utf-8")

    imported = repo.import_payload(
        blob,
        signing_keys={"author": public_key_b64},
    )

    assert imported.preset_id == "alpha"
    assert imported.verification.verified is True
    assert imported.path is not None and imported.path.exists()


def test_import_rejects_invalid_signature(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)
    payload = _build_payload("beta", "1.2.3")
    signature = sign_preset_payload(payload, private_key=private_key, key_id="author")
    broken = signature.as_dict()
    broken["value"] = base64.b64encode(b"invalid").decode("ascii")
    blob = json.dumps({"preset": payload, "signature": broken}).encode("utf-8")

    with pytest.raises(ValueError):
        repo.import_payload(blob, signing_keys={"author": public_key_b64})


def test_import_version_conflict(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)
    payload = _build_payload("gamma", "2.0.0")
    signature = sign_preset_payload(payload, private_key=private_key, key_id="author")
    blob = json.dumps({"preset": payload, "signature": signature.as_dict()}).encode("utf-8")
    repo.import_payload(blob, signing_keys={"author": public_key_b64})

    with pytest.raises(ValueError):
        repo.import_payload(blob, signing_keys={"author": public_key_b64})


def test_export_returns_requested_format(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)
    payload = _build_payload("delta", "3.0.0")
    signature = sign_preset_payload(payload, private_key=private_key, key_id="author")
    repo.import_payload(
        json.dumps({"preset": payload, "signature": signature.as_dict()}).encode("utf-8"),
        signing_keys={"author": public_key_b64},
    )

    document, data = repo.export_preset("delta", format="yaml", signing_keys={"author": public_key_b64})
    assert document.fmt == "yaml"
    assert b"preset:" in data
    assert document.verification.verified is True


def test_parse_unsigned_requires_signature(tmp_path) -> None:
    repo = PresetRepository(tmp_path)
    payload = _build_payload("epsilon", "1.0")
    blob = json.dumps({"preset": payload}).encode("utf-8")

    with pytest.raises(ValueError):
        repo.import_payload(blob)
