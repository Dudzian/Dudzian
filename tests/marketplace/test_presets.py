from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
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


def _write_signed_preset(
    directory: Path,
    *,
    filename: str,
    preset_id: str,
    private_key: ed25519.Ed25519PrivateKey,
    version: str = "1.0.0",
    key_id: str = "author",
) -> None:
    payload = _build_payload(preset_id, version)
    signature = sign_preset_payload(payload, private_key=private_key, key_id=key_id)
    document = {"preset": payload, "signature": signature.as_dict()}
    path = directory / filename
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


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


def test_load_all_supports_json_yaml_and_yml(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    _write_signed_preset(
        tmp_path,
        filename="alpha.json",
        preset_id="alpha",
        private_key=private_key,
        version="1.0.0",
    )
    _write_signed_preset(
        tmp_path,
        filename="beta.yaml",
        preset_id="beta",
        private_key=private_key,
        version="2.0.0",
    )
    _write_signed_preset(
        tmp_path,
        filename="gamma.yml",
        preset_id="gamma",
        private_key=private_key,
        version="3.0.0",
    )
    _write_signed_preset(
        tmp_path,
        filename="delta.YAML",
        preset_id="delta",
        private_key=private_key,
        version="4.0.0",
    )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert {doc.preset_id for doc in documents} == {"alpha", "beta", "gamma", "delta"}
    assert {doc.fmt for doc in documents} == {"json", "yaml"}
    assert all(doc.verification.verified for doc in documents)


def test_load_all_sorts_documents_by_filename(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    for name, preset_id in (
        ("Bravo.json", "bravo"),
        ("alpha.YAML", "alpha"),
        ("CHARLIE.yml", "charlie"),
    ):
        _write_signed_preset(
            tmp_path,
            filename=name,
            preset_id=preset_id,
            private_key=private_key,
        )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert [doc.preset_id for doc in documents] == ["alpha", "bravo", "charlie"]
    assert all(doc.verification.verified for doc in documents)


def test_load_all_uses_casefold_when_sorting(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    for name, preset_id in (
        ("Żuraw.YAML", "zuraw"),
        ("Źrebak.yml", "zrebak"),
        ("alpha.json", "alpha"),
    ):
        _write_signed_preset(
            tmp_path,
            filename=name,
            preset_id=preset_id,
            private_key=private_key,
        )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert [doc.preset_id for doc in documents] == ["alpha", "zrebak", "zuraw"]
    assert all(doc.verification.verified for doc in documents)


def test_load_all_sorts_ignoring_diacritics(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    for name, preset_id in (
        ("Ęlan.yml", "elan"),
        ("Łosoś.json", "losos"),
        ("Ósmy.YAML", "osmy"),
        ("alpha.json", "alpha"),
    ):
        _write_signed_preset(
            tmp_path,
            filename=name,
            preset_id=preset_id,
            private_key=private_key,
        )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert [doc.preset_id for doc in documents] == ["alpha", "elan", "losos", "osmy"]
    assert all(doc.verification.verified for doc in documents)


def test_load_all_uses_natural_ordering_for_numbers(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    for name, preset_id in (
        ("preset10.json", "preset10"),
        ("preset2.yaml", "preset2"),
        ("preset01.yml", "preset01"),
        ("preset9.json", "preset9"),
    ):
        _write_signed_preset(
            tmp_path,
            filename=name,
            preset_id=preset_id,
            private_key=private_key,
        )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert [doc.preset_id for doc in documents] == ["preset01", "preset2", "preset9", "preset10"]
    assert all(doc.verification.verified for doc in documents)


def test_load_all_handles_roman_numerals(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    for name, preset_id in (
        ("preset-IX.yaml", "preset_ix"),
        ("preset-IV.json", "preset_iv"),
        ("preset-V.yml", "preset_v"),
    ):
        _write_signed_preset(
            tmp_path,
            filename=name,
            preset_id=preset_id,
            private_key=private_key,
        )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert [doc.preset_id for doc in documents] == ["preset_iv", "preset_v", "preset_ix"]
    assert all(doc.verification.verified for doc in documents)


def test_load_all_handles_unicode_numerals(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    for name, preset_id in (
        ("preset-6.json", "digit_six"),
        ("preset-VI.yml", "roman_six"),
        ("preset-Ⅶ.yaml", "unicode_roman_seven"),
        ("preset-⑧.json", "circled_eight"),
    ):
        _write_signed_preset(
            tmp_path,
            filename=name,
            preset_id=preset_id,
            private_key=private_key,
        )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert [
        doc.preset_id for doc in documents
    ] == ["digit_six", "roman_six", "unicode_roman_seven", "circled_eight"]
    assert all(doc.verification.verified for doc in documents)


def test_load_all_handles_compound_unicode_roman_numerals(tmp_path, ed25519_keypair) -> None:
    private_key, public_key_b64 = ed25519_keypair
    repo = PresetRepository(tmp_path)

    for name, preset_id in (
        ("preset-XI.json", "roman_eleven"),
        ("preset-Ⅻ.yaml", "unicode_roman_twelve"),
        ("preset-⑬.yml", "circled_thirteen"),
        ("preset-ⅩⅨ.json", "unicode_roman_nineteen"),
    ):
        _write_signed_preset(
            tmp_path,
            filename=name,
            preset_id=preset_id,
            private_key=private_key,
        )

    documents = repo.load_all(signing_keys={"author": public_key_b64})

    assert [
        doc.preset_id for doc in documents
    ] == [
        "roman_eleven",
        "unicode_roman_twelve",
        "circled_thirteen",
        "unicode_roman_nineteen",
    ]
    assert all(doc.verification.verified for doc in documents)


def test_parse_unsigned_requires_signature(tmp_path) -> None:
    repo = PresetRepository(tmp_path)
    payload = _build_payload("epsilon", "1.0")
    blob = json.dumps({"preset": payload}).encode("utf-8")

    with pytest.raises(ValueError):
        repo.import_payload(blob)
