from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("pydantic")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.security.catalog_signatures import (
    verify_catalog_signature,
    verify_catalog_signature_file,
)
from scripts.build_marketplace_catalog import _write_signature, _write_utf8_lf


def test_write_utf8_lf_writes_exact_utf8_bytes(tmp_path: Path) -> None:
    target = tmp_path / "catalog.md"
    payload = "line-1\nline-2\n"

    written = _write_utf8_lf(target, payload)

    assert written == payload.encode("utf-8")
    assert target.read_bytes() == payload.encode("utf-8")


def test_write_utf8_lf_normalizes_crlf_to_lf(tmp_path: Path) -> None:
    target = tmp_path / "catalog.md"
    payload = "line-1\r\nline-2\rline-3\n"

    written = _write_utf8_lf(target, payload)

    assert written == b"line-1\nline-2\nline-3\n"
    assert target.read_bytes() == b"line-1\nline-2\nline-3\n"


def test_write_signature_uses_exact_content_bytes(tmp_path: Path) -> None:
    target = tmp_path / "catalog.md"
    # Simulates a Windows-style CRLF payload; signature must be computed from exact bytes.
    content = b"line-1\r\nline-2\r\n"
    target.write_bytes(content)

    hmac_key_id = "catalog-hmac"
    hmac_key = b"secret"
    ed_key = ed25519.Ed25519PrivateKey.generate()
    ed_key_id = "catalog-ed25519"

    _write_signature(
        target,
        content_bytes=content,
        hmac_key_id=hmac_key_id,
        signing_keys={hmac_key_id: hmac_key},
        ed25519_key=ed_key,
        ed25519_key_id=ed_key_id,
        issuer="tests",
    )

    signature_path = target.with_suffix(".md.sig")
    signature = json.loads(signature_path.read_text(encoding="utf-8"))

    assert signature["sha256"] == hashlib.sha256(content).hexdigest()
    assert signature["hmac"]["value"] == base64.b64encode(
        hmac.new(hmac_key, content, hashlib.sha256).digest()
    ).decode("ascii")

    public_key = ed_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    errors = verify_catalog_signature(
        content=content,
        signature=signature,
        hmac_key=hmac_key,
        ed25519_key=public_key,
        target_path=target,
    )
    assert errors == []


def test_verify_catalog_signature_file_for_json_and_markdown(tmp_path: Path) -> None:
    hmac_key_id = "catalog-hmac"
    hmac_key = b"secret"
    ed_key = ed25519.Ed25519PrivateKey.generate()
    ed_key_id = "catalog-ed25519"
    public_key = ed_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    for name, body in (
        ("catalog.json", '{"schema_version":"1.1"}\n'),
        ("catalog.md", "# Catalog\n"),
    ):
        target = tmp_path / name
        content = _write_utf8_lf(target, body)
        _write_signature(
            target,
            content_bytes=content,
            hmac_key_id=hmac_key_id,
            signing_keys={hmac_key_id: hmac_key},
            ed25519_key=ed_key,
            ed25519_key_id=ed_key_id,
            issuer="tests",
        )

        assert (
            verify_catalog_signature_file(target, hmac_key=hmac_key, ed25519_key=public_key) == []
        )
