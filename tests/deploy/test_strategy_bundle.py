from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import stat
import zipfile
from pathlib import Path


import pytest

from bot_core.security.signing import canonical_json_bytes

from deploy.packaging.build_strategy_bundle import build_from_cli


REPO_ROOT = Path(__file__).resolve().parents[2]
def _read_zip_entry(archive: zipfile.ZipFile, name: str) -> bytes:
    with archive.open(name) as handle:
        return handle.read()


@pytest.mark.usefixtures("tmp_path")
def test_build_strategy_bundle_cli(tmp_path: Path) -> None:
    key_path = tmp_path / "strategy_signing.key"
    key_path.write_bytes(os.urandom(48))
    os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)

    output_dir = tmp_path / "dist"
    version = "2024.06.15"

    archive_path = build_from_cli(
        [
            "--version",
            version,
            "--signing-key-path",
            str(key_path),
            "--output-dir",
            str(output_dir),
            "--signing-key-id",
            "local-test",
            "--log-level",
            "DEBUG",
        ]
    )

    assert archive_path.exists()
    manifest_path = output_dir / f"stage4-strategies-{version}.manifest.json"
    signature_path = output_dir / f"stage4-strategies-{version}.manifest.sig"
    assert manifest_path.exists()
    assert signature_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle"] == "stage4-strategies"
    assert manifest["version"] == version
    assert len(manifest["strategies"]) >= 3
    strategy_names = {entry["name"] for entry in manifest["strategies"]}
    assert {"mean_reversion", "volatility_target", "cross_exchange_arbitrage"}.issubset(
        strategy_names
    )

    bundle_prefix = "stage4-strategies"
    with zipfile.ZipFile(archive_path, "r") as archive:
        zip_members = set(archive.namelist())
        assert f"{bundle_prefix}/manifest.json" in zip_members
        assert f"{bundle_prefix}/manifest.json.sig" in zip_members
        for entry in manifest["strategies"]:
            bundle_path = f"{bundle_prefix}/{entry['bundle_path']}"
            assert bundle_path in zip_members
            payload = _read_zip_entry(archive, bundle_path)
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        for entry in manifest["datasets"]:
            bundle_path = f"{bundle_prefix}/{entry['bundle_path']}"
            assert bundle_path in zip_members
            payload = _read_zip_entry(archive, bundle_path)
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]

        manifest_from_archive = json.loads(
            _read_zip_entry(archive, f"{bundle_prefix}/manifest.json").decode("utf-8")
        )
    assert manifest_from_archive == manifest

    signature_document = json.loads(signature_path.read_text(encoding="utf-8"))
    payload = signature_document["payload"]
    signature = base64.b64decode(signature_document["signature"]["value"])
    expected_signature = hmac.new(
        key_path.read_bytes(), canonical_json_bytes(payload), hashlib.sha384
    ).digest()
    assert signature == expected_signature
    assert signature_document["signature"]["key_id"] == "local-test"
