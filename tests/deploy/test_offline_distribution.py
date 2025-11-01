from __future__ import annotations

import json
import shutil
from pathlib import Path

from deploy.packaging.offline_distribution import build_offline_distribution
from core.update.offline_updater import verify_kbot_package


def test_build_offline_distribution_creates_package(tmp_path: Path) -> None:
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    source_file = payload_dir / "example.txt"
    source_file.write_text("demo", encoding="utf-8")

    output_path = tmp_path / "bundle.kbot"
    manifest_output = tmp_path / "manifest.json"
    rotation_registry = tmp_path / "rotation.json"

    result = build_offline_distribution(
        package_id="demo",
        version="1.0.0",
        payload_dir=payload_dir,
        output_path=output_path,
        fingerprint="HW-123",
        metadata={"channel": "stable"},
        signing_key=b"super-secret",
        signing_key_id="test", 
        rotation_registry_path=rotation_registry,
        manifest_output=manifest_output,
    )

    assert output_path.exists()
    assert result.package_path == output_path
    assert manifest_output.exists()
    summary = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert summary["manifest"]["id"] == "demo"
    assert summary["rotation"]["status"]["key"] == "HW-123"
    assert rotation_registry.exists()

    manifest, signature, staging_dir, _ = verify_kbot_package(
        output_path,
        expected_fingerprint="HW-123",
        hmac_key=b"super-secret",
    )
    try:
        assert manifest.package_id == "demo"
        assert signature is not None
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
