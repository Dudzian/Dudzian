from __future__ import annotations

import json
from pathlib import Path

from bot_core.security.tls_audit import verify_certificate_key_pair
from scripts.generate_mtls_bundle import BundleConfig, generate_bundle


def test_generate_mtls_bundle_creates_expected_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    rotation_registry = tmp_path / "rotation.json"
    config = BundleConfig(
        output_dir=output_dir,
        bundle_name="test-oem",
        common_name="Test OEM",
        organization="Acme",
        valid_days=90,
        key_size=2048,
        server_hostnames=("trading.internal", "127.0.0.1"),
        rotation_registry=rotation_registry,
        ca_key_passphrase=None,
        server_key_passphrase=None,
        client_key_passphrase=None,
    )

    metadata = generate_bundle(config)

    expected_files = {
        "ca_certificate": output_dir / "test-oem-ca.pem",
        "ca_key": output_dir / "test-oem-ca-key.pem",
        "server_certificate": output_dir / "test-oem-server.pem",
        "server_key": output_dir / "test-oem-server-key.pem",
        "client_certificate": output_dir / "test-oem-client.pem",
        "client_key": output_dir / "test-oem-client-key.pem",
        "metadata": output_dir / "test-oem-metadata.json",
    }
    for key, path in expected_files.items():
        assert path.exists(), f"Brak pliku {key}: {path}"

    ok, message = verify_certificate_key_pair(
        expected_files["server_certificate"], expected_files["server_key"]
    )
    assert ok, message

    metadata_content = json.loads(expected_files["metadata"].read_text(encoding="utf-8"))
    assert metadata_content["bundle"] == "test-oem"
    assert metadata["bundle"] == "test-oem"
    assert set(metadata_content["hostnames"]) == {"trading.internal", "127.0.0.1"}
    assert "server" in metadata_content["artifacts"]

    registry_payload = json.loads(rotation_registry.read_text(encoding="utf-8"))
    assert any(key.startswith("test-oem::tls_server") for key in registry_payload)

