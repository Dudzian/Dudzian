"""Testy skryptu generate_mtls_bundle."""

from __future__ import annotations

from pathlib import Path

from bot_core.security.rotation import RotationRegistry
from bot_core.security.tls_audit import audit_mtls_bundle
from scripts.generate_mtls_bundle import generate_mtls_bundle


def test_generate_mtls_bundle_creates_all_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    registry_path = tmp_path / "rotation.json"

    metadata = generate_mtls_bundle(
        output,
        ca_subject="/CN=Test CA/O=BotCore",
        server_subject="/CN=server.bot/O=BotCore",
        client_subject="/CN=client.bot/O=BotCore",
        validity_days=30,
        overwrite=True,
        rotation_registry=registry_path,
    )

    assert (output / "ca" / "ca.pem").exists()
    assert (output / "server" / "server.crt").exists()
    assert (output / "client" / "client.crt").exists()
    assert (output / "bundle.json").exists()
    assert metadata["bundle_path"] == str(output)

    audit_report = audit_mtls_bundle(output)
    assert audit_report["server"]["key_matches_certificate"] is True
    assert audit_report["client"]["key_matches_certificate"] is True

    registry = RotationRegistry(registry_path)
    server_status = registry.status("mtls", "server", interval_days=1)
    assert server_status.last_rotated is not None
