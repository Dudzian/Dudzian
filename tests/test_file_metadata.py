"""Testy modułu bot_core.runtime.file_metadata."""

from __future__ import annotations

from pathlib import Path

from bot_core.runtime.file_metadata import file_reference_metadata


def test_file_reference_metadata_warns_on_world_writable_parent(tmp_path: Path) -> None:
    parent_dir = tmp_path / "insecure"
    parent_dir.mkdir()
    parent_dir.chmod(0o777)

    target = parent_dir / "key.pem"
    target.write_text("dummy", encoding="utf-8")

    metadata = file_reference_metadata(target, role="tls_key")

    warnings = metadata.get("security_warnings", [])
    assert any(
        "Katalog nadrzędny jest zapisywalny dla wszystkich użytkowników" in message
        for message in warnings
    ), warnings
    assert metadata.get("parent_security_flags", {}).get("world_writable") is True
