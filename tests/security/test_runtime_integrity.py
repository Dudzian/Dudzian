from __future__ import annotations

from pathlib import Path

import json

import pytest

from bot_core.security.runtime_integrity import RuntimeIntegrityError, verify_bundle_integrity


def _write_manifest(root: Path, files: dict[str, bytes]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    entries = []
    for relative in sorted(files):
        path = root / relative
        entries.append({"path": relative, "sha384": _hash(path)})

    manifest_path = root / "resources" / "integrity_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"algorithm": "sha384", "files": entries}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha384()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_verify_bundle_integrity_success(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        {
            "daemon/runtime.bin": b"binary",
            "resources/config.json": b"{}",
        },
    )

    assert verify_bundle_integrity(tmp_path) is True


def test_verify_bundle_integrity_mismatch(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        {
            "daemon/runtime.bin": b"binary",
        },
    )

    (tmp_path / "daemon" / "runtime.bin").write_bytes(b"tampered")

    with pytest.raises(RuntimeIntegrityError):
        verify_bundle_integrity(tmp_path, strict=True)

    manifest.unlink()
    with pytest.raises(RuntimeIntegrityError):
        verify_bundle_integrity(tmp_path, strict=True)
