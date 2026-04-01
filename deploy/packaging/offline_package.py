"""Library helpers for building signed offline update packages `.dudzianpkg`."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from bot_core.security.signing import build_hmac_signature


def _hash_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_payload_archive(payload_dir: Path, target_path: Path) -> None:
    with tarfile.open(target_path, mode="w") as archive:
        for entry in sorted(payload_dir.rglob("*")):
            if entry.is_file():
                archive.add(entry, arcname=str(entry.relative_to(payload_dir)))


def build_offline_package(
    *,
    package_id: str,
    version: str,
    payload_dir: Path,
    output_path: Path,
    fingerprint: str | None = None,
    metadata: Mapping[str, object] | None = None,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
) -> Path:
    """Build a signed `.dudzianpkg` from the payload directory."""

    payload_dir = payload_dir.expanduser().resolve()
    if not payload_dir.exists() or not payload_dir.is_dir():
        raise FileNotFoundError(f"Katalog {payload_dir} nie istnieje lub nie jest katalogiem")

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(tempfile.mkdtemp(prefix="offline_pkg_build_"))
    try:
        payload_archive = staging_dir / "payload.tar"
        _build_payload_archive(payload_dir, payload_archive)

        artifacts = [
            {
                "path": "payload.tar",
                "size": payload_archive.stat().st_size,
                "sha256": _hash_sha256(payload_archive),
            }
        ]
        manifest: dict[str, object] = {
            "id": package_id,
            "version": version,
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "artifacts": artifacts,
        }
        if metadata:
            manifest["metadata"] = dict(metadata)

        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        signature_path = staging_dir / "manifest.sig"
        if signing_key is not None:
            signature = build_hmac_signature(manifest, key=signing_key, key_id=signing_key_id)
            signature_path.write_text(
                json.dumps(signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        with tarfile.open(output_path, mode="w:gz") as archive:
            archive.add(manifest_path, arcname="manifest.json")
            if signature_path.exists():
                archive.add(signature_path, arcname="manifest.sig")
            archive.add(payload_archive, arcname="payload.tar")

        return output_path
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


__all__ = ["build_offline_package"]
