"""Budowanie paczek offline `.kbot` wraz z rotacją fingerprintów OEM."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from bot_core.security.rotation import RotationRegistry
from core.update.offline_updater import OfflinePackageManifest, verify_kbot_package
from scripts.package_update import build_kbot_package


def _manifest_to_dict(manifest: OfflinePackageManifest) -> dict[str, Any]:
    return {
        "id": manifest.package_id,
        "version": manifest.version,
        "fingerprint": manifest.fingerprint,
        "created_at": manifest.created_at,
        "artifacts": [
            {"path": artifact.path, "size": artifact.size, "sha256": artifact.sha256}
            for artifact in manifest.artifacts
        ],
        "metadata": dict(manifest.metadata),
    }


def _serialize_rotation_status(status) -> dict[str, Any]:
    return {
        "purpose": status.purpose,
        "key": status.key,
        "last_rotated": status.last_rotated.isoformat().replace("+00:00", "Z") if status.last_rotated else None,
        "days_since_rotation": status.days_since_rotation,
        "due_in_days": status.due_in_days,
        "is_due": status.is_due,
        "is_overdue": status.is_overdue,
    }


@dataclass(slots=True)
class OfflineDistributionResult:
    """Artefakty wygenerowane podczas budowy paczki offline."""

    package_path: Path
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]


def build_offline_distribution(
    *,
    package_id: str,
    version: str,
    payload_dir: Path,
    output_path: Path,
    fingerprint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
    rotation_registry_path: Path | None = None,
    rotation_purpose: str = "offline_distribution",
    manifest_output: Path | None = None,
) -> OfflineDistributionResult:
    """Buduje paczkę `.kbot`, weryfikuje manifest i aktualizuje rejestr rotacji."""

    package_path = build_kbot_package(
        package_id=package_id,
        version=version,
        payload_dir=payload_dir,
        output_path=output_path,
        fingerprint=fingerprint,
        metadata=metadata,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )

    manifest, signature, staging_dir, _artifacts = verify_kbot_package(
        package_path,
        expected_fingerprint=fingerprint,
        hmac_key=signing_key,
    )
    try:
        manifest_dict = _manifest_to_dict(manifest)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    summary: dict[str, Any] = {
        "package_path": str(package_path),
        "package_size": package_path.stat().st_size,
        "manifest": manifest_dict,
        "signature_verified": signature is not None,
    }

    if rotation_registry_path and fingerprint:
        registry = RotationRegistry(rotation_registry_path)
        registry.mark_rotated(fingerprint, rotation_purpose, timestamp=datetime.now(timezone.utc))
        status = registry.status(fingerprint, rotation_purpose)
        summary["rotation"] = {
            "registry_path": str(Path(rotation_registry_path).expanduser()),
            "status": _serialize_rotation_status(status),
        }

    if manifest_output:
        manifest_output = Path(manifest_output).expanduser()
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["manifest_output"] = str(manifest_output)

    return OfflineDistributionResult(
        package_path=package_path,
        manifest=manifest_dict,
        summary=summary,
    )


__all__ = ["OfflineDistributionResult", "build_offline_distribution"]
