from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Mapping

import pytest

from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import CapabilityGuard
from bot_core.security.license import LicenseValidationResult
from bot_core.security.signing import build_hmac_signature
from bot_core.security.update import verify_update_bundle


@pytest.fixture()
def license_result(tmp_path: Path) -> LicenseValidationResult:
    license_path = tmp_path / "license.json"
    license_path.write_text("{}", encoding="utf-8")
    payload: Mapping[str, object] = {
        "edition": "pro",
        "environments": ["demo", "paper", "live"],
        "modules": {"oem_updater": True, "walk_forward": True},
        "runtime": {"multi_strategy_scheduler": True},
        "maintenance_until": "2026-01-01",
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date(2025, 1, 1))
    guard = CapabilityGuard(capabilities)
    return LicenseValidationResult(
        status="ok",
        fingerprint="fp",
        license_path=license_path,
        issued_at=None,
        expires_at=None,
        fingerprint_source="local",
        profile="paper",
        issuer="issuer",
        schema="schema",
        schema_version="1.0",
        license_id="L-1",
        revocation_list_path=None,
        revocation_status=None,
        revocation_reason=None,
        revocation_revoked_at=None,
        revocation_generated_at=None,
        revocation_checked=True,
        revocation_signature_key=None,
        errors=[],
        warnings=[],
        payload=None,
        license_signature_key="key",
        fingerprint_signature_key=None,
        capabilities=capabilities,
        capability_guard=guard,
    )


def _create_manifest(
    tmp_path: Path,
    *,
    allowed_profiles: list[str] | None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[Path, Path, bytes]:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    artifact_path = bundle_dir / "daemon" / "runtime.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"payload")

    manifest: dict[str, object] = {
        "version": "1.0.0",
        "platform": "linux",
        "runtime": "runtime.bin",
        "artifacts": [
            {"path": "daemon/runtime.bin", "sha384": "", "size": artifact_path.stat().st_size},
        ],
        "allowed_profiles": allowed_profiles,
    }
    if metadata:
        manifest["metadata"] = metadata
    # aktualizuj hash
    import hashlib

    digest = hashlib.sha384()
    digest.update(artifact_path.read_bytes())
    manifest["artifacts"][0]["sha384"] = digest.hexdigest()

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    key = b"secret"
    signature = build_hmac_signature(payload=manifest, key=key)
    signature_path = bundle_dir / "manifest.sig"
    signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, signature_path, key


def test_verify_update_success(tmp_path: Path, license_result: LicenseValidationResult) -> None:
    manifest_path, signature_path, key = _create_manifest(tmp_path, allowed_profiles=["paper", "live"])
    bundle_dir = manifest_path.parent

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=bundle_dir,
        signature_path=signature_path,
        hmac_key=key,
        license_result=license_result,
    )

    assert result.is_successful
    assert result.signature_valid
    assert result.signature_checked is True
    assert result.license_ok
    assert result.errors == []


def test_verify_update_rejects_profile(tmp_path: Path, license_result: LicenseValidationResult) -> None:
    manifest_path, signature_path, key = _create_manifest(tmp_path, allowed_profiles=["live"])
    bundle_dir = manifest_path.parent

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=bundle_dir,
        signature_path=signature_path,
        hmac_key=key,
        license_result=license_result,
    )

    assert not result.license_ok
    assert not result.is_successful
    assert result.errors
    assert "live" in result.errors[0]


def test_verify_update_requires_oem_module(tmp_path: Path, license_result: LicenseValidationResult) -> None:
    # usuń moduł OEM Updater z capabilities
    license_result.capabilities = build_capabilities_from_payload(
        {
            "edition": "pro",
            "environments": ["demo", "paper", "live"],
            "modules": {"oem_updater": False, "walk_forward": True},
            "runtime": {"multi_strategy_scheduler": True},
            "maintenance_until": "2026-01-01",
        },
        effective_date=date(2025, 1, 1),
    )
    license_result.capability_guard = CapabilityGuard(license_result.capabilities)

    manifest_path, signature_path, key = _create_manifest(tmp_path, allowed_profiles=["paper"])
    bundle_dir = manifest_path.parent

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=bundle_dir,
        signature_path=signature_path,
        hmac_key=key,
        license_result=license_result,
    )

    assert not result.license_ok
    assert "OEM Updater" in " ".join(result.errors)


def test_verify_update_respects_maintenance(tmp_path: Path, license_result: LicenseValidationResult) -> None:
    license_result.capabilities = build_capabilities_from_payload(
        {
            "edition": "pro",
            "environments": ["demo", "paper", "live"],
            "modules": {"oem_updater": True, "walk_forward": True},
            "runtime": {"multi_strategy_scheduler": True},
            "maintenance_until": "2024-01-01",
        },
        effective_date=date(2025, 1, 1),
    )
    license_result.capability_guard = CapabilityGuard(license_result.capabilities)

    manifest_path, signature_path, key = _create_manifest(tmp_path, allowed_profiles=["paper"])
    bundle_dir = manifest_path.parent

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=bundle_dir,
        signature_path=signature_path,
        hmac_key=key,
        license_result=license_result,
    )

    assert not result.license_ok
    assert any("utrzymaniowa" in err for err in result.errors)


def test_verify_update_checks_metadata_requirements(
    tmp_path: Path, license_result: LicenseValidationResult
) -> None:
    metadata = {
        "required_modules": ["ai_signals", "oem_updater"],
        "min_edition": "commercial",
    }
    manifest_path, signature_path, key = _create_manifest(
        tmp_path,
        allowed_profiles=["paper"],
        metadata=metadata,
    )
    bundle_dir = manifest_path.parent

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=bundle_dir,
        signature_path=signature_path,
        hmac_key=key,
        license_result=license_result,
    )

    assert not result.license_ok
    combined_errors = " ".join(result.errors)
    assert "ai_signals" in combined_errors
    assert "Edycja" in combined_errors
