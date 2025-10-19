from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.security.license import LicenseValidationResult
from bot_core.security.signing import build_hmac_signature
from bot_core.security.update import verify_update_bundle


@pytest.fixture()
def license_result(tmp_path: Path) -> LicenseValidationResult:
    license_path = tmp_path / "license.json"
    license_path.write_text("{}", encoding="utf-8")
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
    )


def _create_manifest(tmp_path: Path, *, allowed_profiles: list[str] | None) -> tuple[Path, Path, bytes]:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    artifact_path = bundle_dir / "daemon" / "runtime.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"payload")

    manifest = {
        "version": "1.0.0",
        "platform": "linux",
        "runtime": "runtime.bin",
        "artifacts": [
            {"path": "daemon/runtime.bin", "sha384": "", "size": artifact_path.stat().st_size},
        ],
        "allowed_profiles": allowed_profiles,
    }
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
