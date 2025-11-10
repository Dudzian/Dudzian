from __future__ import annotations

import hashlib
from pathlib import Path

from bot_core.config_marketplace.schema import load_catalog
from bot_core.security.marketplace_validator import MarketplaceValidator
from scripts.marketplace_cli import validate_release_metadata


def _read_key(path: Path) -> bytes:
    return path.read_bytes()


def test_catalog_integrity_and_release_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog_path = repo_root / "config" / "marketplace" / "catalog.json"
    catalog = load_catalog(catalog_path)

    assert catalog.schema_version == "1.1"
    errors, warnings = validate_release_metadata(catalog, repo_root)
    assert errors == []
    assert warnings == []

    signing_key_path = repo_root / "config" / "marketplace" / "keys" / "dev-hmac.key"
    validator = MarketplaceValidator(signing_keys={"dev-hmac": _read_key(signing_key_path)})
    results = validator.verify_catalog(catalog, repository_root=repo_root / "config" / "marketplace")
    for result in results:
        fingerprint_errors = [err for err in result.errors if "fingerprint" in err.lower()]
        other_errors = [err for err in result.errors if err not in fingerprint_errors]
        assert not other_errors, f"Nieoczekiwane błędy walidacji: {other_errors}"

    for package in catalog.packages:
        if package.versioning.source:
            source_path = repo_root / "config" / "marketplace" / "presets" / package.versioning.source
            assert source_path.exists(), f"Brak pliku źródłowego presetu {package.versioning.source}"
        for artifact in package.distribution:
            artifact_path = repo_root / "config" / "marketplace" / artifact.uri
            assert artifact_path.exists(), f"Brak artefaktu {artifact.uri}"
            blob = artifact_path.read_bytes()
            if artifact.integrity:
                digest = hashlib.new(artifact.integrity.normalized_algorithm(), blob).hexdigest()
                assert digest == artifact.integrity.digest
            if artifact.signature:
                assert artifact.signature.value, "Podpis HMAC musi być obecny"
