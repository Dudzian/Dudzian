from datetime import datetime, timedelta

import pytest

from bot_core.config_marketplace.schema import (
    ComponentDependency,
    ConfigurationMetadata,
    DataRequirement,
    IntegrityInfo,
    LicenseInfo,
    ValidationError,
)


def build_valid_metadata(**overrides):
    base = {
        "schema_version": "1.0.0",
        "config_id": "sample_config",
        "config_version": "2.1.3",
        "title": "Przykładowa konfiguracja",
        "description": "Opis konfiguracji.",
        "author": "Jan Kowalski",
        "author_contact": "jan@example.com",
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
            "spdx_id": "MIT",
        },
        "data_requirements": [
            {
                "name": "market_data",
                "description": "Dane rynkowe",
                "data_format": "json",
                "required": True,
                "schema_uri": "https://example.com/schema.json",
            }
        ],
        "component_dependencies": [
            {
                "component": "execution_engine",
                "min_version": "1.2.0",
                "max_version": "2.0.0",
            }
        ],
        "integrity": {
            "checksum": "a" * 64,
            "signature": "b" * 128,
            "signing_key_id": "sample-key",
            "signature_algorithm": "ed25519",
            "fingerprint_whitelist": ["fingerprint-1"],
        },
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
        "tags": ["strategia", "beta"],
    }
    base.update(overrides)
    return base


def test_configuration_metadata_validates_successfully():
    metadata = ConfigurationMetadata.model_validate(build_valid_metadata())

    assert metadata.config_id == "sample_config"
    assert metadata.license.spdx_id == "MIT"
    assert metadata.data_requirements[0].name == "market_data"
    assert metadata.component_dependencies[0].min_version == "1.2.0"
    assert metadata.integrity.checksum == "a" * 64


def test_configuration_metadata_rejects_invalid_versions():
    payload = build_valid_metadata(config_version="1.0")

    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)


def test_component_dependency_version_range_validation():
    with pytest.raises(ValidationError):
        ComponentDependency.model_validate(
            {
                "component": "risk_engine",
                "min_version": "2.0.0",
                "max_version": "1.0.0",
            }
        )


def test_tags_must_be_unique_and_well_formed():
    payload = build_valid_metadata(tags=["strategia", "strategia"])

    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)

    payload = build_valid_metadata(tags=["!"])
    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)


def test_updated_at_must_not_precede_created_at():
    payload = build_valid_metadata(
        created_at=datetime.utcnow().isoformat(),
        updated_at=(datetime.utcnow() - timedelta(hours=1)).isoformat(),
    )

    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)


def test_integrity_allows_optional_signature_and_fingerprint():
    metadata = ConfigurationMetadata.model_validate(
        build_valid_metadata(
            integrity={
                "checksum": "c" * 64,
                "signature": None,
                "signing_key_id": None,
                "signature_algorithm": None,
                "fingerprint_whitelist": None,
            }
        )
    )

    assert metadata.integrity.signature is None
    assert metadata.integrity.fingerprint_whitelist is None


def test_integrity_requires_complete_signature_information():
    payload = build_valid_metadata(
        integrity={
            "checksum": "d" * 64,
            "signature": "e" * 96,
            "signing_key_id": None,
            "signature_algorithm": "ed25519",
        }
    )

    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)

    payload = build_valid_metadata(
        integrity={
            "checksum": "d" * 64,
            "signature": "e" * 96,
            "signing_key_id": "primary",
            "signature_algorithm": None,
        }
    )

    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)


def test_integrity_rejects_duplicate_fingerprints():
    payload = build_valid_metadata(
        integrity={
            "checksum": "f" * 64,
            "signature": None,
            "signing_key_id": None,
            "signature_algorithm": None,
            "fingerprint_whitelist": ["fp-1", "fp-1"],
        }
    )

    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)


def test_extra_fields_are_forbidden():
    payload = build_valid_metadata()
    payload["unknown"] = "value"

    with pytest.raises(ValidationError):
        ConfigurationMetadata.model_validate(payload)
