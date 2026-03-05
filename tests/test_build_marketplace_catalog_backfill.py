from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bot_core.config_marketplace.schema import MarketplaceCatalog, MarketplacePackageMetadata
from scripts.build_marketplace_catalog import _backfill_exchange_last_verified_at
from scripts.marketplace_cli import validate_release_metadata


def _base_metadata() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "package_id": "test_pkg.v1",
        "display_name": "Test Package",
        "summary": "Test summary",
        "description": "Test description",
        "version": "1.0.0",
        "revision": "2025.01",
        "release_date": "2025-01-01T00:00:00Z",
        "license": {
            "name": "Test License",
            "spdx_id": "Proprietary",
            "url": "https://example.com/license",
            "terms_summary": "Test terms",
            "redistributable": False,
            "commercial_use": False,
        },
        "tags": [],
        "data_requirements": {},
        "distribution": [
            {
                "name": "preset-document",
                "uri": "packages/test.json",
                "kind": "preset",
                "description": "Test preset",
                "size_bytes": 1,
                "integrity": {"algorithm": "sha256", "digest": "0" * 64},
            }
        ],
        "compatibility": [],
        "documentation_url": "https://example.com/doc",
        "release_notes": [],
        "security": {},
        "release": {
            "channel": "public",
            "review_status": "approved",
            "reviewers": [{"name": "QA", "email": "qa@example.com", "role": "qa"}],
            "ticket": "https://example.com/ticket",
            "approved_at": "2025-01-10T10:00:00Z",
            "notes": None,
        },
        "exchange_compatibility": [],
        "versioning": {
            "channel": "public",
            "iteration": "minor",
            "supersedes": [],
            "superseded_by": [],
        },
        "user_preferences": [],
    }


def test_backfill_sets_last_verified_at_from_approved_at() -> None:
    metadata = _base_metadata()
    metadata["exchange_compatibility"] = [
        {
            "exchange": "BINANCE",
            "environments": ["paper", "live"],
            "trading_modes": ["spot"],
            "status": "certified",
            "last_verified_at": None,
        }
    ]

    _backfill_exchange_last_verified_at(metadata)

    entry = metadata["exchange_compatibility"][0]
    assert entry["last_verified_at"] == metadata["release"]["approved_at"]


def test_validate_release_metadata_flags_missing_last_verified_at() -> None:
    metadata = _base_metadata()
    metadata["exchange_compatibility"] = [
        {
            "exchange": "BINANCE",
            "environments": ["paper"],
            "trading_modes": ["spot"],
            "status": "certified",
            "last_verified_at": None,
        }
    ]

    package = MarketplacePackageMetadata.model_validate(metadata)
    catalog = MarketplaceCatalog(
        schema_version="1.1",
        generated_at=datetime.now(timezone.utc),
        packages=[package],
    )

    errors, _warnings = validate_release_metadata(catalog, Path.cwd())

    assert any("wymaga last_verified_at" in error for error in errors)


def test_backfill_skips_when_no_approved_at() -> None:
    metadata = _base_metadata()
    metadata["release"]["approved_at"] = None
    metadata["release"]["review_status"] = "pending"
    metadata["exchange_compatibility"] = [
        {
            "exchange": "BINANCE",
            "environments": ["paper"],
            "trading_modes": ["spot"],
            "status": "certified",
            "last_verified_at": None,
        }
    ]

    _backfill_exchange_last_verified_at(metadata)

    assert metadata["exchange_compatibility"][0]["last_verified_at"] is None
