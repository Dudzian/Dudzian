from __future__ import annotations

import os
from pathlib import Path

import pytest

from bot_core.marketplace.signing_key_policy import (
    DEV_FIXTURE_RELATIVE_PATH,
    DEV_HMAC_FIXTURE_RELATIVE_PATH,
    SigningKeyPolicyError,
    resolve_hmac_signing_key,
    resolve_signing_private_key,
    validate_signing_identity,
    validate_signing_issuer,
)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    fixture = root / DEV_FIXTURE_RELATIVE_PATH
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"DEV TEST FIXTURE - NOT KEY MATERIAL")
    (root / ".git").mkdir()
    return root


def test_production_without_explicit_key_fails_closed(repository: Path) -> None:
    with pytest.raises(SigningKeyPolicyError, match="explicit external --private-key"):
        resolve_signing_private_key(None, environment="production", repo_root=repository)


def test_production_rejects_repository_dev_key(repository: Path) -> None:
    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        resolve_signing_private_key(
            repository / DEV_FIXTURE_RELATIVE_PATH,
            environment="production",
            repo_root=repository,
        )


def test_production_rejects_relative_traversal_to_repository_key(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = repository.parent / "operator"
    outside.mkdir()
    monkeypatch.chdir(outside)

    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        resolve_signing_private_key(
            Path("../repository") / DEV_FIXTURE_RELATIVE_PATH,
            environment="production",
            repo_root=repository,
        )


def test_production_rejects_symlink_into_repository(repository: Path, tmp_path: Path) -> None:
    link = tmp_path / "external-looking.key"
    try:
        link.symlink_to(repository / DEV_FIXTURE_RELATIVE_PATH)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable on this platform: {exc}")

    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        resolve_signing_private_key(link, environment="production", repo_root=repository)


def test_production_rejects_tracked_repository_key(repository: Path, monkeypatch) -> None:
    fixture = (repository / DEV_FIXTURE_RELATIVE_PATH).resolve()
    monkeypatch.setattr(
        "bot_core.marketplace.signing_key_policy._is_tracked",
        lambda path, root: path == fixture and root == repository.resolve(),
    )

    with pytest.raises(SigningKeyPolicyError, match="tracked repository-contained"):
        resolve_signing_private_key(fixture, environment="production", repo_root=repository)


def test_production_accepts_explicit_external_key(repository: Path, tmp_path: Path) -> None:
    external = tmp_path / "protected-custody" / "production.key"
    external.parent.mkdir()
    external.write_bytes(os.urandom(32))

    assert (
        resolve_signing_private_key(external, environment="production", repo_root=repository)
        == external.resolve()
    )


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_dev_and_test_accept_designated_fixture(repository: Path, environment: str) -> None:
    assert (
        resolve_signing_private_key(None, environment=environment, repo_root=repository)
        == (repository / DEV_FIXTURE_RELATIVE_PATH).resolve()
    )


def test_production_without_hmac_fails_closed(repository: Path) -> None:
    with pytest.raises(SigningKeyPolicyError, match="explicit external.*--hmac-key"):
        resolve_hmac_signing_key(None, environment="production", repo_root=repository)


def test_production_rejects_repo_hmac(repository: Path) -> None:
    fixture = repository / DEV_HMAC_FIXTURE_RELATIVE_PATH
    fixture.write_bytes(b"DEV HMAC FIXTURE")

    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        resolve_hmac_signing_key(fixture, environment="production", repo_root=repository)


def test_production_rejects_tracked_hmac(repository: Path, monkeypatch) -> None:
    fixture = repository / DEV_HMAC_FIXTURE_RELATIVE_PATH
    fixture.write_bytes(b"DEV HMAC FIXTURE")
    resolved_fixture = fixture.resolve()
    monkeypatch.setattr(
        "bot_core.marketplace.signing_key_policy._is_tracked",
        lambda path, root: path == resolved_fixture and root == repository.resolve(),
    )

    with pytest.raises(SigningKeyPolicyError, match="tracked repository-contained"):
        resolve_hmac_signing_key(fixture, environment="production", repo_root=repository)


def test_production_rejects_hmac_symlink_into_repository(repository: Path, tmp_path: Path) -> None:
    fixture = repository / DEV_HMAC_FIXTURE_RELATIVE_PATH
    fixture.write_bytes(b"DEV HMAC FIXTURE")
    link = tmp_path / "external-looking-hmac.key"
    try:
        link.symlink_to(fixture)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable on this platform: {exc}")

    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        resolve_hmac_signing_key(link, environment="production", repo_root=repository)


def test_production_accepts_external_hmac(repository: Path, tmp_path: Path) -> None:
    external = tmp_path / "custody" / "catalog-hmac.key"
    external.parent.mkdir()
    external.write_bytes(os.urandom(32))

    assert (
        resolve_hmac_signing_key(external, environment="production", repo_root=repository)
        == external.resolve()
    )


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_dev_and_test_accept_default_hmac_fixture(repository: Path, environment: str) -> None:
    fixture = repository / DEV_HMAC_FIXTURE_RELATIVE_PATH
    fixture.write_bytes(b"DEV HMAC FIXTURE")

    assert (
        resolve_hmac_signing_key(None, environment=environment, repo_root=repository)
        == fixture.resolve()
    )


@pytest.mark.parametrize("environment", ["dev", "test"])
@pytest.mark.parametrize("key_id", ["dev-hmac", "dev-presets", "dev-presets-ed25519"])
def test_dev_and_test_accept_reserved_dev_identities(environment: str, key_id: str) -> None:
    assert (
        validate_signing_identity(key_id, environment=environment, option_name="--key-id") == key_id
    )


@pytest.mark.parametrize("key_id", ["dev-hmac", "dev-presets", "dev-presets-ed25519"])
def test_production_rejects_reserved_dev_identities(key_id: str) -> None:
    with pytest.raises(SigningKeyPolicyError, match=key_id):
        validate_signing_identity(key_id, environment="production", option_name="--key-id")


def test_production_accepts_explicit_authority_identity_and_issuer() -> None:
    assert (
        validate_signing_identity(
            "marketplace-prod-ed25519-v1",
            environment="production",
            option_name="--key-id",
        )
        == "marketplace-prod-ed25519-v1"
    )
    assert (
        validate_signing_issuer("marketplace-production", environment="production")
        == "marketplace-production"
    )


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_dev_and_test_accept_marketplace_ci_issuer(environment: str) -> None:
    assert validate_signing_issuer("marketplace-ci", environment=environment) == "marketplace-ci"
