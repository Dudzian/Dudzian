"""Trust-domain policy for Marketplace preset-signing private keys."""

from __future__ import annotations

import base64
import hashlib
import subprocess
from pathlib import Path

DEV_TEST_ENVIRONMENTS = frozenset({"dev", "test"})
PRODUCTION_ENVIRONMENT = "production"
DEV_FIXTURE_RELATIVE_PATH = Path("config/marketplace/keys/dev-presets-ed25519.key")
DEV_HMAC_FIXTURE_RELATIVE_PATH = Path("config/marketplace/keys/dev-hmac.key")
RESERVED_DEV_SIGNING_IDENTITIES = frozenset({"dev-hmac", "dev-presets", "dev-presets-ed25519"})
RESERVED_DEV_SIGNING_ISSUERS = frozenset({"marketplace-ci"})
KNOWN_DEV_PUBLIC_KEY_SHA256 = frozenset(
    {"c9950e9d553fcff4306531184569aaafacc9ed73b5c8592fee1e820c0f2ad6c6"}
)


class SigningKeyPolicyError(ValueError):
    """Raised when a signing key crosses the DEV/TEST/PRODUCTION trust boundary."""


def is_known_dev_public_key(material: bytes) -> bool:
    """Identify the committed DEV fixture by material, not by its filename or ID."""

    stripped = bytes(material).strip()
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except ValueError:
        decoded = stripped
    return hashlib.sha256(decoded).hexdigest() in KNOWN_DEV_PUBLIC_KEY_SHA256


def validate_signing_identity(key_id: str | None, *, environment: str, option_name: str) -> str:
    """Require an explicit, non-DEV authority identity for production signing."""

    normalized_environment = environment.strip().lower()
    normalized_id = key_id.strip() if key_id else ""
    if normalized_environment == PRODUCTION_ENVIRONMENT:
        if not normalized_id:
            raise SigningKeyPolicyError(f"PRODUCTION signing requires an explicit {option_name}")
        if normalized_id.lower() in RESERVED_DEV_SIGNING_IDENTITIES:
            raise SigningKeyPolicyError(
                f"PRODUCTION signing forbids reserved DEV identity {normalized_id!r}"
            )
    if not normalized_id:
        raise SigningKeyPolicyError(f"Signing identity {option_name} cannot be empty")
    return normalized_id


def validate_signing_issuer(issuer: str | None, *, environment: str) -> str | None:
    """Prevent production artifacts from inheriting a DEV/CI issuer identity."""

    normalized_environment = environment.strip().lower()
    normalized_issuer = issuer.strip() if issuer else ""
    if normalized_environment == PRODUCTION_ENVIRONMENT:
        if not normalized_issuer:
            raise SigningKeyPolicyError("PRODUCTION signing requires an explicit --issuer")
        if normalized_issuer.lower() in RESERVED_DEV_SIGNING_ISSUERS:
            raise SigningKeyPolicyError(
                f"PRODUCTION signing forbids reserved DEV issuer {normalized_issuer!r}"
            )
    return normalized_issuer or None


def repository_root() -> Path:
    """Return the physical repository root containing this module."""

    return Path(__file__).resolve().parents[2]


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_tracked(path: Path, repo_root: Path) -> bool:
    if not _is_within(path, repo_root):
        return False
    relative = path.relative_to(repo_root)
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", str(relative)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def resolve_signing_secret(
    secret_path: str | Path | None,
    *,
    environment: str,
    dev_fixture: str | Path,
    option_name: str,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve and enforce custody policy before private signing material is read.

    DEV and TEST may use the designated repository fixture. PRODUCTION requires an
    explicit path and rejects every path whose physical target is in the repository,
    including symlinks, relative traversal, ``.git`` and tracked repository files.
    ``Path.resolve(strict=True)`` also follows Windows junctions/reparse points.
    """

    normalized_environment = environment.strip().lower()
    if normalized_environment not in (*DEV_TEST_ENVIRONMENTS, PRODUCTION_ENVIRONMENT):
        raise SigningKeyPolicyError(f"Unsupported signing environment: {environment!r}")

    root = Path(repo_root).expanduser().resolve(strict=True) if repo_root else repository_root()
    if secret_path is None:
        if normalized_environment == PRODUCTION_ENVIRONMENT:
            raise SigningKeyPolicyError(
                f"PRODUCTION signing requires an explicit external {option_name}"
            )
        secret_path = root / dev_fixture

    resolved = Path(secret_path).expanduser().resolve(strict=True)
    if normalized_environment == PRODUCTION_ENVIRONMENT:
        # The containment check is physical, not a string-prefix check.  Keep the
        # tracked-file check explicit as a defence-in-depth custody invariant.
        if _is_within(resolved, root):
            tracked = _is_tracked(resolved, root)
            qualifier = "tracked " if tracked else ""
            raise SigningKeyPolicyError(
                f"PRODUCTION signing secret cannot be a {qualifier}repository-contained file: {resolved}"
            )

    return resolved


def resolve_signing_private_key(
    key_path: str | Path | None,
    *,
    environment: str,
    repo_root: str | Path | None = None,
) -> Path:
    """Compatibility wrapper for the Marketplace Ed25519 signing key."""

    return resolve_signing_secret(
        key_path,
        environment=environment,
        dev_fixture=DEV_FIXTURE_RELATIVE_PATH,
        option_name="--private-key",
        repo_root=repo_root,
    )


def resolve_hmac_signing_key(
    key_path: str | Path | None,
    *,
    environment: str,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve a Marketplace HMAC key under the shared signing-secret policy."""

    return resolve_signing_secret(
        key_path,
        environment=environment,
        dev_fixture=DEV_HMAC_FIXTURE_RELATIVE_PATH,
        option_name="--hmac-key/--signing-key",
        repo_root=repo_root,
    )


__all__ = [
    "DEV_FIXTURE_RELATIVE_PATH",
    "DEV_HMAC_FIXTURE_RELATIVE_PATH",
    "DEV_TEST_ENVIRONMENTS",
    "PRODUCTION_ENVIRONMENT",
    "KNOWN_DEV_PUBLIC_KEY_SHA256",
    "RESERVED_DEV_SIGNING_IDENTITIES",
    "RESERVED_DEV_SIGNING_ISSUERS",
    "SigningKeyPolicyError",
    "repository_root",
    "is_known_dev_public_key",
    "resolve_hmac_signing_key",
    "resolve_signing_secret",
    "resolve_signing_private_key",
    "validate_signing_identity",
    "validate_signing_issuer",
]
