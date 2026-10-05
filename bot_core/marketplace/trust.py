"""External trust anchors for Marketplace signatures.

Signature documents are untrusted input.  In particular, an embedded
``public_key`` is compatibility metadata and is never a trust anchor.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Mapping

TRUST_DOMAINS = frozenset({"development", "test", "production"})
_ALIASES = {"dev": "development"}
_RESERVED_DEV_IDENTITIES = frozenset({"dev-hmac", "dev-presets", "dev-presets-ed25519"})
_RESERVED_DEV_ISSUERS = frozenset({"marketplace-ci"})
_KNOWN_DEV_PUBLIC_KEY_SHA256 = frozenset(
    {"c9950e9d553fcff4306531184569aaafacc9ed73b5c8592fee1e820c0f2ad6c6"}
)


def _key_fingerprint(material: bytes | str) -> str:
    raw = material.encode("utf-8") if isinstance(material, str) else bytes(material)
    stripped = raw.strip()
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except (ValueError, TypeError):
        decoded = stripped
    return hashlib.sha256(decoded).hexdigest()


def normalize_trust_domain(value: str) -> str:
    domain = _ALIASES.get(value.strip().lower(), value.strip().lower())
    if domain not in TRUST_DOMAINS:
        raise ValueError(f"unsupported Marketplace trust domain: {value!r}")
    return domain


@dataclass(frozen=True, slots=True)
class TrustedMarketplaceKey:
    public_key: bytes | str
    issuers: frozenset[str]
    environments: frozenset[str]

    def __post_init__(self) -> None:
        if not self.issuers or any(not item.strip() for item in self.issuers):
            raise ValueError("trusted Marketplace key requires non-empty issuers")
        normalized = frozenset(normalize_trust_domain(item) for item in self.environments)
        if not normalized:
            raise ValueError("trusted Marketplace key requires trust domains")
        object.__setattr__(self, "environments", normalized)


@dataclass(frozen=True, slots=True)
class MarketplaceTrustPolicy:
    environment: str
    keys: Mapping[str, TrustedMarketplaceKey]
    allow_legacy_missing_environment: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", normalize_trust_domain(self.environment))
        if self.environment == "production" and not self.keys:
            raise ValueError("production Marketplace trust policy requires a trust anchor")
        if self.environment == "production":
            for key_id, trusted in self.keys.items():
                if key_id.strip().lower() in _RESERVED_DEV_IDENTITIES:
                    raise ValueError(
                        "production Marketplace policy rejects DEV/TEST key identities"
                    )
                if any(issuer.lower() in _RESERVED_DEV_ISSUERS for issuer in trusted.issuers):
                    raise ValueError("production Marketplace policy rejects DEV/TEST issuers")
                if _key_fingerprint(trusted.public_key) in _KNOWN_DEV_PUBLIC_KEY_SHA256:
                    raise ValueError("production Marketplace policy rejects known DEV key material")

    def authorize(
        self, *, key_id: object, issuer: object, environment: object
    ) -> tuple[TrustedMarketplaceKey | None, tuple[str, ...]]:
        key_id_text = key_id.strip() if isinstance(key_id, str) else ""
        if not key_id_text:
            return None, ("missing-key-id",)
        trusted = self.keys.get(key_id_text)
        if trusted is None:
            return None, ("unknown-key-id",)
        if self.environment not in trusted.environments:
            return None, ("key-not-authorized-for-environment",)

        issuer_text = issuer.strip() if isinstance(issuer, str) else ""
        if not issuer_text or issuer_text not in trusted.issuers:
            return None, ("issuer-not-authorized",)

        environment_text = environment.strip() if isinstance(environment, str) else ""
        if (
            not environment_text
            and self.allow_legacy_missing_environment
            and self.environment != "production"
        ):
            environment_text = self.environment
        try:
            artifact_environment = normalize_trust_domain(environment_text)
        except ValueError:
            return None, ("invalid-environment",)
        if artifact_environment != self.environment:
            return None, ("environment-mismatch",)
        return trusted, ()


__all__ = [
    "MarketplaceTrustPolicy",
    "TRUST_DOMAINS",
    "TrustedMarketplaceKey",
    "normalize_trust_domain",
]
