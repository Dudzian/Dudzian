from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.marketplace import sign_preset_payload, verify_preset_signature
from bot_core.marketplace.signed import SignedPresetMarketplace
from bot_core.marketplace.trust import MarketplaceTrustPolicy, TrustedMarketplaceKey


ISSUER = "cryptohunter-marketplace-release"
PAYLOAD = {"name": "Trusted", "metadata": {"id": "trusted", "version": "1.0.0"}}


def _raw_public(key: ed25519.Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _policy(
    key: ed25519.Ed25519PrivateKey,
    *,
    environment: str = "production",
    key_id: str = "production-2026",
    issuer: str = ISSUER,
    key_environments: frozenset[str] | None = None,
) -> MarketplaceTrustPolicy:
    return MarketplaceTrustPolicy(
        environment=environment,
        keys={
            key_id: TrustedMarketplaceKey(
                public_key=_raw_public(key),
                issuers=frozenset({issuer}),
                environments=key_environments or frozenset({environment}),
            )
        },
    )


def _signature(
    key: ed25519.Ed25519PrivateKey,
    *,
    key_id: str = "production-2026",
    issuer: str = ISSUER,
    environment: str = "production",
) -> dict[str, object]:
    return sign_preset_payload(
        PAYLOAD,
        private_key=key,
        key_id=key_id,
        issuer=issuer,
        environment=environment,
    ).as_dict()


def _verify(signature: dict[str, object], policy: MarketplaceTrustPolicy):
    return verify_preset_signature(PAYLOAD, signature, trust_policy=policy)[0]


def test_production_accepts_only_authorized_external_anchor() -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    assert _verify(_signature(key), _policy(key)).verified


def test_test_domain_accepts_authorized_test_key() -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    policy = _policy(key, environment="test", key_id="dev-presets-ed25519", issuer="marketplace-ci")
    signature = _signature(
        key, key_id="dev-presets-ed25519", issuer="marketplace-ci", environment="test"
    )
    assert _verify(signature, policy).verified


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (lambda signature: signature.update(key_id="unknown"), "unknown-key-id"),
        (lambda signature: signature.update(issuer="attacker"), "issuer-not-authorized"),
        (lambda signature: signature.update(environment="test"), "environment-mismatch"),
        (lambda signature: signature.pop("environment"), "invalid-environment"),
    ],
)
def test_production_rejects_untrusted_metadata(mutation, expected_issue: str) -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    signature = _signature(key)
    mutation(signature)
    result = _verify(signature, _policy(key))
    assert not result.verified
    assert expected_issue in result.issues


def test_production_rejects_dev_key_even_if_signature_is_valid() -> None:
    dev_key = ed25519.Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="DEV/TEST"):
        _policy(
            dev_key,
            key_id="dev-presets-ed25519",
            issuer="marketplace-ci",
            key_environments=frozenset({"test"}),
        )


def test_production_rejects_known_dev_material_under_renamed_identity() -> None:
    dev_public = (
        Path(__file__).resolve().parents[2] / "config/marketplace/keys/dev-presets-ed25519.pub"
    ).read_bytes()
    with pytest.raises(ValueError, match="known DEV key material"):
        MarketplaceTrustPolicy(
            environment="production",
            keys={
                "renamed-production-key": TrustedMarketplaceKey(
                    public_key=dev_public,
                    issuers=frozenset({ISSUER}),
                    environments=frozenset({"production"}),
                )
            },
        )


def test_self_signed_and_foreign_key_artifacts_are_rejected() -> None:
    trusted = ed25519.Ed25519PrivateKey.generate()
    attacker = ed25519.Ed25519PrivateKey.generate()
    signature = _signature(attacker)
    result = _verify(signature, _policy(trusted))
    assert not result.verified
    assert "embedded-public-key-mismatch" in result.issues


def test_tampered_embedded_public_key_is_rejected() -> None:
    trusted = ed25519.Ed25519PrivateKey.generate()
    attacker = ed25519.Ed25519PrivateKey.generate()
    signature = _signature(trusted)
    signature["public_key"] = base64.b64encode(_raw_public(attacker)).decode("ascii")
    assert _verify(signature, _policy(trusted)).issues == ("embedded-public-key-mismatch",)


def test_production_policy_without_anchor_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires a trust anchor"):
        MarketplaceTrustPolicy(environment="production", keys={})


def test_runtime_does_not_register_dev_signed_preset(tmp_path: Path) -> None:
    dev_key = ed25519.Ed25519PrivateKey.generate()
    production_key = ed25519.Ed25519PrivateKey.generate()
    document = {
        "preset": PAYLOAD,
        "signature": _signature(
            dev_key, key_id="dev-presets-ed25519", issuer="marketplace-ci", environment="test"
        ),
    }
    (tmp_path / "preset.json").write_text(json.dumps(document), encoding="utf-8")

    class Catalog:
        registered: list[str] = []

        def register_signed_preset(self, document, hwid_provider=None):
            self.registered.append(document.preset_id)
            raise AssertionError("untrusted preset reached registration")

    result = SignedPresetMarketplace(
        tmp_path,
        signing_keys={},
        trust_policy=_policy(production_key),
    ).sync(Catalog())
    assert not result.installed
    assert result.skipped == ("trusted",)
    assert Catalog.registered == []
