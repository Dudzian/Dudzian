from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.api import server as runtime_server
from bot_core.config.models import RuntimeMarketplaceSettings
from bot_core.marketplace import MarketplaceService, PresetRepository, sign_preset_payload
from bot_core.security.hwid import HwIdProvider
from bot_core.strategies.catalog import StrategyCatalog
from bot_core.strategies.presets import MarketplacePresetInstaller


KEY_ID = "marketplace-production-2026"
ISSUER = "cryptohunter-marketplace-release"


def _settings(
    key: ed25519.Ed25519PrivateKey, *, legacy_public_key: str | None = None
) -> RuntimeMarketplaceSettings:
    public = base64.b64encode(key.public_key().public_bytes_raw()).decode("ascii")
    return RuntimeMarketplaceSettings(
        trust_environment="production",
        trusted_key_id=KEY_ID,
        trusted_public_key=public,
        trusted_issuer=ISSUER,
        # A hostile legacy fallback must be ignored by production composition.
        signing_keys={KEY_ID: legacy_public_key or public},
    )


def _payload() -> dict[str, object]:
    return {
        "name": "Production preset",
        "metadata": {"id": "production-preset", "version": "1.0.0"},
        "strategies": [
            {
                "name": "production-strategy",
                "engine": "mean_reversion",
                "parameters": {"budget": 1000, "risk_multiplier": 1.0},
            }
        ],
    }


def _artifact(key: ed25519.Ed25519PrivateKey, **overrides: str) -> bytes:
    signature = sign_preset_payload(
        _payload(),
        private_key=key,
        key_id=overrides.get("key_id", KEY_ID),
        issuer=overrides.get("issuer", ISSUER),
        environment=overrides.get("environment", "production"),
    ).as_dict()
    if overrides.get("remove_environment"):
        signature.pop("environment")
    return json.dumps({"preset": _payload(), "signature": signature}).encode()


def test_production_sign_bundle_install_runtime_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise sign -> release artifact -> installer -> runtime policy -> registration."""
    key = ed25519.Ed25519PrivateKey.generate()
    attacker_key = ed25519.Ed25519PrivateKey.generate()
    attacker_public = base64.b64encode(attacker_key.public_key().public_bytes_raw()).decode("ascii")
    settings = _settings(key, legacy_public_key=attacker_public)
    policy = settings.build_trust_policy()
    assert policy is not None

    bundle = tmp_path / "release-bundle"
    artifacts = bundle / "artifacts"
    catalog = bundle / "catalog"
    licenses = bundle / "licenses"
    artifacts.mkdir(parents=True)
    catalog.mkdir()
    licenses.mkdir()
    artifact = artifacts / "production-preset.json"
    artifact.write_bytes(_artifact(key))
    (catalog / "catalog.yaml").write_text(
        """schema_version: "1.0"
generated_at: "2026-01-01T00:00:00Z"
presets:
  - id: production-preset
    name: Production preset
    version: "1.0.0"
    author: {name: Release}
    artifact: ../artifacts/production-preset.json
""",
        encoding="utf-8",
    )
    (licenses / "production-preset.json").write_text(
        json.dumps({"preset_id": "production-preset", "allowed_fingerprints": ["device"]}),
        encoding="utf-8",
    )

    repository = PresetRepository(tmp_path / "runtime-presets")
    installer = MarketplacePresetInstaller(
        repository,
        catalog_path=catalog,
        licenses_dir=licenses,
        trust_policy=policy,
        environment="production",
        # This must never become the production authority.
        signing_keys={KEY_ID: attacker_key.public_key().public_bytes_raw()},
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "device"),
    )
    result = installer.install_from_catalog("production-preset")
    assert result.success and result.signature_verified

    # Drive the installed artifact through the real server composition root.  A
    # copied runtime tree keeps all relative core/config references realistic,
    # while the marketplace section points at this test's release installation.
    source_config = Path(__file__).resolve().parents[2] / "config"
    runtime_config_dir = tmp_path / "runtime-config"
    shutil.copytree(source_config, runtime_config_dir)
    runtime_config_path = runtime_config_dir / "runtime.yaml"
    raw_runtime = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
    raw_runtime["marketplace"] = {
        "enabled": True,
        "presets_path": str(repository.root),
        "trust_environment": "production",
        "trusted_key_id": KEY_ID,
        "trusted_public_key": settings.trusted_public_key,
        "trusted_issuer": ISSUER,
        "allow_legacy_missing_environment": False,
        "signing_keys": settings.signing_keys,
        "allow_unsigned": False,
    }
    runtime_config_path.write_text(yaml.safe_dump(raw_runtime), encoding="utf-8")

    runtime_catalog = StrategyCatalog(
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "device")
    )
    monkeypatch.setattr(runtime_server, "DEFAULT_STRATEGY_CATALOG", runtime_catalog)
    context = runtime_server.build_local_runtime_context(config_path=runtime_config_path)

    assert context.marketplace_environment == "production"
    assert context.marketplace_trust_policy is not None
    assert context.marketplace_trust_policy.environment == "production"
    assert tuple(context.marketplace_trust_policy.keys) == (KEY_ID,)
    assert context.marketplace_signing_keys[KEY_ID] == attacker_key.public_key().public_bytes_raw()
    assert runtime_catalog.preset("production-preset").signature_verified


@pytest.mark.parametrize(
    "factory",
    [
        lambda repo: MarketplaceService(repository=repo, environment="production"),
        lambda repo: MarketplacePresetInstaller(
            repo, catalog=type("C", (), {"presets": ()})(), environment="production"
        ),
        lambda repo: StrategyCatalog().sync_signed_marketplace(
            repo.root, signing_keys={"legacy": b"legacy"}, environment="production"
        ),
    ],
)
def test_production_composition_rejects_missing_policy(tmp_path: Path, factory) -> None:
    with pytest.raises(ValueError, match="external trust policy"):
        factory(PresetRepository(tmp_path))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"trusted_key_id": KEY_ID},
        {"trusted_key_id": KEY_ID, "trusted_public_key": "malformed", "trusted_issuer": ISSUER},
    ],
)
def test_malformed_production_trust_configuration_fails(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="trust configuration|Ed25519"):
        RuntimeMarketplaceSettings(trust_environment="production", **kwargs).build_trust_policy()
