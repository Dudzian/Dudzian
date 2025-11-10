from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.marketplace import (
    MarketplaceIndex,
    PresetRepository,
    sign_preset_payload,
)
from bot_core.marketplace.signed import SignedPresetMarketplace


class _DummyDescriptor:
    def __init__(self, preset_id: str) -> None:
        self.preset_id = preset_id


class _DummyCatalog:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register_signed_preset(self, document, hwid_provider=None):  # type: ignore[override]
        self.registered.append(document.preset_id)
        return _DummyDescriptor(document.preset_id)


def _write_signed_preset(path: Path, payload: dict[str, object], private_key: ed25519.Ed25519PrivateKey) -> None:
    signature = sign_preset_payload(payload, private_key=private_key, key_id="demo")
    document = {
        "preset": payload,
        "signature": signature.as_dict(),
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_signed_marketplace_import_and_update_plan(tmp_path: Path) -> None:
    repo = PresetRepository(tmp_path)
    key = ed25519.Ed25519PrivateKey.generate()

    beta_payload = {
        "name": "Dependency",
        "metadata": {
            "id": "beta",
            "version": "1.2.0",
        },
    }
    alpha_payload = {
        "name": "Primary",
        "metadata": {
            "id": "alpha",
            "version": "2.0.0",
            "dependencies": [
                {"preset_id": "beta", "version": ">=1.1.0"},
            ],
            "updates": {
                "channels": [{"name": "stable", "version": "2.0.0"}],
            },
        },
    }

    _write_signed_preset(tmp_path / "beta.json", beta_payload, key)
    _write_signed_preset(tmp_path / "alpha.json", alpha_payload, key)

    documents = repo.load_all()
    index = MarketplaceIndex.from_documents(documents)
    plan = index.plan_installation(["alpha"], installed_versions={"beta": "1.0.0"})

    assert plan.install_order == ("beta", "alpha")
    assert plan.upgrades and plan.upgrades[0].preset_id == "beta"
    plan_payload = plan.to_payload()
    assert plan_payload["installOrder"] == ["beta", "alpha"]
    assert plan_payload["upgrades"][0]["presetId"] == "beta"

    catalog = _DummyCatalog()
    marketplace = SignedPresetMarketplace(tmp_path, signing_keys={})
    sync_result = marketplace.sync(catalog)

    assert set(sync_result.installed) == {"alpha", "beta"}
    assert not sync_result.issues
