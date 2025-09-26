from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from KryptoLowca.config_manager import ConfigManager, ValidationError


@pytest.mark.asyncio
async def test_apply_preset_creates_history(tmp_path: Path) -> None:
    marketplace_dir = tmp_path / "marketplace"
    marketplace_dir.mkdir()
    preset_payload = {
        "id": "demo",
        "name": "Demo",
        "description": "Preset testowy",
        "risk_level": "balanced",
        "recommended_min_balance": 1000,
        "timeframe": "1h",
        "exchanges": ["binance"],
        "tags": ["demo"],
        "version": "1.0.1",
        "last_updated": "2024-06-01T12:00:00+00:00",
        "compatibility": {"app": ">=2.9.0"},
        "compliance": {"required_flags": ["compliance_confirmed"]},
        "config": {
            "strategy": {
                "preset": "DEMO",
                "mode": "demo",
            },
            "exchange": {
                "exchange_name": "binance",
                "api_key": "demo-key",
                "api_secret": "demo-secret",
            },
        },
    }
    (marketplace_dir / "demo.json").write_text(
        json.dumps(preset_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    key = Fernet.generate_key()
    manager = ConfigManager(tmp_path / "config.yaml", encryption_key=key)
    manager.set_marketplace_directory(marketplace_dir)
    await manager.load_config()

    config = manager.apply_marketplace_preset(
        "demo",
        actor="user@example.com",
        note="initial import",
    )
    assert config["exchange"]["exchange_name"] == "binance"

    history = manager.get_preset_history("demo")
    assert len(history) == 1
    version_id = history[0]["version_id"]

    snapshot_path = tmp_path / "versions" / "demo" / f"{version_id}.json"
    stored = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert stored["config"]["exchange"]["api_key"] != "demo-key"

    config["trade"]["max_open_positions"] = 7
    await manager.save_config(
        config,
        actor="user@example.com",
        preset_id="demo",
        note="manual tuning",
        source="editor",
    )

    history = manager.get_preset_history("demo")
    assert len(history) == 2

    rolled_back = manager.rollback_preset(
        "demo",
        version_id,
        actor="auditor@example.com",
        note="rollback test",
    )
    assert rolled_back["trade"]["max_open_positions"] == manager._default_config()["trade"]["max_open_positions"]


@pytest.mark.asyncio
async def test_live_mode_requires_confirmation(tmp_path: Path) -> None:
    marketplace_dir = tmp_path / "marketplace"
    marketplace_dir.mkdir()
    payload = {
        "id": "live_demo",
        "name": "Live Demo",
        "description": "Preset wymagający trybu live",
        "risk_level": "balanced",
        "recommended_min_balance": 2500,
        "timeframe": "4h",
        "exchanges": ["binance"],
        "tags": ["live"],
        "version": "0.2.0",
        "last_updated": "2024-06-05T09:00:00+00:00",
        "compatibility": {"app": ">=2.9.0"},
        "compliance": {
            "required_flags": [
                "compliance_confirmed",
                "api_keys_configured",
                "acknowledged_risk_disclaimer",
            ]
        },
        "config": {
            "strategy": {
                "preset": "LIVE_TEST",
                "mode": "live",
                "compliance_confirmed": True,
                "api_keys_configured": True,
                "acknowledged_risk_disclaimer": True,
            },
            "exchange": {"exchange_name": "binance", "testnet": False},
        },
    }
    (marketplace_dir / "live_demo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    key = Fernet.generate_key()
    manager = ConfigManager(tmp_path / "config.yaml", encryption_key=key)
    manager.set_marketplace_directory(marketplace_dir)
    await manager.load_config()

    with pytest.raises(ValidationError):
        manager.apply_marketplace_preset("live_demo", actor="user@example.com")

    config = manager.apply_marketplace_preset(
        "live_demo",
        actor="user@example.com",
        user_confirmed=True,
        note="enable live",
    )
    assert config["strategy"]["mode"] == "live"
