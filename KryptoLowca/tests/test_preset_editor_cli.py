from __future__ import annotations

import asyncio
import json
from pathlib import Path

from cryptography.fernet import Fernet

from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.scripts import preset_editor_cli


def test_cli_editor_applies_preset(tmp_path: Path, capsys) -> None:
    marketplace_dir = tmp_path / "marketplace"
    marketplace_dir.mkdir()
    preset_payload = {
        "id": "cli_demo",
        "name": "CLI Demo",
        "description": "Preset do testów CLI",
        "risk_level": "safe",
        "recommended_min_balance": 500,
        "timeframe": "1h",
        "exchanges": ["binance"],
        "tags": ["cli"],
        "version": "0.0.1",
        "last_updated": "2024-06-10T12:00:00+00:00",
        "compatibility": {"app": ">=2.8.0"},
        "compliance": {"required_flags": ["compliance_confirmed"]},
        "config": {
            "strategy": {
                "preset": "CLI",
                "mode": "demo",
            },
            "trade": {"max_open_positions": 3},
        },
    }
    (marketplace_dir / "cli_demo.json").write_text(
        json.dumps(preset_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    key = Fernet.generate_key().decode()

    exit_code = preset_editor_cli.main(
        [
            "--config-path",
            str(config_path),
            "--marketplace-dir",
            str(marketplace_dir),
            "--preset-id",
            "cli_demo",
            "--encryption-key",
            key,
            "--actor",
            "cli@example.com",
            "--set",
            "trade.max_open_positions=5",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "cli_demo" in output

    manager = ConfigManager(config_path, encryption_key=key.encode())
    asyncio.run(manager.load_config())
    assert manager._current_config["trade"]["max_open_positions"] == 5

    history = manager.get_preset_history("cli_demo")
    assert len(history) >= 2
