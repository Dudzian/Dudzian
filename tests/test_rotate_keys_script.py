from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

from bot_core.security.rotation import RotationRegistry
from scripts import rotate_keys


def _write_minimal_config(path: Path, cache_path: Path) -> None:
    path.write_text(
        f"""
risk_profiles: {{ paper: {{ name: paper, max_daily_loss_pct: 0.1, max_position_pct: 0.5, target_volatility: 0.1, max_leverage: 1.0, stop_loss_atr_multiple: 2.0, max_open_positions: 3, hard_drawdown_pct: 0.2 }} }}
instrument_universes: {{}}
instrument_buckets: {{}}
environments:
  paper:
    name: paper
    exchange: binance
    environment: paper
    keychain_key: binance_paper
    data_cache_path: {cache_path.as_posix()}
    risk_profile: paper
    alert_channels: []
strategies: {{}}
mean_reversion_strategies: {{}}
volatility_target_strategies: {{}}
cross_exchange_arbitrage_strategies: {{}}
multi_strategy_schedulers: {{}}
portfolio_governors: {{}}
reporting: {{}}
""",
        encoding="utf-8",
    )


def test_rotate_keys_updates_registry_and_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    cache_path = tmp_path / "cache"
    _write_minimal_config(config_path, cache_path)

    key_b64 = base64.b64encode(b"stage5_rotation_key").decode("ascii")
    output_path = tmp_path / "rotation.json"

    exit_code = rotate_keys.run(
        [
            "--config",
            str(config_path),
            "--environment",
            "paper",
            "--operator",
            "SecOps",
            "--executed-at",
            "2024-05-20T08:00:00Z",
            "--signing-key",
            key_b64,
            "--signing-key-id",
            "stage5",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["type"] == "stage5_key_rotation"
    assert payload["records"][0]["environment"] == "paper"
    assert payload["signature"]["algorithm"] == "HMAC-SHA256"

    registry_path = cache_path / "security" / "rotation_log.json"
    registry = RotationRegistry(registry_path)
    status = registry.status(
        "binance_paper",
        "trading",
        interval_days=90.0,
        now=datetime.fromisoformat("2024-05-20T08:00:00+00:00"),
    )

    assert status.last_rotated is not None
    assert status.last_rotated.isoformat().startswith("2024-05-20T08:00:00")


def test_rotate_keys_dry_run_does_not_touch_registry(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    cache_path = tmp_path / "cache"
    _write_minimal_config(config_path, cache_path)

    exit_code = rotate_keys.run(
        [
            "--config",
            str(config_path),
            "--operator",
            "Ops",
            "--dry-run",
            "--executed-at",
            "2024-05-01T00:00:00Z",
            "--output",
            str(tmp_path / "dry_run.json"),
        ]
    )

    assert exit_code == 0

    registry_path = cache_path / "security" / "rotation_log.json"
    assert not registry_path.exists()
