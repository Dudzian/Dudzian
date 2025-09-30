from __future__ import annotations

from pathlib import Path

from scripts import validate_config


_VALID_CONFIG = """
risk_profiles:
  balanced:
    max_daily_loss_pct: 0.01
    max_position_pct: 0.02
    target_volatility: 0.06
    max_leverage: 2.0
    stop_loss_atr_multiple: 1.0
    max_open_positions: 3
    hard_drawdown_pct: 0.05

alerts:
  telegram_channels:
    primary:
      chat_id: "123"
      token_secret: telegram_primary_token

environments:
  binance_paper:
    exchange: binance_spot
    environment: paper
    keychain_key: binance_paper_trading
    data_cache_path: ./var/data/binance_paper
    risk_profile: balanced
    alert_channels: ["telegram:primary"]
    required_permissions: [read]
    forbidden_permissions: [withdraw]
"""


_INVALID_CONFIG = """
risk_profiles:
  balanced:
    max_daily_loss_pct: -0.1
    max_position_pct: 0.02
    target_volatility: 0.06
    max_leverage: 2.0
    stop_loss_atr_multiple: 1.0
    max_open_positions: 3
    hard_drawdown_pct: 0.05

environments:
  binance_paper:
    exchange: binance_spot
    environment: paper
    keychain_key: binance_paper_trading
    data_cache_path: ./var/data/binance_paper
    risk_profile: unknown
    alert_channels: ["telegram:primary"]
    required_permissions: [read]
    forbidden_permissions: [read]
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


def test_validate_config_script_returns_success_for_valid_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _VALID_CONFIG)
    exit_code = validate_config.main(["--config", str(config_path), "--json"])
    assert exit_code == 0


def test_validate_config_script_returns_error_for_invalid_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _INVALID_CONFIG)
    exit_code = validate_config.main(["--config", str(config_path)])
    assert exit_code == 1
