# test_config_manager.py
# -*- coding: utf-8 -*-
"""
Unit tests for config_manager.py.
"""
import asyncio
import pytest
import yaml
from pathlib import Path
from cryptography.fernet import Fernet

from config_manager import ConfigManager, ConfigError, ValidationError, AIConfig, DBConfig, TradeConfig, ExchangeConfig

@pytest.fixture
async def config_manager(tmp_path):
    config_path = tmp_path / "config.yaml"
    encryption_key = Fernet.generate_key()
    manager = await ConfigManager.create(config_path=str(config_path), encryption_key=encryption_key)
    return manager

@pytest.mark.asyncio
async def test_load_save_config(config_manager, tmp_path):
    config = {
        "ai": {"threshold_bps": 7.0, "model_types": ["rf", "lstm"], "seq_len": 20, "epochs": 10, "batch_size": 32, "model_dir": "models"},
        "db": {"db_url": "sqlite+aiosqlite:///test.db", "timeout_s": 15.0, "pool_size": 10, "max_overflow": 5},
        "trade": {"risk_per_trade": 0.02, "max_leverage": 2.0, "stop_loss_pct": 0.03, "take_profit_pct": 0.06, "max_open_positions": 3},
        "exchange": {"api_key": "test_key", "api_secret": "test_secret", "exchange_name": "binance", "testnet": True}
    }
    await config_manager.save_config(config)
    loaded = await config_manager.load_config()
    assert loaded["ai"]["threshold_bps"] == 7.0
    assert loaded["exchange"]["api_key"] == "test_key"  # Decrypted
    assert config_manager.config_path.exists()
    with open(config_manager.config_path, "r") as f:
        saved = yaml.safe_load(f)
    assert saved["ai"]["threshold_bps"] == 7.0

@pytest.mark.asyncio
async def test_user_config(config_manager):
    user_id = await config_manager.db_manager.ensure_user("tester@example.com")
    config = {
        "ai": {"threshold_bps": 10.0, "model_types": ["gb"], "seq_len": 30, "epochs": 15, "batch_size": 16},
        "trade": {"risk_per_trade": 0.015}
    }
    await config_manager.save_user_config(user_id, "custom", config)
    loaded = await config_manager.load_config(preset_name="custom", user_id=user_id)
    assert loaded["ai"]["threshold_bps"] == 10.0
    assert loaded["trade"]["risk_per_trade"] == 0.015

@pytest.mark.asyncio
async def test_specific_configs(config_manager):
    config = {
        "ai": {"threshold_bps": 6.0, "model_types": ["rf"], "seq_len": 25, "epochs": 20, "batch_size": 48},
        "db": {"db_url": "sqlite+aiosqlite:///test2.db", "timeout_s": 20.0},
        "trade": {"risk_per_trade": 0.01, "max_leverage": 1.5},
        "exchange": {"exchange_name": "kraken", "testnet": False}
    }
    await config_manager.save_config(config)
    ai_config = config_manager.load_ai_config()
    db_config = config_manager.load_db_config()
    trade_config = config_manager.load_trade_config()
    exchange_config = config_manager.load_exchange_config()
    assert ai_config.threshold_bps == 6.0
    assert db_config.db_url == "sqlite+aiosqlite:///test2.db"
    assert trade_config.max_leverage == 1.5
    assert exchange_config.exchange_name == "kraken"

@pytest.mark.asyncio
async def test_encryption(config_manager):
    config = {
        "exchange": {"api_key": "sensitive_key", "api_secret": "sensitive_secret"}
    }
    await config_manager.save_config(config)
    with open(config_manager.config_path, "r") as f:
        saved = yaml.safe_load(f)
    assert saved["exchange"]["api_key"] != "sensitive_key"  # Encrypted
    loaded = await config_manager.load_config()
    assert loaded["exchange"]["api_key"] == "sensitive_key"  # Decrypted

@pytest.mark.asyncio
async def test_invalid_config(config_manager):
    config = {
        "ai": {"threshold_bps": -1.0},  # Invalid
        "db": {"db_url": ""},  # Invalid
        "trade": {"risk_per_trade": 2.0},  # Invalid
        "exchange": {"api_key": 123}  # Invalid
    }
    with pytest.raises(ValidationError):
        await config_manager.save_config(config)

@pytest.mark.asyncio
async def test_default_config(config_manager):
    config = await config_manager.load_config()
    assert config["ai"]["threshold_bps"] == 5.0
    assert config["db"]["db_url"] == "sqlite+aiosqlite:///trading.db"
    assert config["trade"]["risk_per_trade"] == 0.01
    assert config["exchange"]["exchange_name"] == "binance"