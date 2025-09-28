from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.alerts import EmailChannel, SMSChannel, TelegramChannel
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.repository import FileRiskRepository
from bot_core.runtime import BootstrapContext, bootstrap_environment
from bot_core.security import SecretManager, SecretStorage


class _MemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


def _write_config(tmp_path: Path) -> Path:
    content = """
    risk_profiles:
      balanced:
        max_daily_loss_pct: 0.015
        max_position_pct: 0.05
        target_volatility: 0.11
        max_leverage: 3.0
        stop_loss_atr_multiple: 1.5
        max_open_positions: 5
        hard_drawdown_pct: 0.10
    environments:
      binance_paper:
        exchange: binance_spot
        environment: paper
        keychain_key: binance_paper_key
        credential_purpose: trading
        data_cache_path: ./var/data/binance_paper
        risk_profile: balanced
        alert_channels: ["telegram:primary", "email:ops", "sms:orange_local"]
        ip_allowlist: ["127.0.0.1"]
        alert_throttle:
          window_seconds: 60
          exclude_severities: [critical]
          exclude_categories: [health]
          max_entries: 32
      zonda_paper:
        exchange: zonda_spot
        environment: paper
        keychain_key: zonda_paper_key
        credential_purpose: trading
        data_cache_path: ./var/data/zonda_paper
        risk_profile: balanced
        alert_channels: ["telegram:primary"]
        ip_allowlist: ["127.0.0.1"]
        alert_throttle:
          window_seconds: 120
          exclude_severities: [critical]
          exclude_categories: [health]
          max_entries: 32
    reporting: {}
    alerts:
      telegram_channels:
        primary:
          chat_id: "123456789"
          token_secret: telegram_token
      email_channels:
        ops:
          host: smtp.example.com
          port: 587
          from_address: bot@example.com
          recipients: ["ops@example.com"]
          credential_secret: smtp_credentials
      sms_providers:
        orange_local:
          provider: orange_pl
          api_base_url: https://api.orange.pl/sms/v1
          from_number: BOT-ORANGE
          recipients: ["+48555111222"]
          allow_alphanumeric_sender: true
          sender_id: BOT-ORANGE
          credential_key: sms_orange
    """
    config_path = tmp_path / "core.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_bootstrap_environment_initialises_components(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")

    config_path = _write_config(tmp_path)
    credentials_payload = {
        "key_id": "paper-key",
        "secret": "paper-secret",
        "passphrase": None,
        "permissions": ["read", "trade"],
        "environment": Environment.PAPER.value,
    }
    storage.set_secret("tests:binance_paper_key:trading", json.dumps(credentials_payload))
    storage.set_secret("tests:zonda_paper_key:trading", json.dumps(credentials_payload))
    manager.store_secret_value("telegram_token", "telegram-secret", purpose="alerts:telegram")
    manager.store_secret_value(
        "smtp_credentials",
        json.dumps({"username": "bot", "password": "secret"}),
        purpose="alerts:email",
    )
    manager.store_secret_value(
        "sms_orange",
        json.dumps({"account_sid": "AC123", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert isinstance(context, BootstrapContext)
    assert context.environment.name == "binance_paper"
    assert context.credentials.key_id == "paper-key"
    assert context.adapter.credentials.key_id == "paper-key"

    assert isinstance(context.risk_engine, ThresholdRiskEngine)
    assert isinstance(context.risk_repository, FileRiskRepository)
    result = context.risk_engine.apply_pre_trade_checks(
        OrderRequest(symbol="BTCUSDT", side="buy", quantity=0.2, order_type="limit", price=100.0),
        account=AccountSnapshot(
            balances={"USDT": 1000.0},
            total_equity=1000.0,
            available_margin=1000.0,
            maintenance_margin=0.0,
        ),
        profile_name="balanced",
    )
    assert result.allowed is True

    assert set(context.alert_channels.keys()) == {
        "telegram:primary",
        "email:ops",
        "sms:orange_local",
    }
    assert isinstance(context.alert_channels["telegram:primary"], TelegramChannel)
    assert isinstance(context.alert_channels["email:ops"], EmailChannel)
    assert isinstance(context.alert_channels["sms:orange_local"], SMSChannel)
    assert len(context.alert_router.channels) == 3
    assert context.alert_router.audit_log is context.audit_log
    assert context.alert_router.throttle is not None
    assert context.alert_router.throttle.window.total_seconds() == pytest.approx(60.0)
    assert "critical" in context.alert_router.throttle.exclude_severities

    # Health check nie powinien zgłaszać wyjątków dla świeżo zainicjalizowanych kanałów.
    snapshot = context.alert_router.health_snapshot()
    assert snapshot["telegram:primary"]["status"] == "ok"

    assert context.risk_engine.should_liquidate(profile_name="balanced") is False
    assert context.adapter_settings == {}
    risk_state_path = Path("./var/data/binance_paper/risk_state/balanced.json")
    assert risk_state_path.parent.exists()


def test_bootstrap_environment_supports_zonda(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")

    config_content = """
    risk_profiles:
      conservative:
        max_daily_loss_pct: 0.01
        max_position_pct: 0.03
        target_volatility: 0.07
        max_leverage: 2.0
        stop_loss_atr_multiple: 1.0
        max_open_positions: 3
        hard_drawdown_pct: 0.05
    environments:
      zonda_live:
        exchange: zonda_spot
        environment: live
        keychain_key: zonda_live_key
        credential_purpose: trading
        data_cache_path: ./var/data/zonda_live
        risk_profile: conservative
        alert_channels: ["telegram:primary"]
        ip_allowlist: []
    reporting: {}
    alerts:
      telegram_channels:
        primary:
          chat_id: "123"
          token_secret: telegram_token
      email_channels: {}
      sms_providers: {}
      signal_channels: {}
      whatsapp_channels: {}
      messenger_channels: {}
    """

    config_path = tmp_path / "core.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    credentials_payload = {
        "key_id": "zonda-key",
        "secret": "zonda-secret",
        "permissions": ["read", "trade"],
        "environment": Environment.LIVE.value,
    }
    storage.set_secret("tests:zonda_live_key:trading", json.dumps(credentials_payload))
    manager.store_secret_value("telegram_token", "tg-token", purpose="alerts:telegram")

    context = bootstrap_environment("zonda_live", config_path=config_path, secret_manager=manager)

    assert context.adapter.name == "zonda_spot"
    assert context.credentials.key_id == "zonda-key"
    assert context.environment.exchange == "zonda_spot"
    assert context.adapter_settings == {}
