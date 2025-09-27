"""Testy ładowania konfiguracji."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config import SMSProviderSettings, load_core_config


def test_load_core_config_reads_sms_providers(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        reporting: {}
        alerts:
          sms_providers:
            orange_local:
              provider: orange_pl
              api_base_url: https://api.orange.pl/sms/v1
              from_number: "+48500100100"
              recipients: ["+48555111222"]
              allow_alphanumeric_sender: true
              sender_id: BOT-ORANGE
              credential_key: orange_sms_credentials
          telegram_channels:
            primary:
              chat_id: "123456789"
              token_secret: telegram_primary_token
              parse_mode: MarkdownV2
          email_channels:
            ops:
              host: smtp.example.com
              port: 587
              from_address: bot@example.com
              recipients: ["ops@example.com"]
              credential_secret: smtp_ops_credentials
              use_tls: true
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert "orange_local" in config.sms_providers
    provider = config.sms_providers["orange_local"]
    assert isinstance(provider, SMSProviderSettings)
    assert provider.provider_key == "orange_pl"
    assert provider.api_base_url == "https://api.orange.pl/sms/v1"
    assert provider.allow_alphanumeric_sender is True
    assert provider.sender_id == "BOT-ORANGE"
    assert provider.credential_key == "orange_sms_credentials"
    assert "primary" in config.telegram_channels
    telegram = config.telegram_channels["primary"]
    assert telegram.chat_id == "123456789"
    assert telegram.token_secret == "telegram_primary_token"
    assert telegram.parse_mode == "MarkdownV2"
    email = config.email_channels["ops"]
    assert email.host == "smtp.example.com"
    assert email.port == 587
    assert email.from_address == "bot@example.com"
    assert email.recipients == ("ops@example.com",)
    assert email.credential_secret == "smtp_ops_credentials"
    assert email.use_tls is True
