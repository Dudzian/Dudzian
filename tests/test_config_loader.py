"""Testy ładowania konfiguracji."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config import (
    EmailChannelSettings,
    InstrumentConfig,
    MessengerChannelSettings,
    SMSProviderSettings,
    SignalChannelSettings,
    TelegramChannelSettings,
    WhatsAppChannelSettings,
    load_core_config,
)


def test_load_core_config_reads_sms_providers(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        instrument_universes:
          core_multi_exchange:
            description: "Testowe uniwersum"
            instruments:
              BTC_USDT:
                base_asset: BTC
                quote_asset: USDT
                categories: [core]
                exchanges:
                  binance_spot: BTCUSDT
                  kraken_spot: XBTUSDT
                backfill:
                  - interval: 1d
                    lookback_days: 3650
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
          signal_channels:
            workstation:
              service_url: https://signal-gateway.local
              sender_number: "+48500100999"
              recipients: ["+48555111222"]
              credential_secret: signal_cli_token
              verify_tls: true
          whatsapp_channels:
            business:
              phone_number_id: "10987654321"
              recipients: ["48555111222"]
              token_secret: whatsapp_primary_token
              api_base_url: https://graph.facebook.com
              api_version: v16.0
          messenger_channels:
            ops:
              page_id: "1357924680"
              recipients: ["2468013579"]
              token_secret: messenger_page_token
              api_base_url: https://graph.facebook.com
              api_version: v16.0
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
    assert isinstance(config.telegram_channels["primary"], TelegramChannelSettings)
    telegram = config.telegram_channels["primary"]
    assert telegram.chat_id == "123456789"
    assert telegram.token_secret == "telegram_primary_token"
    assert telegram.parse_mode == "MarkdownV2"
    signal = config.signal_channels["workstation"]
    assert isinstance(signal, SignalChannelSettings)
    assert signal.service_url == "https://signal-gateway.local"
    assert signal.sender_number == "+48500100999"
    assert signal.credential_secret == "signal_cli_token"
    whatsapp = config.whatsapp_channels["business"]
    assert isinstance(whatsapp, WhatsAppChannelSettings)
    assert whatsapp.phone_number_id == "10987654321"
    assert whatsapp.token_secret == "whatsapp_primary_token"
    assert whatsapp.api_version == "v16.0"
    messenger = config.messenger_channels["ops"]
    assert isinstance(messenger, MessengerChannelSettings)
    assert messenger.page_id == "1357924680"
    assert messenger.token_secret == "messenger_page_token"
    email = config.email_channels["ops"]
    assert isinstance(email, EmailChannelSettings)
    assert email.host == "smtp.example.com"
    assert email.port == 587
    assert email.from_address == "bot@example.com"
    assert email.recipients == ("ops@example.com",)
    assert email.credential_secret == "smtp_ops_credentials"
    assert email.use_tls is True
    universe = config.instrument_universes["core_multi_exchange"]
    assert universe.description == "Testowe uniwersum"
    assert len(universe.instruments) == 1
    instrument = universe.instruments[0]
    assert isinstance(instrument, InstrumentConfig)
    assert instrument.exchange_symbols["binance_spot"] == "BTCUSDT"
    assert instrument.backfill_windows[0].interval == "1d"
    assert instrument.backfill_windows[0].lookback_days == 3650



def test_load_core_config_loads_strategies(tmp_path: Path) -> None:
    config_path = tmp_path / 'core.yaml'
    config_path.write_text(
        """
        risk_profiles: {}
        strategies:
          core_daily_trend:
            engine: daily_trend_momentum
            parameters:
              fast_ma: 30
              slow_ma: 120
              breakout_lookback: 60
              momentum_window: 25
              atr_window: 15
              atr_multiplier: 1.8
              min_trend_strength: 0.01
              min_momentum: 0.002
        instrument_universes: {}
        environments: {}
        reporting: {}
        alerts: {}
        """,
        encoding='utf-8',
    )

    config = load_core_config(config_path)

    assert 'core_daily_trend' in config.strategies
    strategy = config.strategies['core_daily_trend']
    assert strategy.fast_ma == 30
    assert strategy.slow_ma == 120
    assert strategy.breakout_lookback == 60
    assert strategy.momentum_window == 25
    assert strategy.atr_window == 15
    assert abs(strategy.atr_multiplier - 1.8) < 1e-9
    assert abs(strategy.min_trend_strength - 0.01) < 1e-9
    assert abs(strategy.min_momentum - 0.002) < 1e-9
