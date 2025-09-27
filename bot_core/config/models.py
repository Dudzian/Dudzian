"""Modele konfiguracji dla nowej architektury."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from bot_core.exchanges.base import Environment


@dataclass(slots=True)
class EnvironmentConfig:
    """Konfiguracja środowiska (np. live, paper, testnet)."""

    name: str
    exchange: str
    environment: Environment
    keychain_key: str
    data_cache_path: str
    risk_profile: str
    alert_channels: Sequence[str]
    ip_allowlist: Sequence[str] = field(default_factory=tuple)
    credential_purpose: str = "trading"


@dataclass(slots=True)
class RiskProfileConfig:
    """Parametry wykorzystywane do inicjalizacji profili ryzyka."""

    name: str
    max_daily_loss_pct: float
    max_position_pct: float
    target_volatility: float
    max_leverage: float
    stop_loss_atr_multiple: float
    max_open_positions: int
    hard_drawdown_pct: float


@dataclass(slots=True)
class SMSProviderSettings:
    """Definicja lokalnego dostawcy SMS z konfiguracji."""

    name: str
    provider_key: str
    api_base_url: str
    from_number: str
    recipients: Sequence[str]
    allow_alphanumeric_sender: bool = False
    sender_id: str | None = None
    credential_key: str | None = None


@dataclass(slots=True)
class TelegramChannelSettings:
    """Konfiguracja kanału Telegram."""

    name: str
    chat_id: str
    token_secret: str
    parse_mode: str = "MarkdownV2"


@dataclass(slots=True)
class EmailChannelSettings:
    """Konfiguracja kanału e-mail."""

    name: str
    host: str
    port: int
    from_address: str
    recipients: Sequence[str]
    credential_secret: str | None = None
    use_tls: bool = True


@dataclass(slots=True)
class CoreConfig:
    """Najwyższego poziomu konfiguracja aplikacji."""

    environments: Mapping[str, EnvironmentConfig]
    risk_profiles: Mapping[str, RiskProfileConfig]
    reporting: Mapping[str, str]
    sms_providers: Mapping[str, SMSProviderSettings]
    telegram_channels: Mapping[str, TelegramChannelSettings]
    email_channels: Mapping[str, EmailChannelSettings]


__all__ = [
    "EnvironmentConfig",
    "RiskProfileConfig",
    "SMSProviderSettings",
    "TelegramChannelSettings",
    "EmailChannelSettings",
    "CoreConfig",
]
