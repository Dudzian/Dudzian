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
    instrument_universe: str | None = None


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
class InstrumentBackfillWindow:
    """Definicja zakresu danych historycznych dla danego interwału."""

    interval: str
    lookback_days: int


@dataclass(slots=True)
class InstrumentConfig:
    """Opis pojedynczego instrumentu w uniwersum."""

    name: str
    base_asset: str
    quote_asset: str
    categories: Sequence[str]
    exchange_symbols: Mapping[str, str]
    backfill_windows: Sequence[InstrumentBackfillWindow] = field(default_factory=tuple)


@dataclass(slots=True)
class InstrumentUniverseConfig:
    """Zbiór instrumentów przypisany do środowisk."""

    name: str
    description: str
    instruments: Sequence[InstrumentConfig]


@dataclass(slots=True)
class DailyTrendMomentumStrategyConfig:
    """Konfiguracja strategii trend/momentum."""

    name: str
    fast_ma: int
    slow_ma: int
    breakout_lookback: int
    momentum_window: int
    atr_window: int
    atr_multiplier: float
    min_trend_strength: float
    min_momentum: float


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
    instrument_universes: Mapping[str, InstrumentUniverseConfig]
    strategies: Mapping[str, DailyTrendMomentumStrategyConfig]
    reporting: Mapping[str, str]
    sms_providers: Mapping[str, SMSProviderSettings]
    telegram_channels: Mapping[str, TelegramChannelSettings]
    email_channels: Mapping[str, EmailChannelSettings]


__all__ = [
    "EnvironmentConfig",
    "RiskProfileConfig",
    "InstrumentBackfillWindow",
    "InstrumentConfig",
    "InstrumentUniverseConfig",
    "DailyTrendMomentumStrategyConfig",
    "SMSProviderSettings",
    "TelegramChannelSettings",
    "EmailChannelSettings",
    "CoreConfig",
]
