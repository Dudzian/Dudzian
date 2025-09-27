"""Ładowanie konfiguracji z plików YAML."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from bot_core.config.models import (
    CoreConfig,
    EmailChannelSettings,
    EnvironmentConfig,
    RiskProfileConfig,
    SMSProviderSettings,
    TelegramChannelSettings,
)
from bot_core.exchanges.base import Environment

# --- opcjonalne typy (mogą nie istnieć w Twojej gałęzi) ---
try:
    from bot_core.config.models import (
        InstrumentBackfillWindow,
        InstrumentConfig,
        InstrumentUniverseConfig,
    )
except Exception:  # brak rozszerzeń instrumentów
    InstrumentBackfillWindow = None  # type: ignore
    InstrumentConfig = None  # type: ignore
    InstrumentUniverseConfig = None  # type: ignore

try:
    from bot_core.config.models import DailyTrendMomentumStrategyConfig
except Exception:  # brak modułu strategii
    DailyTrendMomentumStrategyConfig = None  # type: ignore

# Dodatkowe kanały komunikatorów – w pełni opcjonalne
try:
    from bot_core.config.models import SignalChannelSettings  # type: ignore
except Exception:
    SignalChannelSettings = None  # type: ignore

try:
    from bot_core.config.models import WhatsAppChannelSettings  # type: ignore
except Exception:
    WhatsAppChannelSettings = None  # type: ignore

try:
    from bot_core.config.models import MessengerChannelSettings  # type: ignore
except Exception:
    MessengerChannelSettings = None  # type: ignore


def _core_has(field_name: str) -> bool:
    """Sprawdza, czy CoreConfig posiada dane pole (bezpiecznie dla różnych gałęzi)."""
    return any(f.name == field_name for f in fields(CoreConfig))


def _load_instrument_universes(raw: Mapping[str, Any]):
    if InstrumentUniverseConfig is None or InstrumentConfig is None or InstrumentBackfillWindow is None:
        return {}
    universes: dict[str, InstrumentUniverseConfig] = {}
    for name, entry in raw.get("instrument_universes", {}).items():
        instruments: list[InstrumentConfig] = []
        for instrument_name, instrument_data in (entry.get("instruments", {}) or {}).items():
            backfill_windows = tuple(
                InstrumentBackfillWindow(
                    interval=str(window["interval"]),
                    lookback_days=int(window["lookback_days"]),
                )
                for window in instrument_data.get("backfill", ()) or ()
            )
            instruments.append(
                InstrumentConfig(
                    name=instrument_name,
                    base_asset=str(instrument_data.get("base_asset", "")),
                    quote_asset=str(instrument_data.get("quote_asset", "")),
                    categories=tuple(instrument_data.get("categories", ()) or ()),
                    exchange_symbols={
                        str(ex_name): str(symbol)
                        for ex_name, symbol in (instrument_data.get("exchanges", {}) or {}).items()
                    },
                    backfill_windows=backfill_windows,
                )
            )
        universes[name] = InstrumentUniverseConfig(
            name=name,
            description=str(entry.get("description", "")),
            instruments=tuple(instruments),
        )
    return universes


def _load_sms_providers(raw_alerts: Mapping[str, Any]) -> Mapping[str, SMSProviderSettings]:
    providers: dict[str, SMSProviderSettings] = {}
    for name, entry in raw_alerts.get("sms_providers", {}).items():
        providers[name] = SMSProviderSettings(
            name=name,
            provider_key=str(entry["provider"]),
            api_base_url=str(entry["api_base_url"]),
            from_number=str(entry["from_number"]),
            recipients=tuple(entry.get("recipients", ()) or ()),
            allow_alphanumeric_sender=bool(entry.get("allow_alphanumeric_sender", False)),
            sender_id=entry.get("sender_id"),
            credential_key=entry.get("credential_key"),
        )
    return providers


def _load_signal_channels(raw_alerts: Mapping[str, Any]):
    if SignalChannelSettings is None:
        return {}
    channels: dict[str, SignalChannelSettings] = {}
    for name, entry in raw_alerts.get("signal_channels", {}).items():
        channels[name] = SignalChannelSettings(
            name=name,
            service_url=str(entry["service_url"]),
            sender_number=str(entry["sender_number"]),
            recipients=tuple(entry.get("recipients", ()) or ()),
            credential_secret=entry.get("credential_secret"),
            verify_tls=bool(entry.get("verify_tls", True)),
        )
    return channels


def _load_whatsapp_channels(raw_alerts: Mapping[str, Any]):
    if WhatsAppChannelSettings is None:
        return {}
    channels: dict[str, WhatsAppChannelSettings] = {}
    for name, entry in raw_alerts.get("whatsapp_channels", {}).items():
        channels[name] = WhatsAppChannelSettings(
            name=name,
            phone_number_id=str(entry["phone_number_id"]),
            recipients=tuple(entry.get("recipients", ()) or ()),
            token_secret=str(entry["token_secret"]),
            api_base_url=str(entry.get("api_base_url", "https://graph.facebook.com")),
            api_version=str(entry.get("api_version", "v16.0")),
        )
    return channels


def _load_messenger_channels(raw_alerts: Mapping[str, Any]):
    if MessengerChannelSettings is None:
        return {}
    channels: dict[str, MessengerChannelSettings] = {}
    for name, entry in raw_alerts.get("messenger_channels", {}).items():
        channels[name] = MessengerChannelSettings(
            name=name,
            page_id=str(entry["page_id"]),
            recipients=tuple(entry.get("recipients", ()) or ()),
            token_secret=str(entry["token_secret"]),
            api_base_url=str(entry.get("api_base_url", "https://graph.facebook.com")),
            api_version=str(entry.get("api_version", "v16.0")),
        )
    return channels


def _load_strategies(raw: Mapping[str, Any]):
    if DailyTrendMomentumStrategyConfig is None:
        return {}
    strategies: dict[str, DailyTrendMomentumStrategyConfig] = {}
    for name, entry in raw.get("strategies", {}).items():
        if str(entry.get("engine", "")) != "daily_trend_momentum":
            continue
        params = entry.get("parameters", {}) or {}
        strategies[name] = DailyTrendMomentumStrategyConfig(
            name=name,
            fast_ma=int(params.get("fast_ma", 20)),
            slow_ma=int(params.get("slow_ma", 100)),
            breakout_lookback=int(params.get("breakout_lookback", 55)),
            momentum_window=int(params.get("momentum_window", 20)),
            atr_window=int(params.get("atr_window", 14)),
            atr_multiplier=float(params.get("atr_multiplier", 2.0)),
            min_trend_strength=float(params.get("min_trend_strength", 0.005)),
            min_momentum=float(params.get("min_momentum", 0.0)),
        )
    return strategies


def load_core_config(path: str | Path) -> CoreConfig:
    """Wczytuje plik YAML i mapuje go na dataclasses."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    instrument_universes = _load_instrument_universes(raw)

    environments = {
        name: EnvironmentConfig(
            name=name,
            exchange=entry["exchange"],
            environment=Environment(entry["environment"]),
            keychain_key=entry["keychain_key"],
            data_cache_path=entry["data_cache_path"],
            risk_profile=entry["risk_profile"],
            alert_channels=tuple(entry.get("alert_channels", ()) or ()),
            ip_allowlist=tuple(entry.get("ip_allowlist", ()) or ()),
            credential_purpose=str(entry.get("credential_purpose", "trading")),
            # to pole istnieje tylko w rozszerzonej wersji models.py – ale przekazanie nadmiarowego
            # argumentu do dataclass wywoła błąd, więc ustawiamy je tu bezpiecznie na etapie budowy env
            instrument_universe=entry.get("instrument_universe"),
        )
        for name, entry in raw.get("environments", {}).items()
    }

    risk_profiles = {
        name: RiskProfileConfig(
            name=name,
            max_daily_loss_pct=float(entry["max_daily_loss_pct"]),
            max_position_pct=float(entry["max_position_pct"]),
            target_volatility=float(entry["target_volatility"]),
            max_leverage=float(entry["max_leverage"]),
            stop_loss_atr_multiple=float(entry["stop_loss_atr_multiple"]),
            max_open_positions=int(entry["max_open_positions"]),
            hard_drawdown_pct=float(entry["hard_drawdown_pct"]),
        )
        for name, entry in raw.get("risk_profiles", {}).items()
    }

    strategies = _load_strategies(raw)

    reporting = raw.get("reporting", {}) or {}
    alerts = raw.get("alerts", {}) or {}
    sms_providers = _load_sms_providers(alerts)
    signal_channels = _load_signal_channels(alerts)
    whatsapp_channels = _load_whatsapp_channels(alerts)
    messenger_channels = _load_messenger_channels(alerts)

    telegram_channels = {
        name: TelegramChannelSettings(
            name=name,
            chat_id=str(entry["chat_id"]),
            token_secret=str(entry["token_secret"]),
            parse_mode=str(entry.get("parse_mode", "MarkdownV2")),
        )
        for name, entry in alerts.get("telegram_channels", {}).items()
    }
    email_channels = {
        name: EmailChannelSettings(
            name=name,
            host=str(entry["host"]),
            port=int(entry.get("port", 587)),
            from_address=str(entry["from_address"]),
            recipients=tuple(entry.get("recipients", ()) or ()),
            credential_secret=entry.get("credential_secret"),
            use_tls=bool(entry.get("use_tls", True)),
        )
        for name, entry in alerts.get("email_channels", {}).items()
    }

    # Budujemy kwargs dynamicznie, tylko z polami obecnymi w CoreConfig
    core_kwargs: dict[str, Any] = {
        "environments": environments,
        "risk_profiles": risk_profiles,
        "reporting": reporting,
        "sms_providers": sms_providers,
        "telegram_channels": telegram_channels,
        "email_channels": email_channels,
    }
    if _core_has("instrument_universes"):
        core_kwargs["instrument_universes"] = instrument_universes
    if _core_has("strategies"):
        core_kwargs["strategies"] = strategies
    if _core_has("signal_channels"):
        core_kwargs["signal_channels"] = signal_channels
    if _core_has("whatsapp_channels"):
        core_kwargs["whatsapp_channels"] = whatsapp_channels
    if _core_has("messenger_channels"):
        core_kwargs["messenger_channels"] = messenger_channels

    return CoreConfig(**core_kwargs)  # type: ignore[arg-type]


__all__ = ["load_core_config"]
