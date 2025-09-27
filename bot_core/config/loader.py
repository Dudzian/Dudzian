"""Ładowanie konfiguracji z plików YAML."""
from __future__ import annotations

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


def _load_sms_providers(raw_alerts: Mapping[str, Any]) -> Mapping[str, SMSProviderSettings]:
    providers: dict[str, SMSProviderSettings] = {}
    for name, entry in raw_alerts.get("sms_providers", {}).items():
        providers[name] = SMSProviderSettings(
            name=name,
            provider_key=str(entry["provider"]),
            api_base_url=str(entry["api_base_url"]),
            from_number=str(entry["from_number"]),
            recipients=tuple(entry.get("recipients", ())),
            allow_alphanumeric_sender=bool(entry.get("allow_alphanumeric_sender", False)),
            sender_id=entry.get("sender_id"),
            credential_key=entry.get("credential_key"),
        )
    return providers


def load_core_config(path: str | Path) -> CoreConfig:
    """Wczytuje plik YAML i mapuje go na dataclasses."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    environments = {
        name: EnvironmentConfig(
            name=name,
            exchange=entry["exchange"],
            environment=Environment(entry["environment"]),
            keychain_key=entry["keychain_key"],
            data_cache_path=entry["data_cache_path"],
            risk_profile=entry["risk_profile"],
            alert_channels=tuple(entry.get("alert_channels", ())),
            ip_allowlist=tuple(entry.get("ip_allowlist", ())),
            credential_purpose=str(entry.get("credential_purpose", "trading")),
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

    reporting = raw.get("reporting", {})
    alerts = raw.get("alerts", {})
    sms_providers = _load_sms_providers(alerts)
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
            recipients=tuple(entry.get("recipients", ())),
            credential_secret=entry.get("credential_secret"),
            use_tls=bool(entry.get("use_tls", True)),
        )
        for name, entry in alerts.get("email_channels", {}).items()
    }

    return CoreConfig(
        environments=environments,
        risk_profiles=risk_profiles,
        reporting=reporting,
        sms_providers=sms_providers,
        telegram_channels=telegram_channels,
        email_channels=email_channels,
    )


__all__ = ["load_core_config"]
