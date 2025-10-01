"""Ładowanie konfiguracji z plików YAML."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from bot_core.config.models import (
    AlertThrottleConfig,
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

# Opcjonalna konfiguracja kontrolerów runtime
try:
    from bot_core.config.models import ControllerRuntimeConfig  # type: ignore
except Exception:
    ControllerRuntimeConfig = None  # type: ignore

# Opcjonalna konfiguracja audytu alertów (w nowszych gałęziach)
try:
    from bot_core.config.models import AlertAuditConfig  # type: ignore
except Exception:
    AlertAuditConfig = None  # type: ignore

try:
    from bot_core.config.models import DecisionJournalConfig  # type: ignore
except Exception:
    DecisionJournalConfig = None  # type: ignore

try:
    from bot_core.config.models import (
        CoreReportingConfig,
        SmokeArchiveLocalConfig,
        SmokeArchiveS3Config,
        SmokeArchiveUploadConfig,
    )  # type: ignore
except Exception:
    CoreReportingConfig = None  # type: ignore
    SmokeArchiveLocalConfig = None  # type: ignore
    SmokeArchiveS3Config = None  # type: ignore
    SmokeArchiveUploadConfig = None  # type: ignore


def _core_has(field_name: str) -> bool:
    """Sprawdza, czy CoreConfig posiada dane pole (bezpiecznie dla różnych gałęzi)."""
    return any(f.name == field_name for f in fields(CoreConfig))


def _env_has(field_name: str) -> bool:
    """Sprawdza, czy EnvironmentConfig posiada dane pole (bezpiecznie dla różnych gałęzi)."""
    return any(f.name == field_name for f in fields(EnvironmentConfig))


def _load_instrument_universes(raw: Mapping[str, Any]):
    if InstrumentUniverseConfig is None or InstrumentConfig is None or InstrumentBackfillWindow is None:
        return {}
    universes: dict[str, InstrumentUniverseConfig] = {}
    for name, entry in (raw.get("instrument_universes", {}) or {}).items():
        instruments: list[InstrumentConfig] = []
        for instrument_name, instrument_data in (entry.get("instruments", {}) or {}).items():
            backfill_windows = tuple(
                InstrumentBackfillWindow(
                    interval=str(window["interval"]),
                    lookback_days=int(window["lookback_days"]),
                )
                for window in (instrument_data.get("backfill", ()) or ())
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
    for name, entry in (raw_alerts.get("sms_providers", {}) or {}).items():
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
    for name, entry in (raw_alerts.get("signal_channels", {}) or {}).items():
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
    for name, entry in (raw_alerts.get("whatsapp_channels", {}) or {}).items():
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
    for name, entry in (raw_alerts.get("messenger_channels", {}) or {}).items():
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
    for name, entry in (raw.get("strategies", {}) or {}).items():
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


def _load_alert_throttle(entry: Optional[Mapping[str, Any]]) -> AlertThrottleConfig | None:
    if not entry:
        return None
    window_seconds = float(entry.get("window_seconds", entry.get("window", 0.0)))
    if window_seconds <= 0:
        raise ValueError("alert_throttle.window_seconds musi być dodatnie")
    exclude_severities = tuple(str(value).lower() for value in (entry.get("exclude_severities", ()) or ()))
    exclude_categories = tuple(str(value).lower() for value in (entry.get("exclude_categories", ()) or ()))
    max_entries = int(entry.get("max_entries", 2048))
    return AlertThrottleConfig(
        window_seconds=window_seconds,
        exclude_severities=exclude_severities,
        exclude_categories=exclude_categories,
        max_entries=max_entries,
    )


def _load_alert_audit(entry: Optional[Mapping[str, Any]]):
    """Ładuje konfigurację audytu alertów – tylko jeśli klasa istnieje w danej gałęzi."""
    if AlertAuditConfig is None or not entry:
        return None
    backend = str(entry.get("backend", entry.get("type", "memory"))).strip().lower()
    if backend not in {"memory", "file"}:
        raise ValueError("alert_audit.backend musi mieć wartość 'memory' lub 'file'")

    directory_value = entry.get("directory")
    directory = str(directory_value) if directory_value is not None else None
    filename_pattern = str(entry.get("filename_pattern", "alerts-%Y%m%d.jsonl"))
    retention_value = entry.get("retention_days")
    retention_days = None if retention_value in (None, "") else int(retention_value)
    fsync = bool(entry.get("fsync", False))

    if backend == "file" and not directory:
        raise ValueError("alert_audit.directory jest wymagane dla backendu 'file'")

    return AlertAuditConfig(  # type: ignore[call-arg]
        backend=backend,
        directory=directory,
        filename_pattern=filename_pattern,
        retention_days=retention_days,
        fsync=fsync,
    )


def _load_decision_journal(entry: Optional[Mapping[str, Any]]):
    if DecisionJournalConfig is None or not entry:
        return None

    backend = str(entry.get("backend", entry.get("type", "memory"))).strip().lower()
    if backend in {"disabled", "none"}:
        return None
    if backend not in {"memory", "file"}:
        raise ValueError("decision_journal.backend musi być 'memory', 'file' lub 'disabled'")

    directory_value = entry.get("directory")
    directory = str(directory_value) if directory_value is not None else None
    filename_pattern = str(entry.get("filename_pattern", "decisions-%Y%m%d.jsonl"))
    retention_value = entry.get("retention_days")
    retention_days = None if retention_value in (None, "") else int(retention_value)
    fsync = bool(entry.get("fsync", False))

    if backend == "file" and not directory:
        raise ValueError("decision_journal.directory jest wymagane dla backendu 'file'")

    return DecisionJournalConfig(  # type: ignore[call-arg]
        backend=backend,
        directory=directory,
        filename_pattern=filename_pattern,
        retention_days=retention_days,
        fsync=fsync,
    )


def _format_optional_text(value: Any | None) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _load_smoke_archive_upload(entry: Optional[Mapping[str, Any]]):
    if SmokeArchiveUploadConfig is None or entry is None:
        return None

    backend = str(entry.get("backend", entry.get("type", "local"))).strip().lower()
    if backend in {"disabled", "none"}:
        return None
    if backend not in {"local", "s3"}:
        raise ValueError("smoke_archive_upload.backend musi być 'local', 's3' lub 'disabled'")

    credential_secret = entry.get("credential_secret")
    credential_value = str(credential_secret) if credential_secret not in (None, "") else None

    local_cfg = None
    if backend == "local":
        if SmokeArchiveLocalConfig is None:
            raise ValueError("Backend 'local' nie jest obsługiwany w tej gałęzi")
        raw_local = entry.get("local") or {}
        directory_value = raw_local.get("directory")
        if not directory_value:
            raise ValueError("smoke_archive_upload.local.directory jest wymagane dla backendu 'local'")
        filename_pattern = str(raw_local.get("filename_pattern", "{environment}_{date}_{hash}.zip"))
        fsync = bool(raw_local.get("fsync", False))
        local_cfg = SmokeArchiveLocalConfig(
            directory=str(directory_value),
            filename_pattern=filename_pattern,
            fsync=fsync,
        )

    s3_cfg = None
    if backend == "s3":
        if SmokeArchiveS3Config is None:
            raise ValueError("Backend 's3' nie jest obsługiwany w tej gałęzi")
        raw_s3 = entry.get("s3") or {}
        bucket_value = raw_s3.get("bucket")
        if not bucket_value:
            raise ValueError("smoke_archive_upload.s3.bucket jest wymagane dla backendu 's3'")
        prefix_value = raw_s3.get("prefix")
        endpoint_url = _format_optional_text(raw_s3.get("endpoint_url"))
        region = _format_optional_text(raw_s3.get("region"))
        use_ssl = bool(raw_s3.get("use_ssl", True))
        extra_args = {
            str(key): str(value)
            for key, value in (raw_s3.get("extra_args", {}) or {}).items()
        }
        s3_cfg = SmokeArchiveS3Config(
            bucket=str(bucket_value),
            object_prefix=_format_optional_text(prefix_value),
            endpoint_url=endpoint_url,
            region=region,
            use_ssl=use_ssl,
            extra_args=extra_args,
        )

    return SmokeArchiveUploadConfig(
        backend=backend,
        credential_secret=credential_value,
        local=local_cfg,
        s3=s3_cfg,
    )


def _load_reporting(entry: Optional[Mapping[str, Any]]):
    if CoreReportingConfig is None:
        return entry or {}

    payload = entry or {}
    return CoreReportingConfig(
        daily_report_time_utc=_format_optional_text(payload.get("daily_report_time_utc")),
        weekly_report_day=_format_optional_text(payload.get("weekly_report_day")),
        retention_months=_format_optional_text(payload.get("retention_months")),
        smoke_archive_upload=_load_smoke_archive_upload(payload.get("smoke_archive_upload")),
    )


def load_core_config(path: str | Path) -> CoreConfig:
    """Wczytuje plik YAML i mapuje go na dataclasses."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    instrument_universes = _load_instrument_universes(raw)

    # Środowiska – budujemy kwargs dynamicznie, tak by działało na różnych gałęziach modeli.
    environments: dict[str, EnvironmentConfig] = {}
    for name, entry in (raw.get("environments", {}) or {}).items():
        env_kwargs: dict[str, Any] = {
            "name": name,
            "exchange": entry["exchange"],
            "environment": Environment(entry["environment"]),
            "keychain_key": entry["keychain_key"],
            "data_cache_path": entry["data_cache_path"],
            "risk_profile": entry["risk_profile"],
            "alert_channels": tuple(entry.get("alert_channels", ()) or ()),
            "ip_allowlist": tuple(entry.get("ip_allowlist", ()) or ()),
            "credential_purpose": str(entry.get("credential_purpose", "trading")),
            "instrument_universe": entry.get("instrument_universe"),
            "adapter_settings": {
                str(key): value
                for key, value in (entry.get("adapter_settings", {}) or {}).items()
            },
            "required_permissions": tuple(
                str(value).lower() for value in (entry.get("required_permissions", ()) or ())
            ),
            "forbidden_permissions": tuple(
                str(value).lower() for value in (entry.get("forbidden_permissions", ()) or ())
            ),
        }
        if _env_has("alert_throttle"):
            env_kwargs["alert_throttle"] = _load_alert_throttle(entry.get("alert_throttle"))
        if _env_has("alert_audit"):
            env_kwargs["alert_audit"] = _load_alert_audit(entry.get("alert_audit"))
        if _env_has("decision_journal"):
            env_kwargs["decision_journal"] = _load_decision_journal(entry.get("decision_journal"))
        environments[name] = EnvironmentConfig(**env_kwargs)

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
        for name, entry in (raw.get("risk_profiles", {}) or {}).items()
    }

    strategies = _load_strategies(raw)

    reporting = _load_reporting(raw.get("reporting"))
    alerts = (raw.get("alerts", {}) or {})
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
        for name, entry in (alerts.get("telegram_channels", {}) or {}).items()
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
        for name, entry in (alerts.get("email_channels", {}) or {}).items()
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
    if _core_has("runtime_controllers") and ControllerRuntimeConfig is not None:
        controllers_raw = (raw.get("runtime", {}) or {}).get("controllers", {}) or {}
        core_kwargs["runtime_controllers"] = {
            name: ControllerRuntimeConfig(
                tick_seconds=float(entry.get("tick_seconds", entry.get("tick", 60.0))),
                interval=str(entry.get("interval", "1d")),
            )
            for name, entry in controllers_raw.items()
        }

    return CoreConfig(**core_kwargs)  # type: ignore[arg-type]


__all__ = ["load_core_config"]
