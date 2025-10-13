"""Ładowanie konfiguracji z plików YAML."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import re

import yaml

from bot_core.config.models import (
    AlertThrottleConfig,
    CoreConfig,
    CoverageMonitorTargetConfig,
    CoverageMonitoringConfig,
    EmailChannelSettings,
    EnvironmentConfig,
    EnvironmentDataQualityConfig,
    RiskDecisionLogConfig,
    SecurityBaselineConfig,
    SecurityBaselineSigningConfig,
    ServiceTokenConfig,
    RiskProfileConfig,
    RiskServiceConfig,
    RuntimeResourceLimitsConfig,
    SMSProviderSettings,
    TelegramChannelSettings,
)
from bot_core.exchanges.base import Environment

# --- opcjonalne typy (mogą nie istnieć w Twojej gałęzi) ---
try:
    from bot_core.config.models import (
        InstrumentBackfillWindow,
        InstrumentBucketConfig,
        InstrumentConfig,
        InstrumentUniverseConfig,
    )
except Exception:  # brak rozszerzeń instrumentów
    InstrumentBackfillWindow = None  # type: ignore
    InstrumentBucketConfig = None  # type: ignore
    InstrumentConfig = None  # type: ignore
    InstrumentUniverseConfig = None  # type: ignore

try:
    from bot_core.config.models import DailyTrendMomentumStrategyConfig
except Exception:  # brak modułu strategii
    DailyTrendMomentumStrategyConfig = None  # type: ignore

try:
    from bot_core.config.models import (
        CrossExchangeArbitrageStrategyConfig,
        MeanReversionStrategyConfig,
        MultiStrategySchedulerConfig,
        StrategyScheduleConfig,
        VolatilityTargetingStrategyConfig,
    )
except Exception:  # brak rozszerzonej biblioteki strategii
    CrossExchangeArbitrageStrategyConfig = None  # type: ignore
    MeanReversionStrategyConfig = None  # type: ignore
    MultiStrategySchedulerConfig = None  # type: ignore
    StrategyScheduleConfig = None  # type: ignore
    VolatilityTargetingStrategyConfig = None  # type: ignore

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
        PaperSmokeJsonSyncConfig,
        PaperSmokeJsonSyncLocalConfig,
        PaperSmokeJsonSyncS3Config,
        SmokeArchiveLocalConfig,
        SmokeArchiveS3Config,
        SmokeArchiveUploadConfig,
    )  # type: ignore
except Exception:
    CoreReportingConfig = None  # type: ignore
    PaperSmokeJsonSyncConfig = None  # type: ignore
    PaperSmokeJsonSyncLocalConfig = None  # type: ignore
    PaperSmokeJsonSyncS3Config = None  # type: ignore
    SmokeArchiveLocalConfig = None  # type: ignore
    SmokeArchiveS3Config = None  # type: ignore
    SmokeArchiveUploadConfig = None  # type: ignore

try:
    from bot_core.config.models import MetricsServiceConfig  # type: ignore
except Exception:
    MetricsServiceConfig = None  # type: ignore

try:
    from bot_core.config.models import MetricsServiceTlsConfig  # type: ignore
except Exception:
    MetricsServiceTlsConfig = None  # type: ignore


_GRPC_METADATA_KEY_PATTERN = re.compile(r"^[0-9a-z._-]+$")


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


def _load_instrument_buckets(
    raw: Mapping[str, Any],
) -> Mapping[str, "InstrumentBucketConfig"]:
    if InstrumentBucketConfig is None:
        return {}
    buckets: dict[str, InstrumentBucketConfig] = {}
    for name, entry in (raw.get("instrument_buckets", {}) or {}).items():
        buckets[name] = InstrumentBucketConfig(
            name=name,
            universe=str(entry.get("universe", "")),
            symbols=tuple(str(symbol) for symbol in (entry.get("symbols", ()) or ())),
            max_position_pct=(
                float(entry["max_position_pct"])
                if entry.get("max_position_pct") is not None
                else None
            ),
            max_notional_usd=(
                float(entry["max_notional_usd"])
                if entry.get("max_notional_usd") is not None
                else None
            ),
            tags=tuple(str(tag) for tag in (entry.get("tags", ()) or ())),
        )
    return buckets


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


def _load_mean_reversion_strategies(raw: Mapping[str, Any]):
    if MeanReversionStrategyConfig is None:
        return {}
    strategies: dict[str, MeanReversionStrategyConfig] = {}
    for name, entry in (raw.get("mean_reversion_strategies", {}) or {}).items():
        params = entry.get("parameters", entry) or {}
        strategies[name] = MeanReversionStrategyConfig(
            name=name,
            lookback=int(params.get("lookback", 96)),
            entry_zscore=float(params.get("entry_zscore", 1.8)),
            exit_zscore=float(params.get("exit_zscore", 0.4)),
            max_holding_period=int(params.get("max_holding_period", 12)),
            volatility_cap=float(params.get("volatility_cap", 0.04)),
            min_volume_usd=float(params.get("min_volume_usd", 1000.0)),
        )
    return strategies


def _load_volatility_target_strategies(raw: Mapping[str, Any]):
    if VolatilityTargetingStrategyConfig is None:
        return {}
    strategies: dict[str, VolatilityTargetingStrategyConfig] = {}
    for name, entry in (raw.get("volatility_target_strategies", {}) or {}).items():
        params = entry.get("parameters", entry) or {}
        strategies[name] = VolatilityTargetingStrategyConfig(
            name=name,
            target_volatility=float(params.get("target_volatility", 0.12)),
            lookback=int(params.get("lookback", 60)),
            rebalance_threshold=float(params.get("rebalance_threshold", 0.1)),
            min_allocation=float(params.get("min_allocation", 0.1)),
            max_allocation=float(params.get("max_allocation", 1.0)),
            floor_volatility=float(params.get("floor_volatility", 0.02)),
        )
    return strategies


def _load_cross_exchange_arbitrage_strategies(raw: Mapping[str, Any]):
    if CrossExchangeArbitrageStrategyConfig is None:
        return {}
    strategies: dict[str, CrossExchangeArbitrageStrategyConfig] = {}
    for name, entry in (raw.get("cross_exchange_arbitrage_strategies", {}) or {}).items():
        params = entry.get("parameters", entry) or {}
        strategies[name] = CrossExchangeArbitrageStrategyConfig(
            name=name,
            primary_exchange=str(params.get("primary_exchange", "")),
            secondary_exchange=str(params.get("secondary_exchange", "")),
            spread_entry=float(params.get("spread_entry", 0.0015)),
            spread_exit=float(params.get("spread_exit", 0.0005)),
            max_notional=float(params.get("max_notional", 50_000.0)),
            max_open_seconds=int(params.get("max_open_seconds", 120)),
        )
    return strategies


def _load_strategy_schedule(entry_name: str, entry: Mapping[str, Any]) -> StrategyScheduleConfig:
    assert StrategyScheduleConfig is not None
    return StrategyScheduleConfig(
        name=entry_name,
        strategy=str(entry.get("strategy") or entry_name),
        cadence_seconds=int(entry.get("cadence_seconds", entry.get("cadence", 300))),
        max_drift_seconds=int(entry.get("max_drift_seconds", entry.get("max_drift", 30))),
        warmup_bars=int(entry.get("warmup_bars", 0)),
        risk_profile=str(entry.get("risk_profile", "balanced")),
        max_signals=int(entry.get("max_signals", 10)),
        interval=str(entry.get("interval")) if entry.get("interval") else None,
    )


def _load_multi_strategy_schedulers(raw: Mapping[str, Any]):
    if MultiStrategySchedulerConfig is None or StrategyScheduleConfig is None:
        return {}
    schedulers: dict[str, MultiStrategySchedulerConfig] = {}
    sources: list[Mapping[str, Any]] = []
    top_level = raw.get("multi_strategy_schedulers")
    if isinstance(top_level, Mapping):
        sources.append(top_level)
    runtime_section = raw.get("runtime")
    if isinstance(runtime_section, Mapping):
        runtime_schedulers = runtime_section.get("multi_strategy_schedulers")
        if isinstance(runtime_schedulers, Mapping):
            sources.append(runtime_schedulers)

    for source in sources:
        for name, entry in (source or {}).items():
            if not isinstance(entry, Mapping):
                continue
            schedules_raw = entry.get("schedules", {}) or {}
            schedules = [
                _load_strategy_schedule(schedule_name, schedule_entry)
                for schedule_name, schedule_entry in schedules_raw.items()
                if isinstance(schedule_entry, Mapping)
            ]
            schedulers[name] = MultiStrategySchedulerConfig(
                name=name,
                schedules=tuple(schedules),
                telemetry_namespace=str(
                    entry.get("telemetry_namespace", f"scheduler.{name}")
                ),
                decision_log_category=str(
                    entry.get("decision_log_category", "runtime.scheduler")
                ),
                health_check_interval=int(entry.get("health_check_interval", 300)),
                rbac_tokens=_load_service_tokens(entry.get("rbac_tokens")),
            )
    return schedulers


def _load_runtime_resource_limits(runtime_section: Mapping[str, Any]):
    if RuntimeResourceLimitsConfig is None:
        return None
    entry = runtime_section.get("resource_limits")
    if not isinstance(entry, Mapping) or not entry:
        return None
    cpu_percent = float(entry.get("cpu_percent", entry.get("cpu", 0.0)))
    memory_mb = float(entry.get("memory_mb", entry.get("memory", 0.0)))
    io_read = float(entry.get("io_read_mb_s", entry.get("io_read", 0.0)))
    io_write = float(entry.get("io_write_mb_s", entry.get("io_write", 0.0)))
    warning_threshold = float(entry.get("headroom_warning_threshold", entry.get("warning_threshold", 0.85)))
    return RuntimeResourceLimitsConfig(
        cpu_percent=cpu_percent,
        memory_mb=memory_mb,
        io_read_mb_s=io_read,
        io_write_mb_s=io_write,
        headroom_warning_threshold=warning_threshold,
    )


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


def _load_data_quality(entry: Optional[Mapping[str, Any]]):
    """Mapuje ustawienia data_quality na dataclass środowiska."""
    if EnvironmentDataQualityConfig is None or not entry:
        return None

    max_gap = entry.get("max_gap_minutes")
    if max_gap in (None, ""):
        max_gap_value = None
    else:
        max_gap_value = float(max_gap)

    min_ok_ratio = entry.get("min_ok_ratio")
    if min_ok_ratio in (None, ""):
        min_ok_ratio_value = None
    else:
        min_ok_ratio_value = float(min_ok_ratio)

    return EnvironmentDataQualityConfig(
        max_gap_minutes=max_gap_value,
        min_ok_ratio=min_ok_ratio_value,
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


def _normalize_alert_mode(value: Any | None, *, field_name: str) -> str | None:
    """Normalizuje tryby alertów UI na wartość lower-case lub zwraca ``None``."""

    normalized = _format_optional_text(value)
    if normalized is None:
        return None

    normalized = normalized.strip().lower()
    if normalized in {"enable", "jsonl", "disable"}:
        return normalized

    raise ValueError(
        f"{field_name} musi należeć do {{enable,jsonl,disable}} (otrzymano {value!r})"
    )


_UI_ALERT_AUDIT_BACKEND_ALLOWED = {"auto", "file", "memory"}


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


def _load_paper_smoke_json_sync(entry: Optional[Mapping[str, Any]]):
    if PaperSmokeJsonSyncConfig is None or entry is None:
        return None

    backend = str(entry.get("backend", entry.get("type", "local"))).strip().lower()
    if backend in {"disabled", "none"}:
        return None
    if backend not in {"local", "s3"}:
        raise ValueError(
            "paper_smoke_json_sync.backend musi być 'local', 's3' lub 'disabled'"
        )

    credential_secret = entry.get("credential_secret")
    credential_value = str(credential_secret) if credential_secret not in (None, "") else None

    local_cfg = None
    if backend == "local":
        if PaperSmokeJsonSyncLocalConfig is None:
            raise ValueError("Backend 'local' nie jest obsługiwany w tej gałęzi")
        raw_local = entry.get("local") or {}
        directory_value = raw_local.get("directory")
        if not directory_value:
            raise ValueError(
                "paper_smoke_json_sync.local.directory jest wymagane dla backendu 'local'"
            )
        filename_pattern = str(raw_local.get("filename_pattern", "{environment}_{date}.jsonl"))
        fsync = bool(raw_local.get("fsync", False))
        local_cfg = PaperSmokeJsonSyncLocalConfig(
            directory=str(directory_value),
            filename_pattern=filename_pattern,
            fsync=fsync,
        )

    s3_cfg = None
    if backend == "s3":
        if PaperSmokeJsonSyncS3Config is None:
            raise ValueError("Backend 's3' nie jest obsługiwany w tej gałęzi")
        raw_s3 = entry.get("s3") or {}
        bucket_value = raw_s3.get("bucket")
        if not bucket_value:
            raise ValueError(
                "paper_smoke_json_sync.s3.bucket jest wymagane dla backendu 's3'"
            )
        prefix_value = raw_s3.get("prefix")
        endpoint_url = _format_optional_text(raw_s3.get("endpoint_url"))
        region = _format_optional_text(raw_s3.get("region"))
        use_ssl = bool(raw_s3.get("use_ssl", True))
        extra_args = {
            str(key): str(value)
            for key, value in (raw_s3.get("extra_args", {}) or {}).items()
        }
        s3_cfg = PaperSmokeJsonSyncS3Config(
            bucket=str(bucket_value),
            object_prefix=_format_optional_text(prefix_value),
            endpoint_url=endpoint_url,
            region=region,
            use_ssl=use_ssl,
            extra_args=extra_args,
        )

    return PaperSmokeJsonSyncConfig(
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
        paper_smoke_json_sync=_load_paper_smoke_json_sync(payload.get("paper_smoke_json_sync")),
    )


def _load_coverage_monitoring(
    entry: Optional[Mapping[str, Any]]
) -> CoverageMonitoringConfig | None:
    if not entry:
        return None

    enabled = bool(entry.get("enabled", True))
    default_dispatch = bool(entry.get("default_dispatch", True))
    default_category_raw = entry.get("default_category", "data.ohlcv")
    if default_category_raw in (None, ""):
        default_category = "data.ohlcv"
    else:
        default_category = str(default_category_raw)

    targets_raw = entry.get("targets") or ()
    targets: list[CoverageMonitorTargetConfig] = []
    for target_entry in targets_raw:
        if not isinstance(target_entry, Mapping):
            continue
        environment_value = target_entry.get("environment")
        if not environment_value:
            continue
        dispatch_value = target_entry.get("dispatch")
        dispatch_bool: bool | None
        if dispatch_value is None:
            dispatch_bool = None
        else:
            dispatch_bool = bool(dispatch_value)
        category_value = target_entry.get("category")
        severity_value = target_entry.get("severity_override")
        targets.append(
            CoverageMonitorTargetConfig(
                environment=str(environment_value),
                dispatch=dispatch_bool,
                category=str(category_value) if category_value not in (None, "") else None,
                severity_override=str(severity_value) if severity_value not in (None, "") else None,
            )
        )

    return CoverageMonitoringConfig(
        enabled=enabled,
        default_dispatch=default_dispatch,
        default_category=default_category,
        targets=tuple(targets),
    )


def _normalize_runtime_path(
    raw_value: Any, *, base_dir: Path | None
) -> str | None:
    """Zwraca ścieżkę pliku znormalizowaną względem katalogu konfiguracji."""
    if raw_value in (None, "", False):
        return None

    candidate = Path(str(raw_value)).expanduser()
    if candidate.is_absolute() or base_dir is None:
        return str(candidate)

    try:
        normalized_base = base_dir.expanduser().resolve(strict=False)
    except Exception:  # noqa: BLE001 - zachowujemy najlepsze możliwe przybliżenie
        normalized_base = base_dir.expanduser().absolute()

    return str(normalized_base / candidate)


def _normalize_env_var(value: Any) -> str | None:
    if value in (None, "", False):
        return None
    text = str(value).strip()
    return text or None


def _normalize_fingerprint_pin(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError("Fingerprint pining entry nie może być puste")
    if ":" in text:
        algorithm, fingerprint = text.split(":", 1)
        algorithm = algorithm.strip() or "sha256"
    else:
        algorithm, fingerprint = "sha256", text
    fingerprint = fingerprint.replace(":", "").strip()
    if not fingerprint:
        raise ValueError("Fingerprint pinning wymaga wartości hex")
    allowed = set("0123456789abcdef")
    if any(ch not in allowed for ch in fingerprint):
        raise ValueError("Fingerprint pinning powinien zawierać tylko znaki hex")
    return f"{algorithm}:{fingerprint}"


def _normalize_pinned_fingerprints(raw_value: Any) -> tuple[str, ...]:
    if raw_value in (None, ""):
        return ()
    if isinstance(raw_value, str):
        entries = [raw_value]
    else:
        entries = list(raw_value)
    normalized: list[str] = []
    for entry in entries:
        if entry in (None, ""):
            continue
        normalized.append(_normalize_fingerprint_pin(entry))
    # usuwamy duplikaty zachowując kolejność
    return tuple(dict.fromkeys(normalized))


def _load_service_tokens(raw_value: Any) -> tuple[ServiceTokenConfig, ...]:
    if raw_value in (None, ""):
        return ()
    entries = raw_value
    if isinstance(entries, Mapping):
        entries = [entries]
    tokens: list[ServiceTokenConfig] = []
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        token_id = str(entry.get("token_id") or entry.get("id") or "").strip()
        if not token_id:
            raise ValueError("Każdy wpis rbac_tokens wymaga pola token_id")
        token_env = _normalize_env_var(
            entry.get("token_env")
            or entry.get("env")
            or entry.get("token_env_var")
        )
        token_value = entry.get("token_value") or entry.get("value")
        if token_value in (None, ""):
            token_value = None
        else:
            token_value = str(token_value)
        token_hash = entry.get("token_hash") or entry.get("hash")
        if token_hash in (None, ""):
            token_hash = None
        else:
            token_hash = str(token_hash)
        scopes_raw = entry.get("scopes") or ()
        if isinstance(scopes_raw, str):
            scopes_iter = [scopes_raw]
        else:
            scopes_iter = list(scopes_raw)
        scopes = tuple(
            str(scope).strip()
            for scope in scopes_iter
            if isinstance(scope, str) and scope.strip()
        )
        tokens.append(
            ServiceTokenConfig(
                token_id=token_id,
                token_env=token_env,
                token_value=token_value,
                token_hash=token_hash,
                scopes=scopes,
            )
        )
    return tuple(tokens)


def _normalize_grpc_metadata(raw_value: object) -> tuple[tuple[str, str], ...]:
    """Normalizuje wpisy metadata gRPC do listy par (klucz, wartość)."""

    if raw_value in (None, False, ""):
        return ()

    entries: list[tuple[str, str]] = []

    def _append_entry(key: object, value: object) -> None:
        if key is None:
            raise ValueError("grpc_metadata wymaga niepustego klucza")
        key_str = str(key).strip()
        if not key_str:
            raise ValueError("grpc_metadata wymaga niepustego klucza")
        normalized_key = key_str.lower()
        if key_str != normalized_key:
            raise ValueError("grpc_metadata klucz musi być zapisany małymi literami")
        if not _GRPC_METADATA_KEY_PATTERN.fullmatch(normalized_key):
            raise ValueError(
                "grpc_metadata klucz może zawierać wyłącznie [0-9a-z._-]"
            )
        value_str = "" if value is None else str(value)
        entries.append((normalized_key, value_str.strip()))

    if isinstance(raw_value, Mapping):
        for key, value in raw_value.items():
            _append_entry(key, value)
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
        for item in raw_value:
            if isinstance(item, Mapping):
                if "key" not in item or "value" not in item:
                    raise ValueError("grpc_metadata wpis słownika wymaga pól key i value")
                _append_entry(item["key"], item["value"])
            else:
                text = str(item)
                if "=" in text:
                    key, value = text.split("=", 1)
                elif ":" in text:
                    key, value = text.split(":", 1)
                else:
                    raise ValueError(
                        "grpc_metadata element listy musi mieć format klucz=wartość lub klucz:wartość"
                    )
                _append_entry(key, value)
    else:
        raise TypeError("grpc_metadata musi być mapą lub listą wpisów")

    return tuple(entries)


def _load_metrics_service(
    runtime_section: Optional[Mapping[str, Any]], *, base_dir: Path | None = None
) -> MetricsServiceConfig | None:
    """Ładuje sekcję runtime.metrics_service z zachowaniem zgodności między gałęziami."""
    if MetricsServiceConfig is None or not _core_has("metrics_service"):
        return None

    runtime = runtime_section or {}
    metrics_raw = runtime.get("metrics_service")
    if not metrics_raw:
        return None

    # Lista dostępnych pól w aktualnym MetricsServiceConfig (różne gałęzie mogą się różnić)
    available_fields = {f.name for f in fields(MetricsServiceConfig)}  # type: ignore[arg-type]

    # Pola bazowe (występujące w każdej wersji)
    kwargs: dict[str, Any] = {
        "enabled": bool(metrics_raw.get("enabled", True)),
        "host": str(metrics_raw.get("host", "127.0.0.1")),
        "port": int(metrics_raw.get("port", 0)),
        "history_size": int(metrics_raw.get("history_size", 1024)),
    }

    # Opcjonalne: token autoryzacyjny
    if "auth_token" in available_fields:
        kwargs["auth_token"] = (
            str(metrics_raw.get("auth_token")) if metrics_raw.get("auth_token") else None
        )

    if "rbac_tokens" in available_fields:
        kwargs["rbac_tokens"] = _load_service_tokens(metrics_raw.get("rbac_tokens"))

    if "grpc_metadata" in available_fields:
        raw_metadata = metrics_raw.get("grpc_metadata")
        kwargs["grpc_metadata"] = _normalize_grpc_metadata(raw_metadata)

    # Opcjonalne: log sink, jsonl, fsync
    if "log_sink" in available_fields:
        kwargs["log_sink"] = bool(metrics_raw.get("log_sink", True))
    if "jsonl_path" in available_fields:
        kwargs["jsonl_path"] = _normalize_runtime_path(metrics_raw.get("jsonl_path"), base_dir=base_dir)
    if "jsonl_fsync" in available_fields:
        kwargs["jsonl_fsync"] = bool(metrics_raw.get("jsonl_fsync", False))

    # Opcjonalne: osobna ścieżka na alerty UI (jeśli istnieje w modelu)
    if "ui_alerts_jsonl_path" in available_fields:
        kwargs["ui_alerts_jsonl_path"] = _normalize_runtime_path(
            metrics_raw.get("ui_alerts_jsonl_path"), base_dir=base_dir
        )
    if "ui_alerts_audit_backend" in available_fields:
        backend_value = metrics_raw.get("ui_alerts_audit_backend")
        normalized_backend = _format_optional_text(backend_value)
        if normalized_backend is None:
            kwargs["ui_alerts_audit_backend"] = None
        else:
            normalized_backend = normalized_backend.strip().lower()
            if normalized_backend == "auto":
                kwargs["ui_alerts_audit_backend"] = None
            elif normalized_backend in _UI_ALERT_AUDIT_BACKEND_ALLOWED:
                kwargs["ui_alerts_audit_backend"] = normalized_backend
            else:
                raise ValueError(
                    "ui_alerts_audit_backend musi należeć do {auto,file,memory}"
                )
    if "ui_alerts_risk_profile" in available_fields:
        profile_value = _format_optional_text(metrics_raw.get("ui_alerts_risk_profile"))
        if profile_value is None:
            kwargs["ui_alerts_risk_profile"] = None
        else:
            kwargs["ui_alerts_risk_profile"] = profile_value.strip().lower()
    if "ui_alerts_risk_profiles_file" in available_fields:
        kwargs["ui_alerts_risk_profiles_file"] = _normalize_runtime_path(
            metrics_raw.get("ui_alerts_risk_profiles_file"), base_dir=base_dir
        )

    # Opcjonalne: konfiguracja TLS (jeśli dataclass TLS jest dostępny i pole istnieje)
    if "tls" in available_fields and MetricsServiceTlsConfig is not None:
        tls_raw = metrics_raw.get("tls") or {}
        if isinstance(tls_raw, Mapping) and tls_raw:
            certificate_raw = _normalize_runtime_path(tls_raw.get("certificate_path"), base_dir=base_dir)
            private_key_raw = _normalize_runtime_path(tls_raw.get("private_key_path"), base_dir=base_dir)
            client_ca_raw = _normalize_runtime_path(tls_raw.get("client_ca_path"), base_dir=base_dir)
            kwargs["tls"] = MetricsServiceTlsConfig(
                enabled=bool(tls_raw.get("enabled", False)),
                certificate_path=certificate_raw,
                private_key_path=private_key_raw,
                client_ca_path=client_ca_raw,
                require_client_auth=bool(tls_raw.get("require_client_auth", False)),
                private_key_password_env=_normalize_env_var(
                    tls_raw.get("private_key_password_env")
                ),
                pinned_fingerprints=_normalize_pinned_fingerprints(
                    tls_raw.get("pinned_fingerprints")
                ),
            )

    # Opcjonalne: alerty reduce_motion
    if "reduce_motion_alerts" in available_fields:
        kwargs["reduce_motion_alerts"] = bool(metrics_raw.get("reduce_motion_alerts", False))
    if "reduce_motion_mode" in available_fields:
        kwargs["reduce_motion_mode"] = _normalize_alert_mode(
            metrics_raw.get("reduce_motion_mode"), field_name="reduce_motion_mode"
        )
    if "reduce_motion_category" in available_fields:
        kwargs["reduce_motion_category"] = str(
            metrics_raw.get("reduce_motion_category", "ui.performance")
        )
    if "reduce_motion_severity_active" in available_fields:
        kwargs["reduce_motion_severity_active"] = str(
            metrics_raw.get("reduce_motion_severity_active", "warning")
        )
    if "reduce_motion_severity_recovered" in available_fields:
        kwargs["reduce_motion_severity_recovered"] = str(
            metrics_raw.get("reduce_motion_severity_recovered", "info")
        )

    # Opcjonalne: alerty overlay_budget
    if "overlay_alerts" in available_fields:
        kwargs["overlay_alerts"] = bool(metrics_raw.get("overlay_alerts", False))
    if "overlay_alert_mode" in available_fields:
        kwargs["overlay_alert_mode"] = _normalize_alert_mode(
            metrics_raw.get("overlay_alert_mode"), field_name="overlay_alert_mode"
        )
    if "overlay_alert_category" in available_fields:
        kwargs["overlay_alert_category"] = str(
            metrics_raw.get("overlay_alert_category", "ui.performance")
        )
    if "overlay_alert_severity_exceeded" in available_fields:
        kwargs["overlay_alert_severity_exceeded"] = str(
            metrics_raw.get("overlay_alert_severity_exceeded", "warning")
        )
    if "overlay_alert_severity_recovered" in available_fields:
        kwargs["overlay_alert_severity_recovered"] = str(
            metrics_raw.get("overlay_alert_severity_recovered", "info")
        )
    if "overlay_alert_severity_critical" in available_fields:
        kwargs["overlay_alert_severity_critical"] = _format_optional_text(
            metrics_raw.get("overlay_alert_severity_critical")
        )
    if "overlay_alert_critical_threshold" in available_fields:
        threshold_raw = metrics_raw.get("overlay_alert_critical_threshold")
        if threshold_raw in (None, ""):
            kwargs["overlay_alert_critical_threshold"] = None
        else:
            try:
                kwargs["overlay_alert_critical_threshold"] = int(threshold_raw)
            except (TypeError, ValueError) as exc:  # pragma: no cover - walidacja wejścia
                raise ValueError(
                    "overlay_alert_critical_threshold musi być liczbą całkowitą"
                ) from exc

    # Opcjonalne: alerty jank
    if "jank_alerts" in available_fields:
        kwargs["jank_alerts"] = bool(metrics_raw.get("jank_alerts", False))
    if "jank_alert_mode" in available_fields:
        kwargs["jank_alert_mode"] = _normalize_alert_mode(
            metrics_raw.get("jank_alert_mode"), field_name="jank_alert_mode"
        )
    if "jank_alert_category" in available_fields:
        kwargs["jank_alert_category"] = str(
            metrics_raw.get("jank_alert_category", "ui.performance")
        )
    if "jank_alert_severity_spike" in available_fields:
        kwargs["jank_alert_severity_spike"] = str(
            metrics_raw.get("jank_alert_severity_spike", "warning")
        )
    if "jank_alert_severity_critical" in available_fields:
        kwargs["jank_alert_severity_critical"] = _format_optional_text(
            metrics_raw.get("jank_alert_severity_critical")
        )
    if "jank_alert_critical_over_ms" in available_fields:
        jank_threshold = metrics_raw.get("jank_alert_critical_over_ms")
        if jank_threshold in (None, ""):
            kwargs["jank_alert_critical_over_ms"] = None
        else:
            try:
                kwargs["jank_alert_critical_over_ms"] = float(jank_threshold)
            except (TypeError, ValueError) as exc:  # pragma: no cover - walidacja wejścia
                raise ValueError(
                    "jank_alert_critical_over_ms musi być liczbą"
                ) from exc

    return MetricsServiceConfig(**kwargs)  # type: ignore[call-arg]


def _load_risk_service(
    runtime_section: Optional[Mapping[str, Any]], *, base_dir: Path | None = None
) -> RiskServiceConfig | None:
    if not _core_has("risk_service"):
        return None

    runtime = runtime_section or {}
    risk_raw = runtime.get("risk_service")
    if not risk_raw:
        return None

    available_fields = {f.name for f in fields(RiskServiceConfig)}
    kwargs: dict[str, Any] = {
        "enabled": bool(risk_raw.get("enabled", True)),
        "host": str(risk_raw.get("host", "127.0.0.1")),
        "port": int(risk_raw.get("port", 0)),
        "history_size": int(risk_raw.get("history_size", 256)),
        "publish_interval_seconds": float(risk_raw.get("publish_interval_seconds", 5.0)),
    }

    if "auth_token" in available_fields:
        auth_value = risk_raw.get("auth_token")
        kwargs["auth_token"] = str(auth_value) if auth_value not in (None, "") else None

    if "rbac_tokens" in available_fields:
        kwargs["rbac_tokens"] = _load_service_tokens(risk_raw.get("rbac_tokens"))

    if "profiles" in available_fields:
        profiles_raw = risk_raw.get("profiles") or ()
        kwargs["profiles"] = tuple(
            str(profile).strip()
            for profile in profiles_raw
            if isinstance(profile, str) and profile.strip()
        )

    if "tls" in available_fields and MetricsServiceTlsConfig is not None:
        tls_raw = risk_raw.get("tls") or {}
        if isinstance(tls_raw, Mapping) and tls_raw:
            kwargs["tls"] = MetricsServiceTlsConfig(
                enabled=bool(tls_raw.get("enabled", False)),
                certificate_path=_normalize_runtime_path(
                    tls_raw.get("certificate_path"), base_dir=base_dir
                ),
                private_key_path=_normalize_runtime_path(
                    tls_raw.get("private_key_path"), base_dir=base_dir
                ),
                client_ca_path=_normalize_runtime_path(
                    tls_raw.get("client_ca_path"), base_dir=base_dir
                ),
                require_client_auth=bool(tls_raw.get("require_client_auth", False)),
                private_key_password_env=_normalize_env_var(
                    tls_raw.get("private_key_password_env")
                ),
                pinned_fingerprints=_normalize_pinned_fingerprints(
                    tls_raw.get("pinned_fingerprints")
                ),
            )

    return RiskServiceConfig(**kwargs)  # type: ignore[call-arg]


def _load_security_baseline(
    runtime_section: Optional[Mapping[str, Any]], *, base_dir: Path | None = None
) -> SecurityBaselineConfig | None:
    if not _core_has("security_baseline"):
        return None

    runtime = runtime_section or {}
    baseline_raw = runtime.get("security_baseline")
    if not isinstance(baseline_raw, Mapping):
        return None

    signing_raw = baseline_raw.get("signing")
    signing_config: SecurityBaselineSigningConfig | None = None

    if isinstance(signing_raw, Mapping):
        signing_kwargs: dict[str, Any] = {}

        env_value = _normalize_env_var(signing_raw.get("signing_key_env"))
        if env_value:
            signing_kwargs["signing_key_env"] = env_value

        path_value = _normalize_runtime_path(
            signing_raw.get("signing_key_path"), base_dir=base_dir
        )
        if path_value:
            signing_kwargs["signing_key_path"] = path_value

        value_raw = signing_raw.get("signing_key_value")
        if value_raw not in (None, ""):
            signing_kwargs["signing_key_value"] = str(value_raw)

        key_id_raw = signing_raw.get("signing_key_id")
        if key_id_raw not in (None, ""):
            signing_kwargs["signing_key_id"] = str(key_id_raw)

        require_signature = bool(signing_raw.get("require_signature", False))
        if require_signature or signing_kwargs:
            signing_kwargs["require_signature"] = require_signature
            signing_config = SecurityBaselineSigningConfig(**signing_kwargs)

    if signing_config is None and not baseline_raw:
        return None

    return SecurityBaselineConfig(signing=signing_config)


def _load_risk_decision_log(
    runtime_section: Optional[Mapping[str, Any]], *, base_dir: Path | None = None
) -> RiskDecisionLogConfig | None:
    if not _core_has("risk_decision_log"):
        return None

    runtime = runtime_section or {}
    log_raw = runtime.get("risk_decision_log")
    if not log_raw:
        return None

    available_fields = {f.name for f in fields(RiskDecisionLogConfig)}
    kwargs: dict[str, Any] = {
        "enabled": bool(log_raw.get("enabled", True)),
    }

    if "path" in available_fields:
        kwargs["path"] = _normalize_runtime_path(log_raw.get("path"), base_dir=base_dir)
    if "max_entries" in available_fields:
        kwargs["max_entries"] = int(log_raw.get("max_entries", 1_000))
    if "signing_key_env" in available_fields:
        env_value = log_raw.get("signing_key_env")
        kwargs["signing_key_env"] = (
            str(env_value).strip() if isinstance(env_value, str) and env_value.strip() else None
        )
    if "signing_key_path" in available_fields:
        kwargs["signing_key_path"] = _normalize_runtime_path(
            log_raw.get("signing_key_path"), base_dir=base_dir
        )
    if "signing_key_value" in available_fields:
        value = log_raw.get("signing_key_value")
        kwargs["signing_key_value"] = (
            str(value) if value not in (None, "") else None
        )
    if "signing_key_id" in available_fields:
        key_id = log_raw.get("signing_key_id")
        kwargs["signing_key_id"] = str(key_id) if key_id not in (None, "") else None
    if "jsonl_fsync" in available_fields:
        kwargs["jsonl_fsync"] = bool(log_raw.get("jsonl_fsync", False))

    return RiskDecisionLogConfig(**kwargs)  # type: ignore[call-arg]


def load_core_config(path: str | Path) -> CoreConfig:
    """Wczytuje plik YAML i mapuje go na dataclasses."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    try:
        config_absolute_path = config_path.resolve(strict=False)
    except Exception:  # noqa: BLE001 - zachowujemy najlepsze możliwe przybliżenie
        config_absolute_path = config_path.absolute()
    config_base_dir = config_absolute_path.parent

    instrument_universes = _load_instrument_universes(raw)
    instrument_buckets = _load_instrument_buckets(raw)

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
        if _env_has("default_strategy"):
            strategy_value = entry.get("default_strategy")
            env_kwargs["default_strategy"] = (
                str(strategy_value) if strategy_value not in (None, "") else None
            )
        if _env_has("default_controller"):
            controller_value = entry.get("default_controller")
            env_kwargs["default_controller"] = (
                str(controller_value) if controller_value not in (None, "") else None
            )
        if _env_has("alert_throttle"):
            env_kwargs["alert_throttle"] = _load_alert_throttle(entry.get("alert_throttle"))
        if _env_has("alert_audit"):
            env_kwargs["alert_audit"] = _load_alert_audit(entry.get("alert_audit"))
        if _env_has("decision_journal"):
            env_kwargs["decision_journal"] = _load_decision_journal(entry.get("decision_journal"))
        if _env_has("data_quality"):
            env_kwargs["data_quality"] = _load_data_quality(entry.get("data_quality"))
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
            data_quality=_load_data_quality(entry.get("data_quality")),
            strategy_allocations={
                str(bucket): float(weight)
                for bucket, weight in (entry.get("strategy_allocations", {}) or {}).items()
            },
            instrument_buckets=tuple(
                str(bucket) for bucket in (entry.get("instrument_buckets", ()) or ())
            ),
        )
        for name, entry in (raw.get("risk_profiles", {}) or {}).items()
    }

    if risk_profiles:
        for env in environments.values():
            if env.data_quality is not None:
                continue
            profile = risk_profiles.get(env.risk_profile)
            if profile is None or profile.data_quality is None:
                continue
            profile_quality = profile.data_quality
            env.data_quality = EnvironmentDataQualityConfig(
                max_gap_minutes=profile_quality.max_gap_minutes,
                min_ok_ratio=profile_quality.min_ok_ratio,
            )

    strategies = _load_strategies(raw)
    mean_reversion_strategies = _load_mean_reversion_strategies(raw)
    volatility_target_strategies = _load_volatility_target_strategies(raw)
    cross_exchange_arbitrage_strategies = _load_cross_exchange_arbitrage_strategies(raw)
    scheduler_configs = _load_multi_strategy_schedulers(raw)

    reporting = _load_reporting(raw.get("reporting"))
    runtime_section = raw.get("runtime") or {}
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
    if _core_has("instrument_buckets"):
        core_kwargs["instrument_buckets"] = instrument_buckets
    if _core_has("strategies"):
        core_kwargs["strategies"] = strategies
    if _core_has("mean_reversion_strategies"):
        core_kwargs["mean_reversion_strategies"] = mean_reversion_strategies
    if _core_has("volatility_target_strategies"):
        core_kwargs["volatility_target_strategies"] = volatility_target_strategies
    if _core_has("cross_exchange_arbitrage_strategies"):
        core_kwargs["cross_exchange_arbitrage_strategies"] = cross_exchange_arbitrage_strategies
    if _core_has("multi_strategy_schedulers"):
        core_kwargs["multi_strategy_schedulers"] = scheduler_configs
    if _core_has("signal_channels"):
        core_kwargs["signal_channels"] = signal_channels
    if _core_has("whatsapp_channels"):
        core_kwargs["whatsapp_channels"] = whatsapp_channels
    if _core_has("messenger_channels"):
        core_kwargs["messenger_channels"] = messenger_channels
    if _core_has("runtime_controllers") and ControllerRuntimeConfig is not None:
        controllers_raw = (runtime_section.get("controllers") or {})
        core_kwargs["runtime_controllers"] = {
            name: ControllerRuntimeConfig(
                tick_seconds=float(entry.get("tick_seconds", entry.get("tick", 60.0))),
                interval=str(entry.get("interval", "1d")),
            )
            for name, entry in controllers_raw.items()
        }
    if _core_has("coverage_monitoring"):
        core_kwargs["coverage_monitoring"] = _load_coverage_monitoring(
            raw.get("coverage_monitoring")
        )
    metrics_config = _load_metrics_service(runtime_section, base_dir=config_base_dir)
    if metrics_config is not None:
        core_kwargs["metrics_service"] = metrics_config

    risk_service_config = _load_risk_service(runtime_section, base_dir=config_base_dir)
    if risk_service_config is not None:
        core_kwargs["risk_service"] = risk_service_config

    resource_limits_config = _load_runtime_resource_limits(runtime_section)
    if resource_limits_config is not None and _core_has("runtime_resource_limits"):
        core_kwargs["runtime_resource_limits"] = resource_limits_config

    risk_decision_log_config = _load_risk_decision_log(
        runtime_section, base_dir=config_base_dir
    )
    if risk_decision_log_config is not None:
        core_kwargs["risk_decision_log"] = risk_decision_log_config

    security_baseline_config = _load_security_baseline(
        runtime_section, base_dir=config_base_dir
    )
    if security_baseline_config is not None:
        core_kwargs["security_baseline"] = security_baseline_config

    core_kwargs["source_path"] = str(config_absolute_path)
    core_kwargs["source_directory"] = str(config_base_dir)

    return CoreConfig(**core_kwargs)  # type: ignore[arg-type]


__all__ = ["load_core_config"]
