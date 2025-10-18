"""Ładowanie konfiguracji z plików YAML."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import base64
import binascii
import os
import re

import yaml

from bot_core.config.models import (
    AlertThrottleConfig,
    CoreConfig,
    CoverageMonitorTargetConfig,
    CoverageMonitoringConfig,
    EmailChannelSettings,
    EnvironmentAIConfig,
    EnvironmentAIModelConfig,
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
    PortfolioGovernorConfig,
    PortfolioAssetConfig,
    PortfolioRiskBudgetConfig,
    PortfolioDriftToleranceConfig,
    PortfolioSloOverrideConfig,
    PortfolioDecisionLogConfig,
    PortfolioRuntimeInputsConfig,
    PermissionProfileConfig,
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

try:
    from bot_core.config.models import (  # type: ignore
        DecisionEngineConfig,
        DecisionEngineTCOConfig,
        DecisionOrchestratorThresholds,
        DecisionStressTestConfig,
    )
except Exception:
    DecisionEngineConfig = None  # type: ignore
    DecisionEngineTCOConfig = None  # type: ignore
    DecisionOrchestratorThresholds = None  # type: ignore
    DecisionStressTestConfig = None  # type: ignore

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

# --- sekcja z konfliktu: scala oba warianty i zachowuje wszystkie typy opcjonalne ---
try:
    from bot_core.config.models import (  # type: ignore
        ObservabilityConfig,
        SLOThresholdConfig,
        KeyRotationConfig,
        KeyRotationEntryConfig,
    )
except Exception:
    ObservabilityConfig = None  # type: ignore
    SLOThresholdConfig = None  # type: ignore
    KeyRotationConfig = None  # type: ignore
    KeyRotationEntryConfig = None  # type: ignore

try:
    from bot_core.config.models import (  # type: ignore
        PortfolioGovernorConfig,
        PortfolioGovernorStrategyConfig,
        PortfolioGovernorScoringWeights,
    )
except Exception:
    PortfolioGovernorConfig = None  # type: ignore
    PortfolioGovernorStrategyConfig = None  # type: ignore
    PortfolioGovernorScoringWeights = None  # type: ignore

try:
    from bot_core.config.models import (  # type: ignore
        MarketIntelConfig,
        MarketIntelSqliteConfig,
    )
except Exception:
    MarketIntelConfig = None  # type: ignore
    MarketIntelSqliteConfig = None  # type: ignore

try:
    from bot_core.config.models import (  # type: ignore
        StressLabConfig,
        StressLabDatasetConfig,
        StressLabScenarioConfig,
        StressLabShockConfig,
        StressLabThresholdsConfig,
    )
except Exception:
    StressLabConfig = None  # type: ignore
    StressLabDatasetConfig = None  # type: ignore
    StressLabScenarioConfig = None  # type: ignore
    StressLabShockConfig = None  # type: ignore
    StressLabThresholdsConfig = None  # type: ignore

try:
    from bot_core.config.models import (  # type: ignore
        ResilienceConfig,
        ResilienceDrillConfig,
        ResilienceDrillThresholdsConfig,
    )
except Exception:
    ResilienceConfig = None  # type: ignore
    ResilienceDrillConfig = None  # type: ignore
    ResilienceDrillThresholdsConfig = None  # type: ignore

try:
    from bot_core.config.models import LiveRoutingConfig, PrometheusAlertRuleConfig  # type: ignore
except Exception:
    LiveRoutingConfig = None  # type: ignore
    PrometheusAlertRuleConfig = None  # type: ignore


_GRPC_METADATA_KEY_PATTERN = re.compile(r"^[0-9a-z._-]+$")


def _decode_grpc_base64(value: object, *, key: str, source: str) -> bytes:
    """Dekoduje wartość base64 używaną w grpc_metadata."""

    text = "" if value is None else str(value)
    normalized = "".join(text.split())
    if not normalized:
        raise ValueError(
            f"grpc_metadata klucz '{key}' otrzymał pustą wartość base64 (źródło {source})"
        )
    try:
        return base64.b64decode(normalized, validate=True)
    except binascii.Error as exc:  # noqa: PERF203 - chcemy pełny komunikat
        raise ValueError(
            f"grpc_metadata klucz '{key}' zawiera niepoprawną wartość base64 (źródło {source})"
        ) from exc


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


def _load_permission_profiles(raw: Mapping[str, Any]) -> Mapping[str, PermissionProfileConfig]:
    if PermissionProfileConfig is None or not _core_has("permission_profiles"):
        return {}
    profiles: dict[str, PermissionProfileConfig] = {}
    for name, entry in (raw.get("permission_profiles", {}) or {}).items():
        profiles[name] = PermissionProfileConfig(
            name=name,
            required_permissions=tuple(
                str(value).lower() for value in (entry.get("required_permissions", ()) or ())
            ),
            forbidden_permissions=tuple(
                str(value).lower()
                for value in (entry.get("forbidden_permissions", ()) or ())
            ),
        )
    return profiles


def _maybe_float(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value in (None, "", False):
        return None
    return int(value)


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
            inputs_entry = entry.get("portfolio_inputs")
            inputs_config = None
            if isinstance(inputs_entry, Mapping):
                slo_path_value = inputs_entry.get("slo_report_path") or inputs_entry.get("slo_report")
                stress_path_value = (
                    inputs_entry.get("stress_lab_report_path")
                    or inputs_entry.get("stress_lab_report")
                    or inputs_entry.get("stress_report")
                )
                slo_age_value = inputs_entry.get("slo_max_age_minutes") or inputs_entry.get("slo_max_age")
                stress_age_value = (
                    inputs_entry.get("stress_max_age_minutes")
                    or inputs_entry.get("stress_max_age")
                )
                inputs_config = PortfolioRuntimeInputsConfig(
                    slo_report_path=_format_optional_text(slo_path_value),
                    slo_max_age_minutes=_maybe_int(slo_age_value),
                    stress_lab_report_path=_format_optional_text(stress_path_value),
                    stress_max_age_minutes=_maybe_int(stress_age_value),
                )
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
                portfolio_governor=(
                    str(entry.get("portfolio_governor")).strip()
                    if entry.get("portfolio_governor") not in (None, "")
                    else None
                ),
                portfolio_inputs=inputs_config,
            )
    return schedulers


def _load_portfolio_governors(raw: Mapping[str, Any]):
    if PortfolioGovernorConfig is None:
        return {}
    entries = raw.get("portfolio_governors") or {}
    if not isinstance(entries, Mapping):
        return {}
    governors: dict[str, PortfolioGovernorConfig] = {}
    for name, entry in entries.items():
        if not isinstance(entry, Mapping):
            continue
        drift_entry = entry.get("drift_tolerance") or entry.get("drift") or {}
        if not isinstance(drift_entry, Mapping):
            drift_entry = {}
        drift = PortfolioDriftToleranceConfig(
            absolute=float(drift_entry.get("absolute", drift_entry.get("abs", 0.01))),
            relative=float(drift_entry.get("relative", drift_entry.get("rel", 0.25))),
        )

        risk_budgets: dict[str, PortfolioRiskBudgetConfig] = {}
        for budget_name, budget_entry in (entry.get("risk_budgets", {}) or {}).items():
            if not isinstance(budget_entry, Mapping):
                continue
            risk_budgets[budget_name] = PortfolioRiskBudgetConfig(
                name=budget_name,
                max_var_pct=_maybe_float(
                    budget_entry.get("max_var_pct")
                    or budget_entry.get("max_var_percent")
                ),
                max_drawdown_pct=_maybe_float(
                    budget_entry.get("max_drawdown_pct")
                    or budget_entry.get("max_drawdown_percent")
                ),
                max_leverage=_maybe_float(budget_entry.get("max_leverage")),
                severity=str(budget_entry.get("severity", "warning")),
                tags=tuple(
                    str(tag) for tag in (budget_entry.get("tags", ()) or ())
                ),
            )

        slo_overrides: list[PortfolioSloOverrideConfig] = []
        for override_entry in (entry.get("slo_overrides") or ()):  # type: ignore[arg-type]
            if not isinstance(override_entry, Mapping):
                continue
            name_value = override_entry.get("slo") or override_entry.get("slo_name")
            if name_value in (None, ""):
                continue
            apply_on_source = override_entry.get("apply_on") or override_entry.get("statuses")
            apply_on = (
                tuple(str(item) for item in (apply_on_source or ("warning", "breach")))
                or ("warning", "breach")
            )
            slo_overrides.append(
                PortfolioSloOverrideConfig(
                    slo_name=str(name_value),
                    apply_on=apply_on,
                    weight_multiplier=_maybe_float(override_entry.get("weight_multiplier")),
                    min_weight=_maybe_float(override_entry.get("min_weight")),
                    max_weight=_maybe_float(override_entry.get("max_weight")),
                    severity=(
                        str(override_entry.get("severity"))
                        if override_entry.get("severity") not in (None, "")
                        else None
                    ),
                    tags=tuple(
                        str(tag) for tag in (override_entry.get("tags", ()) or ())
                    ),
                    force_rebalance=bool(override_entry.get("force_rebalance", False)),
                )
            )

        assets: list[PortfolioAssetConfig] = []
        for asset_entry in (entry.get("assets") or ()):  # type: ignore[arg-type]
            if not isinstance(asset_entry, Mapping):
                continue
            symbol = asset_entry.get("symbol")
            target_weight = asset_entry.get("target_weight")
            if symbol is None or target_weight is None:
                continue
            risk_budget_value = asset_entry.get("risk_budget")
            risk_budget = (
                str(risk_budget_value)
                if risk_budget_value not in (None, "")
                else None
            )
            notes_value = asset_entry.get("notes")
            notes = str(notes_value) if notes_value not in (None, "") else None
            assets.append(
                PortfolioAssetConfig(
                    symbol=str(symbol),
                    target_weight=float(target_weight),
                    min_weight=_maybe_float(asset_entry.get("min_weight")),
                    max_weight=_maybe_float(asset_entry.get("max_weight")),
                    max_volatility_pct=_maybe_float(
                        asset_entry.get("max_volatility_pct")
                        or asset_entry.get("max_volatility_percent")
                    ),
                    min_liquidity_usd=_maybe_float(
                        asset_entry.get("min_liquidity_usd")
                        or asset_entry.get("min_liquidity")
                    ),
                    risk_budget=risk_budget,
                    notes=notes,
                    tags=tuple(
                        str(tag) for tag in (asset_entry.get("tags", ()) or ())
                    ),
                )
            )

        intel_entry = entry.get("market_intel") if isinstance(entry.get("market_intel"), Mapping) else {}
        interval_value = entry.get("market_intel_interval")
        if isinstance(intel_entry, Mapping):
            interval_value = interval_value or intel_entry.get("interval")
        interval = (
            str(interval_value).strip()
            if interval_value not in (None, "")
            else None
        )
        lookback_value = entry.get("market_intel_lookback_bars")
        if isinstance(intel_entry, Mapping):
            lookback_value = lookback_value or intel_entry.get("lookback_bars") or intel_entry.get("lookback")
        try:
            lookback_bars = int(lookback_value) if lookback_value not in (None, "") else 168
        except (TypeError, ValueError):  # pragma: no cover - diagnostyka konfiguracji
            lookback_bars = 168

        governors[name] = PortfolioGovernorConfig(
            name=name,
            portfolio_id=str(entry.get("portfolio_id", name)),
            drift_tolerance=drift,
            rebalance_cooldown_seconds=int(
                entry.get("rebalance_cooldown_seconds", entry.get("rebalance_cooldown", 900))
            ),
            min_rebalance_value=float(
                entry.get("min_rebalance_value", entry.get("min_rebalance_notional", 0.0) or 0.0)
            ),
            min_rebalance_weight=float(
                entry.get("min_rebalance_weight", entry.get("min_weight_delta", 0.0) or 0.0)
            ),
            assets=tuple(assets),
            risk_budgets=risk_budgets,
            risk_overrides=tuple(
                str(item) for item in (entry.get("risk_overrides", ()) or ())
            ),
            slo_overrides=tuple(override for override in slo_overrides if override.slo_name),
            market_intel_interval=interval,
            market_intel_lookback_bars=lookback_bars,
        )

    return governors


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


def _resolve_optional_path(value: object, *, base_dir: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _load_environment_ai(
    entry: Optional[Mapping[str, Any]], *, base_dir: Path
) -> EnvironmentAIConfig | None:
    if EnvironmentAIConfig is None or EnvironmentAIModelConfig is None or not entry:
        return None

    enabled = bool(entry.get("enabled", True))
    threshold_raw = entry.get("threshold_bps", entry.get("threshold"))
    threshold_bps = float(threshold_raw) if threshold_raw not in (None, "") else 5.0
    model_dir = _resolve_optional_path(entry.get("model_dir"), base_dir=base_dir)
    default_strategy_value = entry.get("default_strategy")
    default_strategy = (
        str(default_strategy_value).strip() if default_strategy_value else None
    )
    default_profile_value = entry.get("default_risk_profile")
    default_risk_profile = (
        str(default_profile_value).strip() if default_profile_value else None
    )
    default_notional_raw = entry.get("default_notional")
    default_notional = (
        float(default_notional_raw)
        if default_notional_raw not in (None, "")
        else None
    )
    default_action_value = entry.get("default_action", "enter")
    default_action = str(default_action_value) or "enter"

    preload = tuple(str(item) for item in (entry.get("preload", ()) or ()))

    models_raw = entry.get("models", ()) or ()
    models: list[EnvironmentAIModelConfig] = []
    for model_entry in models_raw:
        if not isinstance(model_entry, Mapping):
            raise ValueError("environment.ai.models musi zawierać obiekty mapujące")
        symbol_raw = model_entry.get("symbol")
        model_type_raw = model_entry.get("model_type", model_entry.get("type"))
        path_raw = model_entry.get("path")
        if not symbol_raw or not model_type_raw or not path_raw:
            raise ValueError(
                "Każdy model AI musi mieć pola 'symbol', 'model_type' oraz 'path'"
            )
        symbol = str(symbol_raw)
        model_type = str(model_type_raw)
        path = _resolve_optional_path(path_raw, base_dir=base_dir)
        if path is None:
            raise ValueError("environment.ai.models[].path nie może być puste")
        strategy_value = model_entry.get("strategy")
        risk_profile_value = model_entry.get("risk_profile")
        action_value = model_entry.get("action")
        notional_raw = model_entry.get("notional")
        models.append(
            EnvironmentAIModelConfig(
                symbol=symbol,
                model_type=model_type,
                path=path,
                strategy=str(strategy_value).strip() if strategy_value else None,
                risk_profile=(
                    str(risk_profile_value).strip() if risk_profile_value else None
                ),
                notional=(
                    float(notional_raw)
                    if notional_raw not in (None, "")
                    else None
                ),
                action=str(action_value) if action_value else None,
            )
        )

    return EnvironmentAIConfig(
        enabled=enabled,
        model_dir=model_dir,
        threshold_bps=threshold_bps,
        default_strategy=default_strategy,
        default_risk_profile=default_risk_profile,
        default_notional=default_notional,
        default_action=default_action,
        preload=preload,
        models=tuple(models),
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


def _load_market_intel_config(
    section: Mapping[str, Any] | None, *, base_dir: Path | None
) -> MarketIntelConfig | None:
    if MarketIntelConfig is None:
        return None
    if not isinstance(section, Mapping):
        return None

    enabled = bool(section.get("enabled", False))
    output_raw = section.get("output_directory") or "../data/stage6/metrics"
    output_directory = _normalize_runtime_path(output_raw, base_dir=base_dir)
    if output_directory is None:
        output_directory = "../data/stage6/metrics"

    manifest_path = _normalize_runtime_path(section.get("manifest_path"), base_dir=base_dir)
    default_weight = float(section.get("default_weight", 1.0))

    required_raw = section.get("required_symbols") or ()
    if not isinstance(required_raw, Sequence):
        raise ValueError("market_intel.required_symbols musi być listą")
    required_symbols = tuple(
        str(value).strip() for value in required_raw if str(value).strip()
    )

    sqlite_cfg: MarketIntelSqliteConfig | None = None
    sqlite_raw = section.get("sqlite")
    if sqlite_raw not in (None, False):
        if not isinstance(sqlite_raw, Mapping):
            raise ValueError("market_intel.sqlite musi być mapą")
        path_raw = sqlite_raw.get("path")
        if path_raw in (None, ""):
            raise ValueError("market_intel.sqlite wymaga pola 'path'")
        sqlite_path = _normalize_runtime_path(path_raw, base_dir=base_dir)
        if sqlite_path is None:
            raise ValueError("market_intel.sqlite.path jest niepoprawny")
        table_value = sqlite_raw.get("table", "market_metrics")
        table = str(table_value).strip() or "market_metrics"

        sqlite_kwargs: dict[str, Any] = {"path": sqlite_path, "table": table}

        for column_name in (
            "symbol_column",
            "mid_price_column",
            "depth_column",
            "spread_column",
            "funding_column",
            "sentiment_column",
            "volatility_column",
            "weight_column",
        ):
            if column_name not in sqlite_raw:
                continue
            value = sqlite_raw[column_name]
            if column_name == "weight_column":
                sqlite_kwargs[column_name] = (
                    None if value in (None, "", False) else str(value)
                )
                continue
            text = str(value).strip()
            if not text:
                raise ValueError(
                    f"market_intel.sqlite.{column_name} nie może być puste"
                )
            sqlite_kwargs[column_name] = text

        sqlite_cfg = MarketIntelSqliteConfig(**sqlite_kwargs)  # type: ignore[arg-type]

    config_kwargs: dict[str, Any] = {
        "enabled": enabled,
        "output_directory": output_directory,
        "default_weight": default_weight,
    }
    if manifest_path is not None:
        config_kwargs["manifest_path"] = manifest_path
    if sqlite_cfg is not None:
        config_kwargs["sqlite"] = sqlite_cfg
    if required_symbols:
        config_kwargs["required_symbols"] = required_symbols

    return MarketIntelConfig(**config_kwargs)  # type: ignore[arg-type]


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


def _normalize_grpc_metadata(
    raw_value: object, *, base_dir: Path | None
) -> tuple[tuple[str, str | bytes], Mapping[str, str]]:
    """Normalizuje wpisy metadata gRPC do listy par (klucz, wartość).

    Zwraca pary `(key, value)` oraz mapę źródeł, aby można było odnotować
    pochodzenie (inline/env/plik) w decision logu.
    """

    if raw_value in (None, False, ""):
        return (), {}

    entries: list[tuple[str, str | bytes]] = []
    sources: dict[str, str] = {}

    def _append_entry(
        key: object,
        value: object,
        *,
        source: str,
    ) -> None:
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
        if isinstance(value, (bytes, bytearray)):
            normalized_value: str | bytes
            raw_bytes = bytes(value)
            if normalized_key.endswith("-bin"):
                normalized_value = raw_bytes
            else:
                try:
                    normalized_value = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        "grpc_metadata klucz '%s' wymaga tekstu UTF-8 – otrzymano dane binarne"
                        % normalized_key
                    ) from exc
        else:
            value_str = "" if value is None else str(value)
            normalized_value = value_str.strip()
        entries.append((normalized_key, normalized_value))
        sources[normalized_key] = source

    def _resolve_mapping_value(
        entry: Mapping[str, Any],
        *,
        key: str,
    ) -> tuple[Any, str]:
        if "value" in entry:
            return entry.get("value"), "inline"
        if "value_env" in entry or "env" in entry:
            env_name = _normalize_env_var(entry.get("value_env") or entry.get("env"))
            if not env_name:
                raise ValueError("grpc_metadata value_env wymaga niepustej nazwy zmiennej")
            if env_name not in os.environ:
                raise ValueError(
                    f"grpc_metadata value_env '{env_name}' nie jest ustawione w środowisku"
                )
            return os.environ[env_name], f"env:{env_name}"
        if "value_env_base64" in entry or "env_base64" in entry or "value_env64" in entry:
            env_name = _normalize_env_var(
                entry.get("value_env_base64")
                or entry.get("env_base64")
                or entry.get("value_env64")
            )
            if not env_name:
                raise ValueError(
                    "grpc_metadata value_env_base64 wymaga niepustej nazwy zmiennej"
                )
            if env_name not in os.environ:
                raise ValueError(
                    f"grpc_metadata value_env_base64 '{env_name}' nie jest ustawione w środowisku"
                )
            decoded = _decode_grpc_base64(
                os.environ[env_name], key=key, source=f"env:{env_name}"
            )
            return decoded, f"env:{env_name}"
        if "value_file" in entry or "value_path" in entry:
            raw_path = entry.get("value_file") or entry.get("value_path")
            normalized_path = _normalize_runtime_path(raw_path, base_dir=base_dir)
            if not normalized_path:
                raise ValueError("grpc_metadata value_file wymaga ścieżki do pliku")
            file_path = Path(normalized_path)
            try:
                contents = file_path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:  # noqa: PERF203 - informujemy o brakującym pliku
                raise ValueError(
                    f"grpc_metadata value_file '{normalized_path}' nie istnieje"
                ) from exc
            return contents, f"file:{normalized_path}"
        if "value_file_base64" in entry or "value_file64" in entry:
            raw_path = entry.get("value_file_base64") or entry.get("value_file64")
            normalized_path = _normalize_runtime_path(raw_path, base_dir=base_dir)
            if not normalized_path:
                raise ValueError("grpc_metadata value_file_base64 wymaga ścieżki do pliku")
            file_path = Path(normalized_path)
            try:
                contents = file_path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:  # noqa: PERF203 - informujemy o brakującym pliku
                raise ValueError(
                    f"grpc_metadata value_file_base64 '{normalized_path}' nie istnieje"
                ) from exc
            decoded = _decode_grpc_base64(
                contents, key=key, source=f"file:{normalized_path}"
            )
            return decoded, f"file:{normalized_path}"
        if "value_base64" in entry or "value_b64" in entry:
            raw_value = entry.get("value_base64") or entry.get("value_b64")
            decoded = _decode_grpc_base64(raw_value, key=key, source="inline")
            return decoded, "inline"
        raise ValueError(
            "grpc_metadata wpis słownika wymaga pola value, value_env, value_file lub wariantu base64"
        )

    if isinstance(raw_value, Mapping):
        for key, value in raw_value.items():
            if isinstance(value, Mapping):
                key_label = str(key).strip()
                if any(
                    candidate_key in value
                    for candidate_key in ("value", "value_env", "env", "value_file", "value_path")
                ):
                    resolved_value, source = _resolve_mapping_value(
                        value,
                        key=key_label or str(key),
                    )
                    _append_entry(key, resolved_value, source=source)
                    continue
            _append_entry(key, value, source="inline")
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
        for item in raw_value:
            if isinstance(item, Mapping):
                if "key" not in item:
                    raise ValueError("grpc_metadata wpis słownika wymaga pola key")
                key_label = str(item["key"]).strip()
                value, source = _resolve_mapping_value(
                    item,
                    key=key_label or str(item["key"]),
                )
                _append_entry(item["key"], value, source=source)
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
                _append_entry(key, value, source="inline")
    else:
        raise TypeError("grpc_metadata musi być mapą lub listą wpisów")

    return tuple(entries), sources


def _normalize_grpc_metadata_files(
    raw_value: object, *, base_dir: Path | None
) -> tuple[str, ...]:
    """Normalizuje listę plików z nagłówkami gRPC względem katalogu konfiguracyjnego."""

    if raw_value in (None, False, ""):
        return ()

    entries: Sequence[object]
    if isinstance(raw_value, (str, bytes, bytearray)):
        entries = [raw_value]
    elif isinstance(raw_value, Sequence):
        entries = raw_value
    else:
        raise TypeError("grpc_metadata_files musi być listą ścieżek lub pojedynczą ścieżką")

    normalized: list[str] = []
    for entry in entries:
        if entry in (None, "", False):
            continue
        if isinstance(entry, (bytes, bytearray)):
            text_value = entry.decode("utf-8", errors="ignore").strip()
        else:
            text_value = str(entry).strip()
        if not text_value:
            continue
        sentinel = text_value.lower()
        if sentinel in {"none", "null"}:
            continue
        normalized_path = _normalize_runtime_path(text_value, base_dir=base_dir)
        if normalized_path is None:
            continue
        normalized.append(normalized_path)

    return tuple(dict.fromkeys(normalized))


def _normalize_grpc_metadata_directories(
    raw_value: object, *, base_dir: Path | None
) -> tuple[str, ...]:
    """Normalizuje listę katalogów z plikami nagłówków gRPC."""

    if raw_value in (None, False, ""):
        return ()

    entries: Sequence[object]
    if isinstance(raw_value, (str, bytes, bytearray)):
        entries = [raw_value]
    elif isinstance(raw_value, Sequence):
        entries = raw_value
    else:
        raise TypeError(
            "grpc_metadata_directories musi być listą ścieżek lub pojedynczą ścieżką"
        )

    normalized: list[str] = []
    for entry in entries:
        if entry in (None, "", False):
            continue
        if isinstance(entry, (bytes, bytearray)):
            text_value = entry.decode("utf-8", errors="ignore").strip()
        else:
            text_value = str(entry).strip()
        if not text_value:
            continue
        sentinel = text_value.lower()
        if sentinel in {"none", "null"}:
            continue
        normalized_path = _normalize_runtime_path(text_value, base_dir=base_dir)
        if normalized_path is None:
            continue
        normalized.append(normalized_path)

    return tuple(dict.fromkeys(normalized))


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
    token_env_value: str | None = None
    token_file_path: str | None = None
    if "auth_token" in available_fields:
        kwargs["auth_token"] = (
            str(metrics_raw.get("auth_token")) if metrics_raw.get("auth_token") else None
        )
    if "auth_token_env" in available_fields:
        token_env_value = _normalize_env_var(metrics_raw.get("auth_token_env"))
        kwargs["auth_token_env"] = token_env_value
    if "auth_token_file" in available_fields:
        token_file_path = _normalize_runtime_path(metrics_raw.get("auth_token_file"), base_dir=base_dir)
        kwargs["auth_token_file"] = token_file_path
    if kwargs.get("auth_token") is None and token_env_value:
        env_token = os.environ.get(token_env_value)
        if env_token:
            kwargs["auth_token"] = env_token
    if (
        kwargs.get("auth_token") is None
        and token_file_path
    ):
        try:
            file_token = Path(token_file_path).expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            file_token = ""
        if file_token:
            kwargs["auth_token"] = file_token

    if "rbac_tokens" in available_fields:
        kwargs["rbac_tokens"] = _load_service_tokens(metrics_raw.get("rbac_tokens"))

    if "grpc_metadata" in available_fields:
        raw_metadata = metrics_raw.get("grpc_metadata")
        metadata_entries, metadata_sources = _normalize_grpc_metadata(
            raw_metadata, base_dir=base_dir
        )
        kwargs["grpc_metadata"] = metadata_entries
        if "grpc_metadata_sources" in available_fields:
            kwargs["grpc_metadata_sources"] = dict(metadata_sources)
    if "grpc_metadata_files" in available_fields:
        raw_metadata_files = metrics_raw.get("grpc_metadata_files")
        kwargs["grpc_metadata_files"] = _normalize_grpc_metadata_files(
            raw_metadata_files, base_dir=base_dir
        )
    if "grpc_metadata_directories" in available_fields:
        raw_metadata_directories = metrics_raw.get("grpc_metadata_directories")
        kwargs["grpc_metadata_directories"] = _normalize_grpc_metadata_directories(
            raw_metadata_directories, base_dir=base_dir
        )

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

    sentinel = object()

    def _normalize_optional_float_field(
        field_name: str, default: float | None
    ) -> float | None:
        raw_value = metrics_raw.get(field_name, sentinel)
        if raw_value is sentinel:
            return default
        if raw_value in (None, ""):
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - walidacja wejścia
            raise ValueError(f"{field_name} musi być liczbą") from exc

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

    # Opcjonalne: alerty wydajności
    if "performance_alerts" in available_fields:
        kwargs["performance_alerts"] = bool(
            metrics_raw.get("performance_alerts", False)
        )
    if "performance_alert_mode" in available_fields:
        kwargs["performance_alert_mode"] = _normalize_alert_mode(
            metrics_raw.get("performance_alert_mode"), field_name="performance_alert_mode"
        )
    if "performance_category" in available_fields:
        kwargs["performance_category"] = str(
            metrics_raw.get("performance_category", "ui.performance")
        )
    if "performance_severity_warning" in available_fields:
        kwargs["performance_severity_warning"] = str(
            metrics_raw.get("performance_severity_warning", "warning")
        )
    if "performance_severity_critical" in available_fields:
        kwargs["performance_severity_critical"] = str(
            metrics_raw.get("performance_severity_critical", "critical")
        )
    if "performance_severity_recovered" in available_fields:
        kwargs["performance_severity_recovered"] = str(
            metrics_raw.get("performance_severity_recovered", "info")
        )
    if "performance_event_to_frame_warning_ms" in available_fields:
        kwargs["performance_event_to_frame_warning_ms"] = _normalize_optional_float_field(
            "performance_event_to_frame_warning_ms", 45.0
        )
    if "performance_event_to_frame_critical_ms" in available_fields:
        kwargs["performance_event_to_frame_critical_ms"] = _normalize_optional_float_field(
            "performance_event_to_frame_critical_ms", 60.0
        )
    if "cpu_utilization_warning_percent" in available_fields:
        kwargs["cpu_utilization_warning_percent"] = _normalize_optional_float_field(
            "cpu_utilization_warning_percent", 85.0
        )
    if "cpu_utilization_critical_percent" in available_fields:
        kwargs["cpu_utilization_critical_percent"] = _normalize_optional_float_field(
            "cpu_utilization_critical_percent", 95.0
        )
    if "gpu_utilization_warning_percent" in available_fields:
        kwargs["gpu_utilization_warning_percent"] = _normalize_optional_float_field(
            "gpu_utilization_warning_percent", None
        )
    if "gpu_utilization_critical_percent" in available_fields:
        kwargs["gpu_utilization_critical_percent"] = _normalize_optional_float_field(
            "gpu_utilization_critical_percent", None
        )
    if "ram_usage_warning_megabytes" in available_fields:
        kwargs["ram_usage_warning_megabytes"] = _normalize_optional_float_field(
            "ram_usage_warning_megabytes", None
        )
    if "ram_usage_critical_megabytes" in available_fields:
        kwargs["ram_usage_critical_megabytes"] = _normalize_optional_float_field(
            "ram_usage_critical_megabytes", None
        )

    return MetricsServiceConfig(**kwargs)  # type: ignore[call-arg]


def _load_live_routing(
    runtime_section: Optional[Mapping[str, Any]], *, base_dir: Path | None = None
) -> LiveRoutingConfig | None:
    if LiveRoutingConfig is None or not _core_has("live_routing"):
        return None

    runtime = runtime_section or {}
    raw = runtime.get("live_routing")
    if not raw:
        return None

    enabled = bool(raw.get("enabled", False))
    default_route = tuple(
        str(entry).strip()
        for entry in raw.get("default_route", [])
        if str(entry).strip()
    )
    if enabled and not default_route:
        raise ValueError("runtime.live_routing.default_route musi zawierać co najmniej jedną giełdę")

    overrides_raw = raw.get("route_overrides") or {}
    overrides: dict[str, tuple[str, ...]] = {}
    if isinstance(overrides_raw, Mapping):
        for symbol, route in overrides_raw.items():
            if not route:
                continue
            normalized = tuple(
                str(entry).strip()
                for entry in route
                if str(entry).strip()
            )
            if normalized:
                overrides[str(symbol)] = normalized

    buckets_raw = raw.get("latency_histogram_buckets") or ()
    buckets: list[float] = []
    for entry in buckets_raw:
        if entry in (None, ""):
            continue
        try:
            buckets.append(float(entry))
        except (TypeError, ValueError) as exc:
            raise ValueError("latency_histogram_buckets musi zawierać wartości liczbowe") from exc

    alerts: list[PrometheusAlertRuleConfig] = []
    alerts_raw = raw.get("prometheus_alerts") or ()
    for entry in alerts_raw:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()
        expr = str(entry.get("expr", "")).strip()
        if not name or not expr:
            raise ValueError(
                "Każdy alert w runtime.live_routing.prometheus_alerts wymaga pól name i expr"
            )
        duration_raw = entry.get("for") if "for" in entry else entry.get("for_duration")
        duration = str(duration_raw).strip() if duration_raw not in (None, "") else None
        labels_raw = entry.get("labels") or {}
        annotations_raw = entry.get("annotations") or {}
        if not isinstance(labels_raw, Mapping):
            raise ValueError("labels w prometheus_alerts musi być mapą")
        if not isinstance(annotations_raw, Mapping):
            raise ValueError("annotations w prometheus_alerts musi być mapą")
        labels = {str(k): str(v) for k, v in labels_raw.items() if str(k).strip()}
        annotations = {str(k): str(v) for k, v in annotations_raw.items() if str(k).strip()}
        alerts.append(
            PrometheusAlertRuleConfig(
                name=name,
                expr=expr,
                for_duration=duration,
                labels=labels,
                annotations=annotations,
            )
        )

    return LiveRoutingConfig(
        enabled=enabled,
        default_route=default_route,
        route_overrides=overrides,
        latency_histogram_buckets=tuple(buckets),
        prometheus_alerts=tuple(alerts),
    )


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


def _build_decision_threshold(
    entry: Mapping[str, Any] | None,
    *,
    fallback: DecisionOrchestratorThresholds | None = None,
) -> DecisionOrchestratorThresholds:
    if DecisionOrchestratorThresholds is None:
        raise RuntimeError("DecisionOrchestratorThresholds nie jest dostępne w tej gałęzi")
    required_keys = (
        "max_cost_bps",
        "min_net_edge_bps",
        "max_daily_loss_pct",
        "max_drawdown_pct",
        "max_position_ratio",
        "max_open_positions",
    )
    if entry is None:
        if fallback is None:
            raise ValueError("Brak sekcji 'orchestrator' w decision_engine")
        return fallback
    data = dict(entry)
    values: dict[str, float | int | None] = {}
    for key in required_keys:
        if key in data:
            raw_value = data[key]
        elif fallback is not None:
            raw_value = getattr(fallback, key)
        else:
            raise ValueError(
                f"Brak wymaganego pola '{key}' w konfiguracji decision orchestratora"
            )
        if key == "max_open_positions":
            values[key] = int(raw_value)
        else:
            values[key] = float(raw_value)
    max_latency = data.get("max_latency_ms")
    if max_latency is None and fallback is not None:
        max_latency = fallback.max_latency_ms
    max_trade_notional = data.get("max_trade_notional")
    if max_trade_notional is None and fallback is not None:
        max_trade_notional = fallback.max_trade_notional
    return DecisionOrchestratorThresholds(
        max_cost_bps=float(values["max_cost_bps"]),
        min_net_edge_bps=float(values["min_net_edge_bps"]),
        max_daily_loss_pct=float(values["max_daily_loss_pct"]),
        max_drawdown_pct=float(values["max_drawdown_pct"]),
        max_position_ratio=float(values["max_position_ratio"]),
        max_open_positions=int(values["max_open_positions"]),
        max_latency_ms=(None if max_latency is None else float(max_latency)),
        max_trade_notional=(
            None if max_trade_notional in (None, "") else float(max_trade_notional)
        ),
    )


def _load_decision_engine_config(
    raw: Mapping[str, Any] | None, *, base_dir: Path | None
) -> DecisionEngineConfig | None:
    if raw is None or DecisionEngineConfig is None:
        return None
    orchestrator_raw = raw.get("orchestrator")
    if orchestrator_raw is None and raw.get("profile_overrides") is None:
        return None
    base_threshold = _build_decision_threshold(orchestrator_raw, fallback=None)
    overrides_raw = raw.get("profile_overrides") or {}
    overrides: dict[str, DecisionOrchestratorThresholds] = {}
    for profile, entry in overrides_raw.items():
        overrides[str(profile)] = _build_decision_threshold(entry, fallback=base_threshold)
    stress_config: DecisionStressTestConfig | None = None
    stress_raw = raw.get("stress_tests") or {}
    if DecisionStressTestConfig is not None and stress_raw:
        stress_config = DecisionStressTestConfig(
            cost_shock_bps=float(stress_raw.get("cost_shock_bps", 0.0)),
            latency_spike_ms=float(stress_raw.get("latency_spike_ms", 0.0)),
            slippage_multiplier=float(stress_raw.get("slippage_multiplier", 1.0)),
        )
    min_probability = float(raw.get("min_probability", 0.0))
    require_cost_data = bool(raw.get("require_cost_data", False))
    penalty_cost_bps = float(raw.get("penalty_cost_bps", 0.0))
    tco_config: DecisionEngineTCOConfig | None = None
    tco_raw = raw.get("tco")
    if DecisionEngineTCOConfig is not None and tco_raw:
        if not isinstance(tco_raw, Mapping):
            raise ValueError("Sekcja decision_engine.tco musi być mapą")
        paths_raw = tco_raw.get("reports")
        if not paths_raw:
            raise ValueError(
                "Sekcja decision_engine.tco wymaga listy 'reports' z co najmniej jedną ścieżką"
            )
        if isinstance(paths_raw, (str, bytes)):
            reports_source: Sequence[str] = [str(paths_raw)]
        else:
            if not isinstance(paths_raw, Sequence):
                raise ValueError(
                    "Pole decision_engine.tco.reports musi być listą ścieżek"
                )
            reports_source = [str(entry) for entry in paths_raw if str(entry).strip()]
        if not reports_source:
            raise ValueError(
                "Pole decision_engine.tco.reports nie może być puste"
            )
        normalized_reports = tuple(
            str(_normalize_runtime_path(path, base_dir=base_dir)) for path in reports_source
        )
        require_at_startup = bool(tco_raw.get("require_at_startup", False))
        tco_kwargs: dict[str, Any] = {}
        tco_fields = {field.name for field in fields(DecisionEngineTCOConfig)}
        if "report_paths" in tco_fields:
            tco_kwargs["report_paths"] = normalized_reports
        else:
            tco_kwargs["reports"] = normalized_reports
        if "require_at_startup" in tco_fields or require_at_startup:
            tco_kwargs["require_at_startup"] = require_at_startup
        tco_config = DecisionEngineTCOConfig(**tco_kwargs)  # type: ignore[arg-type]
    return DecisionEngineConfig(
        orchestrator=base_threshold,
        profile_overrides=overrides,
        stress_tests=stress_config,
        min_probability=min_probability,
        require_cost_data=require_cost_data,
        penalty_cost_bps=penalty_cost_bps,
        tco=tco_config,
    )


_ALLOWED_SLO_COMPARATORS = {"<=", ">=", "<", ">"}
_ALLOWED_SLO_AGGREGATIONS = {"average", "avg", "p95", "max", "min"}


def _load_key_rotation_config(
    raw: Mapping[str, Any] | None, *, base_dir: Path | None
) -> KeyRotationConfig | None:
    if raw is None or KeyRotationConfig is None:
        return None
    if "registry_path" not in raw or not raw["registry_path"]:
        raise ValueError("Konfiguracja key_rotation wymaga pola 'registry_path'")
    registry_path = _normalize_runtime_path(raw["registry_path"], base_dir=base_dir)
    default_interval = float(raw.get("default_interval_days", 90.0))
    warn_within = float(raw.get("default_warn_within_days", 14.0))
    audit_directory = str(
        _normalize_runtime_path(raw.get("audit_directory"), base_dir=base_dir)
        if raw.get("audit_directory")
        else raw.get("audit_directory", "var/audit/keys")
    )

    entries_raw = raw.get("entries") or []
    entries: list[KeyRotationEntryConfig] = []
    for entry in entries_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Element 'entries' w key_rotation musi być mapą")
        key_value = str(entry.get("key", "")).strip()
        if not key_value:
            raise ValueError("Element 'entries' wymaga pola 'key'")
        purpose_value = str(entry.get("purpose", "")).strip()
        if not purpose_value:
            raise ValueError("Element 'entries' wymaga pola 'purpose'")
        interval_value = entry.get("interval_days")
        warn_value = entry.get("warn_within_days")
        entries.append(
            KeyRotationEntryConfig(
                key=key_value,
                purpose=purpose_value,
                interval_days=(
                    None if interval_value in (None, "") else float(interval_value)
                ),
                warn_within_days=(
                    None if warn_value in (None, "") else float(warn_value)
                ),
            )
        )

    config_kwargs: dict[str, Any] = {
        "registry_path": registry_path,
        "default_interval_days": default_interval,
        "default_warn_within_days": warn_within,
        "entries": tuple(entries),
        "audit_directory": audit_directory,
    }

    for field_name in ("signing_key_env", "signing_key_path", "signing_key_value", "signing_key_id"):
        if field_name in raw and raw[field_name] not in (None, ""):
            value = raw[field_name]
            if field_name.endswith("_path"):
                config_kwargs[field_name] = _normalize_runtime_path(value, base_dir=base_dir)
            else:
                config_kwargs[field_name] = str(value)

    return KeyRotationConfig(**config_kwargs)  # type: ignore[arg-type]


def _load_observability_config(
    raw: Mapping[str, Any] | None, *, base_dir: Path | None
) -> ObservabilityConfig | None:
    if raw is None or ObservabilityConfig is None:
        return None

    slo_entries: dict[str, SLOThresholdConfig] = {}
    slo_raw = raw.get("slo") or {}
    for name, entry in slo_raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError("Definicja SLO musi być mapą z parametrami")
        metric = str(entry.get("metric", "")).strip()
        if not metric:
            raise ValueError(f"SLO '{name}' wymaga pola 'metric'")
        comparator = str(entry.get("comparator", "<=")).strip() or "<="
        if comparator not in _ALLOWED_SLO_COMPARATORS:
            raise ValueError(f"SLO '{name}' używa nieobsługiwanego komparatora: {comparator}")
        aggregation = str(entry.get("aggregation", "average")).strip().lower()
        if aggregation == "avg":
            aggregation = "average"
        if aggregation not in _ALLOWED_SLO_AGGREGATIONS:
            raise ValueError(f"SLO '{name}' używa nieobsługiwanej agregacji: {aggregation}")
        window_minutes = float(entry.get("window_minutes", 1440.0))
        objective = float(entry.get("objective"))
        label_filters_raw = entry.get("label_filters") or {}
        if not isinstance(label_filters_raw, Mapping):
            raise ValueError("label_filters w SLO musi być mapą")
        label_filters = {str(key): str(value) for key, value in label_filters_raw.items()}
        min_samples = int(entry.get("min_samples", 1))
        slo_entries[str(name)] = SLOThresholdConfig(
            name=str(name),
            metric=metric,
            objective=objective,
            comparator=comparator,
            window_minutes=window_minutes,
            aggregation=aggregation,
            label_filters=label_filters,
            min_samples=max(1, min_samples),
        )

    key_rotation = _load_key_rotation_config(raw.get("key_rotation"), base_dir=base_dir)

    if not slo_entries and key_rotation is None:
        return None

    return ObservabilityConfig(slo=slo_entries, key_rotation=key_rotation)


def _load_portfolio_governor_config(
    raw_root: Mapping[str, Any], *, base_dir: Path | None
) -> PortfolioGovernorConfig | None:
    if PortfolioGovernorConfig is None:
        return None

    section: Mapping[str, Any] | None = None
    top_level = raw_root.get("portfolio_governor")
    if isinstance(top_level, Mapping):
        section = top_level
    if section is None:
        runtime_section = raw_root.get("runtime")
        if isinstance(runtime_section, Mapping):
            runtime_entry = runtime_section.get("portfolio_governor")
            if isinstance(runtime_entry, Mapping):
                section = runtime_entry
    if not section:
        return None

    enabled = bool(section.get("enabled", False))
    interval_value = float(section.get("rebalance_interval_minutes", 15.0))
    smoothing_value = float(section.get("smoothing", 0.5))
    default_baseline = float(section.get("default_baseline_weight", 0.25))
    default_min = float(section.get("default_min_weight", 0.05))
    default_max = float(section.get("default_max_weight", 0.5))
    require_complete = bool(section.get("require_complete_metrics", True))
    min_score_threshold = float(section.get("min_score_threshold", 0.0))
    default_cost_bps = float(section.get("default_cost_bps", 0.0))
    max_signal_floor = int(section.get("max_signal_floor", 1))

    scoring_raw = section.get("scoring") or {}
    scoring = PortfolioGovernorScoringWeights(
        alpha=float(scoring_raw.get("alpha", 1.0)),
        cost=float(scoring_raw.get("cost", 1.0)),
        slo=float(scoring_raw.get("slo", 1.0)),
        risk=float(scoring_raw.get("risk", 0.5)),
    )

    strategies_raw = section.get("strategies") or {}
    strategies: dict[str, PortfolioGovernorStrategyConfig] = {}
    for name, entry in strategies_raw.items():
        if not isinstance(entry, Mapping):
            continue
        baseline_weight = float(entry.get("baseline_weight", default_baseline))
        min_weight = float(entry.get("min_weight", default_min))
        max_weight = float(entry.get("max_weight", default_max))
        baseline_max_signals_raw = entry.get("baseline_max_signals")
        baseline_max_signals = None
        if baseline_max_signals_raw not in (None, ""):
            baseline_max_signals = int(baseline_max_signals_raw)
        max_signal_factor = float(entry.get("max_signal_factor", 1.0))
        risk_profile_raw = entry.get("risk_profile")
        risk_profile = (
            str(risk_profile_raw).strip()
            if isinstance(risk_profile_raw, str) and risk_profile_raw.strip()
            else None
        )
        tags = tuple(str(tag) for tag in (entry.get("tags") or ()))
        strategies[str(name)] = PortfolioGovernorStrategyConfig(
            baseline_weight=baseline_weight,
            min_weight=min_weight,
            max_weight=max_weight,
            baseline_max_signals=baseline_max_signals,
            max_signal_factor=max_signal_factor,
            risk_profile=risk_profile,
            tags=tags,
        )

    return PortfolioGovernorConfig(
        enabled=enabled,
        rebalance_interval_minutes=max(0.0, interval_value),
        smoothing=min(max(smoothing_value, 0.0), 1.0),
        scoring=scoring,
        strategies=strategies,
        default_baseline_weight=default_baseline,
        default_min_weight=default_min,
        default_max_weight=default_max,
        require_complete_metrics=require_complete,
        min_score_threshold=min_score_threshold,
        default_cost_bps=default_cost_bps,
        max_signal_floor=max(0, max_signal_floor),
    )


def _parse_stress_lab_thresholds(
    raw: Mapping[str, Any] | None, *, context: str
) -> StressLabThresholdsConfig | None:
    if StressLabThresholdsConfig is None:
        return None

    defaults = StressLabThresholdsConfig()
    values: dict[str, float] = {
        field.name: float(getattr(defaults, field.name))
        for field in fields(StressLabThresholdsConfig)
    }

    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context}: sekcja thresholds musi być mapą")
        for field in fields(StressLabThresholdsConfig):
            if field.name not in raw:
                continue
            value = raw[field.name]
            if value in (None, ""):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{context}: wartość '{field.name}' musi być liczbą"
                ) from exc
            values[field.name] = max(0.0, numeric)

    return StressLabThresholdsConfig(**values)


def _load_stress_lab_config(
    raw_root: Mapping[str, Any], *, base_dir: Path | None
) -> StressLabConfig | None:
    if StressLabConfig is None:
        return None

    section: Mapping[str, Any] | None = None
    top_level = raw_root.get("stress_lab")
    if isinstance(top_level, Mapping):
        section = top_level
    if section is None:
        runtime_section = raw_root.get("runtime")
        if isinstance(runtime_section, Mapping):
            runtime_entry = runtime_section.get("stress_lab")
            if isinstance(runtime_entry, Mapping):
                section = runtime_entry
    if section is None:
        return None

    enabled = bool(section.get("enabled", False))
    require_success = bool(section.get("require_success", True))

    report_dir_raw = section.get("report_directory") or "var/audit/stage6/stress_lab"
    report_directory = _normalize_runtime_path(report_dir_raw, base_dir=base_dir)
    if report_directory is None:
        report_directory = "var/audit/stage6/stress_lab"

    signing_key_env = _normalize_env_var(section.get("signing_key_env"))
    signing_key_path = _normalize_runtime_path(section.get("signing_key_path"), base_dir=base_dir)
    signing_key_id = _normalize_env_var(section.get("signing_key_id"))

    datasets_raw = section.get("datasets") or {}
    if not isinstance(datasets_raw, Mapping):
        raise ValueError("stress_lab.datasets musi być mapą")
    datasets: dict[str, StressLabDatasetConfig] = {}
    for symbol, entry in datasets_raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError("Element stress_lab.datasets musi być mapą")
        metrics_path_raw = entry.get("metrics_path")
        if metrics_path_raw in (None, ""):
            raise ValueError(f"Dataset Stress Lab '{symbol}' wymaga pola 'metrics_path'")
        metrics_path = _normalize_runtime_path(metrics_path_raw, base_dir=base_dir)
        weight_raw = entry.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dataset Stress Lab '{symbol}' posiada niepoprawną wagę") from exc
        dataset_symbol = str(entry.get("symbol", symbol)).strip() or str(symbol)
        allow_synthetic = bool(entry.get("allow_synthetic", False))
        datasets[str(symbol)] = StressLabDatasetConfig(
            symbol=dataset_symbol,
            metrics_path=str(metrics_path),
            weight=max(0.0, weight),
            allow_synthetic=allow_synthetic,
        )

    scenarios_raw = section.get("scenarios") or []
    if not isinstance(scenarios_raw, Sequence):
        raise ValueError("stress_lab.scenarios musi być listą")
    scenarios: list[StressLabScenarioConfig] = []
    for entry in scenarios_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Element stress_lab.scenarios musi być mapą")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("Scenariusz Stress Lab wymaga pola 'name'")
        severity = str(entry.get("severity", "medium")).strip().lower() or "medium"
        markets_raw = entry.get("markets") or ()
        if not isinstance(markets_raw, Sequence):
            raise ValueError(f"Scenariusz '{name}' wymaga listy 'markets'")
        markets = tuple(str(value).strip() for value in markets_raw if str(value).strip())
        if not markets:
            raise ValueError(f"Scenariusz '{name}' musi zawierać co najmniej jeden rynek")
        shocks_raw = entry.get("shocks") or ()
        if not isinstance(shocks_raw, Sequence):
            raise ValueError(f"Scenariusz '{name}' wymaga listy 'shocks'")
        shocks: list[StressLabShockConfig] = []
        for index, shock_entry in enumerate(shocks_raw, start=1):
            if not isinstance(shock_entry, Mapping):
                raise ValueError(
                    f"Scenariusz '{name}' posiada nieprawidłowy shock nr {index} (oczekiwano mapy)"
                )
            shock_type = str(shock_entry.get("type", "")).strip().lower()
            if not shock_type:
                raise ValueError(
                    f"Scenariusz '{name}' wymaga pola 'type' dla shock nr {index}"
                )
            try:
                intensity = float(shock_entry.get("intensity", 1.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Scenariusz '{name}' posiada niepoprawną wartość 'intensity' (shock {index})"
                ) from exc
            duration_raw = shock_entry.get("duration_minutes")
            if duration_raw in (None, ""):
                duration_value = None
            else:
                try:
                    duration_value = float(duration_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Scenariusz '{name}' posiada niepoprawną wartość 'duration_minutes' (shock {index})"
                    ) from exc
            notes_value = shock_entry.get("notes")
            notes = (
                str(notes_value).strip()
                if isinstance(notes_value, str) and notes_value.strip()
                else None
            )
            shocks.append(
                StressLabShockConfig(
                    type=shock_type,
                    intensity=max(0.0, intensity),
                    duration_minutes=None if duration_value is None else max(0.0, duration_value),
                    notes=notes,
                )
            )

        threshold_overrides = _parse_stress_lab_thresholds(
            entry.get("threshold_overrides"),
            context=f"stress_lab.scenarios['{name}']",
        )

        description_value = entry.get("description")
        description = (
            str(description_value).strip()
            if isinstance(description_value, str) and description_value.strip()
            else None
        )

        scenarios.append(
            StressLabScenarioConfig(
                name=name,
                severity=severity,
                markets=markets,
                shocks=tuple(shocks),
                description=description,
                threshold_overrides=threshold_overrides,
            )
        )

    thresholds = _parse_stress_lab_thresholds(
        section.get("thresholds"), context="stress_lab"
    )

    config_kwargs: dict[str, Any] = {
        "enabled": enabled,
        "require_success": require_success,
        "report_directory": report_directory,
        "datasets": datasets,
        "scenarios": tuple(scenarios),
    }
    if thresholds is not None:
        config_kwargs["thresholds"] = thresholds
    if signing_key_env is not None:
        config_kwargs["signing_key_env"] = signing_key_env
    if signing_key_path is not None:
        config_kwargs["signing_key_path"] = signing_key_path
    if signing_key_id is not None:
        config_kwargs["signing_key_id"] = signing_key_id

    return StressLabConfig(**config_kwargs)  # type: ignore[arg-type]


def _parse_resilience_thresholds(
    raw: Mapping[str, Any] | None, *, context: str
) -> ResilienceDrillThresholdsConfig | None:
    if ResilienceDrillThresholdsConfig is None:
        return None

    defaults = ResilienceDrillThresholdsConfig()
    values: dict[str, float | int] = {
        field.name: getattr(defaults, field.name)
        for field in fields(ResilienceDrillThresholdsConfig)
    }

    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context}: sekcja thresholds musi być mapą")
        for field in fields(ResilienceDrillThresholdsConfig):
            if field.name not in raw:
                continue
            value = raw[field.name]
            if value in (None, ""):
                continue
            default_value = getattr(defaults, field.name)
            try:
                if isinstance(default_value, int) and not isinstance(default_value, bool):
                    numeric = int(float(value))
                    values[field.name] = max(0, numeric)
                else:
                    numeric_float = float(value)
                    values[field.name] = max(0.0, numeric_float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{context}: wartość '{field.name}' musi być liczbą"
                ) from exc

    return ResilienceDrillThresholdsConfig(**values)


def _load_resilience_config(
    raw_root: Mapping[str, Any], *, base_dir: Path | None
) -> ResilienceConfig | None:
    if ResilienceConfig is None:
        return None

    section: Mapping[str, Any] | None = None
    top_level = raw_root.get("resilience")
    if isinstance(top_level, Mapping):
        section = top_level
    if section is None:
        runtime_section = raw_root.get("runtime")
        if isinstance(runtime_section, Mapping):
            runtime_entry = runtime_section.get("resilience")
            if isinstance(runtime_entry, Mapping):
                section = runtime_entry
    if section is None:
        return None

    enabled = bool(section.get("enabled", False))
    require_success = bool(section.get("require_success", True))

    report_dir_raw = section.get("report_directory") or "var/audit/stage6/resilience"
    report_directory = _normalize_runtime_path(report_dir_raw, base_dir=base_dir)
    if report_directory is None:
        report_directory = "var/audit/stage6/resilience"

    signing_key_env = _normalize_env_var(section.get("signing_key_env"))
    signing_key_path = _normalize_runtime_path(section.get("signing_key_path"), base_dir=base_dir)
    signing_key_id = _normalize_env_var(section.get("signing_key_id"))

    drills_raw = section.get("drills") or []
    if not isinstance(drills_raw, Sequence):
        raise ValueError("resilience.drills musi być listą")

    drills: list[ResilienceDrillConfig] = []
    for entry in drills_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Element resilience.drills musi być mapą")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("resilience.drills wymaga pola 'name'")
        primary = str(entry.get("primary", "")).strip()
        if not primary:
            raise ValueError(f"resilience.drills['{name}'] wymaga pola 'primary'")
        dataset_path_raw = entry.get("dataset_path")
        if dataset_path_raw in (None, ""):
            raise ValueError(
                f"resilience.drills['{name}'] wymaga pola 'dataset_path'"
            )
        dataset_path = _normalize_runtime_path(dataset_path_raw, base_dir=base_dir)
        if dataset_path is None:
            raise ValueError(
                f"resilience.drills['{name}']: niepoprawna ścieżka datasetu"
            )
        fallbacks_raw = entry.get("fallbacks") or ()
        if not isinstance(fallbacks_raw, Sequence):
            raise ValueError(
                f"resilience.drills['{name}']: pole 'fallbacks' musi być listą"
            )
        fallbacks = tuple(
            str(value).strip()
            for value in fallbacks_raw
            if str(value).strip()
        )
        thresholds = _parse_resilience_thresholds(
            entry.get("thresholds"), context=f"resilience.drills['{name}']"
        )
        description_value = entry.get("description")
        description = (
            str(description_value).strip()
            if isinstance(description_value, str) and description_value.strip()
            else None
        )
        config_kwargs: dict[str, Any] = {
            "name": name,
            "primary": primary,
            "fallbacks": fallbacks,
            "dataset_path": str(dataset_path),
        }
        if thresholds is not None:
            config_kwargs["thresholds"] = thresholds
        if description is not None:
            config_kwargs["description"] = description
        drills.append(ResilienceDrillConfig(**config_kwargs))

    config_kwargs: dict[str, Any] = {
        "enabled": enabled,
        "require_success": require_success,
        "report_directory": report_directory,
        "drills": tuple(drills),
    }
    if signing_key_env is not None:
        config_kwargs["signing_key_env"] = signing_key_env
    if signing_key_path is not None:
        config_kwargs["signing_key_path"] = signing_key_path
    if signing_key_id is not None:
        config_kwargs["signing_key_id"] = signing_key_id

    return ResilienceConfig(**config_kwargs)  # type: ignore[arg-type]


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


def _load_portfolio_decision_log(
    runtime_section: Optional[Mapping[str, Any]], *, base_dir: Path | None = None
) -> PortfolioDecisionLogConfig | None:
    if not _core_has("portfolio_decision_log"):
        return None

    runtime = runtime_section or {}
    log_raw = runtime.get("portfolio_decision_log")
    if not log_raw:
        return None

    available_fields = {f.name for f in fields(PortfolioDecisionLogConfig)}
    kwargs: dict[str, Any] = {
        "enabled": bool(log_raw.get("enabled", True)),
    }

    if "path" in available_fields:
        kwargs["path"] = _normalize_runtime_path(log_raw.get("path"), base_dir=base_dir)
    if "max_entries" in available_fields:
        kwargs["max_entries"] = int(log_raw.get("max_entries", 512))
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
        kwargs["signing_key_value"] = str(value) if value not in (None, "") else None
    if "signing_key_id" in available_fields:
        key_id = log_raw.get("signing_key_id")
        kwargs["signing_key_id"] = str(key_id) if key_id not in (None, "") else None
    if "jsonl_fsync" in available_fields:
        kwargs["jsonl_fsync"] = bool(log_raw.get("jsonl_fsync", False))

    return PortfolioDecisionLogConfig(**kwargs)  # type: ignore[call-arg]


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
    permission_profiles = _load_permission_profiles(raw)

    # Środowiska – budujemy kwargs dynamicznie, tak by działało na różnych gałęziach modeli.
    environments: dict[str, EnvironmentConfig] = {}
    for name, entry in (raw.get("environments", {}) or {}).items():
        permission_profile_name: str | None = None
        profile_required: Sequence[str] = ()
        profile_forbidden: Sequence[str] = ()
        if _env_has("permission_profile"):
            raw_profile = entry.get("permission_profile")
            if isinstance(raw_profile, str) and raw_profile.strip():
                permission_profile_name = raw_profile.strip()
                profile = permission_profiles.get(permission_profile_name)
                if profile is not None:
                    profile_required = profile.required_permissions
                    profile_forbidden = profile.forbidden_permissions
        raw_required = entry.get("required_permissions")
        if raw_required is None:
            required_permissions = tuple(profile_required)
        else:
            required_permissions = tuple(
                str(value).lower() for value in (raw_required or ())
            )
            if not required_permissions and profile_required:
                required_permissions = tuple(profile_required)
        raw_forbidden = entry.get("forbidden_permissions")
        if raw_forbidden is None:
            forbidden_permissions = tuple(profile_forbidden)
        else:
            forbidden_permissions = tuple(
                str(value).lower() for value in (raw_forbidden or ())
            )
            if not forbidden_permissions and profile_forbidden:
                forbidden_permissions = tuple(profile_forbidden)
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
            "required_permissions": required_permissions,
            "forbidden_permissions": forbidden_permissions,
        }
        if _env_has("permission_profile"):
            env_kwargs["permission_profile"] = permission_profile_name
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
        if _env_has("ai"):
            env_kwargs["ai"] = _load_environment_ai(entry.get("ai"), base_dir=config_base_dir)
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
    portfolio_governor_configs = _load_portfolio_governors(raw)

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
    if permission_profiles and _core_has("permission_profiles"):
        core_kwargs["permission_profiles"] = permission_profiles
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
    if _core_has("portfolio_governors"):
        core_kwargs["portfolio_governors"] = portfolio_governor_configs
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

    live_routing_config = _load_live_routing(runtime_section, base_dir=config_base_dir)
    if live_routing_config is not None:
        core_kwargs["live_routing"] = live_routing_config

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

    portfolio_decision_log_config = _load_portfolio_decision_log(
        runtime_section, base_dir=config_base_dir
    )
    if portfolio_decision_log_config is not None:
        core_kwargs["portfolio_decision_log"] = portfolio_decision_log_config

    security_baseline_config = _load_security_baseline(
        runtime_section, base_dir=config_base_dir
    )
    if security_baseline_config is not None:
        core_kwargs["security_baseline"] = security_baseline_config

    observability_config = _load_observability_config(
        raw.get("observability"), base_dir=config_base_dir
    )
    if observability_config is not None and _core_has("observability"):
        core_kwargs["observability"] = observability_config

    market_intel_config = _load_market_intel_config(
        raw.get("market_intel"), base_dir=config_base_dir
    )
    if market_intel_config is not None and _core_has("market_intel"):
        core_kwargs["market_intel"] = market_intel_config

    portfolio_governor_config = _load_portfolio_governor_config(
        raw,
        base_dir=config_base_dir,
    )
    if portfolio_governor_config is not None and _core_has("portfolio_governor"):
        core_kwargs["portfolio_governor"] = portfolio_governor_config

    stress_lab_config = _load_stress_lab_config(raw, base_dir=config_base_dir)
    if stress_lab_config is not None and _core_has("stress_lab"):
        core_kwargs["stress_lab"] = stress_lab_config

    resilience_config = _load_resilience_config(raw, base_dir=config_base_dir)
    if resilience_config is not None and _core_has("resilience"):
        core_kwargs["resilience"] = resilience_config

    decision_engine_config = _load_decision_engine_config(
        raw.get("decision_engine"), base_dir=config_base_dir
    )
    if decision_engine_config is not None and _core_has("decision_engine"):
        core_kwargs["decision_engine"] = decision_engine_config

    core_kwargs["source_path"] = str(config_absolute_path)
    core_kwargs["source_directory"] = str(config_base_dir)

    return CoreConfig(**core_kwargs)  # type: ignore[arg-type]


__all__ = ["load_core_config"]
