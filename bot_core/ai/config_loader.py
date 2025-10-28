"""Utilities for loading AI-related risk threshold configuration."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def _get_supported_signal_threshold_metrics() -> frozenset[str]:
    from bot_core.trading.signal_thresholds import SUPPORTED_SIGNAL_THRESHOLD_METRICS

    return frozenset(name.casefold() for name in SUPPORTED_SIGNAL_THRESHOLD_METRICS)

_DEFAULTS_PACKAGE = "bot_core.ai._defaults"
_DEFAULTS_RESOURCE = "risk_thresholds.yaml"
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OVERRIDE_PATH = _ROOT / "config" / "risk_thresholds.yaml"
_ENV_OVERRIDE_VAR = "BOT_CORE_RISK_THRESHOLDS_PATH"


def _coerce_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return candidate


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _deep_update(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping):
            if isinstance(target.get(key), MutableMapping):
                _deep_update(target[key], value)  # type: ignore[index]
            else:
                target[key] = deepcopy(value)
        else:
            target[key] = value


def _load_default_thresholds() -> dict[str, Any]:
    with resources.files(_DEFAULTS_PACKAGE).joinpath(_DEFAULTS_RESOURCE).open("r", encoding="utf8") as stream:
        data = yaml.safe_load(stream) or {}
    return deepcopy(_ensure_mapping(data, context="Default risk thresholds configuration"))


_DEFAULT_THRESHOLDS = _load_default_thresholds()


def _load_override(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf8") as stream:
        data = yaml.safe_load(stream) or {}
    return _ensure_mapping(data, context=f"Risk thresholds configuration at {path}")


def _validate_thresholds(thresholds: Mapping[str, Any]) -> None:
    market = thresholds.get("market_regime", {})
    if not isinstance(market, Mapping):
        raise ValueError("market_regime section must be a mapping")

    metrics = market.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("market_regime.metrics section must be a mapping")
    for key in ("short_span_min", "short_span_divisor", "long_span_min"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Invalid metrics threshold {key}: {value!r}")

    risk_score = market.get("risk_score", {})
    if not isinstance(risk_score, Mapping):
        raise ValueError("market_regime.risk_score section must be a mapping")
    for key in ("volatility_weight", "intraday_weight", "drawdown_weight", "volatility_mix_weight"):
        value = risk_score.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Invalid risk score weight {key}: {value!r}")

    auto_trader = thresholds.get("auto_trader", {})
    if not isinstance(auto_trader, Mapping):
        raise ValueError("auto_trader section must be a mapping")

    map_cfg = auto_trader.get("map_regime_to_signal", {})
    if not isinstance(map_cfg, Mapping):
        raise ValueError("auto_trader.map_regime_to_signal section must be a mapping")
    for key in (
        "assessment_confidence",
        "summary_confidence",
        "summary_stability",
        "risk_trend",
        "risk_volatility",
        "regime_persistence",
        "transition_rate",
    ):
        value = map_cfg.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Invalid signal threshold {key}: {value!r}")

    adjust_cfg = auto_trader.get("adjust_strategy_parameters", {})
    if not isinstance(adjust_cfg, Mapping):
        raise ValueError("auto_trader.adjust_strategy_parameters section must be a mapping")
    numeric_keys = (
        "high_risk",
        "trend_low_risk",
        "mean_reversion_low_risk",
        "intraday_low_risk",
        "risk_level_elevated",
        "risk_level_elevated_stop_loss",
        "risk_level_elevated_take_profit",
        "risk_level_calm",
        "risk_level_calm_leverage",
        "risk_level_calm_take_profit",
        "resilience_low",
        "resilience_high",
        "stress_balance_high",
        "entropy_high",
        "entropy_low",
        "risk_volatility_high",
        "regime_persistence_low",
        "regime_persistence_high",
        "instability_ceiling",
        "confidence_volatility_high",
        "confidence_trend_low",
        "regime_streak_low",
        "confidence_trend_high",
        "confidence_volatility_low",
        "risk_level_calm_upper",
        "transition_rate_high",
        "summary_risk_cap",
        "summary_risk_trend_high",
        "summary_stability_floor",
        "instability_critical",
        "instability_elevated",
        "instability_low",
        "drawdown_critical",
        "drawdown_elevated",
        "drawdown_low",
        "liquidity_pressure_high",
        "liquidity_pressure_low",
        "low_risk_enhancement_cap",
        "stress_relief_risk_cap",
        "moderate_risk_enhancement_cap",
        "confidence_decay_high",
        "degradation_critical",
        "degradation_elevated",
        "degradation_low",
        "degradation_positive_cap",
        "stability_projection_low",
        "stability_projection_high",
        "volume_trend_volatility_high",
        "volume_trend_volatility_low",
        "volatility_trend_high",
        "volatility_trend_relief",
        "drawdown_trend_high",
        "drawdown_trend_relief",
        "stress_index_critical",
        "stress_index_elevated",
        "stress_index_low",
        "stress_index_relief",
        "stress_index_tail_relief",
        "stress_momentum_high",
        "stress_momentum_low",
        "tail_risk_high",
        "tail_risk_low",
        "shock_frequency_high",
        "shock_frequency_low",
        "regime_persistence_positive",
        "distribution_pressure_adjust_high",
        "distribution_pressure_adjust_low",
        "skewness_bias_adjust_high",
        "skewness_bias_adjust_low",
        "kurtosis_adjust_high",
        "kurtosis_adjust_low",
        "volume_imbalance_adjust_high",
        "volume_imbalance_adjust_low",
        "liquidity_gap_high",
        "liquidity_gap_low",
        "liquidity_gap_relief",
        "resilience_mid",
        "liquidity_trend_high",
        "liquidity_trend_low",
        "stress_projection_critical",
        "stress_projection_elevated",
        "stress_projection_low",
        "confidence_resilience_high",
        "confidence_resilience_mid",
        "confidence_fragility_high",
        "confidence_fragility_low",
        "volatility_of_volatility_high",
        "volatility_of_volatility_low",
    )
    for key in numeric_keys:
        value = adjust_cfg.get(key)
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"Invalid strategy parameter threshold {key}: {value!r}")

    guardrails_cfg = auto_trader.get("signal_guardrails", {})
    if not isinstance(guardrails_cfg, Mapping):
        raise ValueError("auto_trader.signal_guardrails section must be a mapping")
    guardrail_keys = (
        "effective_risk_cap",
        "stress_index",
        "tail_risk_index",
        "shock_frequency",
        "stress_momentum",
        "resilience_score",
        "stress_balance",
        "regime_entropy",
        "confidence_fragility",
        "degradation_score",
        "stability_projection",
        "volume_trend_volatility",
        "volatility_trend",
        "drawdown_trend",
        "liquidity_gap",
        "stress_projection",
        "confidence_resilience",
        "liquidity_trend",
    )
    for key in guardrail_keys:
        value = guardrails_cfg.get(key)
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"Invalid signal guardrail threshold {key}: {value!r}")

    cooldown_cfg = auto_trader.get("cooldown", {})
    if not isinstance(cooldown_cfg, Mapping):
        raise ValueError("auto_trader.cooldown section must be a mapping")

    release = cooldown_cfg.get("release", {})
    if not isinstance(release, Mapping):
        raise ValueError("auto_trader.cooldown.release section must be a mapping")
    for key in ("cooldown_score", "recovery_potential"):
        value = release.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Invalid cooldown release threshold {key}: {value!r}")

    supported_metrics = _get_supported_signal_threshold_metrics()

    signal_thresholds = auto_trader.get("signal_thresholds")
    if signal_thresholds is None:
        normalised_signal_thresholds: dict[str, float] | None = None
    elif isinstance(signal_thresholds, Mapping):
        normalised_signal_thresholds = {}
        for key, raw in signal_thresholds.items():
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"auto_trader.signal_thresholds[{key!r}] must be numeric, got {raw!r}"
                ) from None
            if not math.isfinite(numeric):
                raise ValueError(
                    f"auto_trader.signal_thresholds[{key!r}] must be finite, got {numeric!r}"
                )
            key_norm = str(key).strip().casefold()
            if not key_norm:
                raise ValueError("auto_trader.signal_thresholds keys must be non-empty")
            if key_norm not in supported_metrics:
                raise ValueError(
                    "auto_trader.signal_thresholds contains unsupported metric "
                    f"{key!r}; supported metrics: {sorted(supported_metrics)!r}"
                )
            normalised_signal_thresholds[key_norm] = numeric
    else:
        raise ValueError("auto_trader.signal_thresholds section must be a mapping")

    strategy_thresholds = auto_trader.get("strategy_signal_thresholds")
    if strategy_thresholds is None:
        normalised_strategy_thresholds: dict[str, dict[str, dict[str, float]]] | None = None
    elif isinstance(strategy_thresholds, Mapping):
        normalised_strategy_thresholds = {}
        for exchange_key, strategy_map in strategy_thresholds.items():
            if not isinstance(strategy_map, Mapping):
                raise ValueError(
                    "auto_trader.strategy_signal_thresholds values must be mappings"
                )
            exchange_norm = str(exchange_key).strip().casefold()
            if not exchange_norm:
                continue
            strategies_normalised: dict[str, dict[str, float]] = {}
            for strategy_key, metric_map in strategy_map.items():
                if not isinstance(metric_map, Mapping):
                    raise ValueError(
                        "auto_trader.strategy_signal_thresholds entries must map to metric mappings"
                    )
                strategy_norm = str(strategy_key).strip().casefold()
                if not strategy_norm:
                    continue
                metrics_normalised: dict[str, float] = {}
                for metric_name, raw in metric_map.items():
                    try:
                        numeric = float(raw)
                    except (TypeError, ValueError):
                        raise ValueError(
                            "auto_trader.strategy_signal_thresholds values must be numeric"
                        ) from None
                    if not math.isfinite(numeric):
                        raise ValueError(
                            "auto_trader.strategy_signal_thresholds values must be finite"
                        )
                    metric_norm = str(metric_name).strip().casefold()
                    if not metric_norm:
                        raise ValueError(
                            "auto_trader.strategy_signal_thresholds metric names must be non-empty"
                        )
                    if metric_norm not in supported_metrics:
                        raise ValueError(
                            "auto_trader.strategy_signal_thresholds contains unsupported metric "
                            f"{metric_name!r}; supported metrics: {sorted(supported_metrics)!r}"
                        )
                    metrics_normalised[metric_norm] = numeric
                if metrics_normalised:
                    strategies_normalised[strategy_norm] = metrics_normalised
            if strategies_normalised:
                normalised_strategy_thresholds[exchange_norm] = strategies_normalised
    else:
        raise ValueError("auto_trader.strategy_signal_thresholds section must be a mapping")

    if normalised_signal_thresholds is not None:
        auto_trader["signal_thresholds"] = normalised_signal_thresholds
    if normalised_strategy_thresholds is not None:
        auto_trader["strategy_signal_thresholds"] = normalised_strategy_thresholds


@lru_cache(maxsize=None)
def load_risk_thresholds(config_path: str | Path | None = None) -> Mapping[str, Any]:
    data = deepcopy(_DEFAULT_THRESHOLDS)
    override_path: Path | None
    explicit_override = config_path if config_path is not None else os.getenv(_ENV_OVERRIDE_VAR)
    if explicit_override:
        override_path = _coerce_path(explicit_override)
        if not override_path.exists():
            raise FileNotFoundError(
                f"Risk thresholds override file {override_path} declared but does not exist"
            )
    else:
        override_path = _DEFAULT_OVERRIDE_PATH

    if override_path.exists():
        overrides = _load_override(override_path)
        _deep_update(data, overrides)
    _validate_thresholds(data)
    return data


def reset_threshold_cache() -> None:
    cache_clear = getattr(load_risk_thresholds, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


__all__ = ["load_risk_thresholds", "reset_threshold_cache"]
