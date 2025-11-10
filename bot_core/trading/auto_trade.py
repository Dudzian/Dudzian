"""Rozszerzony silnik autotradingu wspierający wiele strategii i reżimy."""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from collections import Counter, deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from types import MappingProxyType
from typing import Any, Deque, Dict, Iterable as TypingIterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from bot_core.ai import DecisionModelInference, ModelScore
from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeStrategyWeights,
    RegimeSummary,
    RiskLevel,
)
from bot_core.backtest.ma import simulate_trades_ma  # noqa: F401 - zachowaj kompatybilność API
from bot_core.events import DebounceRule, Event, EventBus, EventType, EmitterAdapter
from bot_core.trading.engine import (
    EngineConfig,
    TechnicalIndicators,
    TechnicalIndicatorsService,
    TradingParameters,
)
from bot_core.trading.signal_thresholds import SUPPORTED_SIGNAL_THRESHOLD_METRICS
from bot_core.trading.regime_workflow import RegimeSwitchDecision
from bot_core.trading.strategies import StrategyCatalog
from bot_core.trading.strategy_aliasing import (
    MIGRATION_FALLBACK_SUFFIX,
    StrategyAliasResolver,
    canonical_alias_map,
    normalise_suffixes,
    strategy_key_aliases,
    strategy_name_candidates,
)
from bot_core.strategies import StrategyPresetWizard
from bot_core.strategies.regime_workflow import (
    PresetAvailability,
    RegimePresetActivation,
    StrategyRegimeWorkflow,
)
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal

class _AliasConfigSentinel:
    __slots__ = ()


_ALIAS_UNSET = _AliasConfigSentinel()


def _canonical_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _canonical_metric_name(name: str) -> str:
    return str(name).strip().casefold()


@dataclass
class _AutoRiskFreezeState:
    risk_level: RiskLevel | None = None
    risk_score: float | None = None
    triggered_at: float = 0.0
    last_extension_at: float = 0.0
    reason: str | None = None
    released_at: float = 0.0


@dataclass
class _ManualRiskFreezeState:
    reason: str | None = None
    triggered_at: float | None = None
    last_extension_at: float | None = None
    source_reason: str | None = None
    released_at: float | None = None
@dataclass
class AutoTradeConfig:
    """Podstawowa konfiguracja silnika autotradingu.

    Pola ``primary_exchange`` oraz ``strategy`` są propagowane do dziennika decyzji
    oraz zdarzeń telemetrycznych. Należy je uzupełnić (lub zapewnić w
    ``decision_journal_context``) jeśli downstream wymaga routingu po tych
    identyfikatorach.
    """

    symbol: str = "BTCUSDT"
    qty: float = 0.01
    emit_signals: bool = True
    use_close_only: bool = True
    default_params: Dict[str, int] | None = None
    risk_freeze_seconds: int = 300
    strategy_weights: Mapping[str, Mapping[str, float]] | None = None
    trading_parameters: TradingParameters | Mapping[str, Any] | None = None
    regime_window: int = 60
    activation_threshold: float = 0.2
    breakout_window: int = 24
    mean_reversion_window: int = 20
    mean_reversion_z: float = 1.25
    arbitrage_window: int = 20
    arbitrage_threshold: float = 0.003
    arbitrage_confirmation_window: int = 4
    regime_history_maxlen: int = 5
    regime_history_decay: float = 0.65
    auto_risk_freeze: bool = True
    auto_risk_freeze_level: RiskLevel | str = RiskLevel.CRITICAL
    auto_risk_freeze_score: float = 0.8
    regime_parameter_overrides: Mapping[str, Mapping[str, float | int]] | None = None
    strategy_alias_map: Mapping[str, str] | None = None
    strategy_alias_suffixes: Iterable[str] | None = None
    primary_exchange: str | None = None
    strategy: str | None = None
    signal_thresholds: Mapping[str, float] | None = None
    strategy_signal_thresholds: Mapping[str, Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        if self.default_params is None:
            self.default_params = {"fast": 10, "slow": 50}
        if self.strategy_weights is None:
            defaults = RegimeStrategyWeights.default()
            self.strategy_weights = {
                regime.value: dict(weights)
                for regime, weights in defaults.weights.items()
            }
        params = self.trading_parameters
        if params is None:
            self.trading_parameters = TradingParameters()
        elif isinstance(params, Mapping):
            self.trading_parameters = TradingParameters(**{str(k): v for k, v in params.items()})
        elif not isinstance(params, TradingParameters):
            raise TypeError(
                "trading_parameters must be TradingParameters, mapping or None"
            )
        self.breakout_window = int(self.breakout_window)
        if self.breakout_window < 2:
            raise ValueError("breakout_window must be at least 2")
        self.mean_reversion_window = int(self.mean_reversion_window)
        if self.mean_reversion_window < 3:
            raise ValueError("mean_reversion_window must be at least 3")
        self.mean_reversion_z = float(self.mean_reversion_z)
        self.regime_history_maxlen = int(self.regime_history_maxlen)
        self.regime_history_decay = float(self.regime_history_decay)
        if self.regime_history_maxlen < 1:
            raise ValueError("regime_history_maxlen must be at least 1")
        if not (0.0 < self.regime_history_decay <= 1.0):
            raise ValueError("regime_history_decay must be in the (0, 1] range")
        self.auto_risk_freeze = bool(self.auto_risk_freeze)
        self.arbitrage_window = int(self.arbitrage_window)
        if self.arbitrage_window < 2:
            raise ValueError("arbitrage_window must be at least 2")
        self.arbitrage_confirmation_window = int(self.arbitrage_confirmation_window)
        if self.arbitrage_confirmation_window < 1:
            raise ValueError("arbitrage_confirmation_window must be at least 1")
        self.arbitrage_threshold = float(self.arbitrage_threshold)
        if self.arbitrage_threshold <= 0:
            raise ValueError("arbitrage_threshold must be positive")
        level = self.auto_risk_freeze_level
        if isinstance(level, str):
            try:
                level = RiskLevel(level.lower())
            except ValueError as exc:  # pragma: no cover - walidacja wejścia
                raise ValueError("auto_risk_freeze_level must be a valid RiskLevel") from exc
        elif not isinstance(level, RiskLevel):
            raise TypeError("auto_risk_freeze_level must be RiskLevel or string")
        self.auto_risk_freeze_level = level
        self.auto_risk_freeze_score = float(self.auto_risk_freeze_score)
        if not (0.0 <= self.auto_risk_freeze_score <= 1.0):
            raise ValueError("auto_risk_freeze_score must be in the [0, 1] range")
        if self.regime_parameter_overrides is None:
            self.regime_parameter_overrides = {}
        else:
            cleaned: Dict[str, Dict[str, float | int]] = {}
            for regime_key, payload in self.regime_parameter_overrides.items():
                normalised: Dict[str, float | int] = {}
                for key, value in payload.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, int):
                        normalised[str(key)] = int(value)
                        continue
                    if isinstance(value, float):
                        normalised[str(key)] = float(value)
                        continue
                    try:
                        coerced = float(value)
                    except (TypeError, ValueError):
                        continue
                    normalised[str(key)] = coerced
                if normalised:
                    cleaned[str(regime_key)] = normalised
            self.regime_parameter_overrides = cleaned
        alias_map = canonical_alias_map(self.strategy_alias_map)
        self.strategy_alias_map = alias_map or None
        if self.strategy_alias_suffixes is None:
            self.strategy_alias_suffixes = None
        else:
            normalised_suffixes = normalise_suffixes(self.strategy_alias_suffixes)
            self.strategy_alias_suffixes = normalised_suffixes or None
        if self.primary_exchange is not None:
            self.primary_exchange = str(self.primary_exchange).strip() or None
        if self.strategy is not None:
            self.strategy = str(self.strategy).strip() or None
        if self.signal_thresholds:
            cleaned_signal_thresholds: dict[str, float] = {}
            for key, value in dict(self.signal_thresholds).items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                cleaned_signal_thresholds[_canonical_metric_name(str(key))] = numeric
            self.signal_thresholds = (
                MappingProxyType(cleaned_signal_thresholds)
                if cleaned_signal_thresholds
                else None
            )
        else:
            self.signal_thresholds = None
        if self.strategy_signal_thresholds:
            cleaned_exchanges: dict[str, Mapping[str, float]] = {}
            for exchange_key, strategies in dict(self.strategy_signal_thresholds).items():
                exchange_norm = _canonical_identifier(str(exchange_key))
                if not exchange_norm:
                    continue
                cleaned_strategies: dict[str, Mapping[str, float]] = {}
                for strategy_key, metric_map in dict(strategies).items():
                    strategy_norm = _canonical_identifier(str(strategy_key))
                    if not strategy_norm:
                        continue
                    cleaned_metrics: dict[str, float] = {}
                    for metric_name, value in dict(metric_map).items():
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(numeric):
                            continue
                        cleaned_metrics[_canonical_metric_name(str(metric_name))] = numeric
                    if cleaned_metrics:
                        cleaned_strategies[strategy_norm] = MappingProxyType(cleaned_metrics)
                if cleaned_strategies:
                    cleaned_exchanges[exchange_norm] = MappingProxyType(cleaned_strategies)
            self.strategy_signal_thresholds = (
                MappingProxyType(cleaned_exchanges) if cleaned_exchanges else None
            )
        else:
            self.strategy_signal_thresholds = None


class AutoTradeEngine:
    """Prosty kontroler autotradingu reagujący na ticki z EventBusa."""

    _RISK_LEVEL_ORDER = {
        RiskLevel.CALM: 0,
        RiskLevel.BALANCED: 1,
        RiskLevel.WATCH: 2,
        RiskLevel.ELEVATED: 3,
        RiskLevel.CRITICAL: 4,
    }

    _STRATEGY_SUFFIXES: tuple[str, ...] = ("_probing", MIGRATION_FALLBACK_SUFFIX)
    _STRATEGY_ALIAS_MAP: Mapping[str, str] = MappingProxyType({
        "intraday_breakout": "day_trading",
    })
    _ALIAS_RESOLVER: StrategyAliasResolver | None = None

    _PRESET_ENGINE_MAPPING = {
        "trend_following": "daily_trend_momentum",
        "day_trading": "day_trading",
        "mean_reversion": "mean_reversion",
        "arbitrage": "cross_exchange_arbitrage",
        "grid_trading": "grid_trading",
        "options_income": "options_income",
        "scalping": "scalping",
        "statistical_arbitrage": "statistical_arbitrage",
        "volatility_target": "volatility_target",
    }

    _EVENT_CONTEXT_KEYS: tuple[str, ...] = (
        "schedule",
        "strategy",
        "schedule_run_id",
        "strategy_instance_id",
        "status",
        "primary_exchange",
        "secondary_exchange",
        "base_asset",
        "quote_asset",
        "instrument_type",
        "data_feed",
        "risk_budget_bucket",
    )

    _ROUTING_DETAIL_KEYS: tuple[str, ...] = (
        "primary_exchange",
        "strategy",
    )

    @classmethod
    def _alias_resolver(cls) -> StrategyAliasResolver:
        resolver = cls._ALIAS_RESOLVER
        if (
            resolver is None
            or resolver.base_alias_map is not cls._STRATEGY_ALIAS_MAP
            or resolver.base_suffixes != cls._STRATEGY_SUFFIXES
        ):
            resolver = StrategyAliasResolver(
                cls._STRATEGY_ALIAS_MAP,
                cls._STRATEGY_SUFFIXES,
            )
            cls._ALIAS_RESOLVER = resolver
        return resolver

    def __init__(
        self,
        adapter: EmitterAdapter,
        broker_submit_market,
        cfg: Optional[AutoTradeConfig] = None,
        *,
        regime_classifier: MarketRegimeClassifier | None = None,
        regime_history: RegimeHistory | None = None,
        strategy_catalog: StrategyCatalog | None = None,
        regime_workflow: StrategyRegimeWorkflow | None = None,
        indicator_service: TechnicalIndicatorsService | None = None,
        indicator_config: EngineConfig | None = None,
        decision_inference: DecisionModelInference | None = None,
        decision_journal: TradingDecisionJournal | None = None,
        decision_journal_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.bus: EventBus = adapter.bus
        self.cfg = cfg or AutoTradeConfig()
        self._logger = logging.getLogger(__name__)
        base_resolver = type(self)._alias_resolver()
        override_resolver = base_resolver
        if self.cfg.strategy_alias_map or self.cfg.strategy_alias_suffixes:
            override_resolver = base_resolver.derive(
                alias_map=self.cfg.strategy_alias_map,
                suffixes=self.cfg.strategy_alias_suffixes,
            )
        self._alias_resolver_override: StrategyAliasResolver | None = (
            None if override_resolver is base_resolver else override_resolver
        )
        self._closes: List[float] = []
        self._bars: Deque[Mapping[str, float]] = deque(maxlen=max(self.cfg.regime_window * 3, 200))
        self._params = dict(self.cfg.default_params)
        self._last_signal: Optional[int] = None
        self._enabled: bool = True
        self._risk_frozen_until: float = 0.0
        self._manual_risk_frozen_until: float = 0.0
        self._auto_risk_frozen_until: float = 0.0
        self._auto_risk_frozen: bool = False
        self._auto_risk_state = _AutoRiskFreezeState()
        self._manual_risk_state: _ManualRiskFreezeState | None = None
        self._submit_market = broker_submit_market
        self._regime_classifier = regime_classifier or MarketRegimeClassifier()
        self._regime_history = RegimeHistory(
            thresholds_loader=self._regime_classifier.thresholds_loader
        )
        self._regime_history.reload_thresholds(
            thresholds=self._regime_classifier.thresholds_snapshot()
        )
        self._global_signal_thresholds: Dict[str, float] = {}
        self._strategy_signal_thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._signal_threshold_after_adjustment: float | None = None
        self._activation_threshold = float(self.cfg.activation_threshold)
        self._configure_signal_thresholds(
            signal_thresholds=self.cfg.signal_thresholds,
            strategy_signal_thresholds=self.cfg.strategy_signal_thresholds,
        )
        normalized_weights = self._normalize_strategy_config(self.cfg.strategy_weights)
        self._strategy_weights = RegimeStrategyWeights(
            weights={
                regime: dict(weights)
                for regime, weights in normalized_weights.items()
            }
        )
        catalog = strategy_catalog or StrategyCatalog.default()
        self._strategy_catalog = catalog
        self._engine_key_cache: Dict[str, str | None] = {}
        base_override = {
            "day_trading_momentum_window": int(max(1, self.cfg.breakout_window)),
            "day_trading_volatility_window": int(
                max(1, math.ceil(self.cfg.breakout_window * 1.5))
            ),
            "arbitrage_confirmation_window": int(self.cfg.arbitrage_confirmation_window),
            "arbitrage_spread_threshold": float(self.cfg.arbitrage_threshold),
        }
        workflow_parameter_overrides = {
            regime: dict(base_override) for regime in MarketRegime
        }
        self._workflow_parameter_overrides = {
            regime: dict(values) for regime, values in workflow_parameter_overrides.items()
        }
        self._workflow_signing_key = f"autotrade:{self.cfg.symbol}".encode("utf-8")
        self._workflow_owned = False
        self._regime_workflow: StrategyRegimeWorkflow | None = self._initialize_strategy_workflow(
            regime_workflow
        )
        self._sync_workflow_state()
        if indicator_service is None:
            indicator_cfg = indicator_config or EngineConfig(cache_indicators=False)
            self._indicator_service = TechnicalIndicatorsService(self._logger, indicator_cfg)
        else:
            self._indicator_service = indicator_service
        self._base_trading_params: TradingParameters = self.cfg.trading_parameters
        self._last_trading_parameters: TradingParameters | None = None
        self._last_regime: MarketRegimeAssessment | None = None
        self._last_summary: RegimeSummary | None = None
        self._last_regime_decision: RegimeSwitchDecision | None = None
        self._last_regime_activation: RegimePresetActivation | None = None
        self._regime_activation_history: list[RegimePresetActivation] = []
        self._preset_availability: tuple[PresetAvailability, ...] = ()
        self._decision_inference = decision_inference
        self._decision_journal = decision_journal
        if decision_journal_context is None:
            self._decision_journal_context: Dict[str, Any] = {}
        elif not isinstance(decision_journal_context, Mapping):
            raise TypeError(
                "decision_journal_context must be a mapping or None"
            )
        else:
            self._decision_journal_context = {
                str(key): value for key, value in decision_journal_context.items()
            }
        for key in self._ROUTING_DETAIL_KEYS:
            context_value = self._decision_journal_context.get(key)
            normalised: str | None = None
            if context_value is not None:
                candidate = str(context_value).strip()
                if candidate:
                    normalised = candidate
            if normalised is None:
                cfg_value = getattr(self.cfg, key, None)
                if cfg_value is not None:
                    candidate = str(cfg_value).strip()
                    if candidate:
                        normalised = candidate
            if normalised is not None:
                self._decision_journal_context[key] = normalised
            elif key in self._decision_journal_context:
                del self._decision_journal_context[key]
        self._last_inference_score: ModelScore | None = None
        self._last_inference_metadata: Dict[str, object] | None = None
        self._last_inference_scored_at: dt.datetime | None = None
        self._last_dynamic_preset: Dict[str, object] | None = None

        batch_rule = DebounceRule(window=0.1, max_batch=1)
        self.bus.subscribe(EventType.MARKET_TICK, self._on_ticks_batch, rule=batch_rule)
        self.bus.subscribe(EventType.WFO_STATUS, self._on_wfo_status_batch, rule=batch_rule)
        self.bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert_batch, rule=batch_rule)

    def _routing_snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for key in self._ROUTING_DETAIL_KEYS:
            value = self._decision_journal_context.get(key)
            candidate: str | None = None
            if value is not None:
                candidate = str(value).strip()
            if not candidate:
                cfg_value = getattr(self.cfg, key, None)
                if cfg_value is not None:
                    cfg_candidate = str(cfg_value).strip()
                    if cfg_candidate:
                        candidate = cfg_candidate
            if not candidate:
                candidate = "unknown"
            snapshot[str(key)] = candidate
        return snapshot

    def _augment_status_detail(
        self, detail: Mapping[str, Any] | None
    ) -> Dict[str, Any]:
        payload = {str(key): value for key, value in (detail or {}).items()}
        routing_snapshot = self._routing_snapshot()
        for key, value in routing_snapshot.items():
            payload.setdefault(key, value)
        return payload

    def enable(self) -> None:
        self._enabled = True
        self._manual_risk_frozen_until = 0.0
        self._auto_risk_frozen_until = 0.0
        self._auto_risk_frozen = False
        self._auto_risk_state = _AutoRiskFreezeState()
        self._manual_risk_state = None
        self._recompute_risk_freeze_until()
        detail = self._augment_status_detail({"symbol": self.cfg.symbol})
        self.adapter.push_autotrade_status("enabled", detail=detail)  # type: ignore[attr-defined]

    def disable(self, reason: str = "") -> None:
        self._enabled = False
        detail = self._augment_status_detail(
            {"symbol": self.cfg.symbol, "reason": reason}
        )
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "disabled",
            detail=detail,
            level="WARN",
        )

    def apply_params(self, params: Dict[str, int]) -> None:
        self._params = dict(params)
        detail = self._augment_status_detail(
            {"symbol": self.cfg.symbol, "params": self._params}
        )
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "params_applied",
            detail=detail,
        )

    def freeze_risk(
        self,
        duration: float | int | None = None,
        *,
        reason: str | None = "manual",
    ) -> None:
        """Zainicjuj ręczne zamrożenie ryzyka na określony czas."""

        seconds = float(duration if duration is not None else self.cfg.risk_freeze_seconds)
        if seconds <= 0:
            raise ValueError("duration must be positive")
        now = time.time()
        expiry = now + seconds
        self._apply_manual_risk_freeze(
            reason=str(reason or "manual"),
            expiry=expiry,
            now=now,
            source="manual",
        )
        self._recompute_risk_freeze_until()

    def unfreeze_risk(self, *, reason: str | None = None) -> None:
        """Odblokuj ręczne zamrożenie ryzyka, jeżeli jest aktywne."""

        now = time.time()
        released = self._clear_manual_risk_freeze(now=now, source=reason or "manual")
        if released:
            self._recompute_risk_freeze_until()

    def _apply_manual_risk_freeze(
        self,
        *,
        reason: str,
        expiry: float,
        now: float,
        source: str,
    ) -> None:
        previous_until = self._manual_risk_frozen_until if self._manual_risk_state else 0.0
        state = self._manual_risk_state or _ManualRiskFreezeState()
        state.reason = reason
        state.source_reason = source
        if not state.triggered_at:
            state.triggered_at = now
        state.last_extension_at = now
        state.released_at = None
        self._manual_risk_state = state
        self._manual_risk_frozen_until = float(expiry)
        detail: Dict[str, Any] = {
            "symbol": self.cfg.symbol,
            "reason": reason,
            "source_reason": source,
            "triggered_at": state.triggered_at,
            "last_extension_at": state.last_extension_at,
            "released_at": None,
            "frozen_for": max(0.0, float(expiry) - float(state.triggered_at or now)),
            "until": float(expiry),
        }
        event = "risk_freeze"
        if previous_until and previous_until > now:
            if float(expiry) > previous_until + 1e-6:
                detail["extended_from"] = previous_until
                event = "risk_freeze_extend"
            else:
                detail["previous_until"] = previous_until
        detail = self._augment_status_detail(detail)
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            event,
            detail=detail,
            level="WARN",
        )
        self._record_risk_decision_event(
            event=event,
            mode="manual",
            detail=dict(detail),
        )

    def _clear_manual_risk_freeze(self, *, now: float, source: str) -> bool:
        state_before = self._manual_risk_state
        if not state_before:
            return False
        active = bool(self._manual_risk_frozen_until and self._manual_risk_frozen_until > now)
        self._manual_risk_state = None
        self._manual_risk_frozen_until = 0.0
        if active:
            state = state_before
            released_at = now
            state.released_at = released_at
            frozen_for = max(0.0, released_at - float(state.triggered_at or released_at))
            detail = {
                "symbol": self.cfg.symbol,
                "reason": state.reason,
                "source_reason": source,
                "triggered_at": state.triggered_at,
                "last_extension_at": state.last_extension_at,
                "released_at": released_at,
                "frozen_for": frozen_for,
            }
            detail = self._augment_status_detail(detail)
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_unfreeze",
                detail=detail,
            )
            self._record_risk_decision_event(
                event="risk_unfreeze",
                mode="manual",
                detail=dict(detail),
            )
        return active

    def _auto_risk_freeze_reason(
        self,
        *,
        level_triggered: bool,
        score_triggered: bool,
        summary: RegimeSummary | None,
    ) -> str:
        if self._auto_risk_frozen:
            previous = self._auto_risk_state
            prev_level = previous.risk_level
            prev_score = previous.risk_score if previous.risk_score is not None else 0.0
            if summary is not None and prev_level is not None:
                prev_rank = self._RISK_LEVEL_ORDER.get(prev_level, -1)
                current_rank = self._RISK_LEVEL_ORDER.get(summary.risk_level, -1)
                if current_rank > prev_rank >= 0:
                    return "risk_level_escalated"
            if summary is not None and previous.risk_score is not None:
                if float(summary.risk_score) > float(prev_score) + 1e-6:
                    return "risk_score_increase"
            return "expiry_near"
        if level_triggered and score_triggered:
            return "risk_level_and_score_threshold"
        if level_triggered:
            return "risk_level_threshold"
        return "risk_score_threshold"

    def _auto_risk_recovery_reason(
        self,
        *,
        summary: RegimeSummary | None,
        target_rank: int,
    ) -> str:
        if summary is None:
            return "risk_recovered"
        level_rank = self._RISK_LEVEL_ORDER.get(summary.risk_level, -1)
        score_below = summary.risk_score < self.cfg.auto_risk_freeze_score
        level_below = level_rank < target_rank or target_rank < 0
        if score_below and level_below:
            return "risk_recovered"
        if level_below:
            return "risk_level_recovered"
        if score_below:
            return "risk_score_recovered"
        return "risk_recovered"

    def _emit_auto_risk_freeze(
        self,
        *,
        now: float,
        summary: RegimeSummary | None,
        expiry: float,
        reason: str,
        level: RiskLevel | None,
        score: float | None,
        previous_until: float,
        event: str,
    ) -> None:
        state = self._auto_risk_state
        if event == "auto_risk_freeze":
            state.triggered_at = now
        elif not state.triggered_at:
            state.triggered_at = now
        state.last_extension_at = now
        state.risk_level = level
        state.risk_score = None if score is None else float(score)
        state.reason = reason
        state.released_at = 0.0
        self._auto_risk_state = state
        frozen_until = float(expiry)
        if previous_until and previous_until > frozen_until:
            frozen_until = previous_until
        self._auto_risk_frozen = True
        self._auto_risk_frozen_until = frozen_until
        detail: Dict[str, Any] = {
            "symbol": self.cfg.symbol,
            "reason": reason,
            "risk_level": level.value if isinstance(level, RiskLevel) else None,
            "risk_score": None if score is None else float(score),
            "triggered_at": state.triggered_at,
            "last_extension_at": state.last_extension_at,
            "released_at": None,
            "frozen_for": max(0.0, frozen_until - float(state.triggered_at)),
            "until": frozen_until,
        }
        if event == "auto_risk_freeze_extend" and previous_until:
            detail["extended_from"] = previous_until
        detail = self._augment_status_detail(detail)
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            event,
            detail=detail,
            level="WARN",
        )
        self._record_risk_decision_event(
            event=event,
            mode="auto",
            detail=dict(detail),
            summary=summary,
        )

    def _emit_auto_risk_unfreeze(
        self,
        *,
        now: float,
        reason: str,
        summary: RegimeSummary | None,
    ) -> None:
        state = self._auto_risk_state
        level = summary.risk_level if summary is not None else state.risk_level
        score = summary.risk_score if summary is not None else state.risk_score
        triggered_at = state.triggered_at or now
        last_extension_at = state.last_extension_at or triggered_at
        released_at = now
        state.released_at = released_at
        detail = {
            "symbol": self.cfg.symbol,
            "reason": reason,
            "risk_level": level.value if isinstance(level, RiskLevel) else None,
            "risk_score": None if score is None else float(score),
            "triggered_at": triggered_at,
            "last_extension_at": last_extension_at,
            "released_at": released_at,
            "frozen_for": max(0.0, released_at - triggered_at),
        }
        detail = self._augment_status_detail(detail)
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "auto_risk_unfreeze",
            detail=detail,
        )
        self._record_risk_decision_event(
            event="auto_risk_unfreeze",
            mode="auto",
            detail=dict(detail),
            summary=summary,
        )
        self._auto_risk_state = _AutoRiskFreezeState()
        self._auto_risk_frozen = False
        self._auto_risk_frozen_until = 0.0

    @staticmethod
    def _normalize_strategy_config(
        raw: Mapping[str, Mapping[str, float]] | None
    ) -> Dict[MarketRegime, Dict[str, float]]:
        if raw is None:
            defaults = RegimeStrategyWeights.default()
            return {regime: dict(weights) for regime, weights in defaults.weights.items()}
        normalized: Dict[MarketRegime, Dict[str, float]] = {}
        for regime_name, weights in raw.items():
            try:
                regime = MarketRegime(regime_name)
            except ValueError:
                if isinstance(regime_name, str):
                    try:
                        regime = MarketRegime(regime_name.lower())
                    except ValueError:
                        regime = MarketRegime.TREND
                else:
                    regime = MarketRegime.TREND
            normalized[regime] = {str(name): float(value) for name, value in weights.items()}
        return normalized

    def _configure_signal_thresholds(
        self,
        *,
        signal_thresholds: Mapping[str, float] | None,
        strategy_signal_thresholds: Mapping[str, Mapping[str, float]] | None,
    ) -> None:
        global_thresholds: Dict[str, float] = {}
        if signal_thresholds:
            for metric_name, value in signal_thresholds.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                global_thresholds[_canonical_metric_name(str(metric_name))] = numeric
        strategy_thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}
        if strategy_signal_thresholds:
            for exchange_key, strategy_map in strategy_signal_thresholds.items():
                exchange_norm = _canonical_identifier(str(exchange_key))
                if not exchange_norm:
                    continue
                strategy_payload: Dict[str, Dict[str, float]] = {}
                for strategy_key, metric_map in strategy_map.items():
                    strategy_norm = _canonical_identifier(str(strategy_key))
                    if not strategy_norm:
                        continue
                    metrics_payload: Dict[str, float] = {}
                    for metric_name, value in metric_map.items():
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(numeric):
                            continue
                        metrics_payload[_canonical_metric_name(str(metric_name))] = numeric
                    if metrics_payload:
                        strategy_payload[strategy_norm] = metrics_payload
                if strategy_payload:
                    strategy_thresholds[exchange_norm] = strategy_payload

        self._global_signal_thresholds = global_thresholds
        self._strategy_signal_thresholds = strategy_thresholds

        if global_thresholds:
            self.cfg.signal_thresholds = MappingProxyType(dict(global_thresholds))
        else:
            self.cfg.signal_thresholds = None

        if strategy_thresholds:
            nested_proxy = {
                exchange: MappingProxyType(
                    {
                        strategy: MappingProxyType(dict(metrics))
                        for strategy, metrics in strategy_map.items()
                    }
                )
                for exchange, strategy_map in strategy_thresholds.items()
            }
            self.cfg.strategy_signal_thresholds = MappingProxyType(nested_proxy)
        else:
            self.cfg.strategy_signal_thresholds = None

        self._signal_threshold_after_adjustment = self._resolve_signal_threshold(
            "signal_after_adjustment"
        )
        activation_override = self._resolve_signal_threshold("signal_after_clamp")
        if activation_override is not None:
            self.cfg.activation_threshold = float(abs(activation_override))
        self._activation_threshold = float(abs(self.cfg.activation_threshold))

    def _resolve_signal_threshold(self, metric: str) -> float | None:
        metric_key = _canonical_metric_name(metric)
        exchange_key = _canonical_identifier(self.cfg.primary_exchange)
        strategy_key = _canonical_identifier(self.cfg.strategy)
        if exchange_key and strategy_key:
            strategy_map = self._strategy_signal_thresholds.get(exchange_key)
            if strategy_map:
                metrics_map = strategy_map.get(strategy_key)
                if metrics_map:
                    value = metrics_map.get(metric_key)
                    if value is not None:
                        return float(value)
        value = self._global_signal_thresholds.get(metric_key)
        if value is not None:
            return float(value)
        return None

    def apply_signal_threshold_overrides(
        self,
        *,
        signal_thresholds: Mapping[str, float] | None = None,
        strategy_signal_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self._configure_signal_thresholds(
            signal_thresholds=signal_thresholds or self.cfg.signal_thresholds,
            strategy_signal_thresholds=(
                strategy_signal_thresholds or self.cfg.strategy_signal_thresholds
            ),
        )

    def signal_threshold_snapshot(self) -> Mapping[str, object]:
        strategy_snapshot: Dict[str, Dict[str, Dict[str, float]]] = {
            exchange: {strategy: dict(metrics) for strategy, metrics in strategies.items()}
            for exchange, strategies in self._strategy_signal_thresholds.items()
        }
        effective: Dict[str, float] = {"signal_after_clamp": self._activation_threshold}
        if self._signal_threshold_after_adjustment is not None:
            effective["signal_after_adjustment"] = self._signal_threshold_after_adjustment
        payload: dict[str, object] = {
            "global": dict(self._global_signal_thresholds),
            "per_strategy": strategy_snapshot,
            "effective": effective,
        }
        return MappingProxyType(payload)

    def _initialize_strategy_workflow(
        self, workflow: StrategyRegimeWorkflow | None
    ) -> StrategyRegimeWorkflow | None:
        if workflow is None:
            self._workflow_owned = True
            workflow = StrategyRegimeWorkflow(
                wizard=StrategyPresetWizard(),
                classifier=self._regime_classifier,
                history=self._regime_history,
                logger=self._logger,
            )
        if workflow is not None and self._workflow_owned:
            try:
                self._register_strategy_presets(workflow)
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                self._logger.debug(
                    "Nie udało się zarejestrować presetów workflow: %s", exc, exc_info=True
                )
        return workflow

    def _register_strategy_presets(self, workflow: StrategyRegimeWorkflow) -> None:
        signing_key = getattr(self, "_workflow_signing_key", b"autotrade")
        for regime, weights in self._strategy_weights.weights.items():
            entries, normalized = self._build_preset_entries(weights)
            if not entries:
                continue
            metadata = {
                "ensemble_weights": normalized,
                "parameter_overrides": dict(
                    self._workflow_parameter_overrides.get(regime, {})
                ),
            }
            try:
                workflow.register_preset(
                    regime,
                    name=f"autotrade-{regime.value}",
                    entries=entries,
                    signing_key=signing_key,
                    metadata=metadata,
                )
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                self._logger.debug(
                    "Rejestracja presetu dla reżimu %s nie powiodła się: %s",
                    regime,
                    exc,
                    exc_info=True,
                )
        fallback_entries, fallback_weights = self._build_preset_entries({"day_trading": 1.0})
        if fallback_entries:
            try:
                workflow.register_emergency_preset(
                    name="autotrade-emergency",
                    entries=fallback_entries,
                    signing_key=signing_key,
                    metadata={"ensemble_weights": fallback_weights},
                )
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                self._logger.debug(
                    "Rejestracja presetu awaryjnego nie powiodła się: %s", exc, exc_info=True
                )

    def _build_preset_entries(
        self, weights: Mapping[str, float]
    ) -> tuple[List[Mapping[str, object]], Dict[str, float]]:
        entries: List[Mapping[str, object]] = []
        missing: list[str] = []
        invalid: list[str] = []
        zeroed: list[str] = []
        prepared: list[tuple[str, float]] = []
        for name, weight in weights.items():
            key = str(name)
            try:
                value = float(weight)
            except (TypeError, ValueError):
                invalid.append(key)
                continue
            if not math.isfinite(value):
                invalid.append(key)
                continue
            if value <= 0.0:
                zeroed.append(key)
                continue
            prepared.append((key, value))

        if not prepared:
            if invalid:
                self._logger.warning(
                    "Pominięto strategie z nieprawidłowymi wagami: %s",
                    ", ".join(sorted(set(invalid))),
                )
            if zeroed:
                self._logger.info(
                    "Pominięto strategie z zerowymi wagami: %s",
                    ", ".join(sorted(set(zeroed))),
                )
            return entries, {}

        total = sum(weight for _, weight in prepared)
        if total <= 0.0:
            self._logger.warning(
                "Nie można zbudować presetu: suma wag (%s) jest niepoprawna",
                total,
            )
            return entries, {}

        normalized: Dict[str, float] = {}
        for key, value in prepared:
            normalized[key] = value / total

        sorted_weights = sorted(
            normalized.items(),
            key=lambda item: (-item[1], item[0]),
        )

        for name, weight in sorted_weights:
            engine = self._PRESET_ENGINE_MAPPING.get(name)
            if engine is None:
                missing.append(name)
                continue
            entry: Dict[str, object] = {
                "engine": engine,
                "name": name,
                "parameters": {},
                "tags": ["auto_trade", name],
                "metadata": {
                    "ensemble_weight": weight,
                    "strategy": name,
                },
            }
            entries.append(entry)
        if invalid:
            self._logger.warning(
                "Pominięto strategie z nieprawidłowymi wagami: %s",
                ", ".join(sorted(set(invalid))),
            )
        if zeroed:
            self._logger.info(
                "Pominięto strategie z zerowymi wagami: %s",
                ", ".join(sorted(set(zeroed))),
            )
        if missing:
            self._logger.warning(
                "Pominięto strategie bez zmapowanego silnika: %s", ", ".join(sorted(set(missing)))
            )
        return entries, normalized

    def _refresh_workflow_presets(self) -> None:
        workflow = getattr(self, "_regime_workflow", None)
        if workflow is None or not self._workflow_owned:
            return
        try:
            self._register_strategy_presets(workflow)
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            self._logger.debug(
                "Aktualizacja presetów workflow nie powiodła się: %s", exc, exc_info=True
            )

    def _infer_available_data(self, frame: pd.DataFrame) -> set[str]:
        available: set[str] = {"ohlcv"}
        if not frame.empty:
            available.add("technical_indicators")
            available.add("spread_history")
            available.add("order_book")
            available.add("latency_monitoring")
        return available

    def _activation_weights(
        self, activation: RegimePresetActivation
    ) -> Dict[str, float]:
        metadata = activation.version.metadata
        preset_meta = metadata.get("preset_metadata") if isinstance(metadata, Mapping) else None
        weights: Dict[str, float] = {}
        candidates: Mapping[str, float] | None = None
        if isinstance(preset_meta, Mapping):
            raw = preset_meta.get("ensemble_weights")
            if isinstance(raw, Mapping):
                candidates = raw  # type: ignore[assignment]
        if candidates is None and isinstance(activation.preset, Mapping):
            preset_payload = activation.preset.get("metadata")
            if isinstance(preset_payload, Mapping):
                raw = preset_payload.get("ensemble_weights")
                if isinstance(raw, Mapping):
                    candidates = raw  # type: ignore[assignment]
        if candidates is not None:
            for name, value in candidates.items():
                try:
                    weights[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
        dynamic_candidate: Mapping[str, object] | None = None
        metrics_payload: Dict[str, float] | None = None
        assessment_obj = activation.assessment
        if isinstance(assessment_obj, MarketRegimeAssessment):
            metrics_payload = {}
            raw_metrics = getattr(assessment_obj, "metrics", {})
            if isinstance(raw_metrics, Mapping):
                for key, value in raw_metrics.items():
                    try:
                        metrics_payload[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
            try:
                metrics_payload["confidence"] = float(assessment_obj.confidence)
            except (TypeError, ValueError):
                pass
            try:
                metrics_payload["risk_score"] = float(assessment_obj.risk_score)
            except (TypeError, ValueError):
                pass
            if not metrics_payload:
                metrics_payload = None
        catalog = getattr(self, "_strategy_catalog", None)
        if catalog is not None and hasattr(catalog, "dynamic_preset_for"):
            try:
                dynamic_candidate = catalog.dynamic_preset_for(
                    activation.regime.value,
                    metrics=metrics_payload,
                )
            except Exception:  # pragma: no cover - defensywna degradacja
                dynamic_candidate = None
        if dynamic_candidate:
            dynamic_weights: Dict[str, float] = {}
            strategies_payload = dynamic_candidate.get("strategies")
            if isinstance(strategies_payload, TypingIterable):
                for entry in strategies_payload:
                    if not isinstance(entry, Mapping):
                        continue
                    name = str(entry.get("name") or "").strip()
                    if not name:
                        continue
                    try:
                        weight_value = float(entry.get("weight", 1.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    dynamic_weights[name] = weight_value
            if dynamic_weights:
                weights = dynamic_weights
                preset_metadata = dynamic_candidate.get("metadata")
                preset_metrics = dynamic_candidate.get("metrics")
                stored_preset: Dict[str, object] = {
                    "name": dynamic_candidate.get("name"),
                    "regime": dynamic_candidate.get("regime"),
                    "strategies": [
                        {"name": key, "weight": value}
                        for key, value in dynamic_weights.items()
                    ],
                }
                generated_at = dynamic_candidate.get("generated_at")
                if isinstance(generated_at, str) and generated_at.strip():
                    stored_preset["generated_at"] = generated_at
                if isinstance(preset_metadata, Mapping):
                    stored_preset["metadata"] = {
                        str(key): value for key, value in preset_metadata.items()
                    }
                if isinstance(preset_metrics, Mapping):
                    stored_preset["metrics"] = {
                        str(key): value for key, value in preset_metrics.items()
                    }
                self._last_dynamic_preset = stored_preset
            else:
                self._last_dynamic_preset = None
        else:
            self._last_dynamic_preset = None
        if not weights:
            fallback = self._strategy_weights.weights_for(activation.regime)
            weights = {str(name): float(value) for name, value in fallback.items()}
        total = sum(abs(value) for value in weights.values())
        if total > 0:
            weights = {name: float(value) / total for name, value in weights.items()}
        return weights

    def _build_activation_payload(
        self, activation: RegimePresetActivation
    ) -> Mapping[str, object]:
        version_meta: Dict[str, object] = {}
        for key, value in activation.version.metadata.items():
            if isinstance(value, Mapping):
                version_meta[key] = {str(k): v for k, v in value.items()}
            elif isinstance(value, tuple):
                version_meta[key] = [str(item) for item in value]
            else:
                version_meta[key] = value
        payload: Dict[str, object] = {
            "regime": activation.regime.value,
            "activated_at": activation.activated_at.isoformat(),
            "preset_regime": activation.preset_regime.value
            if isinstance(activation.preset_regime, MarketRegime)
            else None,
            "used_fallback": bool(activation.used_fallback),
            "blocked_reason": activation.blocked_reason,
            "missing_data": list(activation.missing_data),
            "license_issues": list(activation.license_issues),
            "recommendation": activation.recommendation,
            "decision_candidates": len(activation.decision_candidates),
            "version": {
                "hash": activation.version.hash,
                "issued_at": activation.version.issued_at.isoformat(),
                "signature": {str(k): str(v) for k, v in activation.version.signature.items()},
                "metadata": version_meta,
            },
        }
        if isinstance(activation.preset, Mapping):
            preset_name = activation.preset.get("name")
            if preset_name:
                payload["preset_name"] = str(preset_name)
            preset_meta = activation.preset.get("metadata")
            if isinstance(preset_meta, Mapping):
                payload["preset_metadata"] = {
                    str(k): v for k, v in preset_meta.items()
                }
        return MappingProxyType(payload)

    def _build_activation_metadata(
        self,
        activation: RegimePresetActivation,
        weights: Mapping[str, float],
    ) -> tuple[Mapping[str, Mapping[str, object]], Mapping[str, tuple[str, ...]]]:
        catalog_metadata = self._collect_strategy_metadata(weights)
        per_strategy: Dict[str, Dict[str, object]] = {
            name: dict(payload) for name, payload in catalog_metadata["strategies"].items()
        }
        for entry in activation.preset.get("strategies", []) if isinstance(activation.preset, Mapping) else []:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or entry.get("engine") or "").strip()
            if not name:
                continue
            payload = per_strategy.setdefault(name, {})
            payload.setdefault("engine", entry.get("engine"))
            if entry.get("license_tier"):
                payload["license_tier"] = entry.get("license_tier")
            if entry.get("risk_classes"):
                payload["risk_classes"] = tuple(entry.get("risk_classes", ()))
            if entry.get("required_data"):
                payload["required_data"] = tuple(entry.get("required_data", ()))
            if entry.get("capability"):
                payload["capability"] = entry.get("capability")
            if entry.get("tags"):
                payload["tags"] = tuple(entry.get("tags", ()))
            metadata_payload = entry.get("metadata")
            if isinstance(metadata_payload, Mapping):
                combined = dict(payload.get("preset_metadata", {}))
                combined.update({str(k): v for k, v in metadata_payload.items()})
                payload["preset_metadata"] = combined
            payload["preset_weight"] = weights.get(name)
        summary = dict(catalog_metadata["summary"])
        version_meta = activation.version.metadata
        for key in ("license_tiers", "risk_classes", "required_data", "capabilities", "tags"):
            value = version_meta.get(key)
            if isinstance(value, (tuple, list)) and value:
                summary[key] = tuple(str(item) for item in value)
        return (
            {
                name: MappingProxyType(dict(payload))
                for name, payload in per_strategy.items()
            },
            MappingProxyType({key: tuple(values) for key, values in summary.items()}),
        )

    def _sync_workflow_state(self) -> None:
        """Synchronise shared components with an injected workflow."""

        workflow = getattr(self, "_regime_workflow", None)
        if workflow is None:
            return
        classifier = getattr(workflow, "classifier", None)
        if not isinstance(classifier, MarketRegimeClassifier):
            classifier = getattr(workflow, "_classifier", None)
        if isinstance(classifier, MarketRegimeClassifier):
            self._regime_classifier = classifier
        history = getattr(workflow, "history", None)
        if not isinstance(history, RegimeHistory):
            history = getattr(workflow, "_history", None)
        if isinstance(history, RegimeHistory):
            self._regime_history = history
        catalog = getattr(workflow, "catalog", None)
        if isinstance(catalog, StrategyCatalog) and catalog is not self._strategy_catalog:
            self._update_strategy_catalog(catalog)
        last_activation = getattr(workflow, "last_activation", None)
        if isinstance(last_activation, RegimePresetActivation):
            self._last_regime_activation = last_activation
        else:
            self._last_regime_activation = None
        self._last_regime_decision = getattr(workflow, "last_decision", None)

    def _build_base_trading_parameters(self) -> TradingParameters:
        base = self._base_trading_params
        fast = int(self._params.get("fast", base.ema_fast_period))
        slow = int(self._params.get("slow", base.ema_slow_period))
        if slow <= fast:
            slow = fast + 1
        overrides = {
            "ema_fast_period": fast,
            "ema_slow_period": slow,
            "day_trading_momentum_window": int(max(1, self.cfg.breakout_window)),
            "day_trading_volatility_window": int(
                max(1, math.ceil(self.cfg.breakout_window * 1.5))
            ),
            "arbitrage_confirmation_window": int(self.cfg.arbitrage_confirmation_window),
            "arbitrage_spread_threshold": float(self.cfg.arbitrage_threshold),
        }
        return replace(base, **overrides)

    def _compose_trading_parameters(self, weights: Mapping[str, float]) -> TradingParameters:
        base = self._build_base_trading_parameters()
        total = sum(float(v) for v in weights.values()) or 1.0
        normalized = {
            str(name): float(value) / total for name, value in weights.items()
        }
        return replace(base, ensemble_weights=normalized)

    def _collect_strategy_metadata(
        self, weights: Mapping[str, float]
    ) -> Mapping[str, Mapping[str, object] | Mapping[str, tuple[str, ...]]]:
        strategies: Dict[str, Mapping[str, object]] = {}
        license_tiers: list[str] = []
        risk_classes: list[str] = []
        required_data: list[str] = []
        capabilities: list[str] = []
        tags: list[str] = []

        cache = getattr(self, "_strategy_metadata_cache", None)
        if cache is None:
            cache = {}
            self._strategy_metadata_cache = cache

        sentinel = object()

        def _store_metadata_entry(
            metadata: Mapping[str, object],
            resolved: str | None,
            *aliases: str,
        ) -> Mapping[str, object]:
            payload = MappingProxyType(dict(metadata))
            entry = (payload, resolved)
            for alias in aliases:
                if not alias:
                    continue
                for variant in strategy_key_aliases(alias):
                    cache[variant] = entry
            return payload

        def _store_missing_entry(*aliases: str) -> None:
            for alias in aliases:
                if not alias:
                    continue
                for variant in strategy_key_aliases(alias):
                    cache[variant] = None

        def _append_unique(bucket: list[str], values: Iterable[str]) -> None:
            seen = set(bucket)
            for value in values:
                text = str(value).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                bucket.append(text)

        catalog = getattr(self, "_strategy_catalog", None)
        if catalog is None:
            return {
                "strategies": MappingProxyType({}),
                "summary": MappingProxyType(
                    {
                        "license_tiers": tuple(),
                        "risk_classes": tuple(),
                        "required_data": tuple(),
                        "capabilities": tuple(),
                        "tags": tuple(),
                    }
                ),
            }

        resolver = type(self)._alias_resolver()

        for name in sorted(weights):
            lookup_sequence = strategy_name_candidates(
                name,
                resolver.alias_map,
                resolver.suffixes,
                normalised=True,
            ) or (name,)
            metadata_proxy: Mapping[str, object] | None = None
            resolved_name: str | None = None
            for candidate in lookup_sequence:
                cached = cache.get(candidate, sentinel)
                if cached is not sentinel:
                    if cached is None:
                        continue
                    metadata_proxy, cached_resolved = cached
                    resolved_name = cached_resolved or candidate
                    metadata_proxy = _store_metadata_entry(
                        metadata_proxy,
                        resolved_name,
                        candidate,
                        name,
                    )
                    break
                try:
                    metadata = catalog.metadata_for(candidate)
                except Exception as exc:  # pragma: no cover - defensywne logowanie
                    self._logger.debug(
                        "Nie udało się pobrać metadanych strategii %s: %s",
                        candidate,
                        exc,
                        exc_info=True,
                    )
                    _store_missing_entry(candidate)
                    continue
                if metadata:
                    resolved_name = candidate
                    metadata_proxy = _store_metadata_entry(
                        metadata,
                        resolved_name,
                        candidate,
                        name,
                    )
                    break
                _store_missing_entry(candidate)
            if not metadata_proxy:
                continue
            payload = dict(metadata_proxy)
            payload.setdefault("name", name)
            if resolved_name and resolved_name != name:
                payload.setdefault("catalog_name", resolved_name)
                aliases = [
                    alias
                    for alias in lookup_sequence
                    if alias not in {resolved_name, name}
                ]
                if aliases:
                    payload.setdefault("aliases", tuple(aliases))
            metadata_payload = MappingProxyType(payload)
            strategies[name] = metadata_payload
            license_value = metadata_payload.get("license_tier")
            if isinstance(license_value, str):
                _append_unique(license_tiers, (license_value,))
            risk_value = metadata_payload.get("risk_classes")
            if isinstance(risk_value, Iterable):
                _append_unique(risk_classes, risk_value)
            required_value = metadata_payload.get("required_data")
            if isinstance(required_value, Iterable):
                _append_unique(required_data, required_value)
            capability_value = metadata_payload.get("capability")
            if isinstance(capability_value, str):
                _append_unique(capabilities, (capability_value,))
            tags_value = metadata_payload.get("tags")
            if isinstance(tags_value, Iterable):
                _append_unique(tags, tags_value)

        return {
            "strategies": MappingProxyType(
                {name: MappingProxyType(dict(payload)) for name, payload in strategies.items()}
            ),
            "summary": MappingProxyType(
                {
                    "license_tiers": tuple(license_tiers),
                    "risk_classes": tuple(risk_classes),
                    "required_data": tuple(required_data),
                    "capabilities": tuple(capabilities),
                    "tags": tuple(tags),
                }
            ),
        }

    def _normalize_preset_availability(self, report: Any) -> PresetAvailability:
        if isinstance(report, PresetAvailability):
            return report
        regime = getattr(report, "regime", None)
        version = getattr(report, "version", None)
        if version is None:
            raise ValueError("Preset availability report missing version metadata")
        ready = bool(getattr(report, "ready", False))
        blocked_reason = getattr(report, "blocked_reason", None)
        missing_data = tuple(getattr(report, "missing_data", ()))
        license_issues = tuple(getattr(report, "license_issues", ()))
        schedule_blocked = bool(getattr(report, "schedule_blocked", False))
        return PresetAvailability(
            regime=regime,
            version=version,
            ready=ready,
            blocked_reason=blocked_reason,
            missing_data=missing_data,
            license_issues=license_issues,
            schedule_blocked=schedule_blocked,
        )

    def _serialize_preset_availability(
        self, report: PresetAvailability
    ) -> dict[str, Any]:
        metadata = dict(getattr(report.version, "metadata", {}))
        regime_key = (
            report.regime.value if isinstance(report.regime, MarketRegime) else report.regime
        )
        license_tiers = list(metadata.get("license_tiers", ()))
        risk_classes = list(metadata.get("risk_classes", ()))
        required_data = list(metadata.get("required_data", ()))
        capabilities = list(metadata.get("capabilities", ()))
        tags = list(metadata.get("tags", ()))
        preset_metadata = metadata.get("preset_metadata")
        if isinstance(preset_metadata, tuple):
            preset_metadata = dict(preset_metadata)
        elif not isinstance(preset_metadata, Mapping):
            preset_metadata = {}
        return {
            "regime": regime_key,
            "ready": bool(report.ready),
            "blocked_reason": report.blocked_reason,
            "missing_data": list(report.missing_data),
            "license_issues": list(report.license_issues),
            "schedule_blocked": bool(report.schedule_blocked),
            "preset_hash": report.version.hash,
            "preset_signature": dict(report.version.signature),
            "preset_name": metadata.get("name"),
            "license_tiers": license_tiers,
            "risk_classes": risk_classes,
            "required_data": required_data,
            "capabilities": capabilities,
            "tags": tags,
            "preset_metadata": preset_metadata,
        }

    def _handle_regime_status(
        self,
        assessment: MarketRegimeAssessment,
        summary: RegimeSummary | None,
        metadata_summary: Mapping[str, tuple[str, ...]] | None = None,
        activation_payload: Mapping[str, object] | None = None,
    ) -> None:
        previous_assessment = self._last_regime
        previous_summary = self._last_summary
        should_emit = previous_assessment is None or (
            previous_assessment.regime != assessment.regime
        )
        if not should_emit and summary is not None and previous_summary is not None:
            if summary.risk_level != previous_summary.risk_level:
                should_emit = True
            else:
                risk_delta = abs(summary.risk_score - previous_summary.risk_score)
                if risk_delta >= 0.1:
                    should_emit = True
        if should_emit:
            detail = assessment.to_dict()
            if summary is not None:
                detail["summary"] = summary.to_dict()
                detail["thresholds"] = self._regime_history.thresholds_snapshot()
            if metadata_summary:
                detail["metadata"] = {
                    key: list(values)
                    for key, values in metadata_summary.items()
                    if values
                }
            if activation_payload:
                detail["activation"] = {
                    key: value
                    for key, value in activation_payload.items()
                }
            detail = self._augment_status_detail(detail)
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "regime_update",
                detail=detail,
            )
        self._last_regime = assessment
        if summary is not None:
            self._last_summary = summary

    def inspect_regime_presets(
        self,
        available_data: Iterable[str] = (),
        *,
        now: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        workflow = getattr(self, "_regime_workflow", None)
        reports: tuple[Any, ...] = ()
        if workflow is not None:
            inspector = getattr(workflow, "inspect_presets", None)
            if callable(inspector):
                try:
                    reports = inspector(available_data, now=now)
                except TypeError:
                    reports = inspector(available_data)
            else:
                getter = getattr(workflow, "get_availability", None)
                if callable(getter):
                    reports = getter()
                else:
                    reports = getattr(workflow, "_preset_availability", ())

        normalized = tuple(
            self._normalize_preset_availability(report) for report in reports
        )
        self._preset_availability = normalized
        return [self._serialize_preset_availability(report) for report in normalized]

    def summarize_regime_presets(
        self,
        available_data: Iterable[str] = (),
        *,
        now: dt.datetime | None = None,
    ) -> dict[str, Any]:
        reports = self.inspect_regime_presets(available_data, now=now)
        availability = self._preset_availability
        total = len(availability)
        summary: dict[str, Any] = {
            "total_presets": total,
            "ready_presets": 0,
            "blocked_presets": 0,
            "schedule_blocked_presets": 0,
            "missing_data_counts": {},
            "license_issue_counts": {},
            "blocked_reason_counts": {},
            "regimes": {},
            "reports": reports,
        }
        if not availability:
            summary["missing_data_counts"] = {}
            summary["license_issue_counts"] = {}
            summary["blocked_reason_counts"] = {}
            return summary

        missing_counter: Counter[str] = Counter()
        license_counter: Counter[str] = Counter()
        blocked_counter: Counter[str] = Counter()
        regime_stats: dict[str, dict[str, Any]] = {}

        for report in availability:
            if report.ready:
                summary["ready_presets"] += 1
            else:
                summary["blocked_presets"] += 1
            if report.schedule_blocked:
                summary["schedule_blocked_presets"] += 1

            for item in report.missing_data:
                missing_counter[str(item)] += 1
            for issue in report.license_issues:
                license_counter[str(issue)] += 1
            if report.blocked_reason:
                blocked_counter[str(report.blocked_reason)] += 1

            regime_key = (
                report.regime.value if isinstance(report.regime, MarketRegime) else report.regime
            )
            stats = regime_stats.setdefault(
                str(regime_key),
                {
                    "total_presets": 0,
                    "ready_presets": 0,
                    "blocked_presets": 0,
                    "schedule_blocked_presets": 0,
                    "missing_data": set(),
                    "license_issue_counts": Counter[str](),
                    "blocked_reason_counts": Counter[str](),
                },
            )
            stats["total_presets"] += 1
            if report.ready:
                stats["ready_presets"] += 1
            else:
                stats["blocked_presets"] += 1
            if report.schedule_blocked:
                stats["schedule_blocked_presets"] += 1
            stats["missing_data"].update(str(item) for item in report.missing_data)
            stats["license_issue_counts"].update(str(issue) for issue in report.license_issues)
            if report.blocked_reason:
                stats["blocked_reason_counts"].update([str(report.blocked_reason)])

        summary["missing_data_counts"] = dict(missing_counter)
        summary["license_issue_counts"] = dict(license_counter)
        summary["blocked_reason_counts"] = dict(blocked_counter)
        summary["regimes"] = {
            key: {
                "total_presets": value["total_presets"],
                "ready_presets": value["ready_presets"],
                "blocked_presets": value["blocked_presets"],
                "schedule_blocked_presets": value["schedule_blocked_presets"],
                "missing_data": sorted(value["missing_data"]),
                "license_issue_counts": dict(value["license_issue_counts"]),
                "blocked_reason_counts": dict(value["blocked_reason_counts"]),
            }
            for key, value in regime_stats.items()
        }
        return summary

    def _gather_regime_activation_history(self) -> list[RegimePresetActivation]:
        entries: list[RegimePresetActivation] = []
        workflow = getattr(self, "_regime_workflow", None)
        if workflow is not None:
            history_method = getattr(workflow, "activation_history", None)
            if callable(history_method):
                try:
                    entries.extend(history_method())
                except Exception:
                    pass
            manual_entries = getattr(workflow, "_history_entries", None)
            if manual_entries:
                entries.extend(list(manual_entries))
            last_activation = getattr(workflow, "last_activation", None)
            if last_activation is not None:
                entries.append(last_activation)
        entries.extend(self._regime_activation_history)
        deduped: list[RegimePresetActivation] = []
        seen: set[tuple[Any, Any, Any]] = set()
        for entry in entries:
            if entry is None:
                continue
            activated_at = getattr(entry, "activated_at", None)
            version_hash = getattr(entry.version, "hash", None)
            regime = getattr(entry, "regime", None)
            key = (activated_at, version_hash, regime)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        deduped.sort(key=lambda activation: activation.activated_at)
        return deduped

    @staticmethod
    def _normalize_history_export_limit(limit: Any) -> int | None:
        if limit is None:
            return None
        if isinstance(limit, bool):
            raise TypeError("limit must be an integer or None")
        try:
            normalized = int(limit)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive conversion
            raise TypeError("limit must be an integer or None") from exc
        if normalized < 0:
            return None
        return normalized

    @staticmethod
    def _serialize_payload(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return value.to_dict()
            except Exception:  # pragma: no cover - fall through to generic handling
                pass
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        return value

    def _serialize_regime_activation(
        self,
        activation: RegimePresetActivation,
        *,
        include_metadata: bool,
        include_decision: bool,
        include_summary: bool,
        include_preset: bool,
        coerce_timestamps: bool,
        tz: dt.tzinfo | None,
    ) -> dict[str, Any]:
        regime_key = (
            activation.regime.value
            if isinstance(activation.regime, MarketRegime)
            else activation.regime
        )
        preset_regime_key = (
            activation.preset_regime.value
            if isinstance(activation.preset_regime, MarketRegime)
            else activation.preset_regime
        )
        metadata = dict(getattr(activation.version, "metadata", {}))
        record: dict[str, Any] = {
            "regime": regime_key,
            "preset_regime": preset_regime_key,
            "preset_hash": activation.version.hash,
            "preset_signature": dict(activation.version.signature),
            "preset_name": metadata.get("name"),
            "used_fallback": bool(activation.used_fallback),
            "missing_data": list(activation.missing_data),
            "blocked_reason": activation.blocked_reason,
            "license_issues": list(activation.license_issues),
            "recommendation": activation.recommendation,
        }
        activated_at = activation.activated_at
        if coerce_timestamps:
            target_tz = tz or timezone.utc
            if activated_at.tzinfo is None:
                activated_at = activated_at.replace(tzinfo=target_tz)
            else:
                activated_at = activated_at.astimezone(target_tz)
            record["activated_at"] = activated_at
        else:
            record["activated_at"] = activated_at.isoformat()

        if include_metadata:
            record["license_tiers"] = list(metadata.get("license_tiers", ()))
            record["risk_classes"] = list(metadata.get("risk_classes", ()))
            record["required_data"] = list(metadata.get("required_data", ()))
            record["capabilities"] = list(metadata.get("capabilities", ()))
            record["tags"] = list(metadata.get("tags", ()))
        if include_summary and activation.summary is not None:
            record["summary"] = self._serialize_payload(activation.summary)
        if include_decision:
            record["assessment"] = self._serialize_payload(activation.assessment)
            record["decision_candidates"] = [
                self._serialize_payload(candidate)
                for candidate in activation.decision_candidates
            ]
        if include_preset:
            record["preset"] = self._serialize_payload(activation.preset)
        return record

    def regime_activation_history_records(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        include_metadata: bool = True,
        include_decision: bool = True,
        include_summary: bool = True,
        include_preset: bool = True,
        coerce_timestamps: bool = False,
        tz: dt.tzinfo | None = dt.timezone.utc,
    ) -> list[dict[str, Any]]:
        history = self._gather_regime_activation_history()
        normalized_limit = self._normalize_history_export_limit(limit)
        if normalized_limit is not None and normalized_limit >= 0:
            if normalized_limit == 0:
                history = []
            elif len(history) > normalized_limit:
                history = history[-normalized_limit:]
        if reverse:
            history = list(reversed(history))
        return [
            self._serialize_regime_activation(
                activation,
                include_metadata=include_metadata,
                include_decision=include_decision,
                include_summary=include_summary,
                include_preset=include_preset,
                coerce_timestamps=coerce_timestamps,
                tz=tz,
            )
            for activation in history
        ]

    def regime_activation_history_frame(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        include_metadata: bool = True,
        include_decision: bool = True,
        include_summary: bool = True,
        include_preset: bool = False,
        coerce_timestamps: bool = True,
        tz: dt.tzinfo | None = dt.timezone.utc,
    ) -> pd.DataFrame:
        workflow = getattr(self, "_regime_workflow", None)
        if workflow is not None:
            frame_method = getattr(workflow, "activation_history_frame", None)
            if callable(frame_method):
                try:
                    frame = frame_method(limit=limit)
                except TypeError:
                    frame = frame_method()
                except Exception:
                    frame = None
                else:
                    if isinstance(frame, pd.DataFrame):
                        return frame.copy()
        records = self.regime_activation_history_records(
            limit=limit,
            reverse=reverse,
            include_metadata=include_metadata,
            include_decision=include_decision,
            include_summary=include_summary,
            include_preset=include_preset,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )
        if not records:
            return pd.DataFrame()
        return pd.DataFrame.from_records(records)

    def summarize_regime_activation_history(self) -> dict[str, Any]:
        history = self._gather_regime_activation_history()
        if not history:
            return {
                "total_activations": 0,
                "fallback_activations": 0,
                "license_issue_activations": 0,
                "missing_data_counts": {},
                "license_issue_counts": {},
                "blocked_reason_counts": {},
                "first_activation_at": None,
                "last_activation": None,
                "regimes": {},
            }

        missing_counter: Counter[str] = Counter()
        license_counter: Counter[str] = Counter()
        blocked_counter: Counter[str] = Counter()
        regime_summary: dict[str, dict[str, Any]] = {}

        for activation in history:
            for item in activation.missing_data:
                missing_counter[str(item)] += 1
            for issue in activation.license_issues:
                license_counter[str(issue)] += 1
            if activation.blocked_reason:
                blocked_counter[str(activation.blocked_reason)] += 1

            regime_key = (
                activation.regime.value
                if isinstance(activation.regime, MarketRegime)
                else activation.regime
            )
            stats = regime_summary.setdefault(
                str(regime_key),
                {
                    "activations": 0,
                    "fallback_activations": 0,
                    "license_issue_activations": 0,
                    "missing_data": Counter[str](),
                    "license_issue_counts": Counter[str](),
                    "blocked_reason_counts": Counter[str](),
                    "last_activation_at": None,
                },
            )
            stats["activations"] += 1
            if activation.used_fallback:
                stats["fallback_activations"] += 1
            if activation.license_issues:
                stats["license_issue_activations"] += 1
            stats["missing_data"].update(str(item) for item in activation.missing_data)
            stats["license_issue_counts"].update(str(issue) for issue in activation.license_issues)
            if activation.blocked_reason:
                stats["blocked_reason_counts"].update([str(activation.blocked_reason)])
            stats["last_activation_at"] = activation.activated_at.isoformat()

        summary: dict[str, Any] = {
            "total_activations": len(history),
            "fallback_activations": sum(1 for entry in history if entry.used_fallback),
            "license_issue_activations": sum(
                1 for entry in history if entry.license_issues
            ),
            "missing_data_counts": dict(missing_counter),
            "license_issue_counts": dict(license_counter),
            "blocked_reason_counts": dict(blocked_counter),
            "first_activation_at": history[0].activated_at.isoformat(),
            "last_activation": self._serialize_regime_activation(
                history[-1],
                include_metadata=True,
                include_decision=True,
                include_summary=True,
                include_preset=False,
                coerce_timestamps=False,
                tz=None,
            ),
            "regimes": {
                key: {
                    "activations": value["activations"],
                    "fallback_activations": value["fallback_activations"],
                    "license_issue_activations": value["license_issue_activations"],
                    "missing_data": dict(value["missing_data"]),
                    "license_issue_counts": dict(value["license_issue_counts"]),
                    "blocked_reason_counts": dict(value["blocked_reason_counts"]),
                    "last_activation_at": value["last_activation_at"],
                }
                for key, value in regime_summary.items()
            },
        }
        return summary

    def _evaluate_regime_decision(
        self,
        indicator_frame: pd.DataFrame,
        base_parameters: TradingParameters,
    ) -> tuple[
        MarketRegimeAssessment,
        RegimeSummary | None,
        Dict[str, float],
        TradingParameters,
        Mapping[str, Mapping[str, object]],
        Mapping[str, tuple[str, ...]],
        Mapping[str, object] | None,
    ]:
        workflow = getattr(self, "_regime_workflow", None)
        if workflow is not None:
            activate = getattr(workflow, "activate", None)
            if callable(activate):
                try:
                    activation = activate(
                        indicator_frame,
                        available_data=self._infer_available_data(indicator_frame),
                        symbol=self.cfg.symbol,
                    )
                except Exception as exc:  # pragma: no cover - defensywne logowanie
                    self._logger.debug("Błąd workflow reżimu: %s", exc, exc_info=True)
                else:
                    self._last_regime_activation = activation
                    self._regime_activation_history.append(activation)
                    weights = self._activation_weights(activation)
                    normalized_weights = {
                        str(name): float(value) for name, value in weights.items()
                    }
                    parameters = self._compose_trading_parameters(normalized_weights)
                    strategy_metadata, metadata_summary = self._build_activation_metadata(
                        activation, normalized_weights
                    )
                    activation_payload = self._build_activation_payload(activation)
                    timestamp = pd.Timestamp(activation.activated_at)
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.tz_convert(None)
                    decision = RegimeSwitchDecision(
                        regime=activation.regime,
                        assessment=activation.assessment,
                        summary=activation.summary,
                        weights=normalized_weights,
                        parameters=parameters,
                        timestamp=timestamp,
                        strategy_metadata=strategy_metadata,
                        license_tiers=tuple(metadata_summary.get("license_tiers", ())),
                        risk_classes=tuple(metadata_summary.get("risk_classes", ())),
                        required_data=tuple(metadata_summary.get("required_data", ())),
                        capabilities=tuple(metadata_summary.get("capabilities", ())),
                        tags=tuple(metadata_summary.get("tags", ())),
                    )
                    self._last_regime_decision = decision
                    self._handle_regime_status(
                        decision.assessment,
                        decision.summary,
                        metadata_summary,
                        activation_payload,
                    )
                    return (
                        decision.assessment,
                        decision.summary,
                        normalized_weights,
                        parameters,
                        strategy_metadata,
                        metadata_summary,
                        activation_payload,
                    )
        assessment = self._classify_regime(indicator_frame)
        summary = self._regime_history.summarise()
        weights = {
            str(name): float(value)
            for name, value in self._strategy_weights.weights_for(assessment.regime).items()
        }
        parameters = self._compose_trading_parameters(weights)
        normalized = {
            str(name): float(value) for name, value in parameters.ensemble_weights.items()
        }
        metadata = self._collect_strategy_metadata(normalized)
        return (
            assessment,
            summary,
            normalized,
            parameters,
            metadata["strategies"],
            metadata["summary"],
            None,
        )

    def _prepare_indicator_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.copy()
        if "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"], unit="s", errors="coerce")
            if ts.notna().all():
                data = data.drop(columns=["timestamp"]).set_index(ts)
            else:
                data = data.drop(columns=["timestamp"], errors="ignore")
                data.index = pd.RangeIndex(len(data))
        elif "open_time" in data.columns:
            ts = pd.to_datetime(data["open_time"], unit="s", errors="coerce")
            if ts.notna().all():
                data = data.drop(columns=["open_time"]).set_index(ts)
            else:
                data.index = pd.RangeIndex(len(data))
        else:
            data.index = pd.RangeIndex(len(data))
        data = data.sort_index()
        for column in ("open", "high", "low", "close", "volume"):
            if column not in data.columns:
                if column == "open":
                    data[column] = data.get("close", 0.0)
                else:
                    data[column] = 0.0
            series = pd.to_numeric(data[column], errors="coerce")
            if column == "open":
                series = series.ffill().bfill()
                if "close" in data.columns:
                    series = series.fillna(pd.to_numeric(data["close"], errors="coerce"))
            else:
                series = series.fillna(0.0)
            data[column] = series.astype(float)
        data = data.loc[~data.index.duplicated(keep="last")]
        return data[["open", "high", "low", "close", "volume"]]

    def _generate_plugin_signals(
        self,
        indicators: TechnicalIndicators,
        params: TradingParameters,
        data: pd.DataFrame,
        weights: Mapping[str, float],
    ) -> Dict[str, float]:
        candidate_names = set(self._strategy_catalog.available()) | set(weights)
        signals: Dict[str, float] = {}
        for name in sorted(candidate_names):
            plugin = self._strategy_catalog.create(name)
            if plugin is None:
                continue
            try:
                series = plugin.generate(indicators, params, market_data=data)
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                self._logger.debug(
                    "Strategia plugin '%s' zgłosiła wyjątek: %s", name, exc, exc_info=True
                )
                continue
            if series.empty:
                continue
            value = float(series.iloc[-1])
            if not np.isfinite(value):
                continue
            signals[name] = float(np.clip(value, -1.0, 1.0))
        return signals

    @staticmethod
    def _stringify_metadata(metadata: Mapping[str, Any]) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            key_str = str(key)
            if isinstance(value, bool):
                payload[key_str] = "true" if value else "false"
            elif isinstance(value, (int, np.integer)):
                payload[key_str] = str(int(value))
            elif isinstance(value, (float, np.floating)):
                formatted = f"{float(value):.10f}".rstrip("0").rstrip(".")
                payload[key_str] = formatted if formatted else "0"
            elif isinstance(value, (dt.datetime, pd.Timestamp)):
                timestamp = value
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
                payload[key_str] = timestamp.astimezone(dt.timezone.utc).isoformat()
            elif isinstance(value, Mapping):
                try:
                    payload[key_str] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                except TypeError:
                    payload[key_str] = str(value)
            elif isinstance(value, (list, tuple, set)):
                try:
                    payload[key_str] = json.dumps(list(value), ensure_ascii=False)
                except TypeError:
                    payload[key_str] = ",".join(str(item) for item in value)
            else:
                payload[key_str] = str(value)
        return payload

    def _build_inference_features(
        self,
        frame: pd.DataFrame | None,
        indicators: TechnicalIndicators | None = None,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if frame is not None and not frame.empty:
            last_row = frame.iloc[-1]
            for column in ("open", "high", "low", "close", "volume"):
                if column not in last_row.index:
                    continue
                try:
                    value = float(last_row[column])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    features[f"price_{column}"] = value
        if indicators is not None:
            for name in (
                "rsi",
                "ema_fast",
                "ema_slow",
                "sma_trend",
                "atr",
                "bollinger_upper",
                "bollinger_lower",
                "bollinger_middle",
                "macd",
                "macd_signal",
                "stochastic_k",
                "stochastic_d",
            ):
                series = getattr(indicators, name, None)
                if not isinstance(series, pd.Series) or series.empty:
                    continue
                try:
                    value = float(series.iloc[-1])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    features[f"indicator_{name}"] = value
        return features

    def _apply_inference_adjustment(
        self,
        weights: Mapping[str, float],
        score: ModelScore,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        base_weights = {str(name): float(value) for name, value in weights.items()}
        normalized_signal = math.tanh(float(score.expected_return_bps) / 100.0)
        influence = float(score.success_probability)
        scaling = max(0.0, 1.0 + normalized_signal * influence)
        adjusted = {
            name: float(value) * scaling
            for name, value in base_weights.items()
        }
        base_abs = sum(abs(value) for value in base_weights.values())
        adjusted_abs = sum(abs(value) for value in adjusted.values())
        metadata: Dict[str, float] = {
            "normalized_signal": normalized_signal,
            "scaling_factor": scaling,
            "base_abs_weight": base_abs,
            "adjusted_abs_weight": adjusted_abs,
        }
        return adjusted, metadata

    def _record_ai_inference_event(
        self,
        score: ModelScore,
        *,
        features: Mapping[str, float],
        weights_before: Mapping[str, float],
        weights_after: Mapping[str, float],
        assessment: MarketRegimeAssessment,
        adjustment_metadata: Mapping[str, float],
        scored_at: dt.datetime | None = None,
    ) -> None:
        if self._decision_journal is None:
            return
        try:
            event_ts = dt.datetime.now(dt.timezone.utc)
            context = dict(self._decision_journal_context)
            environment = str(context.get("environment", "auto-trade"))
            portfolio = str(context.get("portfolio", "autotrader"))
            risk_profile = str(context.get("risk_profile", assessment.regime.value))
            routing_snapshot = self._routing_snapshot()
            metadata_payload: Dict[str, Any] = {
                "symbol": self.cfg.symbol,
                "regime": assessment.regime.value,
                "expected_return_bps": float(score.expected_return_bps),
                "success_probability": float(score.success_probability),
                "feature_count": len(features),
                "weights_before": {
                    str(name): float(value) for name, value in weights_before.items()
                },
                "weights_after": {
                    str(name): float(value) for name, value in weights_after.items()
                },
            }
            metadata_payload.update({str(k): v for k, v in adjustment_metadata.items()})
            for key, value in routing_snapshot.items():
                metadata_payload.setdefault(key, value)
            scored_at_iso = self._isoformat(scored_at or self._last_inference_scored_at)
            if scored_at_iso is not None:
                metadata_payload["scored_at"] = scored_at_iso
            sample = {
                str(name): float(value)
                for name, value in sorted(features.items())[:8]
            }
            if sample:
                metadata_payload["feature_sample"] = sample
            event_kwargs: Dict[str, Any] = {
                "event_type": "ai_inference",
                "timestamp": event_ts,
                "environment": environment,
                "portfolio": portfolio,
                "risk_profile": risk_profile,
                "symbol": self.cfg.symbol,
                "metadata": self._stringify_metadata(metadata_payload),
            }
            for key in self._EVENT_CONTEXT_KEYS:
                value = context.get(key)
                if value is None:
                    value = routing_snapshot.get(key)
                if value is not None:
                    event_kwargs[key] = str(value)
            event = TradingDecisionEvent(**event_kwargs)
        except Exception:  # pragma: no cover - log defensively, do not break trading
            self._logger.debug(
                "Nie udało się zbudować zdarzenia decision journalu dla inference",
                exc_info=True,
            )
            return
        try:
            self._decision_journal.record(event)
        except Exception:  # pragma: no cover - avoid breaking execution on IO issues
            self._logger.debug(
                "Nie udało się zapisać zdarzenia decision journalu dla inference",
                exc_info=True,
            )

    def _record_risk_decision_event(
        self,
        *,
        event: str,
        mode: str,
        detail: Mapping[str, Any],
        summary: RegimeSummary | None = None,
    ) -> None:
        if self._decision_journal is None:
            return
        try:
            event_ts = dt.datetime.now(dt.timezone.utc)
            context = dict(self._decision_journal_context)
            environment = str(context.get("environment", "auto-trade"))
            portfolio = str(context.get("portfolio", "autotrader"))
            summary_regime = None
            if summary is not None:
                summary_regime = getattr(summary, "regime", None)
            if isinstance(summary_regime, MarketRegime):
                inferred_profile = summary_regime.value
            elif isinstance(summary_regime, str):
                inferred_profile = summary_regime
            else:
                inferred_profile = "autotrade"
            risk_profile = str(context.get("risk_profile", inferred_profile))
            routing_snapshot = self._routing_snapshot()
            metadata_payload: Dict[str, Any] = {
                "symbol": self.cfg.symbol,
                "mode": mode,
                "event": event,
            }
            metadata_payload.update({str(k): v for k, v in detail.items()})
            for key, value in routing_snapshot.items():
                metadata_payload.setdefault(key, value)
            if summary is not None:
                level = getattr(summary, "risk_level", None)
                if isinstance(level, RiskLevel):
                    metadata_payload.setdefault("risk_level", level.value)
                elif level is not None:
                    metadata_payload.setdefault("risk_level", level)
                score = getattr(summary, "risk_score", None)
                if score is not None:
                    metadata_payload.setdefault("risk_score", score)
            event_kwargs: Dict[str, Any] = {
                "event_type": event,
                "timestamp": event_ts,
                "environment": environment,
                "portfolio": portfolio,
                "risk_profile": risk_profile,
                "symbol": self.cfg.symbol,
                "status": event,
                "metadata": self._stringify_metadata(metadata_payload),
            }
            for key in self._EVENT_CONTEXT_KEYS:
                value = context.get(key)
                if value is None:
                    value = routing_snapshot.get(key)
                if value is not None:
                    event_kwargs[key] = str(value)
            event_obj = TradingDecisionEvent(**event_kwargs)
        except Exception:  # pragma: no cover - log defensively, do not break trading
            self._logger.debug(
                "Nie udało się zbudować zdarzenia decision journalu dla risk control",
                exc_info=True,
            )
            return
        try:
            self._decision_journal.record(event_obj)
        except Exception:  # pragma: no cover - avoid breaking execution on IO issues
            self._logger.debug(
                "Nie udało się zapisać zdarzenia decision journalu dla risk control",
                exc_info=True,
            )

    def _stage6_signal_bundle(self, closes: List[float], frame: pd.DataFrame) -> Dict[str, float]:
        trend_signal = float(self._trend_following_signal(closes))
        day_signal = float(self._day_trading_signal(frame))
        mean_signal = float(self._mean_reversion_signal(closes))
        arbitrage_signal = float(self._arbitrage_signal(frame))
        return {
            "trend_following": trend_signal,
            "day_trading": day_signal,
            "mean_reversion": mean_signal,
            "arbitrage": arbitrage_signal,
            "daily_breakout": day_signal,
        }

    def _install_regime_components(
        self,
        classifier: MarketRegimeClassifier,
        history: RegimeHistory | None = None,
    ) -> None:
        """Powiąż klasyfikator oraz historię, zapewniając spójną konfigurację progów."""

        loader = getattr(classifier, "thresholds_loader", None)
        if loader is None or not callable(loader):  # pragma: no cover - defensywne strażniki
            raise TypeError("classifier must expose a callable thresholds_loader")
        snapshot_getter = getattr(classifier, "thresholds_snapshot", None)
        if snapshot_getter is None or not callable(snapshot_getter):  # pragma: no cover - strażnik
            raise TypeError("classifier must provide thresholds_snapshot()")

        self._regime_classifier = classifier
        supplied_history = history
        if supplied_history is None:
            existing_history = getattr(self, "_regime_history", None)
            if isinstance(existing_history, RegimeHistory):
                history = existing_history
                history.reconfigure(
                    maxlen=self.cfg.regime_history_maxlen,
                    decay=self.cfg.regime_history_decay,
                    keep_history=True,
                )
                history.reload_thresholds(loader=loader)
            else:
                history = RegimeHistory(
                    thresholds_loader=loader,
                    maxlen=self.cfg.regime_history_maxlen,
                    decay=self.cfg.regime_history_decay,
                )
        else:
            history = supplied_history
            history.reload_thresholds(loader=loader)
        thresholds = snapshot_getter()
        history.reload_thresholds(thresholds=thresholds)
        self._regime_history = history

    def set_regime_components(
        self,
        *,
        classifier: MarketRegimeClassifier,
        history: RegimeHistory | None = None,
        reset_state: bool = True,
    ) -> None:
        """Zastąp aktywny klasyfikator i historię autotradera."""

        target_history = history or getattr(self, "_regime_history", None)
        self._install_regime_components(classifier, target_history)
        if reset_state:
            self._last_regime = None
            self._last_summary = None
            self._regime_history.clear()
            self._auto_risk_frozen = False
            self._auto_risk_frozen_until = 0.0
            self._auto_risk_state = _AutoRiskFreezeState()
            self._recompute_risk_freeze_until()

    def configure_regime_history(
        self,
        *,
        maxlen: int | None = None,
        decay: float | None = None,
        reset: bool = False,
    ) -> None:
        """Zmień parametry wygładzania historii reżimu."""

        update_maxlen = maxlen is not None
        update_decay = decay is not None
        if maxlen is None and decay is None:
            if not reset:
                return
            maxlen = self._regime_history.maxlen
            decay = self._regime_history.decay
        keep_history = not reset
        self._regime_history.reconfigure(
            maxlen=maxlen,
            decay=decay,
            keep_history=keep_history,
        )
        if update_maxlen and maxlen is not None:
            self.cfg.regime_history_maxlen = int(maxlen)
        if update_decay and decay is not None:
            self.cfg.regime_history_decay = float(decay)
        if reset:
            self._last_regime = None
            self._last_summary = None

    def configure_strategy_weights(
        self,
        overrides: Mapping[MarketRegime | str, Mapping[str, float]],
        *,
        replace: bool = False,
    ) -> None:
        """Zaktualizuj domyślne wagi strategii używane przez silnik i workflow."""

        if not overrides:
            return
        normalised = self._normalize_strategy_config(overrides)
        if not normalised:
            return

        current = {
            regime: dict(weights)
            for regime, weights in self._strategy_weights.weights.items()
        }
        for regime, payload in normalised.items():
            if replace:
                current[regime] = {str(name): float(value) for name, value in payload.items()}
            else:
                regime_weights = dict(current.get(regime, {}))
                regime_weights.update({str(name): float(value) for name, value in payload.items()})
                current[regime] = regime_weights

        self._strategy_weights = RegimeStrategyWeights(weights=current)
        self.cfg.strategy_weights = {
            regime.value: dict(weights) for regime, weights in current.items()
        }

        workflow = getattr(self, "_regime_workflow", None)
        update = getattr(workflow, "update_default_weights", None)
        if callable(update):
            update(normalised, replace=replace)
        else:
            self._refresh_workflow_presets()

    def configure_parameter_overrides(
        self,
        overrides: Mapping[MarketRegime | str, Mapping[str, float | int]],
        *,
        replace: bool = False,
    ) -> None:
        """Zmień domyślne nadpisania parametrów używane przez silnik i workflow."""

        if not overrides:
            return
        normalised = self._normalize_parameter_config(overrides)
        if not normalised:
            return

        current = {
            regime: dict(values)
            for regime, values in self._parameter_overrides.items()
        }
        for regime, payload in normalised.items():
            if replace:
                current[regime] = {str(name): value for name, value in payload.items()}
            else:
                regime_overrides = dict(current.get(regime, {}))
                regime_overrides.update({str(name): value for name, value in payload.items()})
                current[regime] = regime_overrides

        self._parameter_overrides = current
        self.cfg.regime_parameter_overrides = {
            regime.value: dict(values) for regime, values in current.items()
        }

        workflow = getattr(self, "_regime_workflow", None)
        update = getattr(workflow, "update_parameter_overrides", None)
        if callable(update):
            update(current, replace=replace)
        else:
            self._refresh_workflow_presets()

    def _on_wfo_status_batch(self, events: List[Event]) -> None:
        for ev in events:
            st = ev.payload.get("status") if ev.payload else None
            if st is None and ev.payload:
                st = ev.payload.get("state") or ev.payload.get("kind")
            if st == "applied":
                payload = ev.payload or {}
                params = payload.get("params") or payload.get("detail", {}).get("params")
                if params:
                    self.apply_params(params)
                self.enable()

    def _on_risk_alert_batch(self, events: List[Event]) -> None:
        for ev in events:
            now = time.time()
            payload = ev.payload or {}
            if payload.get("symbol") != self.cfg.symbol:
                continue
            expiry = now + self.cfg.risk_freeze_seconds
            reason_code = str(payload.get("kind") or "risk_alert")
            source = str(payload.get("source") or reason_code)
            self._apply_manual_risk_freeze(
                reason=reason_code,
                expiry=float(expiry),
                now=now,
                source=source,
            )
            self._recompute_risk_freeze_until()

    def _on_ticks_batch(self, events: List[Event]) -> None:
        for ev in events:
            payload = ev.payload or {}
            if payload.get("symbol") != self.cfg.symbol:
                continue
            bar = payload.get("bar") or {}
            px = float(bar.get("close", payload.get("price", 0.0)))
            high = float(bar.get("high", px))
            low = float(bar.get("low", px))
            volume = float(bar.get("volume", bar.get("quoteVolume", 0.0) or 0.0))
            timestamp = float(bar.get("open_time") or payload.get("timestamp") or time.time())
            self._closes.append(px)
            self._bars.append(
                {
                    "timestamp": timestamp,
                    "close": px,
                    "high": high,
                    "low": low,
                    "volume": volume,
                }
            )
            self._maybe_trade()

    def _maybe_trade(self) -> None:
        self._sync_freeze_state()
        closes = self._closes
        if len(closes) < max(self._params.get("fast", 10), self._params.get("slow", 50)) + 2:
            return
        if not self._enabled:
            return
        if time.time() < self._risk_frozen_until:
            return
        if len(self._bars) < self.cfg.regime_window:
            return
        frame = pd.DataFrame(list(self._bars)[-self.cfg.regime_window :])
        indicator_frame = self._prepare_indicator_frame(frame)
        data_for_regime = indicator_frame if not indicator_frame.empty else frame
        base_parameters = self._build_base_trading_parameters()
        (
            assessment,
            summary,
            weights,
            parameters,
            strategy_metadata,
            metadata_summary,
            activation_payload,
        ) = self._evaluate_regime_decision(data_for_regime, base_parameters)
        self._last_trading_parameters = parameters
        summary_payload: dict[str, object] | None = None
        if summary is not None:
            summary_payload = summary.to_dict()
        inference_features = self._build_inference_features(indicator_frame)
        plugin_signals: Dict[str, float] = {}
        indicators: TechnicalIndicators | None = None
        if not indicator_frame.empty:
            try:
                indicators = self._indicator_service.calculate_indicators(indicator_frame, parameters)
            except Exception as exc:  # pragma: no cover - defensywna degradacja
                self._logger.debug("Błąd wyliczania wskaźników: %s", exc, exc_info=True)
            else:
                plugin_signals = self._generate_plugin_signals(
                    indicators,
                    parameters,
                    indicator_frame,
                    weights,
                )
                inference_features = self._build_inference_features(indicator_frame, indicators)
        fallback_signals = self._stage6_signal_bundle(closes, frame)
        signals = dict(fallback_signals)
        if plugin_signals:
            signals.update(plugin_signals)
        signals["daily_breakout"] = signals.get("day_trading", fallback_signals["day_trading"])
        base_weights = dict(weights)
        base_abs_weight = sum(abs(value) for value in base_weights.values())
        base_signal_numerator = sum(
            base_weights.get(name, 0.0) * signals.get(name, 0.0)
            for name in base_weights
        )
        base_signal_value = (
            base_signal_numerator / base_abs_weight if base_abs_weight else 0.0
        )
        ai_metadata: dict[str, object] | None = None
        self._last_inference_score = None
        self._last_inference_metadata = None
        self._last_inference_scored_at = None
        adjustment_metadata: Dict[str, float] | None = None
        denominator_override: float | None = None
        ai_score: ModelScore | None = None
        weights_before_adjustment: Dict[str, float] | None = None
        adjusted_signal_value: float | None = None
        inference_service = self._decision_inference
        if (
            inference_service is not None
            and getattr(inference_service, "is_ready", True)
            and inference_features
        ):
            try:
                ai_score = inference_service.score(
                    inference_features,
                    context={"symbol": self.cfg.symbol, "regime": assessment.regime.value},
                )
            except Exception as exc:  # pragma: no cover - inference should never break trading
                self._logger.debug(
                    "DecisionModelInference score failed: %s", exc, exc_info=True
                )
            else:
                self._last_inference_scored_at = dt.datetime.now(dt.timezone.utc)
                scored_at_iso = self._isoformat(self._last_inference_scored_at)
                weights_before_adjustment = dict(weights)
                weights, adjustment_metadata = self._apply_inference_adjustment(weights, ai_score)
                adjustment_metadata = dict(adjustment_metadata)
                parameters = self._compose_trading_parameters(weights)
                self._last_trading_parameters = parameters
                denominator_override = base_abs_weight if base_abs_weight > 0 else None
                adjusted_numerator = sum(
                    weights.get(name, 0.0) * signals.get(name, 0.0)
                    for name in weights
                )
                denominator_for_adjusted = denominator_override
                if denominator_for_adjusted is None:
                    denominator_for_adjusted = sum(abs(value) for value in weights.values())
                adjusted_signal_value = (
                    adjusted_numerator / denominator_for_adjusted
                    if denominator_for_adjusted
                    else 0.0
                )
                adjustment_metadata["signal_before_adjustment"] = base_signal_value
                adjustment_metadata["signal_after_adjustment"] = adjusted_signal_value
                ai_metadata = {
                    "expected_return_bps": float(ai_score.expected_return_bps),
                    "success_probability": float(ai_score.success_probability),
                    "weight_scaling": adjustment_metadata.get("scaling_factor", 1.0),
                    "normalized_signal": adjustment_metadata.get("normalized_signal", 0.0),
                    "feature_count": len(inference_features),
                    "base_abs_weight": adjustment_metadata.get("base_abs_weight"),
                    "adjusted_abs_weight": adjustment_metadata.get("adjusted_abs_weight"),
                    "signal_before_adjustment": base_signal_value,
                    "signal_after_adjustment": adjusted_signal_value,
                }
                if scored_at_iso is not None:
                    ai_metadata["scored_at"] = scored_at_iso
                self._last_inference_score = ai_score
        numerator = 0.0
        for name, weight in weights.items():
            signal_value = signals.get(name, 0.0)
            numerator += weight * signal_value
        if denominator_override is not None:
            denominator = denominator_override
        else:
            denominator = sum(abs(weight) for weight in weights.values())
        combined_unclamped = numerator / denominator if denominator else 0.0
        raw_combined_unclamped = combined_unclamped
        adjustment_threshold_triggered = False
        if self._signal_threshold_after_adjustment is not None:
            reference_value = (
                adjusted_signal_value
                if adjusted_signal_value is not None
                else combined_unclamped
            )
            if abs(reference_value) < self._signal_threshold_after_adjustment:
                combined_unclamped = 0.0
                adjustment_threshold_triggered = abs(reference_value) > 0.0
                if adjusted_signal_value is not None:
                    adjusted_signal_value = 0.0
                    if adjustment_metadata is not None:
                        adjustment_metadata["signal_after_adjustment"] = 0.0
        combined = max(-1.0, min(1.0, combined_unclamped))
        if ai_metadata is not None:
            ai_metadata.setdefault("signal_before_adjustment", base_signal_value)
            ai_metadata.setdefault(
                "signal_after_adjustment",
                adjusted_signal_value if adjusted_signal_value is not None else combined_unclamped,
            )
            ai_metadata["signal_after_clamp"] = combined
            ai_metadata["signal_after_clamp_threshold"] = self._activation_threshold
            if self._signal_threshold_after_adjustment is not None:
                ai_metadata["signal_after_adjustment_threshold"] = (
                    self._signal_threshold_after_adjustment
                )
            if adjustment_metadata is not None:
                adjustment_metadata["signal_after_clamp"] = combined
            if (
                ai_score is not None
                and adjustment_metadata is not None
                and weights_before_adjustment is not None
            ):
                self._record_ai_inference_event(
                    ai_score,
                    features=inference_features,
                    weights_before=weights_before_adjustment,
                    weights_after=weights,
                    assessment=assessment,
                    adjustment_metadata=adjustment_metadata,
                    scored_at=self._last_inference_scored_at,
                )
                self._last_inference_metadata = dict(ai_metadata)
        clamp_threshold_triggered = (
            0.0 < abs(raw_combined_unclamped) < self._activation_threshold
        )
        metadata_payload: dict[str, object] | None = None
        metadata_container: dict[str, object] = {}
        if strategy_metadata:
            metadata_container["per_strategy"] = {
                name: dict(payload)
                for name, payload in strategy_metadata.items()
            }
            metadata_container["license_tiers"] = list(
                metadata_summary.get("license_tiers", ())
            )
            metadata_container["risk_classes"] = list(
                metadata_summary.get("risk_classes", ())
            )
            metadata_container["required_data"] = list(
                metadata_summary.get("required_data", ())
            )
            metadata_container["capabilities"] = list(
                metadata_summary.get("capabilities", ())
            )
            metadata_container["tags"] = list(metadata_summary.get("tags", ()))
        if activation_payload:
            metadata_container["activation"] = {
                key: value for key, value in activation_payload.items()
            }
        if ai_metadata is not None:
            metadata_container["ai_inference"] = dict(ai_metadata)
        if self._last_dynamic_preset is not None:
            metadata_container.setdefault(
                "adaptive_preset",
                {key: value for key, value in self._last_dynamic_preset.items()},
            )
        if summary_payload is not None:
            metadata_container["regime_summary"] = summary_payload
        signal_thresholds_metadata: dict[str, object] = {
            "signal_after_clamp": self._activation_threshold
        }
        if self._signal_threshold_after_adjustment is not None:
            signal_thresholds_metadata["signal_after_adjustment"] = (
                self._signal_threshold_after_adjustment
            )
            if adjustment_threshold_triggered:
                signal_thresholds_metadata["signal_after_adjustment_triggered"] = True
        if clamp_threshold_triggered:
            signal_thresholds_metadata["signal_after_clamp_triggered"] = True
        if signal_thresholds_metadata:
            metadata_container.setdefault("signal_thresholds", signal_thresholds_metadata)
        if metadata_container:
            metadata_payload = metadata_container

        if self.cfg.emit_signals:
            payload = {
                "symbol": self.cfg.symbol,
                "direction": combined,
                "params": dict(self._params),
                "regime": assessment.regime.value,
                "weights": weights,
                "signals": signals,
                "strategy_parameters": {
                    "ema_fast_period": parameters.ema_fast_period,
                    "ema_slow_period": parameters.ema_slow_period,
                    "ensemble_weights": parameters.ensemble_weights,
                    "day_trading_momentum_window": parameters.day_trading_momentum_window,
                    "day_trading_volatility_window": parameters.day_trading_volatility_window,
                    "arbitrage_confirmation_window": parameters.arbitrage_confirmation_window,
                    "arbitrage_spread_threshold": parameters.arbitrage_spread_threshold,
                },
            }
            if metadata_payload:
                payload["metadata"] = metadata_payload
            payload.update(self._routing_snapshot())
            self.adapter.publish(EventType.SIGNAL, payload)
        direction = 0
        activation_threshold = self._activation_threshold
        if combined > activation_threshold:
            direction = +1
        elif combined < -activation_threshold:
            direction = -1
        if self._last_signal is None:
            self._last_signal = 0
        if direction > 0 and self._last_signal <= 0:
            self._submit_market("buy", self.cfg.qty)
            self._last_signal = +1
            detail = {
                "symbol": self.cfg.symbol,
                "qty": self.cfg.qty,
                "regime": assessment.to_dict(),
                "summary": summary_payload,
            }
            if metadata_payload:
                detail["metadata"] = metadata_payload
            if activation_payload:
                detail["activation"] = {
                    key: value for key, value in activation_payload.items()
                }
            detail = self._augment_status_detail(detail)
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_long",
                detail=detail,
            )
        elif direction < 0 and self._last_signal >= 0:
            self._submit_market("sell", self.cfg.qty)
            self._last_signal = -1
            detail = {
                "symbol": self.cfg.symbol,
                "qty": self.cfg.qty,
                "regime": assessment.to_dict(),
                "summary": summary_payload,
            }
            if metadata_payload:
                detail["metadata"] = metadata_payload
            if activation_payload:
                detail["activation"] = {
                    key: value for key, value in activation_payload.items()
                }
            detail = self._augment_status_detail(detail)
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_short",
                detail=detail,
            )

    @property
    def last_regime_decision(self) -> RegimeSwitchDecision | None:
        """Return the most recent decision produced by the regime workflow."""

        return self._last_regime_decision

    @property
    def last_regime_activation(self) -> RegimePresetActivation | None:
        """Return the raw preset activation reported by the workflow, if any."""

        return self._last_regime_activation

    @staticmethod
    def _sma_tail(xs: List[float], n: int) -> Optional[float]:
        if len(xs) < n:
            return None
        return sum(xs[-n:]) / n

    def _last_cross_signal(self, xs: List[float], fast: int, slow: int) -> Optional[int]:
        if fast >= slow or len(xs) < slow + 2:
            return None
        f_prev = self._sma_tail(xs[:-1], fast)
        s_prev = self._sma_tail(xs[:-1], slow)
        f_now = self._sma_tail(xs, fast)
        s_now = self._sma_tail(xs, slow)
        if None in (f_prev, s_prev, f_now, s_now):
            return None
        cross_up = f_now > s_now and f_prev <= s_prev
        cross_dn = f_now < s_now and f_prev >= s_prev
        if cross_up:
            return +1
        if cross_dn:
            return -1
        return 0

    def _trend_following_signal(self, closes: List[float]) -> int:
        fast = int(self._params.get("fast", 10))
        slow = int(self._params.get("slow", 50))
        signal = self._last_cross_signal(closes, fast, slow)
        return 0 if signal is None else signal

    def _day_trading_signal(self, frame: pd.DataFrame) -> int:
        window = max(2, int(self.cfg.breakout_window))
        if frame.empty or len(frame) < window:
            return 0
        recent = frame.tail(window)
        high = float(recent["high"].max())
        low = float(recent["low"].min())
        last_close = float(frame["close"].iloc[-1])
        if last_close >= high * 0.999:
            return +1
        if last_close <= low * 1.001:
            return -1
        return 0

    def _daily_breakout_signal(self, frame: pd.DataFrame) -> int:
        """Alias zachowujący kompatybilność z wcześniejszym API."""

        return self._day_trading_signal(frame)

    def _mean_reversion_signal(self, closes: List[float]) -> float:
        window = max(3, int(self.cfg.mean_reversion_window))
        if len(closes) < window:
            return 0.0
        subset = np.asarray(closes[-window:], dtype=float)
        mean = float(subset.mean())
        std = float(subset.std())
        if std == 0.0:
            return 0.0
        zscore = (subset[-1] - mean) / std
        if zscore > self.cfg.mean_reversion_z:
            return -1.0
        if zscore < -self.cfg.mean_reversion_z:
            return +1.0
        return 0.0

    def _arbitrage_signal(self, frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        closes = frame["close"].astype(float)
        window = max(2, int(self.cfg.arbitrage_window))
        reference = closes.rolling(window=window, min_periods=1).mean()
        spread = (closes - reference) / (reference.abs() + 1e-9)
        confirm = max(1, int(self.cfg.arbitrage_confirmation_window))
        confirmed = spread.rolling(window=confirm, min_periods=1).mean().iloc[-1]
        threshold = float(self.cfg.arbitrage_threshold)
        if confirmed > threshold:
            return -1.0
        if confirmed < -threshold:
            return +1.0
        if threshold <= 0:
            return 0.0
        latest = float(spread.iloc[-1])
        return float(np.tanh(latest / threshold))

    def _classify_regime(self, frame: pd.DataFrame) -> MarketRegimeAssessment:
        if frame.empty:
            if self._last_regime is None:
                metrics: Dict[str, float] = {}
                self._last_regime = MarketRegimeAssessment(
                    regime=MarketRegime.TREND,
                    confidence=0.0,
                    risk_score=0.0,
                    metrics=metrics,
                    symbol=self.cfg.symbol,
                )
            return self._last_regime
        try:
            assessment = self._regime_classifier.assess(frame, symbol=self.cfg.symbol)
        except ValueError:
            if self._last_regime is not None:
                return self._last_regime
            assessment = MarketRegimeAssessment(
                regime=MarketRegime.TREND,
                confidence=0.0,
                risk_score=0.0,
                metrics={},
                symbol=self.cfg.symbol,
            )
        self._regime_history.reload_thresholds(
            thresholds=self._regime_classifier.thresholds_snapshot()
        )
        self._regime_history.update(assessment)
        summary = self._regime_history.summarise()
        self._handle_regime_status(assessment, summary)
        return assessment

    def _sync_freeze_state(self) -> None:
        now = time.time()

        if self._manual_risk_frozen_until and now >= self._manual_risk_frozen_until:
            self._clear_manual_risk_freeze(now=now, source="expiry")

        summary = self._regime_history.summarise()
        auto_until = self._auto_risk_frozen_until if self._auto_risk_frozen else 0.0
        target_rank = self._RISK_LEVEL_ORDER.get(self.cfg.auto_risk_freeze_level, 99)
        level_triggered = False
        score_triggered = False
        triggered = False
        if summary is not None:
            level_rank = self._RISK_LEVEL_ORDER.get(summary.risk_level, -1)
            level_triggered = level_rank >= target_rank >= 0
            score_triggered = summary.risk_score >= self.cfg.auto_risk_freeze_score
            triggered = level_triggered or score_triggered

        if self._auto_risk_frozen and self.cfg.auto_risk_freeze:
            if not triggered:
                reason = self._auto_risk_recovery_reason(summary=summary, target_rank=target_rank)
                self._emit_auto_risk_unfreeze(now=now, reason=reason, summary=summary)
                auto_until = 0.0
            elif auto_until and now >= auto_until:
                # Allow extension logic below to refresh the expiry without unfreezing first
                pass

        if self.cfg.auto_risk_freeze and triggered:
            new_expiry = now + float(self.cfg.risk_freeze_seconds)
            previous_until = self._auto_risk_frozen_until if self._auto_risk_frozen else 0.0
            reason = self._auto_risk_freeze_reason(
                level_triggered=level_triggered,
                score_triggered=score_triggered,
                summary=summary,
            )
            event = "auto_risk_freeze_extend" if self._auto_risk_frozen else "auto_risk_freeze"
            if event == "auto_risk_freeze_extend" and previous_until and new_expiry <= previous_until + 1e-6:
                new_expiry = previous_until
            self._emit_auto_risk_freeze(
                now=now,
                summary=summary,
                expiry=new_expiry,
                reason=reason,
                level=summary.risk_level if summary else self._auto_risk_state.risk_level,
                score=summary.risk_score if summary is not None else self._auto_risk_state.risk_score,
                previous_until=previous_until,
                event=event,
            )
            auto_until = self._auto_risk_frozen_until
        elif auto_until and now >= auto_until:
            self._emit_auto_risk_unfreeze(now=now, reason="expired", summary=summary)

        self._recompute_risk_freeze_until()

    def _recompute_risk_freeze_until(self) -> None:
        now = time.time()
        manual_until = self._manual_risk_frozen_until
        if manual_until and now >= manual_until:
            manual_until = 0.0
            self._manual_risk_frozen_until = 0.0
            self._manual_risk_state = None
        auto_until = self._auto_risk_frozen_until if self._auto_risk_frozen else 0.0
        if auto_until and now >= auto_until:
            auto_until = 0.0
            self._auto_risk_frozen_until = 0.0
            self._auto_risk_frozen = False
            self._auto_risk_state = _AutoRiskFreezeState()
        self._risk_frozen_until = float(max(manual_until, auto_until, 0.0))

    def get_last_regime_assessment(self) -> MarketRegimeAssessment | None:
        """Zwróć ostatnią ocenę reżimu (bez możliwości modyfikacji stanu)."""

        if self._last_regime is None:
            return None
        return deepcopy(self._last_regime)

    def get_regime_summary(self) -> RegimeSummary | None:
        """Zwróć wygładzoną historię reżimu jako kopię defensywną."""

        summary = self._regime_history.summarise()
        if summary is None:
            return None
        return deepcopy(summary)

    def get_regime_thresholds(self) -> Mapping[str, Any]:
        """Udostępnij aktualnie aktywne progi klasyfikatora."""

        return self._regime_history.thresholds_snapshot()

    def _build_risk_snapshot(self) -> "RiskFreezeSnapshot":
        now = time.time()
        manual_active = bool(
            self._manual_risk_state
            and self._manual_risk_frozen_until
            and self._manual_risk_frozen_until > now
        )
        manual_reason = (
            getattr(self._manual_risk_state, "reason", None) if manual_active else None
        )
        manual_until = (
            float(self._manual_risk_frozen_until) if manual_active else None
        )
        auto_active = bool(self._auto_risk_frozen and self._auto_risk_frozen_until > now)
        auto_until = float(self._auto_risk_frozen_until) if auto_active else None
        auto_level = self._auto_risk_state.risk_level if auto_active else None
        auto_score = self._auto_risk_state.risk_score if auto_active else None
        combined_until = float(self._risk_frozen_until)
        return RiskFreezeSnapshot(
            manual_active=manual_active,
            manual_reason=manual_reason,
            manual_until=manual_until,
            auto_active=auto_active,
            auto_until=auto_until,
            auto_risk_level=auto_level,
            auto_risk_score=auto_score,
            combined_until=combined_until,
        )

    def snapshot(self) -> "AutoTradeSnapshot":
        """Zbuduj migawkę stanu autotradera do celów monitoringu/UI."""

        params = self._last_trading_parameters or self._build_base_trading_parameters()
        weights = dict(params.ensemble_weights)
        metadata = self._collect_strategy_metadata(weights)
        catalog_entries = tuple(dict(entry) for entry in self._strategy_catalog.describe())
        metadata_payload = {
            "per_strategy": {
                name: dict(payload) for name, payload in metadata["strategies"].items()
            },
            "license_tiers": metadata["summary"]["license_tiers"],
            "risk_classes": metadata["summary"]["risk_classes"],
            "required_data": metadata["summary"]["required_data"],
            "capabilities": metadata["summary"]["capabilities"],
            "tags": metadata["summary"]["tags"],
        }
        ai_payload: Mapping[str, object] | None = None
        if self._last_inference_metadata or self._last_inference_score:
            inference_payload: Dict[str, object] = {}
            if self._last_inference_metadata:
                inference_payload.update(
                    {str(key): value for key, value in self._last_inference_metadata.items()}
                )
            if self._last_inference_score is not None:
                inference_payload.setdefault(
                    "expected_return_bps", float(self._last_inference_score.expected_return_bps)
                )
                inference_payload.setdefault(
                    "success_probability", float(self._last_inference_score.success_probability)
                )
            if self._last_inference_scored_at is not None:
                inference_payload["scored_at"] = self._isoformat(
                    self._last_inference_scored_at
                )
            if inference_payload:
                ai_payload = MappingProxyType(dict(inference_payload))
        workflow = getattr(self, "_regime_workflow", None)
        overrides_obj = None
        if workflow is not None:
            overrides_obj = getattr(workflow, "default_parameter_overrides", None)
            if callable(overrides_obj):
                overrides_obj = overrides_obj()
        overrides = overrides_obj or self._workflow_parameter_overrides
        overrides_payload: dict[str, Mapping[str, float | int]] = {
            regime.value if isinstance(regime, MarketRegime) else str(regime): dict(values)
            for regime, values in (overrides or {}).items()
        }
        activation_payload = None
        if self._last_regime_activation is not None:
            activation_payload = self._build_activation_payload(self._last_regime_activation)
        return AutoTradeSnapshot(
            symbol=self.cfg.symbol,
            enabled=bool(self._enabled),
            trading_parameters=params,
            strategy_weights=dict(weights),
            regime_decision=self._last_regime_decision,
            regime_thresholds=MappingProxyType(dict(self._regime_history.thresholds_snapshot())),
            regime_parameter_overrides=MappingProxyType(overrides_payload),
            strategy_catalog=catalog_entries,
            metadata=MappingProxyType(metadata_payload),
            regime_activation=activation_payload,
            risk=self._build_risk_snapshot(),
            ai_inference=ai_payload,
        )

    @staticmethod
    def _isoformat(moment: dt.datetime | None) -> str | None:
        if isinstance(moment, dt.datetime):
            try:
                return moment.astimezone(dt.timezone.utc).isoformat()
            except ValueError:
                return moment.isoformat()
        return None

    @staticmethod
    def _normalize_regime_value(regime: MarketRegime | str | None) -> str:
        if isinstance(regime, MarketRegime):
            return regime.value
        if regime is None:
            return "unknown"
        return str(regime)

@dataclass(frozen=True)
class RiskFreezeSnapshot:
    manual_active: bool
    manual_reason: str | None
    manual_until: float | None
    auto_active: bool
    auto_until: float | None
    auto_risk_level: RiskLevel | None
    auto_risk_score: float | None
    combined_until: float


@dataclass(frozen=True)
class AutoTradeSnapshot:
    symbol: str
    enabled: bool
    trading_parameters: TradingParameters
    strategy_weights: Mapping[str, float]
    regime_decision: RegimeSwitchDecision | None
    regime_thresholds: Mapping[str, Any]
    regime_parameter_overrides: Mapping[str, Mapping[str, float | int]]
    strategy_catalog: tuple[Mapping[str, object], ...]
    metadata: Mapping[str, object]
    regime_activation: Mapping[str, object] | None
    risk: RiskFreezeSnapshot
    ai_inference: Mapping[str, object] | None


__all__ = [
    "AutoTradeConfig",
    "AutoTradeEngine",
    "PresetAvailability",
    "AutoTradeSnapshot",
    "RiskFreezeSnapshot",
]
