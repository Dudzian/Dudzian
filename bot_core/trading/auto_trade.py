"""Rozszerzony silnik autotradingu wspierający wiele strategii i reżimy."""
from __future__ import annotations

import datetime as dt
import logging
import math
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

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
from bot_core.trading.regime_workflow import RegimeSwitchDecision, RegimeSwitchWorkflow
from bot_core.trading.strategies import StrategyCatalog


@dataclass
class _AutoRiskFreezeState:
    risk_level: RiskLevel | None = None
    risk_score: float | None = None
    triggered_at: float = 0.0
    last_extension_at: float = 0.0


@dataclass
class _ManualRiskFreezeState:
    reason: str | None = None
    triggered_at: float | None = None
    last_extension_at: float | None = None


@dataclass(frozen=True)
class RiskFreezeSnapshot:
    """Stan zamrożenia ryzyka składowany w migawce autotradera."""

    manual_active: bool
    manual_until: float | None
    manual_reason: str | None
    manual_triggered_at: float | None
    manual_last_extension_at: float | None
    auto_active: bool
    auto_until: float | None
    auto_risk_level: RiskLevel | None
    auto_risk_score: float | None
    auto_triggered_at: float | None
    auto_last_extension_at: float | None
    combined_until: float


@dataclass(frozen=True)
class AutoTradeSnapshot:
    """Ustrukturyzowany podgląd stanu :class:`AutoTradeEngine`."""

    symbol: str
    enabled: bool
    params: Mapping[str, int]
    trading_parameters: TradingParameters | None
    strategy_weights: Mapping[str, float]
    last_signal: int | None
    risk: RiskFreezeSnapshot
    regime_assessment: MarketRegimeAssessment | None
    regime_summary: RegimeSummary | None
    regime_decision: RegimeSwitchDecision | None
    regime_thresholds: Mapping[str, Any]
    regime_parameter_overrides: Mapping[str, Mapping[str, float | int]]
    strategy_catalog: Tuple[Mapping[str, str], ...]


@dataclass
class AutoTradeConfig:
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


class AutoTradeEngine:
    """Prosty kontroler autotradingu reagujący na ticki z EventBusa."""

    _RISK_LEVEL_ORDER = {
        RiskLevel.CALM: 0,
        RiskLevel.BALANCED: 1,
        RiskLevel.WATCH: 2,
        RiskLevel.ELEVATED: 3,
        RiskLevel.CRITICAL: 4,
    }

    def __init__(
        self,
        adapter: EmitterAdapter,
        broker_submit_market,
        cfg: Optional[AutoTradeConfig] = None,
        *,
        regime_classifier: MarketRegimeClassifier | None = None,
        regime_history: RegimeHistory | None = None,
        strategy_catalog: StrategyCatalog | None = None,
        regime_workflow: RegimeSwitchWorkflow | None = None,
        indicator_service: TechnicalIndicatorsService | None = None,
        indicator_config: EngineConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.bus: EventBus = adapter.bus
        self.cfg = cfg or AutoTradeConfig()
        self._logger = logging.getLogger(__name__)
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
        normalized_weights = self._normalize_strategy_config(self.cfg.strategy_weights)
        self._strategy_weights = RegimeStrategyWeights(
            weights={
                regime: dict(weights)
                for regime, weights in normalized_weights.items()
            }
        )
        catalog = strategy_catalog or StrategyCatalog.default()
        self._strategy_catalog = catalog
        workflow_weight_defaults = {
            regime: dict(weights) for regime, weights in normalized_weights.items()
        }
        base_override = {
            "day_trading_momentum_window": int(max(1, self.cfg.breakout_window)),
            "day_trading_volatility_window": int(
                max(1, math.ceil(self.cfg.breakout_window * 1.5))
            ),
            "arbitrage_confirmation_window": int(self.cfg.arbitrage_confirmation_window),
            "arbitrage_spread_threshold": float(self.cfg.arbitrage_threshold),
        }
        config_parameter_overrides = self._normalize_parameter_config(
            self.cfg.regime_parameter_overrides
        )
        workflow_parameter_overrides: Dict[MarketRegime, Dict[str, float | int]] = {}
        for regime in MarketRegime:
            payload = dict(base_override)
            payload.update(config_parameter_overrides.get(regime, {}))
            workflow_parameter_overrides[regime] = payload
        self._parameter_overrides = {
            regime: dict(values)
            for regime, values in workflow_parameter_overrides.items()
        }
        if regime_workflow is None:
            self._regime_workflow: RegimeSwitchWorkflow | None = RegimeSwitchWorkflow(
                classifier=self._regime_classifier,
                history=self._regime_history,
                catalog=self._strategy_catalog,
                default_weights=workflow_weight_defaults,
                default_parameter_overrides=self._parameter_overrides,
                logger=self._logger,
            )
        else:
            self._regime_workflow = regime_workflow
        self._sync_workflow_state()
        self.cfg.regime_parameter_overrides = {
            regime.value: dict(values)
            for regime, values in self._parameter_overrides.items()
        }
        self._sync_workflow_parameter_overrides()
        if indicator_service is None:
            indicator_cfg = indicator_config or EngineConfig(cache_indicators=False)
            self._indicator_service = TechnicalIndicatorsService(self._logger, indicator_cfg)
        else:
            self._indicator_service = indicator_service
        self._base_trading_params: TradingParameters = self.cfg.trading_parameters
        self._last_trading_parameters: TradingParameters | None = None
        self._last_regime: MarketRegimeAssessment | None = None
        self._last_summary: RegimeSummary | None = None
        self._last_regime_decision: RegimeSwitchDecision | None = getattr(
            self._regime_workflow, "last_decision", None
        )

        batch_rule = DebounceRule(window=0.1, max_batch=1)
        self.bus.subscribe(EventType.MARKET_TICK, self._on_ticks_batch, rule=batch_rule)
        self.bus.subscribe(EventType.WFO_STATUS, self._on_wfo_status_batch, rule=batch_rule)
        self.bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert_batch, rule=batch_rule)

    def enable(self) -> None:
        self._enabled = True
        self._manual_risk_frozen_until = 0.0
        self._auto_risk_frozen_until = 0.0
        self._auto_risk_frozen = False
        self._auto_risk_state = _AutoRiskFreezeState()
        self._manual_risk_state = None
        self._recompute_risk_freeze_until()
        self.adapter.push_autotrade_status("enabled", detail={"symbol": self.cfg.symbol})  # type: ignore[attr-defined]

    def disable(self, reason: str = "") -> None:
        self._enabled = False
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "disabled",
            detail={"symbol": self.cfg.symbol, "reason": reason},
            level="WARN",
        )

    def apply_params(self, params: Dict[str, int]) -> None:
        self._params = dict(params)
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "params_applied",
            detail={"symbol": self.cfg.symbol, "params": self._params},
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

    @staticmethod
    def _normalize_parameter_config(
        raw: Mapping[MarketRegime | str, Mapping[str, float | int]] | None
    ) -> Dict[MarketRegime, Dict[str, float | int]]:
        if raw is None:
            return {}
        normalised: Dict[MarketRegime, Dict[str, float | int]] = {}
        for regime_key, payload in raw.items():
            try:
                regime = (
                    regime_key
                    if isinstance(regime_key, MarketRegime)
                    else MarketRegime(str(regime_key).lower())
                )
            except ValueError:
                regime = MarketRegime.TREND
            cleaned: Dict[str, float | int] = {}
            for key, value in payload.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    cleaned[str(key)] = int(value)
                    continue
                if isinstance(value, float):
                    cleaned[str(key)] = float(value)
                    continue
                try:
                    coerced = float(value)
                except (TypeError, ValueError):
                    continue
                cleaned[str(key)] = coerced
            if cleaned:
                normalised[regime] = cleaned
        return normalised

    def _sync_workflow_state(self) -> None:
        """Synchronise shared components with an injected workflow."""

        workflow = getattr(self, "_regime_workflow", None)
        if workflow is None:
            return
        classifier = getattr(workflow, "classifier", None)
        if isinstance(classifier, MarketRegimeClassifier):
            self._regime_classifier = classifier
        history = getattr(workflow, "history", None)
        if isinstance(history, RegimeHistory):
            self._regime_history = history
        catalog = getattr(workflow, "catalog", None)
        if isinstance(catalog, StrategyCatalog):
            self._strategy_catalog = catalog
        self._last_regime_decision = getattr(workflow, "last_decision", None)

    def _sync_workflow_parameter_overrides(self) -> None:
        workflow = getattr(self, "_regime_workflow", None)
        if workflow is None:
            return
        update = getattr(workflow, "update_parameter_overrides", None)
        if callable(update) and self._parameter_overrides:
            update(self._parameter_overrides, replace=False)

    def _build_risk_snapshot(self, now: float) -> RiskFreezeSnapshot:
        manual_active = bool(self._manual_risk_frozen_until and now < self._manual_risk_frozen_until)
        manual_state = self._manual_risk_state if manual_active else None
        manual_until: float | None = (
            float(self._manual_risk_frozen_until) if manual_active else None
        )
        auto_active = bool(self._auto_risk_frozen and now < self._auto_risk_frozen_until)
        auto_state = self._auto_risk_state if auto_active else _AutoRiskFreezeState()
        auto_until: float | None = (
            float(self._auto_risk_frozen_until) if auto_active else None
        )
        combined_until = float(self._risk_frozen_until) if self._risk_frozen_until else 0.0
        return RiskFreezeSnapshot(
            manual_active=manual_active,
            manual_until=manual_until,
            manual_reason=manual_state.reason if manual_state else None,
            manual_triggered_at=manual_state.triggered_at if manual_state else None,
            manual_last_extension_at=manual_state.last_extension_at if manual_state else None,
            auto_active=auto_active,
            auto_until=auto_until,
            auto_risk_level=auto_state.risk_level,
            auto_risk_score=auto_state.risk_score,
            auto_triggered_at=auto_state.triggered_at,
            auto_last_extension_at=auto_state.last_extension_at,
            combined_until=combined_until,
        )

    def _apply_manual_risk_freeze(
        self,
        *,
        reason: str,
        expiry: float,
        now: float,
        source: str,
    ) -> None:
        manual_active = now < self._manual_risk_frozen_until
        previous_until = self._manual_risk_frozen_until if manual_active else 0.0
        state = self._manual_risk_state if manual_active else None
        reason_code = str(reason)
        if not manual_active or state is None:
            self._manual_risk_state = _ManualRiskFreezeState(
                reason=reason_code,
                triggered_at=now,
                last_extension_at=now,
            )
            self._manual_risk_frozen_until = float(expiry)
            detail = {
                "symbol": self.cfg.symbol,
                "until": self._manual_risk_frozen_until,
                "reason": reason_code,
                "triggered_at": now,
                "last_extension_at": now,
                "released_at": None,
                "frozen_for": None,
                "source": source,
            }
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_freeze",
                detail=detail,
                level="WARN",
            )
            return
        previous_reason = state.reason
        should_extend = expiry > self._manual_risk_frozen_until + 1e-6
        state.reason = reason_code
        state.last_extension_at = now
        if should_extend:
            self._manual_risk_frozen_until = float(expiry)
            extend_detail = {
                "symbol": self.cfg.symbol,
                "extended_from": previous_until,
                "until": self._manual_risk_frozen_until,
                "reason": reason_code,
                "triggered_at": state.triggered_at or now,
                "last_extension_at": state.last_extension_at,
                "released_at": None,
                "frozen_for": None,
                "source": source,
            }
            if previous_reason and previous_reason != reason_code:
                extend_detail["previous_reason"] = previous_reason
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_freeze_extend",
                detail=extend_detail,
                level="WARN",
            )

    def _clear_manual_risk_freeze(
        self,
        *,
        now: float,
        source: str | None = None,
    ) -> bool:
        if not self._manual_risk_frozen_until:
            return False
        state = self._manual_risk_state
        detail = {
            "symbol": self.cfg.symbol,
            "reason": state.reason if state else None,
            "triggered_at": state.triggered_at if state else None,
            "last_extension_at": state.last_extension_at if state else None,
            "released_at": now,
            "frozen_for": (
                float(now - state.triggered_at)
                if state and state.triggered_at is not None
                else None
            ),
        }
        if source is not None:
            detail["source"] = source
        previous_until = self._manual_risk_frozen_until
        if previous_until:
            detail["until"] = previous_until
        self._manual_risk_frozen_until = 0.0
        self._manual_risk_state = None
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "risk_unfreeze",
            detail=detail,
        )
        return True

    def snapshot(self) -> AutoTradeSnapshot:
        """Zwróć bezpieczną do odczytu migawkę stanu autotradera."""

        now = time.time()
        params = {str(key): int(value) for key, value in self._params.items()}
        trading_params = deepcopy(self._last_trading_parameters)
        if trading_params is not None:
            strategy_weights = {
                str(name): float(value)
                for name, value in trading_params.ensemble_weights.items()
            }
        else:
            last_regime = self._last_regime.regime if self._last_regime else MarketRegime.TREND
            strategy_weights = {
                str(name): float(weight)
                for name, weight in self._strategy_weights.weights_for(last_regime).items()
            }
        thresholds = {
            str(key): deepcopy(value)
            for key, value in self._regime_history.thresholds_snapshot().items()
        }
        parameter_overrides = {
            regime.value: dict(values)
            for regime, values in self._parameter_overrides.items()
        }
        return AutoTradeSnapshot(
            symbol=self.cfg.symbol,
            enabled=bool(self._enabled),
            params=params,
            trading_parameters=trading_params,
            strategy_weights=strategy_weights,
            last_signal=self._last_signal,
            risk=self._build_risk_snapshot(now),
            regime_assessment=deepcopy(self._last_regime),
            regime_summary=deepcopy(self._last_summary),
            regime_decision=deepcopy(self._last_regime_decision),
            regime_thresholds=thresholds,
            regime_parameter_overrides=parameter_overrides,
            strategy_catalog=self._strategy_catalog.describe(),
        )

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

    def _handle_regime_status(
        self,
        assessment: MarketRegimeAssessment,
        summary: RegimeSummary | None,
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
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "regime_update",
                detail=detail,
            )
        self._last_regime = assessment
        if summary is not None:
            self._last_summary = summary

    def _evaluate_regime_decision(
        self,
        indicator_frame: pd.DataFrame,
        base_parameters: TradingParameters,
    ) -> tuple[
        MarketRegimeAssessment,
        RegimeSummary | None,
        Dict[str, float],
        TradingParameters,
    ]:
        workflow = getattr(self, "_regime_workflow", None)
        if workflow is not None:
            try:
                decision = workflow.decide(
                    indicator_frame,
                    base_parameters,
                    symbol=self.cfg.symbol,
                    parameter_overrides=self._parameter_overrides,
                )
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                self._logger.debug("Błąd workflow reżimu: %s", exc, exc_info=True)
            else:
                self._last_regime_decision = decision
                self._handle_regime_status(decision.assessment, decision.summary)
                weights = {
                    str(name): float(value)
                    for name, value in decision.weights.items()
                }
                return (
                    decision.assessment,
                    decision.summary,
                    weights,
                    decision.parameters,
                )
        assessment = self._classify_regime(indicator_frame)
        summary = self._regime_history.summarise()
        weights = {
            str(name): float(value)
            for name, value in self._strategy_weights.weights_for(assessment.regime).items()
        }
        parameters = self._compose_trading_parameters(weights)
        overrides = self._parameter_overrides.get(assessment.regime, {})
        if overrides:
            cleaned = {
                str(key): value
                for key, value in overrides.items()
                if str(key) != "ensemble_weights"
            }
            if cleaned:
                parameters = replace(parameters, **cleaned)
        normalized = {
            str(name): float(value) for name, value in parameters.ensemble_weights.items()
        }
        return assessment, summary, normalized, parameters

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

    def _legacy_signal_bundle(self, closes: List[float], frame: pd.DataFrame) -> Dict[str, float]:
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
        assessment, summary, weights, parameters = self._evaluate_regime_decision(
            data_for_regime, base_parameters
        )
        self._last_trading_parameters = parameters
        plugin_signals: Dict[str, float] = {}
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
        fallback_signals = self._legacy_signal_bundle(closes, frame)
        signals = dict(fallback_signals)
        if plugin_signals:
            signals.update(plugin_signals)
        signals["daily_breakout"] = signals.get("day_trading", fallback_signals["day_trading"])
        numerator = 0.0
        denominator = 0.0
        for name, weight in weights.items():
            signal_value = signals.get(name, 0.0)
            numerator += weight * signal_value
            denominator += abs(weight)
        combined = numerator / denominator if denominator else 0.0
        if self.cfg.emit_signals:
            self.adapter.publish(
                EventType.SIGNAL,
                {
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
                },
            )
        direction = 0
        if combined > self.cfg.activation_threshold:
            direction = +1
        elif combined < -self.cfg.activation_threshold:
            direction = -1
        if self._last_signal is None:
            self._last_signal = 0
        if direction > 0 and self._last_signal <= 0:
            self._submit_market("buy", self.cfg.qty)
            self._last_signal = +1
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_long",
                detail={
                    "symbol": self.cfg.symbol,
                    "qty": self.cfg.qty,
                    "regime": assessment.to_dict(),
                    "summary": summary.to_dict() if summary is not None else None,
                },
            )
        elif direction < 0 and self._last_signal >= 0:
            self._submit_market("sell", self.cfg.qty)
            self._last_signal = -1
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_short",
                detail={
                    "symbol": self.cfg.symbol,
                    "qty": self.cfg.qty,
                    "regime": assessment.to_dict(),
                    "summary": summary.to_dict() if summary is not None else None,
                },
            )

    @property
    def last_regime_decision(self) -> RegimeSwitchDecision | None:
        """Return the most recent decision produced by the regime workflow."""

        return self._last_regime_decision

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

        auto_until = self._auto_risk_frozen_until if self._auto_risk_frozen else 0.0
        if auto_until and now >= auto_until:
            self._auto_risk_frozen = False
            self._auto_risk_frozen_until = 0.0
            self._auto_risk_state = _AutoRiskFreezeState()
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "auto_risk_unfreeze",
                detail={"symbol": self.cfg.symbol},
            )

        if self.cfg.auto_risk_freeze:
            summary = self._regime_history.summarise()
            triggered = False
            if summary is not None:
                level_rank = self._RISK_LEVEL_ORDER.get(summary.risk_level, -1)
                target_rank = self._RISK_LEVEL_ORDER.get(self.cfg.auto_risk_freeze_level, 99)
                level_triggered = level_rank >= target_rank >= 0
                score_triggered = summary.risk_score >= self.cfg.auto_risk_freeze_score
                triggered = level_triggered or score_triggered
            if triggered:
                new_expiry = now + float(self.cfg.risk_freeze_seconds)
                previous_until = self._auto_risk_frozen_until if self._auto_risk_frozen else 0.0
                detail = {
                    "symbol": self.cfg.symbol,
                    "risk_level": summary.risk_level.value if summary else None,
                    "risk_score": summary.risk_score if summary else None,
                    "until": new_expiry,
                }
                if not self._auto_risk_frozen:
                    self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                        "auto_risk_freeze",
                        detail=detail,
                        level="WARN",
                    )
                else:
                    if new_expiry > previous_until + 1e-6:
                        extend_detail = dict(detail)
                        extend_detail["extended_from"] = previous_until
                        extend_detail["until"] = new_expiry
                        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                            "auto_risk_freeze_extend",
                            detail=extend_detail,
                            level="WARN",
                        )
                self._auto_risk_frozen = True
                self._auto_risk_frozen_until = max(previous_until, new_expiry)
                if summary is not None:
                    if not self._auto_risk_state.triggered_at:
                        self._auto_risk_state.triggered_at = now
                    self._auto_risk_state.last_extension_at = now
                    self._auto_risk_state.risk_level = summary.risk_level
                    self._auto_risk_state.risk_score = float(summary.risk_score)

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


__all__ = [
    "AutoTradeConfig",
    "AutoTradeEngine",
    "AutoTradeSnapshot",
    "RiskFreezeSnapshot",
]
