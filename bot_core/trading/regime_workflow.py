"""Workflow orchestrating regime-aware strategy selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, Mapping, MutableMapping

import pandas as pd

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeSummary,
)
from bot_core.trading.engine import TradingParameters
from bot_core.trading.strategies import StrategyCatalog


@dataclass(frozen=True)
class RegimeSwitchDecision:
    """Result of a workflow evaluation for a single timestep."""

    regime: MarketRegime
    assessment: MarketRegimeAssessment
    summary: RegimeSummary | None
    weights: Dict[str, float]
    parameters: TradingParameters
    timestamp: pd.Timestamp


class RegimeSwitchWorkflow:
    """High-level controller combining classifier outputs with strategy plugins."""

    def __init__(
        self,
        *,
        classifier: MarketRegimeClassifier | None = None,
        history: RegimeHistory | None = None,
        catalog: StrategyCatalog | None = None,
        confidence_threshold: float = 0.55,
        persistence_threshold: float = 0.35,
        min_switch_cooldown: int = 5,
        default_weights: Mapping[
            MarketRegime | str, Mapping[str, float]
        ] | None = None,
        default_parameter_overrides: Mapping[
            MarketRegime | str, Mapping[str, float | int]
        ] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if confidence_threshold <= 0 or confidence_threshold > 1:
            raise ValueError("confidence_threshold must be in the (0, 1] interval")
        if persistence_threshold <= 0 or persistence_threshold > 1:
            raise ValueError("persistence_threshold must be in the (0, 1] interval")
        if min_switch_cooldown < 0:
            raise ValueError("min_switch_cooldown must be non-negative")

        self._logger = logger or logging.getLogger(__name__)
        self._classifier = classifier or MarketRegimeClassifier()
        self._history = history or RegimeHistory(
            thresholds_loader=self._classifier.thresholds_loader
        )
        self._history.reload_thresholds(
            thresholds=self._classifier.thresholds_snapshot()
        )
        self._catalog = catalog or StrategyCatalog.default()
        self._confidence_threshold = float(confidence_threshold)
        self._persistence_threshold = float(persistence_threshold)
        self._min_switch_cooldown = int(min_switch_cooldown)
        self._last_decision: RegimeSwitchDecision | None = None
        self._last_switch_step: int | None = None
        self._step_counter: int = 0

        self._default_strategy_weights = self._build_default_weights(default_weights)
        self._parameter_overrides = self._build_parameter_overrides(
            default_parameter_overrides
        )

    @property
    def classifier(self) -> MarketRegimeClassifier:
        """Expose the classifier used by the workflow."""

        return self._classifier

    @property
    def history(self) -> RegimeHistory:
        """Expose the shared regime history buffer."""

        return self._history

    @property
    def catalog(self) -> StrategyCatalog:
        """Expose the strategy catalog consulted by the workflow."""

        return self._catalog

    def decide(
        self,
        market_data: pd.DataFrame,
        base_parameters: TradingParameters,
        *,
        symbol: str | None = None,
        parameter_overrides: Mapping[MarketRegime, Mapping[str, float | int]] | None = None,
    ) -> RegimeSwitchDecision:
        """Evaluate the current regime and return updated parameters."""

        if market_data is None or market_data.empty:
            raise ValueError("market_data must contain OHLCV history")

        self._step_counter += 1

        assessment = self._classifier.assess(market_data, symbol=symbol)
        self._history.update(assessment)
        summary = self._history.summarise()
        timestamp = pd.Timestamp(market_data.index[-1])

        previous_decision = self._last_decision
        candidate_regime = self._select_regime(assessment, summary)
        if self._should_defer_switch(candidate_regime, summary):
            if previous_decision is not None:
                candidate_regime = previous_decision.regime

        weights = self._resolve_weights(candidate_regime, assessment)
        overrides = self._prepare_overrides(candidate_regime, parameter_overrides)
        overrides["ensemble_weights"] = weights
        tuned_params = replace(base_parameters, **overrides)

        decision = RegimeSwitchDecision(
            regime=candidate_regime,
            assessment=assessment,
            summary=summary,
            weights=weights,
            parameters=tuned_params,
            timestamp=timestamp,
        )

        if previous_decision is None or previous_decision.regime != decision.regime:
            self._last_switch_step = self._step_counter
        self._last_decision = decision
        self._logger.debug(
            "Regime workflow decision: %s (confidence=%.2f, risk=%.2f)",
            candidate_regime.value,
            assessment.confidence,
            assessment.risk_score,
        )
        return decision

    def _select_regime(
        self,
        assessment: MarketRegimeAssessment,
        summary: RegimeSummary | None,
    ) -> MarketRegime:
        if summary and summary.confidence >= self._confidence_threshold:
            return summary.regime
        if (
            self._last_decision is not None
            and assessment.confidence < self._confidence_threshold
        ):
            return self._last_decision.regime
        return assessment.regime

    def _should_defer_switch(
        self,
        regime: MarketRegime,
        summary: RegimeSummary | None,
    ) -> bool:
        if self._last_decision is None:
            return False
        if regime == self._last_decision.regime:
            return False
        if summary and summary.regime == regime:
            if summary.regime_persistence < self._persistence_threshold:
                return True
        if self._min_switch_cooldown > 0 and self._last_switch_step is not None:
            if (self._step_counter - self._last_switch_step) <= self._min_switch_cooldown:
                return True
        return False

    def _resolve_weights(
        self,
        regime: MarketRegime,
        assessment: MarketRegimeAssessment,
    ) -> Dict[str, float]:
        available = set(self._catalog.available())
        default = dict(self._default_strategy_weights.get(regime, {}))
        weights = {name: weight for name, weight in default.items() if name in available}
        if not weights:
            if not available:
                raise RuntimeError("No strategy plugins registered")
            weights = {name: 1.0 for name in available}

        risk = float(assessment.risk_score)
        if risk > 0.6:
            weights["arbitrage"] = weights.get("arbitrage", 0.0) + 0.1
            weights["trend_following"] = max(weights.get("trend_following", 0.0) - 0.1, 0.0)

        total = sum(weights.values()) or 1.0
        return {name: float(value) / total for name, value in weights.items()}

    def _prepare_overrides(
        self,
        regime: MarketRegime,
        parameter_overrides: Mapping[MarketRegime, Mapping[str, float | int]] | None,
    ) -> MutableMapping[str, float | int | Dict[str, float]]:
        overrides: MutableMapping[str, float | int | Dict[str, float]] = {}
        base = self._parameter_overrides.get(regime)
        if base:
            overrides.update({str(key): value for key, value in base.items()})
        if parameter_overrides and regime in parameter_overrides:
            overrides.update({str(k): v for k, v in parameter_overrides[regime].items()})
        return overrides

    @property
    def last_decision(self) -> RegimeSwitchDecision | None:
        return self._last_decision

    @property
    def default_strategy_weights(self) -> Mapping[MarketRegime, Mapping[str, float]]:
        """Udostępnia skopiowaną konfigurację wag strategii per reżim."""

        return {
            regime: dict(weights)
            for regime, weights in self._default_strategy_weights.items()
        }

    @property
    def default_parameter_overrides(
        self,
    ) -> Mapping[MarketRegime, Mapping[str, float | int]]:
        """Udostępnia domyślne nadpisania parametrów wykorzystywane przy strojeniu."""

        return {
            regime: dict(values)
            for regime, values in self._parameter_overrides.items()
        }

    def _build_default_weights(
        self,
        custom: Mapping[MarketRegime | str, Mapping[str, float]] | None,
    ) -> Dict[MarketRegime, Dict[str, float]]:
        base: Dict[MarketRegime, Dict[str, float]] = {
            MarketRegime.TREND: {
                "trend_following": 0.55,
                "day_trading": 0.15,
                "mean_reversion": 0.15,
                "arbitrage": 0.15,
            },
            MarketRegime.DAILY: {
                "trend_following": 0.25,
                "day_trading": 0.45,
                "mean_reversion": 0.15,
                "arbitrage": 0.15,
            },
            MarketRegime.MEAN_REVERSION: {
                "trend_following": 0.2,
                "day_trading": 0.15,
                "mean_reversion": 0.45,
                "arbitrage": 0.2,
            },
        }
        if not custom:
            return {regime: dict(weights) for regime, weights in base.items()}
        normalised = self._normalise_regime_mapping(custom)
        merged = {regime: dict(weights) for regime, weights in base.items()}
        merged.update(normalised)
        return merged

    def _build_parameter_overrides(
        self,
        custom: Mapping[MarketRegime | str, Mapping[str, float | int]] | None,
    ) -> Dict[MarketRegime, Dict[str, float | int]]:
        base: Dict[MarketRegime, Dict[str, float | int]] = {
            MarketRegime.TREND: {
                "signal_threshold": 0.08,
                "stop_loss_atr_mult": 2.7,
                "take_profit_atr_mult": 3.5,
            },
            MarketRegime.DAILY: {
                "signal_threshold": 0.14,
                "day_trading_momentum_window": 4,
                "day_trading_volatility_window": 10,
            },
            MarketRegime.MEAN_REVERSION: {
                "signal_threshold": 0.06,
                "rsi_oversold": 35.0,
                "rsi_overbought": 65.0,
            },
        }
        if not custom:
            return {regime: dict(values) for regime, values in base.items()}
        normalised = self._normalise_regime_mapping(custom)
        merged = {regime: dict(values) for regime, values in base.items()}
        merged.update(normalised)
        return merged

    def _normalise_regime_mapping(
        self,
        mapping: Mapping[MarketRegime | str, Mapping[str, float | int]],
    ) -> Dict[MarketRegime, Dict[str, float | int]]:
        normalised: Dict[MarketRegime, Dict[str, float | int]] = {}
        for regime_key, payload in mapping.items():
            regime = self._resolve_regime(regime_key)
            cleaned: Dict[str, float | int] = {}
            for key, value in payload.items():
                cleaned_value: float | int | None
                if isinstance(value, bool):
                    # Pomijamy wartości bool, aby uniknąć przypadkowej konwersji do 0/1.
                    cleaned_value = None
                elif isinstance(value, int):
                    cleaned_value = int(value)
                elif isinstance(value, float):
                    cleaned_value = float(value)
                else:
                    try:
                        cleaned_value = float(value)
                    except (TypeError, ValueError):
                        cleaned_value = None
                if cleaned_value is not None:
                    cleaned[str(key)] = cleaned_value
            if cleaned:
                normalised[regime] = cleaned
        return normalised

    @staticmethod
    def _resolve_regime(regime: MarketRegime | str) -> MarketRegime:
        if isinstance(regime, MarketRegime):
            return regime
        try:
            return MarketRegime(str(regime).lower())
        except ValueError as exc:  # pragma: no cover - walidacja wejścia
            raise ValueError(f"Unknown regime key: {regime!r}") from exc

