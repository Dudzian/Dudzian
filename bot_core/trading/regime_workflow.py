"""Workflow orchestrating regime-aware strategy selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from collections.abc import Iterable
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
from bot_core.trading.strategy_aliasing import (
    MIGRATION_FALLBACK_SUFFIX,
    StrategyAliasResolver,
    strategy_key_aliases,
    strategy_name_candidates,
)


@dataclass(frozen=True)
class RegimeSwitchDecision:
    """Result of a workflow evaluation for a single timestep."""

    regime: MarketRegime
    assessment: MarketRegimeAssessment
    summary: RegimeSummary | None
    weights: Dict[str, float]
    parameters: TradingParameters
    timestamp: pd.Timestamp
    strategy_metadata: Mapping[str, Mapping[str, object]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    license_tiers: tuple[str, ...] = ()
    risk_classes: tuple[str, ...] = ()
    required_data: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def metadata_summary(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                "license_tiers": self.license_tiers,
                "risk_classes": self.risk_classes,
                "required_data": self.required_data,
                "capabilities": self.capabilities,
                "tags": self.tags,
            }
        )


@dataclass(frozen=True)
class RegimeSwitchActivation(RegimeSwitchDecision):
    """Decision enriched with metadata about fallback execution."""

    used_fallback: bool = False
    missing_data: tuple[str, ...] = ()
    blocked_reason: str | None = None


class RegimeSwitchWorkflow:
    """High-level controller combining classifier outputs with strategy plugins."""

    _STRATEGY_SUFFIXES: tuple[str, ...] = ("_probing", MIGRATION_FALLBACK_SUFFIX)
    _STRATEGY_ALIAS_MAP: Mapping[str, str] = MappingProxyType(
        {
            "intraday_breakout": "day_trading",
        }
    )
    _ALIAS_RESOLVER: StrategyAliasResolver | None = None

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
        strategy_alias_map: Mapping[str, str] | None = None,
        strategy_alias_suffixes: Iterable[str] | None = None,
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
        self._strategy_metadata_cache: Dict[
            str, tuple[Mapping[str, object], str | None] | None
        ] = {}

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
        metadata = self._collect_strategy_metadata(weights)
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
            strategy_metadata=metadata["strategies"],
            license_tiers=metadata["license_tiers"],
            risk_classes=metadata["risk_classes"],
            required_data=metadata["required_data"],
            capabilities=metadata["capabilities"],
            tags=metadata["tags"],
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

    def activate(
        self,
        market_data: pd.DataFrame,
        *,
        available_data: Iterable[str] = (),
        symbol: str | None = None,
        base_parameters: TradingParameters | None = None,
        parameter_overrides: Mapping[MarketRegime, Mapping[str, float | int]] | None = None,
    ) -> RegimeSwitchActivation:
        """Evaluate the current regime and annotate the result with missing data info."""

        parameters = base_parameters or TradingParameters()
        decision = self.decide(
            market_data,
            parameters,
            symbol=symbol,
            parameter_overrides=parameter_overrides,
        )
        available = {
            str(item).strip().lower()
            for item in available_data
            if str(item).strip()
        }
        missing = self._compute_missing_data(decision.required_data, available)
        blocked_reason = "missing_data" if missing else None
        return RegimeSwitchActivation(
            regime=decision.regime,
            assessment=decision.assessment,
            summary=decision.summary,
            weights=decision.weights,
            parameters=decision.parameters,
            timestamp=decision.timestamp,
            strategy_metadata=decision.strategy_metadata,
            license_tiers=decision.license_tiers,
            risk_classes=decision.risk_classes,
            required_data=decision.required_data,
            capabilities=decision.capabilities,
            tags=decision.tags,
            used_fallback=bool(missing),
            missing_data=missing,
            blocked_reason=blocked_reason,
        )

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

    def _collect_strategy_metadata(
        self, weights: Mapping[str, float]
    ) -> Mapping[str, tuple[str, ...] | Mapping[str, Mapping[str, object]]]:
        strategies: Dict[str, Mapping[str, object]] = {}
        license_tiers: list[str] = []
        risk_classes: list[str] = []
        required_data: list[str] = []
        capabilities: list[str] = []
        tags: list[str] = []

        cache = self._strategy_metadata_cache
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
                    metadata = self._catalog.metadata_for(candidate)
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
            "license_tiers": tuple(license_tiers),
            "risk_classes": tuple(risk_classes),
            "required_data": tuple(required_data),
            "capabilities": tuple(capabilities),
            "tags": tuple(tags),
        }

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

    def _compute_missing_data(
        self, required_data: Iterable[str], available: set[str]
    ) -> tuple[str, ...]:
        missing = []
        for item in required_data:
            text = str(item).strip()
            if not text:
                continue
            if text.lower() not in available:
                missing.append(text)
        return tuple(dict.fromkeys(missing))

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
                "trend_following": 0.35,
                "day_trading": 0.15,
                "mean_reversion": 0.1,
                "arbitrage": 0.1,
                "volatility_target": 0.2,
                "grid_trading": 0.05,
                "options_income": 0.05,
            },
            MarketRegime.DAILY: {
                "day_trading": 0.4,
                "scalping": 0.2,
                "trend_following": 0.1,
                "volatility_target": 0.1,
                "grid_trading": 0.1,
                "arbitrage": 0.05,
                "statistical_arbitrage": 0.05,
            },
            MarketRegime.MEAN_REVERSION: {
                "mean_reversion": 0.35,
                "statistical_arbitrage": 0.25,
                "arbitrage": 0.15,
                "grid_trading": 0.1,
                "options_income": 0.1,
                "scalping": 0.05,
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

