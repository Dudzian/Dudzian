"""Integracja Decision Engine z asynchronicznym `AIManagerem` z KryptoLowca."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd

from bot_core.ai import ModelScore
from bot_core.decision.models import DecisionCandidate


@dataclass(slots=True)
class AIManagerDecisionConnector:
    """Buduje kandydatów Decision Engine na podstawie prognoz `AIManagera`."""

    ai_manager: Any
    strategy: str
    risk_profile: str
    default_notional: float
    action: str = "enter"
    min_probability: float = 0.0
    cost_bps_override: float | None = None
    threshold_bps: float | None = None

    def __post_init__(self) -> None:
        self.strategy = str(self.strategy)
        self.risk_profile = str(self.risk_profile)
        self.action = str(self.action)
        self.default_notional = float(self.default_notional)
        if self.default_notional <= 0:
            raise ValueError("default_notional musi być dodatnie")
        self.min_probability = max(0.0, min(1.0, float(self.min_probability)))
        if self.threshold_bps is None:
            self.threshold_bps = float(getattr(self.ai_manager, "ai_threshold_bps", 0.0) or 0.0)

    async def generate_candidates(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        *,
        model_types: Sequence[str] | None = None,
        feature_columns: Sequence[str] | None = None,
        notional: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[DecisionCandidate]:
        """Zwraca kandydatów Decision Engine na podstawie prognoz `AIManagera`."""

        symbol_key = str(symbol)
        feature_cols = list(feature_columns or ["open", "high", "low", "close", "volume"])
        prediction_series = await self.ai_manager.predict_series(  # type: ignore[func-returns-value]
            symbol_key,
            market_data,
            model_types=model_types,
            feature_cols=feature_cols,
        )
        if not isinstance(prediction_series, pd.Series):
            prediction_series = pd.Series(prediction_series, index=market_data.index[-len(prediction_series) :])
        candidates: list[DecisionCandidate] = []
        base_metadata = dict(metadata or {})
        for ts, signal in prediction_series.items():
            score = self._score_from_signal(float(signal))
            if score.success_probability < self.min_probability:
                continue
            enriched_metadata = self._build_metadata(base_metadata, timestamp=ts, signal=float(signal), score=score)
            candidate_notional = float(notional or self.default_notional)
            candidate = DecisionCandidate(
                strategy=self.strategy,
                action=self.action,
                risk_profile=self.risk_profile,
                symbol=symbol_key,
                notional=candidate_notional,
                expected_return_bps=score.expected_return_bps,
                expected_probability=score.success_probability,
                cost_bps_override=self.cost_bps_override,
                metadata=enriched_metadata,
            )
            candidates.append(candidate)
        return candidates

    def candidate_from_signal(
        self,
        *,
        symbol: str,
        signal: float,
        timestamp: object | None = None,
        notional: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> DecisionCandidate | None:
        """Buduje pojedynczego kandydata dla zadanego sygnału modelu."""

        score = self._score_from_signal(float(signal))
        if score.success_probability < self.min_probability:
            return None
        enriched_metadata = self._build_metadata(
            metadata or {},
            timestamp=timestamp,
            signal=float(signal),
            score=score,
        )
        candidate_notional = float(notional or self.default_notional)
        return DecisionCandidate(
            strategy=self.strategy,
            action=self.action,
            risk_profile=self.risk_profile,
            symbol=str(symbol),
            notional=candidate_notional,
            expected_return_bps=score.expected_return_bps,
            expected_probability=score.success_probability,
            cost_bps_override=self.cost_bps_override,
            metadata=enriched_metadata,
        )

    def _score_from_signal(self, signal: float) -> ModelScore:
        expected_return_bps = signal * 10_000.0
        threshold = abs(float(self.threshold_bps or 0.0))
        if threshold <= 0:
            normalized = min(1.0, max(0.0, abs(signal)))
        else:
            normalized = min(1.0, max(0.0, abs(expected_return_bps) / threshold))
        success_probability = 0.5 + 0.5 * normalized
        if signal == 0:
            success_probability = max(success_probability, 0.5)
        return ModelScore(
            expected_return_bps=expected_return_bps,
            success_probability=max(0.0, min(1.0, success_probability)),
        )

    def _build_metadata(
        self,
        base_metadata: Mapping[str, object],
        *,
        timestamp: object,
        signal: float,
        score: ModelScore,
    ) -> Mapping[str, object]:
        metadata: MutableMapping[str, object] = dict(base_metadata)
        ai_section: MutableMapping[str, object] = {
            "signal": signal,
            "expected_return_bps": score.expected_return_bps,
            "success_probability": score.success_probability,
        }
        metadata.setdefault("ai_manager", {}).update(ai_section)  # type: ignore[arg-type]
        metadata.setdefault("decision_engine", {})  # type: ignore[arg-type]
        decision_section = metadata["decision_engine"]  # type: ignore[index]
        if isinstance(decision_section, MutableMapping):
            decision_section.setdefault("source", "ai_manager")
            decision_section.setdefault("generated_at", timestamp)
        return metadata


__all__ = ["AIManagerDecisionConnector"]
