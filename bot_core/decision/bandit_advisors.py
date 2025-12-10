"""Domyślna implementacja doradcy bandytów dla strategii i ryzyka."""

from __future__ import annotations

import logging
import math
import random
from typing import Mapping, MutableMapping

import numpy as np

from bot_core.ai import MarketRegime, ModelScore
from bot_core.decision.bandits import (
    BanditRecommendation,
    StrategyAdvisor,
    _LinUCBArm,
    _ThompsonArm,
)
from bot_core.decision.models import (
    DecisionCandidate,
    DecisionEvaluation,
    ModelSelectionMetadata,
)

_LOGGER = logging.getLogger(__name__)


class DefaultStrategyBanditAdvisor(StrategyAdvisor):
    """Łączy LinUCB i Thompson sampling dla rekomendacji strategii oraz ryzyka."""

    def __init__(self) -> None:
        self._linucb: MutableMapping[tuple[str, str], _LinUCBArm] = {}
        self._thompson: MutableMapping[tuple[str, str], _ThompsonArm] = {}
        self._rng = random.Random(1729)
        self._pending: MutableMapping[int, tuple[tuple[str, str], np.ndarray, float | None]] = {}

    def recommend(
        self,
        candidate: DecisionCandidate,
        *,
        regime: MarketRegime | str,
        model_score: ModelScore | None,
        selection: ModelSelectionMetadata | None,
        cost_bps: float | None,
        net_edge_bps: float,
    ) -> BanditRecommendation:
        regime_value = self._normalise_regime(regime)
        key = (candidate.strategy.lower(), regime_value)
        context = self._build_context(
            candidate,
            regime_value,
            model_score,
            selection,
            cost_bps,
            net_edge_bps,
        )
        arm = self._linucb.setdefault(key, _LinUCBArm(context.size, alpha=0.75))
        score = arm.predict(context)
        risk_score = self._risk_score(
            key,
            candidate,
            regime_value,
            model_score,
            net_edge_bps,
        )
        modes = self._modes_from_scores(score, risk_score)
        position_size = self._position_from_risk(candidate.notional, risk_score)
        self._pending[id(candidate)] = (key, context, self._extract_meta_label(candidate))
        return BanditRecommendation(
            modes=modes, position_size=position_size, risk_score=risk_score
        )

    def observe(
        self,
        candidate: DecisionCandidate,
        evaluation: DecisionEvaluation,
    ) -> None:
        payload = self._pending.pop(id(candidate), None)
        if payload is None:
            return
        key, context, meta_label = payload
        arm = self._linucb.get(key)
        if arm is None:
            return
        reward = self._reward_from_evaluation(evaluation.net_edge_bps, evaluation.accepted)
        arm.update(context, reward)
        thompson = self._thompson.setdefault(key, _ThompsonArm())
        success = meta_label
        if success is None:
            success = 1.0 if evaluation.accepted else 0.0
        else:
            success = max(0.0, min(1.0, success))
            if not evaluation.accepted and success > 0.5:
                success *= 0.75
        thompson.update(success)

    def _reward_from_evaluation(self, net_edge_bps: float | None, accepted: bool) -> float:
        if net_edge_bps is None:
            net_edge_bps = 0.0
        scaled = net_edge_bps / 25.0
        if not accepted:
            scaled *= 0.5
        return float(max(-1.0, min(1.0, scaled)))

    def _position_from_risk(self, notional: float, risk_score: float) -> float:
        multiplier = 0.4 + 0.8 * max(0.0, min(1.0, risk_score))
        return float(max(0.0, min(notional * 1.6, notional * multiplier)))

    def _modes_from_scores(self, score: float, risk_score: float) -> tuple[str, ...]:
        if score >= 1.6 and risk_score >= 0.65:
            return ("live", "aggressive")
        if score >= 1.1 and risk_score >= 0.5:
            return ("live", "balanced")
        if score >= 0.6 and risk_score >= 0.3:
            return ("shadow", "defensive")
        return ("disabled", "monitor")

    def _risk_score(
        self,
        key: tuple[str, str],
        candidate: DecisionCandidate,
        regime: str,
        model_score: ModelScore | None,
        net_edge_bps: float,
    ) -> float:
        thompson = self._thompson.setdefault(key, _ThompsonArm())
        base_mean = thompson.posterior_mean()
        meta_label = self._extract_meta_label(candidate)
        meta_component = base_mean if meta_label is None else float(max(0.0, min(1.0, meta_label)))
        probability = (
            model_score.success_probability
            if model_score is not None and model_score.success_probability is not None
            else candidate.expected_probability
        )
        probability = float(max(0.0, min(1.0, probability)))
        net_component = 0.5 + 0.5 * max(-1.0, min(1.0, net_edge_bps / 20.0))
        regime_bias = {
            MarketRegime.TREND.value: 0.65,
            MarketRegime.MEAN_REVERSION.value: 0.55,
            MarketRegime.DAILY.value: 0.45,
        }.get(regime, 0.5)
        risk = (
            0.35 * base_mean
            + 0.25 * meta_component
            + 0.2 * probability
            + 0.2 * regime_bias
        )
        risk = 0.5 * risk + 0.5 * net_component
        return max(0.0, min(1.0, risk))

    def _build_context(
        self,
        candidate: DecisionCandidate,
        regime: str,
        model_score: ModelScore | None,
        selection: ModelSelectionMetadata | None,
        cost_bps: float | None,
        net_edge_bps: float,
    ) -> np.ndarray:
        probability = (
            model_score.success_probability
            if model_score is not None and model_score.success_probability is not None
            else candidate.expected_probability
        )
        expected_return = (
            model_score.expected_return_bps
            if model_score is not None and model_score.expected_return_bps is not None
            else candidate.expected_return_bps
        )
        net_edge = net_edge_bps
        if math.isfinite(net_edge) is False:
            net_edge = expected_return - (cost_bps or 0.0)
        model_weight = 0.0
        if selection is not None and selection.selected:
            detail = selection.find(selection.selected)
            if detail is not None and detail.weight is not None and detail.weight > 0:
                model_weight = float((detail.effective_score or 0.0) / detail.weight)
        metadata_confidence = self._extract_confidence(candidate)
        meta_label = self._extract_meta_label(candidate)
        regime_vector = self._encode_regime(regime)
        context = np.array(
            [
                1.0,
                probability,
                expected_return / 100.0,
                net_edge / 100.0,
                model_weight,
                metadata_confidence,
                meta_label if meta_label is not None else 0.5,
            ]
            + list(regime_vector),
            dtype=float,
        )
        return np.nan_to_num(context, nan=0.0, posinf=1.0, neginf=-1.0)

    def _encode_regime(self, regime: str) -> tuple[float, float, float]:
        regime = regime or MarketRegime.TREND.value
        return (
            1.0 if regime == MarketRegime.TREND.value else 0.0,
            1.0 if regime == MarketRegime.MEAN_REVERSION.value else 0.0,
            1.0 if regime == MarketRegime.DAILY.value else 0.0,
        )

    def _normalise_regime(self, regime: MarketRegime | str) -> str:
        if isinstance(regime, MarketRegime):
            return regime.value
        try:
            return MarketRegime(regime).value
        except (ValueError, TypeError):
            return MarketRegime.TREND.value

    def _extract_confidence(self, candidate: DecisionCandidate) -> float:
        metadata = candidate.metadata or {}
        if not isinstance(metadata, Mapping):
            return 0.5
        candidates = [
            metadata.get("model_confidence"),
            metadata.get("confidence"),
            metadata.get("score"),
        ]
        model_meta = metadata.get("model_metadata")
        if isinstance(model_meta, Mapping):
            candidates.extend(
                model_meta.get(key)
                for key in ("confidence", "model_score", "quality")
            )
        for value in candidates:
            try:
                confidence = float(value)
            except (TypeError, ValueError):
                continue
            return max(0.0, min(1.0, confidence))
        return 0.5

    def _extract_meta_label(self, candidate: DecisionCandidate) -> float | None:
        metadata = candidate.metadata or {}
        if not isinstance(metadata, Mapping):
            return None
        for key in ("meta_label", "meta_label_score", "meta_probability"):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                label = float(value)
            except (TypeError, ValueError):
                continue
            return max(0.0, min(1.0, label))
        meta_block = metadata.get("meta_labeling")
        if isinstance(meta_block, Mapping):
            for key in ("label", "probability", "score"):
                value = meta_block.get(key)
                if value is None:
                    continue
                try:
                    label = float(value)
                except (TypeError, ValueError):
                    continue
                return max(0.0, min(1.0, label))
        return None


__all__ = ["DefaultStrategyBanditAdvisor"]
