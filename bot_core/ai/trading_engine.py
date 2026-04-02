"""Osobny moduł AI do scoringu i rankingu okazji tradingowych.

Ten komponent działa obok istniejącego Decision Makera w trybie assist/shadow.
Dostarcza pełny pionowy slice: featuryzacja -> trening -> artefakt -> inference -> ranking.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Sequence

from .models import ModelArtifact
from .repository import FilesystemModelRepository, ModelRepository
from .training import SimpleGradientBoostingModel

_FEATURE_NAMES: tuple[str, ...] = (
    "signal_strength",
    "momentum_5m",
    "volatility_30m",
    "spread_bps",
    "fee_bps",
    "slippage_bps",
    "liquidity_score",
    "risk_penalty_bps",
)


@dataclass(slots=True)
class OpportunitySnapshot:
    """Historyczny snapshot okazji używany do treningu."""

    symbol: str
    signal_strength: float
    momentum_5m: float
    volatility_30m: float
    spread_bps: float
    fee_bps: float
    slippage_bps: float
    liquidity_score: float
    risk_penalty_bps: float
    realized_return_bps: float
    as_of: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class OpportunityCandidate:
    """Bieżący kandydat do oceny przez model."""

    symbol: str
    signal_strength: float
    momentum_5m: float
    volatility_30m: float
    spread_bps: float
    fee_bps: float
    slippage_bps: float
    liquidity_score: float
    risk_penalty_bps: float
    direction_hint: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OpportunityDecision:
    """Audytowalny kontrakt wyniku inferencji/rankingu."""

    symbol: str
    decision_source: str
    model_version: str
    expected_edge_bps: float
    success_probability: float
    confidence: float
    proposed_direction: str
    accepted: bool
    rejection_reason: str | None
    rank: int
    provenance: Mapping[str, object]


class TradingOpportunityAI:
    """Lekki silnik AI do rankingu okazji tradingowych."""

    def __init__(self, repository: ModelRepository | None = None) -> None:
        self._repository = repository
        self._artifact: ModelArtifact | None = None
        self._model: SimpleGradientBoostingModel | None = None

    @property
    def artifact(self) -> ModelArtifact | None:
        return self._artifact

    def fit(self, samples: Sequence[OpportunitySnapshot]) -> ModelArtifact:
        if len(samples) < 5:
            raise ValueError("Do treningu potrzeba co najmniej 5 snapshotów")
        self._validate_training_samples(samples)

        matrix = [self._vector_from_snapshot(sample) for sample in samples]
        targets = [self._target(sample) for sample in samples]

        model = SimpleGradientBoostingModel(learning_rate=0.12, n_estimators=32, min_samples_leaf=2)
        model.fit_matrix(matrix, _FEATURE_NAMES, targets)
        predictions = [model.predict(self._features_from_vector(vector)) for vector in matrix]

        mae = sum(abs(p - y) for p, y in zip(predictions, targets)) / len(targets)
        mse = sum((p - y) ** 2 for p, y in zip(predictions, targets)) / len(targets)
        rmse = math.sqrt(mse)
        directional_hits = sum(
            1 for p, y in zip(predictions, targets) if (p >= 0 and y >= 0) or (p < 0 and y < 0)
        )
        direction_acc = directional_hits / len(targets)
        target_scale = max(10.0, self._stddev(targets))

        trained_at = datetime.now(timezone.utc)
        version = self._build_version(samples, trained_at)
        metadata: MutableMapping[str, object] = {
            "model_version": version,
            "objective": "maximize_expected_net_edge_bps_after_costs",
            "target_definition": "realized_return_bps - spread_bps - fee_bps - slippage_bps - risk_penalty_bps",
            "decision_source": "model",
            "probability_scale_bps": target_scale,
            "training_scope": "opportunity_ranking_shadow",
        }

        artifact = ModelArtifact(
            feature_names=_FEATURE_NAMES,
            model_state=model.to_state(),
            trained_at=trained_at,
            metrics={
                "summary": {
                    "mae_bps": mae,
                    "rmse_bps": rmse,
                    "directional_accuracy": direction_acc,
                }
            },
            metadata=metadata,
            target_scale=target_scale,
            training_rows=len(samples),
            validation_rows=0,
            test_rows=0,
            feature_scalers=model.feature_scalers,
            backend="builtin",
        )
        self._artifact = artifact
        self._model = model
        return artifact

    def save_model(self, *, version: str | None = None, activate: bool = False) -> str:
        if self._artifact is None:
            raise RuntimeError("Brak wytrenowanego artefaktu do zapisu")
        if self._repository is None:
            raise RuntimeError("Brak skonfigurowanego repozytorium modeli")

        resolved_version = version or str(self._artifact.metadata.get("model_version", "v1"))
        path = self._repository.publish(
            self._artifact,
            version=resolved_version,
            filename=f"trading-opportunity-{resolved_version}.json",
            aliases=("latest", "opportunity_ranker"),
            activate=activate,
        )
        return str(path)

    def load_model(self, reference: str = "latest") -> ModelArtifact:
        if self._repository is None:
            raise RuntimeError("Brak skonfigurowanego repozytorium modeli")
        artifact = self._repository.load_model(reference)
        model = artifact.build_model()
        if not isinstance(model, SimpleGradientBoostingModel):
            raise TypeError("TradingOpportunityAI wymaga backendu builtin")
        self._artifact = artifact
        self._model = model
        return artifact

    def rank(
        self,
        candidates: Sequence[OpportunityCandidate],
        *,
        min_expected_edge_bps: float = 0.0,
        min_probability: float = 0.50,
    ) -> list[OpportunityDecision]:
        self._validate_rank_thresholds(
            min_expected_edge_bps=min_expected_edge_bps,
            min_probability=min_probability,
        )
        self._validate_candidates(candidates)
        model, artifact = self._require_model()
        decisions: list[OpportunityDecision] = []
        prob_scale = self._safe_probability_scale(
            artifact.metadata.get("probability_scale_bps", artifact.target_scale)
        )
        model_version = str(artifact.metadata.get("model_version", "unknown"))

        for candidate in candidates:
            features = self._features_from_vector(self._vector_from_candidate(candidate))
            edge = float(model.predict(features))
            probability = self._edge_to_probability(edge=edge, scale=prob_scale)
            confidence = abs(probability - 0.5) * 2.0

            accepted = edge >= min_expected_edge_bps and probability >= min_probability
            if accepted:
                direction = self._resolve_direction(candidate)
                rejection_reason = None
            else:
                direction = "skip"
                if edge < min_expected_edge_bps:
                    rejection_reason = "edge_below_threshold"
                else:
                    rejection_reason = "probability_below_threshold"

            decisions.append(
                OpportunityDecision(
                    symbol=candidate.symbol,
                    decision_source="model",
                    model_version=model_version,
                    expected_edge_bps=edge,
                    success_probability=probability,
                    confidence=confidence,
                    proposed_direction=direction,
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    rank=0,
                    provenance={
                        "objective": artifact.metadata.get("objective", ""),
                        "target_definition": artifact.metadata.get("target_definition", ""),
                        "trained_at": artifact.trained_at.isoformat(),
                        "feature_names": list(artifact.feature_names),
                        "probability_method": "heuristic_sigmoid_scaled_edge",
                        "confidence_method": "distance_from_probability_midpoint",
                        "calibration": {
                            "type": "heuristic",
                            "scale_bps": prob_scale,
                        },
                    },
                )
            )

        ordered = sorted(decisions, key=lambda item: item.expected_edge_bps, reverse=True)
        return [
            OpportunityDecision(**{**asdict(entry), "rank": idx})
            for idx, entry in enumerate(ordered, start=1)
        ]

    def _require_model(self) -> tuple[SimpleGradientBoostingModel, ModelArtifact]:
        if self._model is None or self._artifact is None:
            raise RuntimeError("Model nie został wytrenowany ani załadowany")
        return self._model, self._artifact

    @staticmethod
    def _resolve_direction(candidate: OpportunityCandidate) -> str:
        if candidate.direction_hint in {"long", "short", "hold"}:
            return str(candidate.direction_hint)
        return "long" if candidate.signal_strength >= 0 else "short"

    @staticmethod
    def _target(snapshot: OpportunitySnapshot) -> float:
        return float(
            snapshot.realized_return_bps
            - snapshot.spread_bps
            - snapshot.fee_bps
            - snapshot.slippage_bps
            - snapshot.risk_penalty_bps
        )

    @staticmethod
    def _vector_from_snapshot(snapshot: OpportunitySnapshot) -> list[float]:
        return [float(value) for value in TradingOpportunityAI._raw_vector_from_snapshot(snapshot)]

    @staticmethod
    def _vector_from_candidate(candidate: OpportunityCandidate) -> list[float]:
        return [float(value) for value in TradingOpportunityAI._raw_vector_from_candidate(candidate)]

    @staticmethod
    def _raw_vector_from_snapshot(snapshot: OpportunitySnapshot) -> list[object]:
        return [
            snapshot.signal_strength,
            snapshot.momentum_5m,
            snapshot.volatility_30m,
            snapshot.spread_bps,
            snapshot.fee_bps,
            snapshot.slippage_bps,
            snapshot.liquidity_score,
            snapshot.risk_penalty_bps,
        ]

    @staticmethod
    def _raw_vector_from_candidate(candidate: OpportunityCandidate) -> list[object]:
        return [
            candidate.signal_strength,
            candidate.momentum_5m,
            candidate.volatility_30m,
            candidate.spread_bps,
            candidate.fee_bps,
            candidate.slippage_bps,
            candidate.liquidity_score,
            candidate.risk_penalty_bps,
        ]

    @staticmethod
    def _features_from_vector(vector: Sequence[float]) -> Mapping[str, float]:
        return {name: float(value) for name, value in zip(_FEATURE_NAMES, vector)}

    @staticmethod
    def _stddev(values: Sequence[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _build_version(samples: Sequence[OpportunitySnapshot], trained_at: datetime) -> str:
        payload = f"{trained_at.isoformat()}:{len(samples)}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"opp-{trained_at.strftime('%Y%m%d%H%M%S')}-{digest}"

    @staticmethod
    def _is_finite(value: object) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(numeric)

    @classmethod
    def _validate_symbol(cls, symbol: str, *, context: str) -> None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"Nieprawidłowy symbol ({context})")

    @classmethod
    def _validate_feature_vector(
        cls, values: Sequence[object], *, context: str, item_index: int
    ) -> None:
        for feature_name, value in zip(_FEATURE_NAMES, values):
            if not cls._is_finite(value):
                raise ValueError(
                    f"Niefinitywna cecha {feature_name} ({context}#{item_index})"
                )

    @classmethod
    def _validate_training_samples(cls, samples: Sequence[OpportunitySnapshot]) -> None:
        for index, sample in enumerate(samples):
            cls._validate_symbol(sample.symbol, context=f"snapshot#{index}")
            vector = cls._raw_vector_from_snapshot(sample)
            cls._validate_feature_vector(vector, context="snapshot", item_index=index)
            try:
                target = cls._target(sample)
            except (TypeError, ValueError):
                raise ValueError(f"Niefinitywny target (snapshot#{index})") from None
            if not cls._is_finite(target):
                raise ValueError(f"Niefinitywny target (snapshot#{index})")

    @classmethod
    def _validate_candidates(cls, candidates: Sequence[OpportunityCandidate]) -> None:
        for index, candidate in enumerate(candidates):
            cls._validate_symbol(candidate.symbol, context=f"candidate#{index}")
            vector = cls._raw_vector_from_candidate(candidate)
            cls._validate_feature_vector(vector, context="candidate", item_index=index)

    @classmethod
    def _validate_rank_thresholds(
        cls, *, min_expected_edge_bps: float, min_probability: float
    ) -> None:
        if not cls._is_finite(min_expected_edge_bps):
            raise ValueError("Nieprawidłowy min_expected_edge_bps")
        if not cls._is_finite(min_probability):
            raise ValueError("Nieprawidłowy min_probability")
        if not 0.0 <= float(min_probability) <= 1.0:
            raise ValueError("min_probability poza zakresem [0, 1]")

    @classmethod
    def _safe_probability_scale(cls, scale_value: object) -> float:
        try:
            scale = float(scale_value)
        except (TypeError, ValueError):
            return 10.0
        if not cls._is_finite(scale) or scale <= 0.0:
            return 10.0
        return max(scale, 1e-6)

    @classmethod
    def _edge_to_probability(cls, *, edge: float, scale: float) -> float:
        if not cls._is_finite(edge):
            return 0.5
        safe_scale = cls._safe_probability_scale(scale)
        z = float(edge) / safe_scale
        if z >= 60.0:
            return 1.0
        if z <= -60.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))


__all__ = [
    "OpportunityCandidate",
    "OpportunityDecision",
    "OpportunitySnapshot",
    "TradingOpportunityAI",
    "FilesystemModelRepository",
]
