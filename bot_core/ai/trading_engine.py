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
from .trading_opportunity_shadow import (
    OpportunityShadowContext,
    OpportunityShadowRecord,
    OpportunityThresholdConfig,
)
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
        self._classifier_model: SimpleGradientBoostingModel | None = None

    @property
    def artifact(self) -> ModelArtifact | None:
        return self._artifact

    def fit(self, samples: Sequence[OpportunitySnapshot]) -> ModelArtifact:
        if len(samples) < 5:
            raise ValueError("Do treningu potrzeba co najmniej 5 snapshotów")
        self._validate_training_samples(samples)

        matrix = [self._vector_from_snapshot(sample) for sample in samples]
        targets = [self._target(sample) for sample in samples]
        classifier_targets = [self._success_target(edge) for edge in targets]

        model = SimpleGradientBoostingModel(learning_rate=0.12, n_estimators=32, min_samples_leaf=2)
        model.fit_matrix(matrix, _FEATURE_NAMES, targets)
        classifier = SimpleGradientBoostingModel(
            learning_rate=0.1,
            n_estimators=28,
            min_samples_leaf=2,
        )
        classifier.fit_matrix(matrix, _FEATURE_NAMES, classifier_targets)
        predictions = [model.predict(self._features_from_vector(vector)) for vector in matrix]
        probability_predictions = [
            self._clip_probability(classifier.predict(self._features_from_vector(vector)))
            for vector in matrix
        ]

        mae = sum(abs(p - y) for p, y in zip(predictions, targets)) / len(targets)
        mse = sum((p - y) ** 2 for p, y in zip(predictions, targets)) / len(targets)
        rmse = math.sqrt(mse)
        directional_hits = sum(
            1 for p, y in zip(predictions, targets) if (p >= 0 and y >= 0) or (p < 0 and y < 0)
        )
        direction_acc = directional_hits / len(targets)
        classification_hits = sum(
            1
            for predicted, expected in zip(probability_predictions, classifier_targets)
            if (predicted >= 0.5 and expected >= 0.5) or (predicted < 0.5 and expected < 0.5)
        )
        classification_acc = classification_hits / len(classifier_targets)
        brier = sum(
            (predicted - expected) ** 2
            for predicted, expected in zip(probability_predictions, classifier_targets)
        ) / len(classifier_targets)
        target_scale = max(10.0, self._stddev(targets))

        trained_at = datetime.now(timezone.utc)
        version = self._build_version(samples, trained_at)
        metadata: MutableMapping[str, object] = {
            "model_version": version,
            "objective": "maximize_expected_net_edge_bps_after_costs",
            "target_definition": "realized_return_bps - spread_bps - fee_bps - slippage_bps - risk_penalty_bps",
            "classification_target_definition": "1 if target_definition > 0 else 0",
            "decision_source": "model",
            "probability_scale_bps": target_scale,
            "training_scope": "opportunity_ranking_shadow",
            "artifact_schema_version": "opportunity_dual_head_v1",
            "heads": {
                "edge_regressor": {
                    "type": "regression",
                    "prediction_field": "expected_edge_bps",
                    "target_definition": "realized_return_bps - spread_bps - fee_bps - slippage_bps - risk_penalty_bps",
                },
                "success_classifier": {
                    "type": "classification",
                    "prediction_field": "success_probability",
                    "target_definition": "1 if target_definition > 0 else 0",
                },
            },
        }
        model_state: MutableMapping[str, object] = dict(model.to_state())
        model_state["classifier_head_state"] = classifier.to_state()

        artifact = ModelArtifact(
            feature_names=_FEATURE_NAMES,
            model_state=model_state,
            trained_at=trained_at,
            metrics={
                "summary": {
                    "mae_bps": mae,
                    "rmse_bps": rmse,
                    "directional_accuracy": direction_acc,
                    "classifier_accuracy": classification_acc,
                    "classifier_brier_score": brier,
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
        self._classifier_model = classifier
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
        classifier_state = artifact.model_state.get("classifier_head_state")
        classifier: SimpleGradientBoostingModel | None = None
        if isinstance(classifier_state, Mapping):
            loaded_classifier = SimpleGradientBoostingModel()
            loaded_classifier.load_state(classifier_state)
            if loaded_classifier.feature_names:
                classifier = loaded_classifier
        self._artifact = artifact
        self._model = model
        self._classifier_model = classifier
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
        classifier = self._classifier_model if self._classifier_model is not None else None
        prob_scale = self._safe_probability_scale(
            artifact.metadata.get("probability_scale_bps", artifact.target_scale)
        )
        model_version = str(artifact.metadata.get("model_version", "unknown"))

        for candidate in candidates:
            features = self._features_from_vector(self._vector_from_candidate(candidate))
            edge = float(model.predict(features))
            if classifier is not None:
                probability = self._clip_probability(classifier.predict(features))
                probability_method = "model_success_classifier"
                calibration = {
                    "type": "classifier",
                    "target_definition": artifact.metadata.get(
                        "classification_target_definition",
                        "1 if target_definition > 0 else 0",
                    ),
                }
            else:
                probability = self._edge_to_probability(edge=edge, scale=prob_scale)
                probability_method = "heuristic_sigmoid_scaled_edge_fallback"
                calibration = {
                    "type": "heuristic_fallback",
                    "scale_bps": prob_scale,
                }
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
                        "probability_method": probability_method,
                        "confidence_method": "distance_from_probability_midpoint",
                        "calibration": calibration,
                    },
                )
            )

        ordered = sorted(decisions, key=lambda item: item.expected_edge_bps, reverse=True)
        return [
            OpportunityDecision(**{**asdict(entry), "rank": idx})
            for idx, entry in enumerate(ordered, start=1)
        ]

    @classmethod
    def build_shadow_records(
        cls,
        decisions: Sequence[OpportunityDecision],
        *,
        decision_timestamp: datetime | None = None,
        threshold_config: OpportunityThresholdConfig | None = None,
        snapshot: Mapping[str, object] | None = None,
        context: OpportunityShadowContext | None = None,
    ) -> list[OpportunityShadowRecord]:
        """Buduje audytowalne rekordy shadow bez ingerencji w execution path."""

        resolved_timestamp = decision_timestamp or datetime.now(timezone.utc)
        if resolved_timestamp.tzinfo is None:
            resolved_timestamp = resolved_timestamp.replace(tzinfo=timezone.utc)
        resolved_timestamp = resolved_timestamp.astimezone(timezone.utc)
        resolved_thresholds = threshold_config or OpportunityThresholdConfig()
        resolved_context = context or OpportunityShadowContext()

        records: list[OpportunityShadowRecord] = []
        for entry in decisions:
            cls._validate_symbol(entry.symbol, context=f"decision#{entry.rank}")
            record_key = OpportunityShadowRecord.build_record_key(
                symbol=entry.symbol,
                decision_timestamp=resolved_timestamp,
                model_version=entry.model_version,
                rank=entry.rank,
            )
            records.append(
                OpportunityShadowRecord(
                    record_key=record_key,
                    symbol=entry.symbol,
                    decision_timestamp=resolved_timestamp,
                    model_version=entry.model_version,
                    decision_source=entry.decision_source,
                    expected_edge_bps=float(entry.expected_edge_bps),
                    success_probability=float(entry.success_probability),
                    confidence=float(entry.confidence),
                    proposed_direction=entry.proposed_direction,
                    accepted=bool(entry.accepted),
                    rejection_reason=entry.rejection_reason,
                    rank=int(entry.rank),
                    provenance=cls._deep_copy_payload(entry.provenance),
                    threshold_config=resolved_thresholds,
                    snapshot=cls._deep_copy_payload(snapshot or {}),
                    context=OpportunityShadowContext(
                        run_id=resolved_context.run_id,
                        environment=resolved_context.environment,
                        notes=cls._deep_copy_payload(resolved_context.notes),
                    ),
                )
            )
        return records

    @classmethod
    def _deep_copy_payload(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): cls._deep_copy_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._deep_copy_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._deep_copy_payload(item) for item in value)
        return value

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
    def _success_target(edge_bps: float) -> float:
        return 1.0 if float(edge_bps) > 0.0 else 0.0

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

    @classmethod
    def _clip_probability(cls, value: float) -> float:
        if not cls._is_finite(value):
            return 0.5
        return max(0.0, min(1.0, float(value)))


__all__ = [
    "OpportunityCandidate",
    "OpportunityDecision",
    "OpportunitySnapshot",
    "TradingOpportunityAI",
    "FilesystemModelRepository",
]
