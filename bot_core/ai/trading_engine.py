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

_RAW_FEATURE_NAMES: tuple[str, ...] = (
    "signal_strength",
    "momentum_5m",
    "volatility_30m",
    "spread_bps",
    "fee_bps",
    "slippage_bps",
    "liquidity_score",
    "risk_penalty_bps",
)
_REGIME_FEATURE_NAMES: tuple[str, ...] = (
    "regime_trend_score",
    "regime_volatility_score",
    "regime_liquidity_score",
)
_FEATURE_NAMES: tuple[str, ...] = _RAW_FEATURE_NAMES + _REGIME_FEATURE_NAMES
_FEATURE_SPEC_VERSION = "opportunity_features_v1"


@dataclass(slots=True, frozen=True)
class MarketRegimeSnapshot:
    trend_regime: str
    volatility_regime: str
    liquidity_regime: str

    @classmethod
    def from_inputs(
        cls, *, momentum_5m: float, volatility_30m: float, liquidity_score: float
    ) -> "MarketRegimeSnapshot":
        if momentum_5m > 0.15:
            trend = "uptrend"
        elif momentum_5m < -0.15:
            trend = "downtrend"
        else:
            trend = "range"

        if volatility_30m >= 0.45:
            volatility = "high"
        elif volatility_30m >= 0.25:
            volatility = "medium"
        else:
            volatility = "low"

        if liquidity_score >= 0.75:
            liquidity = "deep"
        elif liquidity_score >= 0.40:
            liquidity = "normal"
        else:
            liquidity = "thin"

        return cls(
            trend_regime=trend,
            volatility_regime=volatility,
            liquidity_regime=liquidity,
        )

    def to_feature_values(self) -> Mapping[str, float]:
        trend_map = {"downtrend": -1.0, "range": 0.0, "uptrend": 1.0}
        volatility_map = {"low": 0.0, "medium": 1.0, "high": 2.0}
        liquidity_map = {"thin": 0.0, "normal": 1.0, "deep": 2.0}
        return {
            "regime_trend_score": trend_map[self.trend_regime],
            "regime_volatility_score": volatility_map[self.volatility_regime],
            "regime_liquidity_score": liquidity_map[self.liquidity_regime],
        }

    def to_mapping(self) -> Mapping[str, str]:
        return {
            "trend_regime": self.trend_regime,
            "volatility_regime": self.volatility_regime,
            "liquidity_regime": self.liquidity_regime,
        }


@dataclass(slots=True, frozen=True)
class OpportunityFeaturePayload:
    spec_version: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    regime: MarketRegimeSnapshot
    freshness_seconds: float | None
    provenance: Mapping[str, object]


class _OpportunityFeaturePipeline:
    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        return _FEATURE_NAMES

    @classmethod
    def feature_spec_version(cls) -> str:
        return _FEATURE_SPEC_VERSION

    @classmethod
    def from_snapshot(cls, snapshot: "OpportunitySnapshot") -> OpportunityFeaturePayload:
        raw_values = TradingOpportunityAI._raw_vector_from_snapshot(snapshot)
        raw_mapping = {name: value for name, value in zip(_RAW_FEATURE_NAMES, raw_values)}
        freshness = cls._freshness_seconds(snapshot.as_of)
        return cls._build_payload(raw_mapping, freshness_seconds=freshness, context="snapshot")

    @classmethod
    def from_candidate(cls, candidate: "OpportunityCandidate") -> OpportunityFeaturePayload:
        raw_values = TradingOpportunityAI._raw_vector_from_candidate(candidate)
        raw_mapping = {name: value for name, value in zip(_RAW_FEATURE_NAMES, raw_values)}
        as_of = candidate.metadata.get("as_of") if isinstance(candidate.metadata, Mapping) else None
        freshness = cls._freshness_seconds(as_of)
        return cls._build_payload(raw_mapping, freshness_seconds=freshness, context="candidate")

    @classmethod
    def _build_payload(
        cls,
        raw_mapping: Mapping[str, object],
        *,
        freshness_seconds: float | None,
        context: str,
    ) -> OpportunityFeaturePayload:
        missing = [name for name in _RAW_FEATURE_NAMES if name not in raw_mapping]
        if missing:
            raise ValueError(f"Brak wymaganych cech ({context}): {', '.join(missing)}")
        numeric_raw: dict[str, float] = {}
        for feature_name in _RAW_FEATURE_NAMES:
            value = raw_mapping.get(feature_name)
            if not TradingOpportunityAI._is_finite(value):
                raise ValueError(f"Niefinitywna cecha {feature_name} ({context})")
            numeric_raw[feature_name] = float(value)

        regime = MarketRegimeSnapshot.from_inputs(
            momentum_5m=numeric_raw["momentum_5m"],
            volatility_30m=numeric_raw["volatility_30m"],
            liquidity_score=numeric_raw["liquidity_score"],
        )
        regime_values = regime.to_feature_values()
        values = tuple(float({**numeric_raw, **regime_values}[name]) for name in _FEATURE_NAMES)
        return OpportunityFeaturePayload(
            spec_version=_FEATURE_SPEC_VERSION,
            feature_names=_FEATURE_NAMES,
            feature_values=values,
            regime=regime,
            freshness_seconds=freshness_seconds,
            provenance={
                "pipeline": "opportunity_feature_pipeline",
                "feature_spec_version": _FEATURE_SPEC_VERSION,
                "freshness_seconds": freshness_seconds,
            },
        )

    @staticmethod
    def _freshness_seconds(value: object) -> float | None:
        if not isinstance(value, datetime):
            return None
        as_of = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        freshness = (datetime.now(timezone.utc) - as_of.astimezone(timezone.utc)).total_seconds()
        return max(0.0, freshness)


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


@dataclass(slots=True, frozen=True)
class _PlattSuccessCalibrator:
    """Lekki kalibrator Platta dla prawdopodobieństw sukcesu."""

    slope: float
    intercept: float

    def apply(self, raw_probability: float) -> float:
        clipped = TradingOpportunityAI._clip_probability(raw_probability)
        logit_input = min(max(clipped, 1e-6), 1.0 - 1e-6)
        logit = math.log(logit_input / (1.0 - logit_input))
        z = self.slope * logit + self.intercept
        if z >= 60.0:
            return 1.0
        if z <= -60.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def to_mapping(self) -> Mapping[str, float]:
        return {"method": "platt_scaling", "slope": float(self.slope), "intercept": float(self.intercept)}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "_PlattSuccessCalibrator | None":
        if str(payload.get("method", "")).strip() != "platt_scaling":
            return None
        slope = payload.get("slope")
        intercept = payload.get("intercept")
        if not TradingOpportunityAI._is_finite(slope) or not TradingOpportunityAI._is_finite(intercept):
            return None
        return cls(slope=float(slope), intercept=float(intercept))

    @classmethod
    def fit(
        cls,
        *,
        raw_probabilities: Sequence[float],
        targets: Sequence[float],
        iterations: int = 200,
        learning_rate: float = 0.05,
    ) -> "_PlattSuccessCalibrator | None":
        if len(raw_probabilities) != len(targets) or len(raw_probabilities) < 4:
            return None
        positives = sum(1 for value in targets if float(value) >= 0.5)
        negatives = len(targets) - positives
        if positives == 0 or negatives == 0:
            return None

        logits: list[float] = []
        ys: list[float] = []
        for raw, target in zip(raw_probabilities, targets):
            clipped = min(max(TradingOpportunityAI._clip_probability(raw), 1e-6), 1.0 - 1e-6)
            logits.append(math.log(clipped / (1.0 - clipped)))
            ys.append(1.0 if float(target) >= 0.5 else 0.0)

        slope = 1.0
        intercept = 0.0
        n = len(logits)
        for _ in range(max(1, iterations)):
            grad_slope = 0.0
            grad_intercept = 0.0
            for x, y in zip(logits, ys):
                z = slope * x + intercept
                if z >= 60.0:
                    pred = 1.0
                elif z <= -60.0:
                    pred = 0.0
                else:
                    pred = 1.0 / (1.0 + math.exp(-z))
                err = pred - y
                grad_slope += err * x
                grad_intercept += err
            slope -= learning_rate * (grad_slope / n)
            intercept -= learning_rate * (grad_intercept / n)

        if not TradingOpportunityAI._is_finite(slope) or not TradingOpportunityAI._is_finite(intercept):
            return None
        return cls(slope=float(slope), intercept=float(intercept))


class TradingOpportunityAI:
    """Lekki silnik AI do rankingu okazji tradingowych."""

    def __init__(self, repository: ModelRepository | None = None) -> None:
        self._repository = repository
        self._artifact: ModelArtifact | None = None
        self._model: SimpleGradientBoostingModel | None = None
        self._classifier_model: SimpleGradientBoostingModel | None = None
        self._success_calibrator: _PlattSuccessCalibrator | None = None

    @property
    def artifact(self) -> ModelArtifact | None:
        return self._artifact

    def fit(self, samples: Sequence[OpportunitySnapshot]) -> ModelArtifact:
        if len(samples) < 5:
            raise ValueError("Do treningu potrzeba co najmniej 5 snapshotów")
        self._validate_training_samples(samples)

        ordered_samples = list(samples)
        all_targets = [self._target(sample) for sample in ordered_samples]
        all_classifier_targets = [self._success_target(edge) for edge in all_targets]
        positive_indices = [idx for idx, value in enumerate(all_classifier_targets) if value >= 0.5]
        negative_indices = [idx for idx, value in enumerate(all_classifier_targets) if value < 0.5]

        validation_indices: set[int] = set()
        if positive_indices and negative_indices:
            positive_count = max(1, int(len(positive_indices) * 0.2))
            negative_count = max(1, int(len(negative_indices) * 0.2))
            validation_indices.update(positive_indices[-positive_count:])
            validation_indices.update(negative_indices[-negative_count:])
        else:
            validation_size = max(1, int(len(ordered_samples) * 0.2))
            validation_size = min(validation_size, len(ordered_samples) - 1)
            validation_indices.update(range(len(ordered_samples) - validation_size, len(ordered_samples)))

        validation_samples = [sample for idx, sample in enumerate(ordered_samples) if idx in validation_indices]
        train_samples = [sample for idx, sample in enumerate(ordered_samples) if idx not in validation_indices]

        train_matrix = [self._vector_from_snapshot(sample) for sample in train_samples]
        train_targets = [self._target(sample) for sample in train_samples]
        train_classifier_targets = [self._success_target(edge) for edge in train_targets]

        model = SimpleGradientBoostingModel(learning_rate=0.12, n_estimators=32, min_samples_leaf=2)
        model.fit_matrix(train_matrix, _FEATURE_NAMES, train_targets)
        classifier = SimpleGradientBoostingModel(
            learning_rate=0.1,
            n_estimators=28,
            min_samples_leaf=2,
        )
        classifier.fit_matrix(train_matrix, _FEATURE_NAMES, train_classifier_targets)

        matrix = [self._vector_from_snapshot(sample) for sample in ordered_samples]
        targets = [self._target(sample) for sample in ordered_samples]
        classifier_targets = [self._success_target(edge) for edge in targets]
        predictions = [model.predict(self._features_from_vector(vector)) for vector in matrix]
        raw_probability_predictions = [
            self._clip_probability(classifier.predict(self._features_from_vector(vector)))
            for vector in matrix
        ]

        validation_matrix = [self._vector_from_snapshot(sample) for sample in validation_samples]
        validation_probability_raw = [
            self._clip_probability(classifier.predict(self._features_from_vector(vector)))
            for vector in validation_matrix
        ]
        validation_targets = [self._success_target(self._target(sample)) for sample in validation_samples]
        success_calibrator = _PlattSuccessCalibrator.fit(
            raw_probabilities=validation_probability_raw,
            targets=validation_targets,
        )
        probability_predictions = (
            [success_calibrator.apply(value) for value in raw_probability_predictions]
            if success_calibrator is not None
            else raw_probability_predictions
        )

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
            "probability_raw_method": "model_success_classifier",
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
            "feature_spec": {
                "version": _OpportunityFeaturePipeline.feature_spec_version(),
                "names": list(_OpportunityFeaturePipeline.feature_names()),
            },
        }
        if success_calibrator is not None:
            metadata["probability_calibration"] = {
                **success_calibrator.to_mapping(),
                "fit_split": "validation",
                "validation_rows": len(validation_samples),
                "raw_probability_method": "model_success_classifier",
            }
        else:
            metadata["probability_calibration"] = {
                "method": "none",
                "fit_split": "validation",
                "validation_rows": len(validation_samples),
                "reason": "insufficient_validation_label_diversity_or_rows",
                "raw_probability_method": "model_success_classifier",
            }
        model_state: MutableMapping[str, object] = dict(model.to_state())
        model_state["classifier_head_state"] = classifier.to_state()
        if success_calibrator is not None:
            model_state["success_probability_calibrator"] = dict(success_calibrator.to_mapping())

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
                    "classifier_raw_brier_score": (
                        sum((predicted - expected) ** 2 for predicted, expected in zip(raw_probability_predictions, classifier_targets))
                        / len(classifier_targets)
                    ),
                }
            },
            metadata=metadata,
            target_scale=target_scale,
            training_rows=len(samples),
            validation_rows=len(validation_samples),
            test_rows=0,
            feature_scalers=model.feature_scalers,
            backend="builtin",
        )
        self._artifact = artifact
        self._model = model
        self._classifier_model = classifier
        self._success_calibrator = success_calibrator
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
        self._validate_artifact_feature_spec(artifact)
        classifier_state = artifact.model_state.get("classifier_head_state")
        classifier: SimpleGradientBoostingModel | None = None
        if isinstance(classifier_state, Mapping):
            loaded_classifier = SimpleGradientBoostingModel()
            loaded_classifier.load_state(classifier_state)
            if loaded_classifier.feature_names:
                classifier = loaded_classifier
        calibrator_payload = artifact.model_state.get("success_probability_calibrator")
        calibrator: _PlattSuccessCalibrator | None = None
        if isinstance(calibrator_payload, Mapping):
            calibrator = _PlattSuccessCalibrator.from_mapping(calibrator_payload)
        self._artifact = artifact
        self._model = model
        self._classifier_model = classifier
        self._success_calibrator = calibrator
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
        calibrator = self._success_calibrator if classifier is not None else None
        prob_scale = self._safe_probability_scale(
            artifact.metadata.get("probability_scale_bps", artifact.target_scale)
        )
        model_version = str(artifact.metadata.get("model_version", "unknown"))

        for candidate in candidates:
            feature_payload = _OpportunityFeaturePipeline.from_candidate(candidate)
            features = self._features_from_vector(feature_payload.feature_values)
            edge = float(model.predict(features))
            if classifier is not None:
                raw_probability = self._clip_probability(classifier.predict(features))
                if calibrator is not None:
                    probability = self._clip_probability(calibrator.apply(raw_probability))
                    probability_method = "model_success_classifier_calibrated"
                    calibration = {
                        "type": "probability_calibrator",
                        "path": "calibrated",
                        "method": "platt_scaling",
                        "raw_probability_method": "model_success_classifier",
                    }
                else:
                    probability = raw_probability
                    probability_method = "model_success_classifier_uncalibrated"
                    calibration = {
                        "type": "classifier",
                        "path": "uncalibrated",
                        "method": "none",
                        "raw_probability_method": "model_success_classifier",
                        "target_definition": artifact.metadata.get(
                            "classification_target_definition",
                            "1 if target_definition > 0 else 0",
                        ),
                    }
            else:
                raw_probability = self._edge_to_probability(edge=edge, scale=prob_scale)
                probability = raw_probability
                probability_method = "heuristic_sigmoid_scaled_edge_fallback"
                calibration = {
                    "type": "heuristic_fallback",
                    "path": "uncalibrated",
                    "method": "none",
                    "scale_bps": prob_scale,
                    "raw_probability_method": "heuristic_sigmoid_scaled_edge_fallback",
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
                        "feature_spec_version": _FEATURE_SPEC_VERSION,
                        "feature_freshness_seconds": feature_payload.freshness_seconds,
                        "market_regime": feature_payload.regime.to_mapping(),
                        "probability_method": probability_method,
                        "raw_probability_method": calibration.get("raw_probability_method", probability_method),
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
        payload = _OpportunityFeaturePipeline.from_snapshot(snapshot)
        return [float(value) for value in payload.feature_values]

    @staticmethod
    def _vector_from_candidate(candidate: OpportunityCandidate) -> list[float]:
        payload = _OpportunityFeaturePipeline.from_candidate(candidate)
        return [float(value) for value in payload.feature_values]

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
        padded = list(vector)
        if len(padded) < len(_FEATURE_NAMES):
            padded.extend([0.0] * (len(_FEATURE_NAMES) - len(padded)))
        return {name: float(value) for name, value in zip(_FEATURE_NAMES, padded)}

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
        for feature_name, value in zip(_RAW_FEATURE_NAMES, values):
            if not cls._is_finite(value):
                raise ValueError(
                    f"Niefinitywna cecha {feature_name} ({context}#{item_index})"
                )

    @classmethod
    def _validate_training_samples(cls, samples: Sequence[OpportunitySnapshot]) -> None:
        for index, sample in enumerate(samples):
            cls._validate_symbol(sample.symbol, context=f"snapshot#{index}")
            _OpportunityFeaturePipeline.from_snapshot(sample)
            raw_vector = cls._raw_vector_from_snapshot(sample)
            cls._validate_feature_vector(raw_vector, context="snapshot", item_index=index)
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
            _OpportunityFeaturePipeline.from_candidate(candidate)
            raw_vector = cls._raw_vector_from_candidate(candidate)
            cls._validate_feature_vector(raw_vector, context="candidate", item_index=index)

    @classmethod
    def _validate_artifact_feature_spec(cls, artifact: ModelArtifact) -> None:
        top_level_names = tuple(artifact.feature_names)
        if top_level_names != _FEATURE_NAMES:
            raise ValueError("Feature spec mismatch: artifact feature names are incompatible")
        feature_spec = artifact.metadata.get("feature_spec")
        if not isinstance(feature_spec, Mapping):
            raise ValueError("Feature version mismatch: missing feature spec metadata")
        version = str(feature_spec.get("version", "")).strip()
        if version != _FEATURE_SPEC_VERSION:
            raise ValueError("Feature version mismatch: unsupported opportunity feature spec version")
        metadata_names_raw = feature_spec.get("names")
        if not isinstance(metadata_names_raw, Sequence) or isinstance(metadata_names_raw, (str, bytes)):
            raise ValueError("Feature spec mismatch: metadata.feature_spec.names is invalid")
        metadata_names = tuple(str(name) for name in metadata_names_raw)
        if metadata_names != _FEATURE_NAMES:
            raise ValueError("Feature spec mismatch: metadata feature names are incompatible")
        if metadata_names != top_level_names:
            raise ValueError("Feature spec mismatch: metadata/top-level feature names are inconsistent")

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
