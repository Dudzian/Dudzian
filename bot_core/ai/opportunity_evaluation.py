"""Temporal evaluation utilities for opportunity AI artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import ModelArtifact
from .repository import ModelRepository
from .training import SimpleGradientBoostingModel


@dataclass(slots=True, frozen=True)
class OpportunityTemporalEvaluation:
    model_version: str
    sample_count: int
    edge_mae_bps: float
    edge_rmse_bps: float
    directional_accuracy: float
    success_probability_brier: float
    success_probability_ece: float
    avg_success_probability: float
    probability_method: str
    calibration_method: str = "none"
    reliability_summary: tuple[Mapping[str, float], ...] = ()
    matched_outcomes: int = 0
    label_coverage: float = 0.0


@dataclass(slots=True, frozen=True)
class OpportunityTemporalComparison:
    latest: OpportunityTemporalEvaluation
    previous: OpportunityTemporalEvaluation
    common_sample_count: int
    fairness_applied: bool
    delta_edge_mae_bps: float
    delta_success_probability_brier: float


class OpportunityTemporalEvaluator:
    """Existing public temporal evaluator for opportunity models."""

    def split_temporal(
        self,
        samples: Sequence["OpportunitySnapshot"],
        *,
        train_ratio: float = 0.7,
    ) -> tuple[list["OpportunitySnapshot"], list["OpportunitySnapshot"]]:
        if not 0.0 < float(train_ratio) < 1.0:
            raise ValueError("train_ratio musi należeć do (0, 1)")
        ordered = self._sort_samples(samples)
        split_index = max(1, min(len(ordered) - 1, int(len(ordered) * train_ratio)))
        return ordered[:split_index], ordered[split_index:]

    def walk_forward_splits(
        self,
        samples: Sequence["OpportunitySnapshot"],
        *,
        folds: int = 3,
        min_train_size: int = 5,
    ) -> list[tuple[list["OpportunitySnapshot"], list["OpportunitySnapshot"]]]:
        if folds < 1:
            raise ValueError("folds musi być >= 1")
        ordered = self._sort_samples(samples)
        total = len(ordered)
        if total < min_train_size + 1:
            raise ValueError("Za mało próbek do walk-forward")

        test_window = max(1, (total - min_train_size) // folds)
        splits: list[tuple[list["OpportunitySnapshot"], list["OpportunitySnapshot"]]] = []
        start = min_train_size
        while start < total and len(splits) < folds:
            end = min(total, start + test_window)
            train = ordered[:start]
            test = ordered[start:end]
            if test:
                splits.append((train, test))
            start = end
        return splits

    def evaluate(
        self,
        artifact: ModelArtifact,
        samples: Sequence["OpportunitySnapshot"],
    ) -> OpportunityTemporalEvaluation:
        from .trading_engine import OpportunitySnapshot, TradingOpportunityAI

        if not samples:
            raise ValueError("Brak próbek do oceny temporalnej")
        if not all(isinstance(sample, OpportunitySnapshot) for sample in samples):
            raise TypeError("OpportunityTemporalEvaluator wymaga listy OpportunitySnapshot")

        model = artifact.build_model()
        if not isinstance(model, SimpleGradientBoostingModel):
            raise TypeError("OpportunityTemporalEvaluator wymaga backendu builtin")

        classifier = self._build_classifier_from_artifact(artifact)
        calibrator = self._build_calibrator_from_artifact(artifact)
        calibration_method = (
            "platt_scaling" if classifier is not None and calibrator is not None else "none"
        )
        probability_method = "heuristic_sigmoid_scaled_edge_fallback"
        if classifier is not None and calibrator is not None and calibration_method != "none":
            probability_method = "model_success_classifier_calibrated"
        elif classifier is not None:
            probability_method = "model_success_classifier_uncalibrated"
        probability_scale = TradingOpportunityAI._safe_probability_scale(
            artifact.metadata.get("probability_scale_bps", artifact.target_scale)
        )

        edge_predictions: list[float] = []
        edge_targets: list[float] = []
        probability_predictions: list[float] = []
        probability_targets: list[float] = []

        for sample in samples:
            features = TradingOpportunityAI._features_from_vector(
                TradingOpportunityAI._vector_from_snapshot(sample)
            )
            edge_pred = float(model.predict(features))
            edge_target = TradingOpportunityAI._target(sample)
            if classifier is not None:
                raw_probability = TradingOpportunityAI._clip_probability(classifier.predict(features))
                probability_pred = (
                    TradingOpportunityAI._clip_probability(calibrator.apply(raw_probability))
                    if calibrator is not None
                    else raw_probability
                )
            else:
                probability_pred = TradingOpportunityAI._edge_to_probability(
                    edge=edge_pred,
                    scale=probability_scale,
                )

            edge_predictions.append(edge_pred)
            edge_targets.append(edge_target)
            probability_predictions.append(probability_pred)
            probability_targets.append(TradingOpportunityAI._success_target(edge_target))

        sample_count = len(samples)
        edge_mae = sum(abs(p - y) for p, y in zip(edge_predictions, edge_targets)) / sample_count
        edge_rmse = math.sqrt(
            sum((p - y) ** 2 for p, y in zip(edge_predictions, edge_targets)) / sample_count
        )
        directional_hits = sum(
            1
            for prediction, target in zip(edge_predictions, edge_targets)
            if (prediction >= 0 and target >= 0) or (prediction < 0 and target < 0)
        )
        directional_accuracy = directional_hits / sample_count
        probability_brier = sum(
            (prediction - target) ** 2
            for prediction, target in zip(probability_predictions, probability_targets)
        ) / sample_count
        probability_ece, reliability_summary = self._calibration_error_summary(
            probability_predictions,
            probability_targets,
        )
        avg_probability = sum(probability_predictions) / sample_count

        return OpportunityTemporalEvaluation(
            model_version=str(artifact.metadata.get("model_version", "unknown")),
            sample_count=sample_count,
            edge_mae_bps=edge_mae,
            edge_rmse_bps=edge_rmse,
            directional_accuracy=directional_accuracy,
            success_probability_brier=probability_brier,
            success_probability_ece=probability_ece,
            avg_success_probability=avg_probability,
            probability_method=probability_method,
            calibration_method=calibration_method,
            reliability_summary=reliability_summary,
            matched_outcomes=sample_count,
            label_coverage=1.0,
        )

    def evaluate_from_shadow_labels(
        self,
        shadow_records: Sequence["OpportunityShadowRecord"],
        outcome_labels: Sequence["OpportunityOutcomeLabel"],
    ) -> OpportunityTemporalEvaluation:
        from .trading_opportunity_shadow import OpportunityOutcomeLabel, OpportunityShadowRecord

        if not shadow_records:
            raise ValueError("Brak shadow_records do oceny")
        if not all(isinstance(row, OpportunityShadowRecord) for row in shadow_records):
            raise TypeError("shadow_records muszą być listą OpportunityShadowRecord")
        if not all(isinstance(row, OpportunityOutcomeLabel) for row in outcome_labels):
            raise TypeError("outcome_labels muszą być listą OpportunityOutcomeLabel")

        labels_by_key = {str(label.correlation_key): label for label in outcome_labels}
        matched: list[tuple[float, float, float, float]] = []
        probability_methods: set[str] = set()
        for record in shadow_records:
            label = labels_by_key.get(str(record.record_key))
            if label is None:
                continue
            edge_prediction = float(record.expected_edge_bps)
            probability_prediction = max(0.0, min(1.0, float(record.success_probability)))
            edge_target = float(label.realized_return_bps)
            probability_target = 1.0 if edge_target > 0.0 else 0.0
            matched.append((edge_prediction, edge_target, probability_prediction, probability_target))
            method = record.provenance.get("probability_method") if isinstance(record.provenance, Mapping) else None
            if isinstance(method, str) and method.strip():
                probability_methods.add(method.strip())

        if not matched:
            raise ValueError("Brak par shadow_records/outcome_labels po wspólnym correlation_key")

        edge_predictions = [row[0] for row in matched]
        edge_targets = [row[1] for row in matched]
        probability_predictions = [row[2] for row in matched]
        probability_targets = [row[3] for row in matched]
        sample_count = len(edge_predictions)

        edge_mae = sum(abs(p - y) for p, y in zip(edge_predictions, edge_targets)) / sample_count
        edge_rmse = math.sqrt(
            sum((p - y) ** 2 for p, y in zip(edge_predictions, edge_targets)) / sample_count
        )
        directional_hits = sum(
            1
            for prediction, target in zip(edge_predictions, edge_targets)
            if (prediction >= 0 and target >= 0) or (prediction < 0 and target < 0)
        )
        directional_accuracy = directional_hits / sample_count
        probability_brier = sum(
            (prediction - target) ** 2
            for prediction, target in zip(probability_predictions, probability_targets)
        ) / sample_count
        probability_ece, reliability_summary = self._calibration_error_summary(
            probability_predictions,
            probability_targets,
        )
        avg_probability = sum(probability_predictions) / sample_count

        probability_method = (
            next(iter(probability_methods))
            if len(probability_methods) == 1
            else "mixed_from_shadow_records"
        )
        model_version = str(shadow_records[0].model_version)
        return OpportunityTemporalEvaluation(
            model_version=model_version,
            sample_count=sample_count,
            edge_mae_bps=edge_mae,
            edge_rmse_bps=edge_rmse,
            directional_accuracy=directional_accuracy,
            success_probability_brier=probability_brier,
            success_probability_ece=probability_ece,
            avg_success_probability=avg_probability,
            probability_method=probability_method,
            calibration_method="unknown_from_shadow_records",
            reliability_summary=reliability_summary,
            matched_outcomes=sample_count,
            label_coverage=sample_count / len(shadow_records),
        )

    def evaluate_with_model_comparison(
        self,
        latest_artifact: ModelArtifact,
        previous_artifact: ModelArtifact,
        samples: Sequence["OpportunitySnapshot"],
    ) -> OpportunityTemporalComparison:
        latest_model = latest_artifact.build_model()
        previous_model = previous_artifact.build_model()
        if not isinstance(latest_model, SimpleGradientBoostingModel):
            raise TypeError("OpportunityTemporalEvaluator wymaga backendu builtin (latest)")
        if not isinstance(previous_model, SimpleGradientBoostingModel):
            raise TypeError("OpportunityTemporalEvaluator wymaga backendu builtin (previous)")

        aligned, fairness_applied = self._aligned_common_subset(
            samples=samples,
            latest_model=latest_model,
            previous_model=previous_model,
        )
        if not aligned:
            raise ValueError("Brak wspólnego podzbioru próbek do porównania")
        latest = self.evaluate(latest_artifact, aligned)
        previous = self.evaluate(previous_artifact, aligned)
        return OpportunityTemporalComparison(
            latest=latest,
            previous=previous,
            common_sample_count=len(aligned),
            fairness_applied=fairness_applied,
            delta_edge_mae_bps=latest.edge_mae_bps - previous.edge_mae_bps,
            delta_success_probability_brier=(
                latest.success_probability_brier - previous.success_probability_brier
            ),
        )

    def compare_latest_vs_previous(
        self,
        *,
        latest_artifact: ModelArtifact,
        previous_artifact: ModelArtifact,
        samples: Sequence["OpportunitySnapshot"],
    ) -> OpportunityTemporalComparison:
        return self.evaluate_with_model_comparison(
            latest_artifact=latest_artifact,
            previous_artifact=previous_artifact,
            samples=samples,
        )

    def evaluate_latest_vs_previous(
        self,
        repository: ModelRepository,
        samples: Sequence["OpportunitySnapshot"],
        *,
        latest_ref: str = "latest",
        previous_ref: str = "previous",
    ) -> OpportunityTemporalComparison:
        latest = repository.load_model(latest_ref)
        previous = repository.load_model(previous_ref)
        return self.evaluate_with_model_comparison(latest, previous, samples)

    @staticmethod
    def _sort_samples(samples: Sequence["OpportunitySnapshot"]) -> list["OpportunitySnapshot"]:
        return sorted(
            samples,
            key=lambda item: (
                item.as_of.timestamp() if item.as_of is not None else float("-inf"),
                item.symbol,
            ),
        )

    @classmethod
    def _aligned_common_subset(
        cls,
        samples: Sequence["OpportunitySnapshot"],
        *,
        latest_model: SimpleGradientBoostingModel,
        previous_model: SimpleGradientBoostingModel,
    ) -> tuple[list["OpportunitySnapshot"], bool]:
        from .trading_engine import TradingOpportunityAI

        ordered = cls._sort_samples(samples)
        deduped: list["OpportunitySnapshot"] = []
        seen: set[tuple[str, float | None]] = set()
        for item in ordered:
            key = (
                str(item.symbol),
                item.as_of.timestamp() if item.as_of is not None else None,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        aligned: list["OpportunitySnapshot"] = []
        known_features = set(TradingOpportunityAI._features_from_vector([0.0] * 8).keys())
        for item in deduped:
            if not cls._is_scoreable_for_model(item, latest_model, known_features):
                continue
            if not cls._is_scoreable_for_model(item, previous_model, known_features):
                continue
            aligned.append(item)
        fairness_applied = len(aligned) != len(ordered)
        return aligned, fairness_applied

    @staticmethod
    def _is_scoreable_for_model(
        sample: "OpportunitySnapshot",
        model: SimpleGradientBoostingModel,
        known_features: set[str],
    ) -> bool:
        from .trading_engine import TradingOpportunityAI

        if any(name not in known_features for name in model.feature_names):
            return False
        vector = TradingOpportunityAI._raw_vector_from_snapshot(sample)
        if not all(TradingOpportunityAI._is_finite(value) for value in vector):
            return False
        features = TradingOpportunityAI._features_from_vector(
            TradingOpportunityAI._vector_from_snapshot(sample)
        )
        try:
            prediction = float(model.predict(features))
        except Exception:
            return False
        return math.isfinite(prediction)

    @staticmethod
    def _build_classifier_from_artifact(
        artifact: ModelArtifact,
    ) -> SimpleGradientBoostingModel | None:
        classifier_state = artifact.model_state.get("classifier_head_state")
        if not isinstance(classifier_state, Mapping):
            return None
        classifier = SimpleGradientBoostingModel()
        classifier.load_state(classifier_state)
        if not classifier.feature_names:
            return None
        return classifier

    @staticmethod
    def _build_calibrator_from_artifact(
        artifact: ModelArtifact,
    ) -> "_PlattSuccessCalibrator | None":
        from .trading_engine import _PlattSuccessCalibrator

        payload = artifact.model_state.get("success_probability_calibrator")
        if not isinstance(payload, Mapping):
            return None
        return _PlattSuccessCalibrator.from_mapping(payload)

    @staticmethod
    def _calibration_error_summary(
        predictions: Sequence[float],
        targets: Sequence[float],
        *,
        buckets: int = 10,
    ) -> tuple[float, tuple[Mapping[str, float], ...]]:
        if not predictions:
            return 0.0, ()
        effective_buckets = max(2, int(buckets))
        grouped: list[list[tuple[float, float]]] = [[] for _ in range(effective_buckets)]
        for prediction, target in zip(predictions, targets):
            clipped = max(0.0, min(1.0, float(prediction)))
            idx = min(effective_buckets - 1, int(clipped * effective_buckets))
            grouped[idx].append((clipped, float(target)))
        total = len(predictions)
        ece = 0.0
        summary: list[Mapping[str, float]] = []
        for idx, rows in enumerate(grouped):
            if not rows:
                continue
            count = len(rows)
            avg_conf = sum(pred for pred, _ in rows) / count
            avg_acc = sum(target for _, target in rows) / count
            gap = abs(avg_conf - avg_acc)
            ece += (count / total) * gap
            summary.append(
                {
                    "bucket_index": float(idx),
                    "bucket_start": idx / effective_buckets,
                    "bucket_end": (idx + 1) / effective_buckets,
                    "count": float(count),
                    "avg_confidence": avg_conf,
                    "avg_accuracy": avg_acc,
                    "abs_gap": gap,
                }
            )
        return ece, tuple(summary)


__all__ = [
    "OpportunityTemporalComparison",
    "OpportunityTemporalEvaluation",
    "OpportunityTemporalEvaluator",
]
