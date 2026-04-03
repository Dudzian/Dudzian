"""Temporalna ewaluacja modeli okazji tradingowych na danych shadow/outcome."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from .models import ModelArtifact
from .repository import ModelRepository
from .trading_opportunity_shadow import OpportunityOutcomeLabel, OpportunityShadowRecord


@dataclass(slots=True, frozen=True)
class OpportunitySplitConfig:
    train_fraction: float = 0.7
    validation_fraction: float = 0.15


@dataclass(slots=True, frozen=True)
class OpportunityEvaluationMetrics:
    count: int
    mae_bps: float
    rmse_bps: float
    directional_accuracy: float
    hit_rate: float
    average_realized_return_bps: float
    net_edge_realization_bps: float
    precision_at_k: float
    brier_score: float | None


@dataclass(slots=True, frozen=True)
class OpportunityWalkForwardFoldReport:
    fold_index: int
    train_count: int
    test_count: int
    metrics: OpportunityEvaluationMetrics


@dataclass(slots=True, frozen=True)
class OpportunityModelComparison:
    champion_reference: str
    candidate_reference: str
    champion_test_metrics: OpportunityEvaluationMetrics
    candidate_test_metrics: OpportunityEvaluationMetrics
    mae_delta_bps: float
    directional_accuracy_delta: float
    hit_rate_delta: float


@dataclass(slots=True, frozen=True)
class OpportunityEvaluationReport:
    total_records: int
    train_count: int
    validation_count: int
    test_count: int
    train_metrics: OpportunityEvaluationMetrics
    validation_metrics: OpportunityEvaluationMetrics
    test_metrics: OpportunityEvaluationMetrics
    walk_forward: tuple[OpportunityWalkForwardFoldReport, ...]
    comparison: OpportunityModelComparison | None


@dataclass(slots=True, frozen=True)
class _EvaluationRow:
    timestamp: datetime
    expected_edge_bps: float
    success_probability: float
    realized_return_bps: float
    snapshot: Mapping[str, object]


class OpportunityTemporalEvaluator:
    """Lokalny harness temporal evaluation dla opportunity engine."""

    def __init__(
        self,
        *,
        split_config: OpportunitySplitConfig | None = None,
        walk_forward_folds: int = 3,
        precision_at_k: int = 5,
    ) -> None:
        self._split_config = split_config or OpportunitySplitConfig()
        self._walk_forward_folds = max(1, int(walk_forward_folds))
        self._precision_at_k = max(1, int(precision_at_k))

    def evaluate(
        self,
        shadow_records: Sequence[OpportunityShadowRecord],
        outcome_labels: Sequence[OpportunityOutcomeLabel],
    ) -> OpportunityEvaluationReport:
        rows = self._build_rows(shadow_records, outcome_labels)
        train_rows, validation_rows, test_rows = self._temporal_split(rows)
        walk_forward = self._walk_forward(rows)
        return OpportunityEvaluationReport(
            total_records=len(rows),
            train_count=len(train_rows),
            validation_count=len(validation_rows),
            test_count=len(test_rows),
            train_metrics=self._compute_metrics(train_rows),
            validation_metrics=self._compute_metrics(validation_rows),
            test_metrics=self._compute_metrics(test_rows),
            walk_forward=tuple(walk_forward),
            comparison=None,
        )

    def evaluate_with_model_comparison(
        self,
        shadow_records: Sequence[OpportunityShadowRecord],
        outcome_labels: Sequence[OpportunityOutcomeLabel],
        *,
        repository: ModelRepository,
        champion_reference: str = "active",
        candidate_reference: str = "latest",
    ) -> OpportunityEvaluationReport:
        rows = self._build_rows(shadow_records, outcome_labels)
        train_rows, validation_rows, test_rows = self._temporal_split(rows)

        champion_artifact = repository.load_model(champion_reference)
        candidate_artifact = repository.load_model(candidate_reference)
        champion_rows, candidate_rows = self._score_common_subset(
            test_rows,
            champion_artifact=champion_artifact,
            candidate_artifact=candidate_artifact,
        )
        if not champion_rows:
            raise ValueError(
                "Comparison fairness guard: brak wspólnego scored test subset "
                "dla champion i candidate po wyrównaniu cech"
            )
        champion_metrics = self._compute_metrics(champion_rows)
        candidate_metrics = self._compute_metrics(candidate_rows)

        comparison = OpportunityModelComparison(
            champion_reference=champion_reference,
            candidate_reference=candidate_reference,
            champion_test_metrics=champion_metrics,
            candidate_test_metrics=candidate_metrics,
            mae_delta_bps=(champion_metrics.mae_bps - candidate_metrics.mae_bps),
            directional_accuracy_delta=(
                candidate_metrics.directional_accuracy - champion_metrics.directional_accuracy
            ),
            hit_rate_delta=(candidate_metrics.hit_rate - champion_metrics.hit_rate),
        )

        walk_forward = self._walk_forward(rows)
        return OpportunityEvaluationReport(
            total_records=len(rows),
            train_count=len(train_rows),
            validation_count=len(validation_rows),
            test_count=len(test_rows),
            train_metrics=self._compute_metrics(train_rows),
            validation_metrics=self._compute_metrics(validation_rows),
            test_metrics=self._compute_metrics(test_rows),
            walk_forward=tuple(walk_forward),
            comparison=comparison,
        )

    def evaluate_latest_vs_previous(
        self,
        shadow_records: Sequence[OpportunityShadowRecord],
        outcome_labels: Sequence[OpportunityOutcomeLabel],
        *,
        repository: ModelRepository,
    ) -> OpportunityEvaluationReport:
        versions = list(repository.list_versions())
        if len(versions) < 2:
            raise ValueError("Do porównania latest vs previous potrzeba co najmniej 2 wersji modelu")
        previous, latest = versions[-2], versions[-1]
        return self.evaluate_with_model_comparison(
            shadow_records,
            outcome_labels,
            repository=repository,
            champion_reference=previous,
            candidate_reference=latest,
        )

    def _build_rows(
        self,
        shadow_records: Sequence[OpportunityShadowRecord],
        outcome_labels: Sequence[OpportunityOutcomeLabel],
    ) -> list[_EvaluationRow]:
        labels_by_key = {label.correlation_key: label for label in outcome_labels}
        rows: list[_EvaluationRow] = []
        for record in shadow_records:
            label = labels_by_key.get(record.record_key)
            if label is None:
                continue
            rows.append(
                _EvaluationRow(
                    timestamp=record.decision_timestamp,
                    expected_edge_bps=float(record.expected_edge_bps),
                    success_probability=float(record.success_probability),
                    realized_return_bps=float(label.realized_return_bps),
                    snapshot=dict(record.snapshot),
                )
            )
        rows.sort(key=lambda item: item.timestamp)
        if not rows:
            raise ValueError("Brak sparowanych rekordów shadow/outcome do ewaluacji")
        return rows

    def _temporal_split(
        self,
        rows: Sequence[_EvaluationRow],
    ) -> tuple[list[_EvaluationRow], list[_EvaluationRow], list[_EvaluationRow]]:
        total = len(rows)
        train_end = max(1, int(total * self._split_config.train_fraction))
        validation_end = max(train_end + 1, int(total * (self._split_config.train_fraction + self._split_config.validation_fraction)))
        validation_end = min(validation_end, total - 1) if total > 2 else min(validation_end, total)

        train = list(rows[:train_end])
        validation = list(rows[train_end:validation_end])
        test = list(rows[validation_end:])

        if not validation:
            validation = list(rows[train_end:train_end + 1])
            test = list(rows[train_end + 1 :])
        if not test:
            test = list(rows[-1:])
            if validation:
                validation = validation[:-1]
            elif train:
                train = train[:-1]
        return train, validation, test

    def _walk_forward(self, rows: Sequence[_EvaluationRow]) -> list[OpportunityWalkForwardFoldReport]:
        if len(rows) < 4:
            return [
                OpportunityWalkForwardFoldReport(
                    fold_index=1,
                    train_count=max(1, len(rows) - 1),
                    test_count=1,
                    metrics=self._compute_metrics(list(rows[-1:])),
                )
            ]

        initial_train = max(2, len(rows) // 2)
        remaining = len(rows) - initial_train
        step = max(1, remaining // self._walk_forward_folds)
        reports: list[OpportunityWalkForwardFoldReport] = []

        train_end = initial_train
        fold_index = 1
        while train_end < len(rows) and fold_index <= self._walk_forward_folds:
            test_end = min(len(rows), train_end + step)
            test_rows = list(rows[train_end:test_end])
            if not test_rows:
                break
            reports.append(
                OpportunityWalkForwardFoldReport(
                    fold_index=fold_index,
                    train_count=train_end,
                    test_count=len(test_rows),
                    metrics=self._compute_metrics(test_rows),
                )
            )
            fold_index += 1
            train_end = test_end

        if not reports:
            reports.append(
                OpportunityWalkForwardFoldReport(
                    fold_index=1,
                    train_count=max(1, len(rows) - 1),
                    test_count=1,
                    metrics=self._compute_metrics(list(rows[-1:])),
                )
            )
        return reports

    def _score_with_artifact(
        self,
        rows: Sequence[_EvaluationRow],
        artifact: ModelArtifact,
    ) -> list[_EvaluationRow]:
        model = artifact.build_model()
        probability_scale = float(artifact.metadata.get("probability_scale_bps", artifact.target_scale or 10.0))
        probability_scale = max(1e-6, probability_scale)

        scored: list[_EvaluationRow] = []
        for row in rows:
            features = self._extract_features(row.snapshot, artifact.feature_names)
            if features is None:
                continue
            edge = float(model.predict(features))
            probability = 1.0 / (1.0 + math.exp(-edge / probability_scale))
            scored.append(
                _EvaluationRow(
                    timestamp=row.timestamp,
                    expected_edge_bps=edge,
                    success_probability=probability,
                    realized_return_bps=row.realized_return_bps,
                    snapshot=row.snapshot,
                )
            )
        return scored

    def _score_common_subset(
        self,
        rows: Sequence[_EvaluationRow],
        *,
        champion_artifact: ModelArtifact,
        candidate_artifact: ModelArtifact,
    ) -> tuple[list[_EvaluationRow], list[_EvaluationRow]]:
        champion_model = champion_artifact.build_model()
        candidate_model = candidate_artifact.build_model()
        champion_scale = float(
            champion_artifact.metadata.get(
                "probability_scale_bps",
                champion_artifact.target_scale or 10.0,
            )
        )
        candidate_scale = float(
            candidate_artifact.metadata.get(
                "probability_scale_bps",
                candidate_artifact.target_scale or 10.0,
            )
        )
        champion_scale = max(1e-6, champion_scale)
        candidate_scale = max(1e-6, candidate_scale)

        champion_rows: list[_EvaluationRow] = []
        candidate_rows: list[_EvaluationRow] = []
        for row in rows:
            champion_features = self._extract_features(row.snapshot, champion_artifact.feature_names)
            candidate_features = self._extract_features(row.snapshot, candidate_artifact.feature_names)
            if champion_features is None or candidate_features is None:
                continue
            champion_edge = float(champion_model.predict(champion_features))
            candidate_edge = float(candidate_model.predict(candidate_features))
            champion_probability = 1.0 / (1.0 + math.exp(-champion_edge / champion_scale))
            candidate_probability = 1.0 / (1.0 + math.exp(-candidate_edge / candidate_scale))

            champion_rows.append(
                _EvaluationRow(
                    timestamp=row.timestamp,
                    expected_edge_bps=champion_edge,
                    success_probability=champion_probability,
                    realized_return_bps=row.realized_return_bps,
                    snapshot=row.snapshot,
                )
            )
            candidate_rows.append(
                _EvaluationRow(
                    timestamp=row.timestamp,
                    expected_edge_bps=candidate_edge,
                    success_probability=candidate_probability,
                    realized_return_bps=row.realized_return_bps,
                    snapshot=row.snapshot,
                )
            )
        return champion_rows, candidate_rows

    def _extract_features(
        self,
        snapshot: Mapping[str, object],
        feature_names: Sequence[str],
    ) -> dict[str, float] | None:
        features: dict[str, float] = {}
        for name in feature_names:
            value = snapshot.get(name)
            if not isinstance(value, (int, float)):
                return None
            features[name] = float(value)
        return features

    def _compute_metrics(self, rows: Sequence[_EvaluationRow]) -> OpportunityEvaluationMetrics:
        if not rows:
            return OpportunityEvaluationMetrics(
                count=0,
                mae_bps=0.0,
                rmse_bps=0.0,
                directional_accuracy=0.0,
                hit_rate=0.0,
                average_realized_return_bps=0.0,
                net_edge_realization_bps=0.0,
                precision_at_k=0.0,
                brier_score=None,
            )

        errors = [row.expected_edge_bps - row.realized_return_bps for row in rows]
        mae = sum(abs(err) for err in errors) / len(errors)
        rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
        directional_hits = sum(
            1
            for row in rows
            if (row.expected_edge_bps >= 0.0 and row.realized_return_bps >= 0.0)
            or (row.expected_edge_bps < 0.0 and row.realized_return_bps < 0.0)
        )
        directional_accuracy = directional_hits / len(rows)

        predicted_positive = [row for row in rows if row.expected_edge_bps > 0.0]
        if predicted_positive:
            hit_rate = sum(1 for row in predicted_positive if row.realized_return_bps > 0.0) / len(predicted_positive)
            avg_realized = sum(row.realized_return_bps for row in predicted_positive) / len(predicted_positive)
            net_realization = sum(row.realized_return_bps - row.expected_edge_bps for row in predicted_positive) / len(
                predicted_positive
            )
        else:
            hit_rate = 0.0
            avg_realized = 0.0
            net_realization = 0.0

        top_k = sorted(rows, key=lambda row: row.expected_edge_bps, reverse=True)[: self._precision_at_k]
        precision_at_k = sum(1 for row in top_k if row.realized_return_bps > 0.0) / len(top_k)

        brier = sum((row.success_probability - (1.0 if row.realized_return_bps > 0.0 else 0.0)) ** 2 for row in rows) / len(rows)

        return OpportunityEvaluationMetrics(
            count=len(rows),
            mae_bps=mae,
            rmse_bps=rmse,
            directional_accuracy=directional_accuracy,
            hit_rate=hit_rate,
            average_realized_return_bps=avg_realized,
            net_edge_realization_bps=net_realization,
            precision_at_k=precision_at_k,
            brier_score=brier,
        )


__all__ = [
    "OpportunityEvaluationMetrics",
    "OpportunityEvaluationReport",
    "OpportunityModelComparison",
    "OpportunitySplitConfig",
    "OpportunityTemporalEvaluator",
    "OpportunityWalkForwardFoldReport",
]
