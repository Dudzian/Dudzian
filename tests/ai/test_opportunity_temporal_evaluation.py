from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot_core.ai import (
    FilesystemModelRepository,
    OpportunityOutcomeLabel,
    OpportunityShadowContext,
    OpportunityShadowRecord,
    OpportunitySplitConfig,
    OpportunityTemporalEvaluator,
    OpportunityThresholdConfig,
    OpportunitySnapshot,
    TradingOpportunityAI,
)


def _build_shadow_and_labels() -> tuple[list[OpportunityShadowRecord], list[OpportunityOutcomeLabel]]:
    records: list[OpportunityShadowRecord] = []
    labels: list[OpportunityOutcomeLabel] = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for idx in range(20):
        timestamp = start + timedelta(minutes=15 * idx)
        signal = (idx - 10) / 8.0
        expected_edge = signal * 6.0
        realized = expected_edge * 0.8 + (0.5 if idx % 2 == 0 else -0.5)
        probability = 0.5 + max(-0.45, min(0.45, expected_edge / 20.0))
        key = OpportunityShadowRecord.build_record_key(
            symbol="BTC/USDT",
            decision_timestamp=timestamp,
            model_version="shadow-v1",
            rank=idx + 1,
        )
        snapshot = {
            "signal_strength": signal,
            "momentum_5m": signal * 0.7,
            "volatility_30m": 0.3 + (idx % 3) * 0.05,
            "spread_bps": 0.5,
            "fee_bps": 0.2,
            "slippage_bps": 0.1,
            "liquidity_score": 0.8,
            "risk_penalty_bps": 0.2,
        }
        records.append(
            OpportunityShadowRecord(
                record_key=key,
                symbol="BTC/USDT",
                decision_timestamp=timestamp,
                model_version="shadow-v1",
                decision_source="model",
                expected_edge_bps=expected_edge,
                success_probability=probability,
                confidence=abs(probability - 0.5) * 2.0,
                proposed_direction="long" if expected_edge >= 0 else "short",
                accepted=expected_edge > 0,
                rejection_reason=None,
                rank=idx + 1,
                provenance={},
                threshold_config=OpportunityThresholdConfig(),
                snapshot=snapshot,
                context=OpportunityShadowContext(environment="shadow"),
            )
        )
        labels.append(
            OpportunityOutcomeLabel(
                symbol="BTC/USDT",
                decision_timestamp=timestamp,
                correlation_key=key,
                horizon_minutes=60,
                realized_return_bps=realized,
                max_favorable_excursion_bps=max(0.0, realized + 1.0),
                max_adverse_excursion_bps=min(0.0, realized - 1.0),
            )
        )
    return records, labels


def _build_training_samples(scale: float) -> list[OpportunitySnapshot]:
    rows: list[OpportunitySnapshot] = []
    for idx in range(30):
        signal = (idx - 15) / 10.0
        momentum = signal * 0.7
        volatility = 0.2 + (idx % 4) * 0.05
        rows.append(
            OpportunitySnapshot(
                symbol="BTC/USDT",
                signal_strength=signal,
                momentum_5m=momentum,
                volatility_30m=volatility,
                spread_bps=0.5,
                fee_bps=0.2,
                slippage_bps=0.1,
                liquidity_score=0.8,
                risk_penalty_bps=0.2,
                realized_return_bps=scale * (signal * 8.0 + momentum * 2.0 - volatility),
            )
        )
    return rows


def _artifact_path_for_version(repository: FilesystemModelRepository, version: str) -> Path:
    manifest = repository.get_manifest()
    versions = manifest.get("versions", {})
    entry = dict(versions).get(version, {})
    relative = str(dict(entry).get("file", ""))
    return repository.base_path / relative


def _require_additional_feature(repository: FilesystemModelRepository, version: str, feature_name: str) -> None:
    path = _artifact_path_for_version(repository, version)
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = list(payload.get("feature_names", []))
    if feature_name not in features:
        features.append(feature_name)
    payload["feature_names"] = features
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _with_optional_feature(
    records: list[OpportunityShadowRecord],
    *,
    feature_name: str,
    test_indexes_with_feature: set[int],
) -> list[OpportunityShadowRecord]:
    updated: list[OpportunityShadowRecord] = []
    for idx, record in enumerate(records):
        snapshot = dict(record.snapshot)
        if idx in test_indexes_with_feature:
            snapshot[feature_name] = float(idx) / 10.0
        updated.append(
            OpportunityShadowRecord(
                record_key=record.record_key,
                symbol=record.symbol,
                decision_timestamp=record.decision_timestamp,
                model_version=record.model_version,
                decision_source=record.decision_source,
                expected_edge_bps=record.expected_edge_bps,
                success_probability=record.success_probability,
                confidence=record.confidence,
                proposed_direction=record.proposed_direction,
                accepted=record.accepted,
                rejection_reason=record.rejection_reason,
                rank=record.rank,
                provenance=record.provenance,
                threshold_config=record.threshold_config,
                snapshot=snapshot,
                context=record.context,
            )
        )
    return updated


def test_temporal_split_and_report_contract() -> None:
    records, labels = _build_shadow_and_labels()
    evaluator = OpportunityTemporalEvaluator(
        split_config=OpportunitySplitConfig(train_fraction=0.6, validation_fraction=0.2),
        walk_forward_folds=3,
        precision_at_k=4,
    )

    report = evaluator.evaluate(records, labels)

    assert report.total_records == 20
    assert report.train_count == 12
    assert report.validation_count == 4
    assert report.test_count == 4
    assert report.test_metrics.mae_bps >= 0.0
    assert report.test_metrics.rmse_bps >= 0.0
    assert 0.0 <= report.test_metrics.directional_accuracy <= 1.0
    assert 0.0 <= report.test_metrics.hit_rate <= 1.0
    assert 0.0 <= report.test_metrics.precision_at_k <= 1.0
    assert report.test_metrics.brier_score is not None


def test_walk_forward_reports_incremental_windows() -> None:
    records, labels = _build_shadow_and_labels()
    evaluator = OpportunityTemporalEvaluator(walk_forward_folds=4)

    report = evaluator.evaluate(records, labels)

    assert len(report.walk_forward) >= 2
    assert report.walk_forward[0].train_count < report.walk_forward[-1].train_count
    assert all(fold.test_count > 0 for fold in report.walk_forward)


def test_latest_vs_previous_model_comparison(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "models")

    champion = TradingOpportunityAI(repository=repository)
    champion.fit(_build_training_samples(scale=1.0))
    champion.save_model(version="v1", activate=True)

    candidate = TradingOpportunityAI(repository=repository)
    candidate.fit(_build_training_samples(scale=0.7))
    candidate.save_model(version="v2", activate=False)

    records, labels = _build_shadow_and_labels()
    evaluator = OpportunityTemporalEvaluator()

    report = evaluator.evaluate_latest_vs_previous(records, labels, repository=repository)

    assert report.comparison is not None
    assert report.comparison.champion_reference == "v1"
    assert report.comparison.candidate_reference == "v2"
    assert report.comparison.candidate_test_metrics.count > 0
    assert isinstance(report.comparison.mae_delta_bps, float)


def test_comparison_uses_common_scored_subset_when_features_differ(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "models")
    champion = TradingOpportunityAI(repository=repository)
    champion.fit(_build_training_samples(scale=1.0))
    champion.save_model(version="v1", activate=True)
    candidate = TradingOpportunityAI(repository=repository)
    candidate.fit(_build_training_samples(scale=0.9))
    candidate.save_model(version="v2", activate=False)
    _require_additional_feature(repository, "v2", "candidate_only_feature")

    records, labels = _build_shadow_and_labels()
    records = _with_optional_feature(
        records,
        feature_name="candidate_only_feature",
        test_indexes_with_feature={16, 17},
    )
    evaluator = OpportunityTemporalEvaluator(
        split_config=OpportunitySplitConfig(train_fraction=0.6, validation_fraction=0.2)
    )

    report = evaluator.evaluate_with_model_comparison(records, labels, repository=repository, champion_reference="v1", candidate_reference="v2")

    assert report.comparison is not None
    assert report.comparison.champion_test_metrics.count == 2
    assert report.comparison.candidate_test_metrics.count == 2


def test_comparison_raises_when_no_common_scored_subset(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "models")
    champion = TradingOpportunityAI(repository=repository)
    champion.fit(_build_training_samples(scale=1.0))
    champion.save_model(version="v1", activate=True)
    candidate = TradingOpportunityAI(repository=repository)
    candidate.fit(_build_training_samples(scale=0.8))
    candidate.save_model(version="v2", activate=False)
    _require_additional_feature(repository, "v2", "candidate_only_feature")

    records, labels = _build_shadow_and_labels()
    evaluator = OpportunityTemporalEvaluator(
        split_config=OpportunitySplitConfig(train_fraction=0.6, validation_fraction=0.2)
    )

    try:
        evaluator.evaluate_with_model_comparison(
            records,
            labels,
            repository=repository,
            champion_reference="v1",
            candidate_reference="v2",
        )
        raise AssertionError("Expected ValueError for empty common scored subset")
    except ValueError as exc:
        assert "brak wspólnego scored test subset" in str(exc)


def test_comparison_deltas_are_computed_on_aligned_common_rows(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "models")
    champion = TradingOpportunityAI(repository=repository)
    champion.fit(_build_training_samples(scale=1.0))
    champion.save_model(version="v1", activate=True)
    candidate = TradingOpportunityAI(repository=repository)
    candidate.fit(_build_training_samples(scale=0.6))
    candidate.save_model(version="v2", activate=False)
    _require_additional_feature(repository, "v2", "candidate_only_feature")

    records, labels = _build_shadow_and_labels()
    records = _with_optional_feature(
        records,
        feature_name="candidate_only_feature",
        test_indexes_with_feature={16, 18},
    )
    evaluator = OpportunityTemporalEvaluator(
        split_config=OpportunitySplitConfig(train_fraction=0.6, validation_fraction=0.2)
    )
    report = evaluator.evaluate_with_model_comparison(
        records,
        labels,
        repository=repository,
        champion_reference="v1",
        candidate_reference="v2",
    )

    rows = evaluator._build_rows(records, labels)
    _, _, test_rows = evaluator._temporal_split(rows)
    champion_rows, candidate_rows = evaluator._score_common_subset(
        test_rows,
        champion_artifact=repository.load_model("v1"),
        candidate_artifact=repository.load_model("v2"),
    )
    champion_metrics = evaluator._compute_metrics(champion_rows)
    candidate_metrics = evaluator._compute_metrics(candidate_rows)

    assert report.comparison is not None
    assert report.comparison.mae_delta_bps == champion_metrics.mae_bps - candidate_metrics.mae_bps
    assert report.comparison.directional_accuracy_delta == (
        candidate_metrics.directional_accuracy - champion_metrics.directional_accuracy
    )
