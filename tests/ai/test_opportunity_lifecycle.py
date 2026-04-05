from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bot_core.ai.opportunity_lifecycle import OpportunityLifecycleService
from bot_core.ai.opportunity_evaluation import OpportunityPromotionGateConfig
from bot_core.ai.repository import FilesystemModelRepository
from bot_core.ai.trading_engine import (
    OpportunityDecision,
    OpportunitySnapshot,
    TradingOpportunityAI,
)
from bot_core.ai.trading_opportunity_shadow import (
    OpportunityOutcomeLabel,
    OpportunityShadowRepository,
)


def _build_samples(scale: float = 1.0) -> list[OpportunitySnapshot]:
    return [
        OpportunitySnapshot(
            symbol="BTCUSDT",
            signal_strength=0.3 + idx * 0.1,
            momentum_5m=0.1 + idx * 0.01,
            volatility_30m=0.2 + idx * 0.01,
            spread_bps=2.0,
            fee_bps=1.0,
            slippage_bps=0.5,
            liquidity_score=0.7,
            risk_penalty_bps=0.2,
            realized_return_bps=(8.0 + idx) * scale,
            as_of=datetime(2024, 1, 1, 12, idx, tzinfo=timezone.utc),
        )
        for idx in range(12)
    ]


def _shadow_decision(model_version: str, expected_edge_bps: float) -> OpportunityDecision:
    return OpportunityDecision(
        symbol="BTCUSDT",
        decision_source="model",
        model_version=model_version,
        expected_edge_bps=expected_edge_bps,
        success_probability=0.7,
        confidence=0.3,
        proposed_direction="long",
        accepted=True,
        rejection_reason=None,
        rank=1,
        provenance={"probability_method": "test_probability"},
    )


def _snapshot_metadata() -> dict[str, float]:
    return {
        "signal_strength": 0.8,
        "momentum_5m": 0.2,
        "volatility_30m": 0.3,
        "spread_bps": 2.0,
        "fee_bps": 1.0,
        "slippage_bps": 0.5,
        "liquidity_score": 0.85,
        "risk_penalty_bps": 0.2,
    }


def test_attach_outcome_label_requires_existing_shadow_record(tmp_path: Path) -> None:
    service = OpportunityLifecycleService()
    repository = OpportunityShadowRepository(tmp_path / "shadow")
    label = OpportunityOutcomeLabel(
        symbol="BTCUSDT",
        decision_timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        correlation_key="missing-key",
        horizon_minutes=30,
        realized_return_bps=3.0,
        max_favorable_excursion_bps=4.0,
        max_adverse_excursion_bps=-1.0,
    )
    attached, reason = service.attach_outcome_label(repository, label)
    assert attached is False
    assert reason == "missing_shadow_record:missing-key"


def test_build_persisted_promotion_readiness_uses_real_shadow_outcome_flow(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.2))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    champion_records = engine.build_shadow_records(
        [_shadow_decision("champion-v1", 5.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )
    challenger_records = engine.build_shadow_records(
        [_shadow_decision("challenger-v2", 7.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )
    shadow_repo.append_shadow_records(champion_records)
    shadow_repo.append_shadow_records(challenger_records)
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=decision_time,
                correlation_key=champion_records[0].record_key,
                horizon_minutes=30,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=6.0,
                max_adverse_excursion_bps=-2.0,
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=decision_time,
                correlation_key=challenger_records[0].record_key,
                horizon_minutes=30,
                realized_return_bps=9.0,
                max_favorable_excursion_bps=10.0,
                max_adverse_excursion_bps=-1.0,
                label_quality="final",
            ),
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        gate_config=OpportunityPromotionGateConfig(require_shadow_no_regression=False),
    )

    assert report.champion_shadow_evaluation is not None
    assert report.challenger_shadow_evaluation is not None
    assert report.promotion_report is not None
    assert report.matched_outcomes["champion"] == 1
    assert report.matched_outcomes["challenger"] == 1


def test_build_persisted_promotion_readiness_is_explicitly_degraded_without_labels(
    tmp_path: Path,
) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples())
    engine.save_model(version="single-v1", activate=False)

    shadow_repo.append_shadow_records(
        engine.build_shadow_records(
            [_shadow_decision("single-v1", 5.0)],
            decision_timestamp=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
            snapshot={"candidate_metadata": _snapshot_metadata()},
        )
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="single-v1",
        challenger_version="single-v1",
    )

    assert report.promotion_report is None
    assert report.activation_readiness.activation_ready is False
    assert report.degraded_reasons


def test_build_persisted_promotion_readiness_flags_incomplete_feature_context(
    tmp_path: Path,
) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.1))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    shadow_repo.append_shadow_records(
        engine.build_shadow_records(
            [_shadow_decision("champion-v1", 5.0)],
            decision_timestamp=decision_time,
            snapshot={"candidate_metadata": {"signal_strength": 0.8}},
        )
    )
    shadow_repo.append_shadow_records(
        engine.build_shadow_records(
            [_shadow_decision("challenger-v2", 6.0)],
            decision_timestamp=decision_time,
            snapshot={"candidate_metadata": _snapshot_metadata()},
        )
    )
    records = shadow_repo.load_shadow_records()
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=row.symbol,
                decision_timestamp=row.decision_timestamp,
                correlation_key=row.record_key,
                horizon_minutes=30,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=5.0,
                max_adverse_excursion_bps=-2.0,
                label_quality="final",
            )
            for row in records
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        gate_config=OpportunityPromotionGateConfig(require_shadow_no_regression=False),
    )

    assert any(
        reason.startswith("incomplete_feature_context_for_snapshot_reconstruction:")
        for reason in report.degraded_reasons
    )


def test_build_persisted_promotion_readiness_excludes_proxy_only_outcomes(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.1))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    champion = engine.build_shadow_records(
        [_shadow_decision("champion-v1", 5.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    challenger = engine.build_shadow_records(
        [_shadow_decision("challenger-v2", 6.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    shadow_repo.append_shadow_records([champion, challenger])
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=champion.symbol,
                decision_timestamp=champion.decision_timestamp,
                correlation_key=champion.record_key,
                horizon_minutes=0,
                realized_return_bps=0.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                label_quality="execution_proxy_pending_exit",
            ),
            OpportunityOutcomeLabel(
                symbol=challenger.symbol,
                decision_timestamp=challenger.decision_timestamp,
                correlation_key=challenger.record_key,
                horizon_minutes=0,
                realized_return_bps=0.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                label_quality="execution_proxy_pending_exit",
            ),
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
    )
    assert report.promotion_report is None
    assert "proxy_only_outcomes_excluded_from_governance" in report.degraded_reasons
