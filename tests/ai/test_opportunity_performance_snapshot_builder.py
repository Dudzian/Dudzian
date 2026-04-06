from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot_core.ai.opportunity_lifecycle import (
    OpportunityAutonomyMode,
    OpportunityPerformanceSnapshotBuilder,
    OpportunityPerformanceSnapshotConfig,
    evaluate_autonomy_performance_guard,
)
from bot_core.ai.trading_opportunity_shadow import (
    OpportunityOutcomeLabel,
    OpportunityShadowRepository,
)


def test_snapshot_builder_empty_labels_returns_safe_zero_snapshot(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)

    snapshot = OpportunityPerformanceSnapshotBuilder().load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 0
    assert snapshot.recent_loss_streak == 0
    assert snapshot.recent_realized_return_bps_sum == 0.0
    assert snapshot.recent_avg_realized_return_bps == 0.0
    assert snapshot.recent_worst_realized_return_bps == 0.0
    assert snapshot.recent_negative_outcomes_count == 0
    assert snapshot.recent_partial_only_count == 0


def test_snapshot_builder_final_positive_only_metrics(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    labels = [
        _label(i, realized_return_bps=float(i + 1), label_quality="final_close") for i in range(3)
    ]
    repository.append_outcome_labels(labels)

    snapshot = OpportunityPerformanceSnapshotBuilder().load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 3
    assert snapshot.recent_realized_return_bps_sum == 6.0
    assert snapshot.recent_avg_realized_return_bps == 2.0
    assert snapshot.recent_worst_realized_return_bps == 1.0
    assert snapshot.recent_negative_outcomes_count == 0
    assert snapshot.recent_loss_streak == 0


def test_snapshot_builder_final_mixed_win_loss_metrics(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=10.0, label_quality="final"),
            _label(1, realized_return_bps=-5.0, label_quality="final"),
            _label(2, realized_return_bps=-2.0, label_quality="final"),
            _label(3, realized_return_bps=4.0, label_quality="final"),
        ]
    )

    snapshot = OpportunityPerformanceSnapshotBuilder().load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 4
    assert snapshot.recent_realized_return_bps_sum == 7.0
    assert snapshot.recent_avg_realized_return_bps == 1.75
    assert snapshot.recent_worst_realized_return_bps == -5.0
    assert snapshot.recent_negative_outcomes_count == 2


def test_snapshot_builder_recent_loss_streak_is_counted_from_latest_backwards(
    tmp_path: Path,
) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=11.0, label_quality="final"),
            _label(1, realized_return_bps=-1.0, label_quality="final"),
            _label(2, realized_return_bps=-2.0, label_quality="final"),
            _label(3, realized_return_bps=-3.0, label_quality="final"),
        ]
    )

    snapshot = OpportunityPerformanceSnapshotBuilder().load_recent_performance_snapshot(repository)

    assert snapshot.recent_loss_streak == 3


def test_snapshot_builder_partial_only_labels_are_counted_separately(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=-7.0, label_quality="partial_exit"),
            _label(1, realized_return_bps=12.0, label_quality="final_close"),
            _label(2, realized_return_bps=-1.0, label_quality="partial_followup"),
        ]
    )

    snapshot = OpportunityPerformanceSnapshotBuilder().load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 1
    assert snapshot.recent_realized_return_bps_sum == 12.0
    assert snapshot.recent_partial_only_count == 2


def test_snapshot_builder_proxy_only_labels_do_not_imitate_final_outcomes(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=-7.0, label_quality="execution_proxy_pending_exit"),
            _label(1, realized_return_bps=2.0, label_quality="unknown"),
        ]
    )

    snapshot = OpportunityPerformanceSnapshotBuilder().load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 0
    assert snapshot.recent_realized_return_bps_sum == 0.0
    assert snapshot.recent_loss_streak == 0


def test_snapshot_builder_window_uses_only_last_n_final_outcomes(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=7.0, label_quality="final"),
            _label(1, realized_return_bps=-4.0, label_quality="final"),
            _label(2, realized_return_bps=1.0, label_quality="partial"),
            _label(3, realized_return_bps=-1.0, label_quality="final"),
            _label(4, realized_return_bps=3.0, label_quality="final"),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(recent_final_window_size=2)
    )

    snapshot = builder.load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 2
    assert snapshot.recent_realized_return_bps_sum == 2.0
    assert snapshot.recent_avg_realized_return_bps == 1.0
    assert snapshot.recent_worst_realized_return_bps == -1.0
    assert snapshot.recent_partial_only_count == 0
    assert snapshot.recent_window_label == "last_2_final_outcomes"


def test_snapshot_builder_old_partial_labels_outside_recent_final_window_are_not_counted(
    tmp_path: Path,
) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=1.0, label_quality="partial_exit"),
            _label(1, realized_return_bps=-2.0, label_quality="partial_exit"),
            _label(2, realized_return_bps=3.0, label_quality="partial_exit"),
            _label(3, realized_return_bps=4.0, label_quality="final"),
            _label(4, realized_return_bps=5.0, label_quality="final"),
            _label(5, realized_return_bps=-1.0, label_quality="final"),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(recent_final_window_size=2, max_scan_labels=64)
    )

    snapshot = builder.load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 2
    assert snapshot.recent_realized_return_bps_sum == 4.0
    assert snapshot.recent_partial_only_count == 0


def test_snapshot_builder_window_consistency_counts_partial_only_from_same_recent_span(
    tmp_path: Path,
) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=9.0, label_quality="partial_exit"),
            _label(1, realized_return_bps=8.0, label_quality="final"),
            _label(2, realized_return_bps=-3.0, label_quality="final"),
            _label(3, realized_return_bps=7.0, label_quality="partial_exit"),
            _label(4, realized_return_bps=2.0, label_quality="final"),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(recent_final_window_size=2, max_scan_labels=64)
    )

    snapshot = builder.load_recent_performance_snapshot(repository)

    assert snapshot.recent_final_outcomes_count == 2
    assert snapshot.recent_realized_return_bps_sum == -1.0
    assert snapshot.recent_partial_only_count == 1


def test_snapshot_payload_shape_is_stable_and_serializable(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [_label(0, realized_return_bps=5.0, label_quality="final_close")]
    )
    snapshot = OpportunityPerformanceSnapshotBuilder().load_recent_performance_snapshot(repository)

    payload = snapshot.to_dict()

    assert set(payload) == {
        "recent_final_outcomes_count",
        "recent_loss_streak",
        "recent_realized_return_bps_sum",
        "recent_avg_realized_return_bps",
        "recent_worst_realized_return_bps",
        "recent_negative_outcomes_count",
        "recent_partial_only_count",
        "recent_window_label",
    }


def test_performance_guard_accepts_snapshot_from_builder_without_transform(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=6.0, label_quality="final"),
            _label(1, realized_return_bps=3.0, label_quality="final"),
            _label(2, realized_return_bps=2.0, label_quality="final"),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(recent_final_window_size=3)
    )
    snapshot = builder.load_recent_performance_snapshot(repository)

    decision = evaluate_autonomy_performance_guard(
        requested_mode=OpportunityAutonomyMode.LIVE_AUTONOMOUS,
        input_effective_mode=OpportunityAutonomyMode.LIVE_AUTONOMOUS,
        snapshot=snapshot,
    )

    assert decision.performance_guard_applied is True
    assert "insufficient_recent_final_outcomes_for_live" in decision.reasons


def test_snapshot_builder_scope_filters_by_environment_and_portfolio(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(
                0,
                realized_return_bps=6.0,
                label_quality="final",
                provenance={"environment": "live", "portfolio_id": "live-1"},
            ),
            _label(
                1,
                realized_return_bps=5.0,
                label_quality="final",
                provenance={"environment": "live", "portfolio_id": "live-1"},
            ),
            _label(
                2,
                realized_return_bps=-20.0,
                label_quality="final",
                provenance={"environment": "paper", "portfolio_id": "paper-1"},
            ),
            _label(
                3,
                realized_return_bps=-18.0,
                label_quality="final",
                provenance={"environment": "live", "portfolio_id": "live-2"},
            ),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(
            recent_final_window_size=8,
            scope_environment="live",
            scope_portfolio="live-1",
            require_scope_provenance=True,
        )
    )

    snapshot, diagnostics = builder.load_recent_performance_snapshot_with_scope_diagnostics(repository)

    assert snapshot.recent_final_outcomes_count == 2
    assert snapshot.recent_realized_return_bps_sum == 11.0
    assert diagnostics.scoped_label_count == 2
    assert diagnostics.excluded_label_count == 2


def test_snapshot_builder_scope_requires_provenance_and_excludes_unscoped_labels(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(0, realized_return_bps=7.0, label_quality="final"),
            _label(1, realized_return_bps=6.0, label_quality="final"),
            _label(
                2,
                realized_return_bps=5.0,
                label_quality="final",
                provenance={"environment": "live", "portfolio_id": "live-1"},
            ),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(
            recent_final_window_size=8,
            scope_environment="live",
            scope_portfolio="live-1",
            require_scope_provenance=True,
        )
    )

    snapshot, diagnostics = builder.load_recent_performance_snapshot_with_scope_diagnostics(repository)

    assert snapshot.recent_final_outcomes_count == 1
    assert snapshot.recent_realized_return_bps_sum == 5.0
    assert diagnostics.scoped_label_count == 1
    assert diagnostics.excluded_label_count == 2
    assert diagnostics.missing_scope_provenance_count == 2


def test_snapshot_builder_scope_filters_by_model_version(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(
                0,
                realized_return_bps=6.0,
                label_quality="final",
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                },
            ),
            _label(
                1,
                realized_return_bps=5.0,
                label_quality="final",
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                },
            ),
            _label(
                2,
                realized_return_bps=-20.0,
                label_quality="final",
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "B",
                },
            ),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(
            scope_environment="live",
            scope_portfolio="live-1",
            scope_model_version="A",
            require_scope_provenance=True,
            require_lineage_provenance=True,
        )
    )

    snapshot, diagnostics = builder.load_recent_performance_snapshot_with_scope_diagnostics(repository)

    assert snapshot.recent_final_outcomes_count == 2
    assert snapshot.recent_realized_return_bps_sum == 11.0
    assert diagnostics.scoped_label_count == 2
    assert diagnostics.excluded_label_count == 1


def test_snapshot_builder_scope_filters_by_decision_source_and_counts_missing_lineage(tmp_path: Path) -> None:
    repository = OpportunityShadowRepository(tmp_path)
    repository.append_outcome_labels(
        [
            _label(
                0,
                realized_return_bps=6.0,
                label_quality="final",
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "decision_source": "opportunity_ai_shadow",
                },
            ),
            _label(
                1,
                realized_return_bps=5.0,
                label_quality="final",
                provenance={"environment": "live", "portfolio_id": "live-1"},
            ),
            _label(
                2,
                realized_return_bps=-20.0,
                label_quality="final",
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "decision_source": "other_source",
                },
            ),
        ]
    )
    builder = OpportunityPerformanceSnapshotBuilder(
        OpportunityPerformanceSnapshotConfig(
            scope_environment="live",
            scope_portfolio="live-1",
            scope_decision_source="opportunity_ai_shadow",
            require_scope_provenance=True,
            require_lineage_provenance=True,
        )
    )

    snapshot, diagnostics = builder.load_recent_performance_snapshot_with_scope_diagnostics(repository)

    assert snapshot.recent_final_outcomes_count == 1
    assert snapshot.recent_realized_return_bps_sum == 6.0
    assert diagnostics.scoped_label_count == 1
    assert diagnostics.excluded_label_count == 2
    assert diagnostics.missing_lineage_provenance_count == 1


def _label(
    index: int,
    *,
    realized_return_bps: float,
    label_quality: str,
    provenance: dict[str, object] | None = None,
) -> OpportunityOutcomeLabel:
    return OpportunityOutcomeLabel(
        symbol="BTCUSDT",
        decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index),
        correlation_key=f"key-{index}",
        horizon_minutes=15,
        realized_return_bps=realized_return_bps,
        max_favorable_excursion_bps=max(realized_return_bps, 0.0),
        max_adverse_excursion_bps=min(realized_return_bps, 0.0),
        provenance=dict(provenance or {}),
        label_quality=label_quality,
    )
