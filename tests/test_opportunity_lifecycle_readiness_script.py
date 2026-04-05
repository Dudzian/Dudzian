from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from bot_core.ai.repository import FilesystemModelRepository
from bot_core.ai.trading_engine import OpportunityDecision, OpportunitySnapshot, TradingOpportunityAI
from bot_core.ai.trading_opportunity_shadow import OpportunityOutcomeLabel, OpportunityShadowRepository
from scripts.opportunity_lifecycle_readiness import build_opportunity_lifecycle_readiness_report
from tests._subprocess import run_cli_utf8


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


def _decision(model_version: str, edge: float) -> OpportunityDecision:
    return OpportunityDecision(
        symbol="BTCUSDT",
        decision_source="model",
        model_version=model_version,
        expected_edge_bps=edge,
        success_probability=0.7,
        confidence=0.3,
        proposed_direction="long",
        accepted=True,
        rejection_reason=None,
        rank=1,
        provenance={"probability_method": "test_probability"},
    )


def test_build_opportunity_lifecycle_readiness_report_returns_payload(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.2))
    engine.save_model(version="challenger-v2", activate=False)
    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    champion_record = engine.build_shadow_records(
        [_decision("champion-v1", 5.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": {"signal_strength": 0.9}},
    )[0]
    challenger_record = engine.build_shadow_records(
        [_decision("challenger-v2", 7.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": {"signal_strength": 0.8}},
    )[0]
    shadow_repo.append_shadow_records([champion_record, challenger_record])
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=champion_record.symbol,
                decision_timestamp=decision_time,
                correlation_key=champion_record.record_key,
                horizon_minutes=30,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=6.0,
                max_adverse_excursion_bps=-2.0,
            ),
            OpportunityOutcomeLabel(
                symbol=challenger_record.symbol,
                decision_timestamp=decision_time,
                correlation_key=challenger_record.record_key,
                horizon_minutes=30,
                realized_return_bps=8.0,
                max_favorable_excursion_bps=9.0,
                max_adverse_excursion_bps=-1.0,
            ),
        ]
    )

    report = build_opportunity_lifecycle_readiness_report(
        model_repository_path=tmp_path / "models",
        shadow_repository_path=tmp_path / "shadow",
        champion_version="champion-v1",
        challenger_version="challenger-v2",
    )
    assert report["champion_version"] == "champion-v1"
    assert report["challenger_version"] == "challenger-v2"
    assert "activation_readiness" in report


def test_opportunity_lifecycle_readiness_cli_writes_output(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples())
    engine.save_model(version="single-v1", activate=False)
    record = engine.build_shadow_records(
        [_decision("single-v1", 5.0)],
        decision_timestamp=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
        snapshot={"candidate_metadata": {"signal_strength": 0.9}},
    )[0]
    shadow_repo.append_shadow_records([record])
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=record.symbol,
                decision_timestamp=record.decision_timestamp,
                correlation_key=record.record_key,
                horizon_minutes=30,
                realized_return_bps=3.0,
                max_favorable_excursion_bps=4.0,
                max_adverse_excursion_bps=-1.0,
            )
        ]
    )
    output_path = tmp_path / "readiness.json"
    result = run_cli_utf8(
        [
            sys.executable,
            "scripts/opportunity_lifecycle_readiness.py",
            "--model-repo",
            str(tmp_path / "models"),
            "--shadow-repo",
            str(tmp_path / "shadow"),
            "--champion-version",
            "single-v1",
            "--challenger-version",
            "single-v1",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["champion_version"] == "single-v1"
