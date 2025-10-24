from __future__ import annotations

from collections import deque
from datetime import date, datetime, time, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from bot_core.ai import MarketRegime, MarketRegimeAssessment, RegimeHistory
from bot_core.auto_trader.schedule import ScheduleWindow
from bot_core.strategies import StrategyPresetWizard
from bot_core.strategies.regime_workflow import (
    ActivationCadenceStats,
    ActivationBlockerStats,
    ActivationAssessmentStats,
    ActivationCapabilityStats,
    ActivationDataStats,
    ActivationDecisionStats,
    ActivationHistoryStats,
    ActivationLicenseStats,
    ActivationRiskStats,
    ActivationOutcomeStats,
    ActivationPresetStats,
    ActivationRecommendationStats,
    ActivationReliabilityStats,
    ActivationTagStats,
    ActivationTransitionStats,
    ActivationUptimeStats,
    PresetAvailability,
    StrategyRegimeWorkflow,
)
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import install_capability_guard, reset_capability_guard


class _SequenceClassifier:
    """Deterministyczny klasyfikator zwracający przygotowane reżimy."""

    def __init__(self, regimes: list[MarketRegime]) -> None:
        self._regimes = deque(regimes)

    def assess(self, market_data: pd.DataFrame, *, symbol: str | None = None) -> MarketRegimeAssessment:
        if not self._regimes:
            regime = MarketRegime.TREND
        elif len(self._regimes) == 1:
            regime = self._regimes[0]
        else:
            regime = self._regimes.popleft()
        return MarketRegimeAssessment(
            regime=regime,
            confidence=0.9,
            risk_score=0.2,
            metrics={"volatility": 0.01, "drawdown": 0.02},
            symbol=symbol,
        )


class _AssessmentSequenceClassifier:
    """Klasyfikator zwracający przygotowane oceny reżimu."""

    def __init__(self, assessments: list[MarketRegimeAssessment]) -> None:
        self._assessments = deque(assessments)

    def assess(
        self,
        market_data: pd.DataFrame,
        *,
        symbol: str | None = None,
    ) -> MarketRegimeAssessment:
        if not self._assessments:
            return MarketRegimeAssessment(
                regime=MarketRegime.TREND,
                confidence=0.0,
                risk_score=0.0,
                metrics={},
                symbol=symbol,
            )
        if len(self._assessments) == 1:
            assessment = self._assessments[0]
        else:
            assessment = self._assessments.popleft()
        return MarketRegimeAssessment(
            regime=assessment.regime,
            confidence=assessment.confidence,
            risk_score=assessment.risk_score,
            metrics=assessment.metrics,
            symbol=symbol,
        )


class _StubDecisionEngine:
    """Minimalny orchestrator zwracający przygotowane rekomendacje."""

    def __init__(self, recommendations: list[str | None]) -> None:
        self._recommendations = deque(recommendations)

    def select_strategy(self, regime: MarketRegime | str) -> str | None:  # pragma: no cover - prosty stub
        if not self._recommendations:
            return None
        return self._recommendations.popleft()


def _market_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=60, freq="min", tz="UTC")
    return pd.DataFrame({"close": np.linspace(100, 101, len(index))}, index=index)


def _workflow(regimes: list[MarketRegime]) -> StrategyRegimeWorkflow:
    classifier = _SequenceClassifier(regimes)
    history = RegimeHistory(thresholds_loader=lambda: {})
    schedule = [ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)]
    return StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
        schedule_windows=schedule,
    )


def test_regime_workflow_switches_between_presets() -> None:
    workflow = _workflow([MarketRegime.TREND, MarketRegime.MEAN_REVERSION])
    signing_key = b"license-key"

    trend_version = workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[{"engine": "daily_trend_momentum", "parameters": {"window": 20}}],
        signing_key=signing_key,
        key_id="regime",
    )
    mean_version = workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[{"engine": "mean_reversion", "parameters": {"lookback": 15}}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="emergency",
        entries=[{"engine": "scalping"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    available = {"ohlcv", "technical_indicators", "spread_history", "order_book"}

    first = workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert first.regime is MarketRegime.TREND
    assert first.version.hash == trend_version.hash
    assert not first.used_fallback
    assert "daily_trend_momentum" in first.version.metadata["strategy_keys"]
    assert first.version.signature.get("key_id") == "regime"
    assert first.decision_candidates[0].strategy == "daily_trend_momentum"
    assert first.license_issues == ()

    second = workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc),
    )
    assert second.regime is MarketRegime.MEAN_REVERSION
    assert second.version.hash == mean_version.hash
    assert not second.used_fallback
    assert second.missing_data == ()
    assert second.license_issues == ()

    history = workflow.activation_history()
    assert len(history) == 2
    assert history[0].regime is MarketRegime.TREND
    assert history[1].regime is MarketRegime.MEAN_REVERSION


def test_regime_workflow_falls_back_when_data_missing() -> None:
    workflow = _workflow([MarketRegime.TREND, MarketRegime.MEAN_REVERSION])
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[{"engine": "daily_trend_momentum"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        key_id="regime",
    )
    emergency = workflow.register_emergency_preset(
        name="emergency",
        entries=[{"engine": "scalping"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()

    # Pierwsza aktywacja – trend.
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "order_book"},
        now=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
    )

    # Brak danych spread_history wymusza fallback.
    fallback_activation = workflow.activate(
        data,
        available_data={"ohlcv", "order_book"},
        now=datetime(2024, 1, 1, 10, 1, tzinfo=timezone.utc),
    )
    assert fallback_activation.used_fallback
    assert fallback_activation.version.hash == emergency.hash
    assert "spread_history" in fallback_activation.missing_data
    assert fallback_activation.version.signature.get("key_id") == "regime"
    assert fallback_activation.license_issues == ()
    assert fallback_activation.blocked_reason == "missing_data"
    history = workflow.activation_history()
    assert history[-1].used_fallback


def test_regime_workflow_respects_license_guard() -> None:
    workflow = _workflow([MarketRegime.MEAN_REVERSION])
    signing_key = b"license-key"

    mean_version = workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-pro",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        key_id="regime",
    )
    emergency = workflow.register_emergency_preset(
        name="day-fallback",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {
                "mean_reversion": True,
                "day_trading": True,
            },
            "environments": ["paper"],
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        activation = workflow.activate(
            _market_frame(),
            available_data={
                "ohlcv",
                "order_book",
                "spread_history",
                "technical_indicators",
            },
            now=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    assert activation.used_fallback
    assert activation.version.hash == emergency.hash
    assert activation.license_issues
    assert activation.blocked_reason == "license_blocked"
    assert any("professional" in issue or "licenc" in issue for issue in activation.license_issues)
    assert mean_version.hash != activation.version.hash
    assert workflow.activation_history()[-1].blocked_reason == "license_blocked"


def test_regime_workflow_uses_fallback_when_regime_missing() -> None:
    workflow = _workflow([MarketRegime.DAILY])
    signing_key = b"license-key"

    workflow.register_emergency_preset(
        name="emergency",
        entries=[{"engine": "scalping"}],
        signing_key=signing_key,
        key_id="regime",
    )

    activation = workflow.activate(
        _market_frame(),
        available_data={"ohlcv", "spread_history", "order_book"},
        now=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
    )

    assert activation.used_fallback
    assert activation.blocked_reason == "no_preset"
    assert activation.missing_data == ()
    assert activation.license_issues == ()
    assert workflow.activation_history()[-1].blocked_reason == "no_preset"


def test_inspect_presets_reports_license_and_missing_data_issues() -> None:
    workflow = _workflow([MarketRegime.TREND])
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="rescue",
        entries=[{"engine": "scalping"}],
        signing_key=signing_key,
        key_id="regime",
    )

    capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {
                "mean_reversion": True,
            },
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        reports = workflow.inspect_presets(
            available_data={"ohlcv", "spread_history", "order_book"},
            now=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    assert reports
    by_regime = {report.regime: report for report in reports}
    trend_report = by_regime[MarketRegime.TREND]
    assert not trend_report.ready
    assert trend_report.blocked_reason == "license_blocked"
    assert trend_report.missing_data == ()
    assert trend_report.license_issues
    assert not trend_report.schedule_blocked

    fallback_report = by_regime[None]
    assert fallback_report.blocked_reason == "license_blocked"
    assert fallback_report.license_issues
    assert fallback_report.missing_data == ()


def test_inspect_presets_marks_schedule_and_missing_data_constraints() -> None:
    classifier = _SequenceClassifier([MarketRegime.TREND])
    history = RegimeHistory(thresholds_loader=lambda: {})
    blocked_schedule = [
        ScheduleWindow(
            start=time(0, 0),
            end=time(0, 0),
            allow_trading=False,
        )
    ]
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
        schedule_windows=blocked_schedule,
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-standard",
        entries=[{"engine": "grid_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[
            {
                "engine": "day_trading",
                "license_tier": "standard",
                "required_data": ["ohlcv", "technical_indicators"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )

    reports = workflow.inspect_presets(
        available_data={"ohlcv"},
        now=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    assert reports
    for report in reports:
        assert isinstance(report, PresetAvailability)
        assert not report.ready
        assert report.schedule_blocked
        assert report.blocked_reason == "schedule_blocked"

    # Po zmianie harmonogramu brakujące dane są raportowane.
    open_schedule = [
        ScheduleWindow(
            start=time(0, 0),
            end=time(0, 0),
            allow_trading=True,
        )
    ]
    workflow_open = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_SequenceClassifier([MarketRegime.TREND]),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=open_schedule,
    )
    workflow_open.register_preset(
        MarketRegime.TREND,
        name="trend-standard",
        entries=[{"engine": "grid_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    open_reports = workflow_open.inspect_presets(
        available_data={"ohlcv"},
        now=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    trend_open = next(report for report in open_reports if report.regime == MarketRegime.TREND)
    assert trend_open.blocked_reason == "missing_data"
    assert "order_book" in trend_open.missing_data


def test_activation_history_limit_and_reset() -> None:
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_SequenceClassifier(
            [MarketRegime.TREND, MarketRegime.MEAN_REVERSION, MarketRegime.TREND]
        ),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=[ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)],
        activation_history_limit=2,
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend",
        entries=[{"engine": "daily_trend_momentum"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="emergency",
        entries=[{"engine": "scalping"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    available = {"ohlcv", "technical_indicators", "spread_history", "order_book"}

    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 1, 8, 1, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 1, 8, 2, tzinfo=timezone.utc),
    )

    history = workflow.activation_history()
    assert len(history) == 2
    assert history[0].regime is MarketRegime.MEAN_REVERSION
    assert history[1].regime is MarketRegime.TREND

    workflow.clear_history()
    assert workflow.activation_history() == ()


def test_activation_history_stats_reports_recent_summary() -> None:
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_SequenceClassifier(
            [MarketRegime.TREND, MarketRegime.MEAN_REVERSION, MarketRegime.MEAN_REVERSION]
        ),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=[ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)],
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "spread_history"],
                "license_tier": "standard",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-pro",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
                "license_tier": "professional",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[
            {
                "engine": "day_trading",
                "required_data": ["ohlcv", "technical_indicators"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()

    workflow.activate(
        data,
        available_data={"ohlcv", "spread_history", "order_book", "technical_indicators"},
        now=datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators"},
        now=datetime(2024, 1, 2, 8, 1, tzinfo=timezone.utc),
    )

    capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {
                "mean_reversion": True,
                "day_trading": True,
                "daily_trend_momentum": True,
            },
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        workflow.activate(
            data,
            available_data={
                "ohlcv",
                "spread_history",
                "order_book",
                "technical_indicators",
            },
            now=datetime(2024, 1, 2, 8, 2, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    stats = workflow.activation_history_stats()
    assert isinstance(stats, ActivationHistoryStats)
    assert stats.total == 3
    assert stats.fallback_count == 2
    assert stats.regime_counts[MarketRegime.TREND] == 1
    assert stats.regime_counts[MarketRegime.MEAN_REVERSION] == 2
    assert stats.preset_regime_counts[MarketRegime.TREND] == 1
    assert stats.preset_regime_counts[None] == 2
    assert stats.blocked_reasons["missing_data"] == 1
    assert stats.blocked_reasons["license_blocked"] == 1
    assert stats.missing_data["spread_history"] == 1
    assert any("professional" in issue for issue in stats.license_issue_counts)

    tail_stats = workflow.activation_history_stats(limit=1)
    assert tail_stats.total == 1
    assert tail_stats.fallback_count == 1
    assert tail_stats.blocked_reasons["license_blocked"] == 1


def test_activation_transition_stats_reports_transition_counts() -> None:
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_SequenceClassifier(
            [MarketRegime.TREND, MarketRegime.MEAN_REVERSION, MarketRegime.MEAN_REVERSION]
        ),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=[ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)],
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "spread_history"],
                "license_tier": "standard",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-pro",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
                "license_tier": "professional",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "day_trading", "required_data": ["ohlcv"]}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    full = {"ohlcv", "spread_history", "order_book", "technical_indicators"}

    workflow.activate(
        data,
        available_data=full,
        now=datetime(2024, 1, 4, 8, 0, tzinfo=timezone.utc),
    )

    workflow.activate(
        data,
        available_data={"ohlcv", "order_book", "technical_indicators"},
        now=datetime(2024, 1, 4, 8, 1, tzinfo=timezone.utc),
    )

    capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {
                "daily_trend_momentum": True,
                "day_trading": True,
            },
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        workflow.activate(
            data,
            available_data=full,
            now=datetime(2024, 1, 4, 8, 2, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    stats = workflow.activation_transition_stats()
    assert isinstance(stats, ActivationTransitionStats)
    assert stats.total == 2
    assert stats.fallback_transitions == 2
    assert stats.regime_transitions[(MarketRegime.TREND, MarketRegime.MEAN_REVERSION)] == 1
    assert stats.regime_transitions[(MarketRegime.MEAN_REVERSION, MarketRegime.MEAN_REVERSION)] == 1
    assert stats.preset_regime_transitions[(MarketRegime.TREND, None)] == 1
    assert stats.preset_regime_transitions[(None, None)] == 1
    assert stats.blocked_transitions[(None, "missing_data")] == 1
    assert stats.blocked_transitions[("missing_data", "license_blocked")] == 1

    tail_stats = workflow.activation_transition_stats(limit=2)
    assert tail_stats.total == 1
    assert tail_stats.fallback_transitions == 1
    assert tail_stats.regime_transitions[(MarketRegime.MEAN_REVERSION, MarketRegime.MEAN_REVERSION)] == 1
    assert tail_stats.blocked_transitions[("missing_data", "license_blocked")] == 1


def test_activation_history_frame_returns_enriched_dataframe() -> None:
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_SequenceClassifier(
            [MarketRegime.TREND, MarketRegime.MEAN_REVERSION, MarketRegime.MEAN_REVERSION]
        ),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=[ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)],
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "spread_history"],
                "license_tier": "standard",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-pro",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
                "license_tier": "professional",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[
            {
                "engine": "day_trading",
                "required_data": ["ohlcv", "technical_indicators"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()

    workflow.activate(
        data,
        available_data={"ohlcv", "spread_history", "order_book", "technical_indicators"},
        now=datetime(2024, 1, 3, 8, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators"},
        now=datetime(2024, 1, 3, 8, 1, tzinfo=timezone.utc),
    )

    capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {
                "mean_reversion": True,
                "day_trading": True,
                "daily_trend_momentum": True,
            },
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        workflow.activate(
            data,
            available_data={
                "ohlcv",
                "spread_history",
                "order_book",
                "technical_indicators",
            },
            now=datetime(2024, 1, 3, 8, 2, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    frame = workflow.activation_history_frame()
    assert list(frame.columns) == [
        "activated_at",
        "regime",
        "preset_regime",
        "preset_name",
        "preset_version",
        "used_fallback",
        "blocked_reason",
        "missing_data",
        "license_issues",
        "recommendation",
    ]
    assert len(frame) == 3
    assert list(frame["regime"]) == [
        MarketRegime.TREND,
        MarketRegime.MEAN_REVERSION,
        MarketRegime.MEAN_REVERSION,
    ]
    assert list(frame["preset_name"]) == ["trend-core", "fallback", "fallback"]
    assert frame.loc[1, "used_fallback"]
    assert frame.loc[1, "missing_data"] == ("spread_history",)
    assert frame.loc[2, "blocked_reason"] == "license_blocked"
    assert frame.loc[2, "license_issues"]

    tail_frame = workflow.activation_history_frame(limit=1)
    assert len(tail_frame) == 1
    assert tail_frame.iloc[0]["preset_name"] == "fallback"

    empty_workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_SequenceClassifier([MarketRegime.TREND]),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=[ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)],
    )
    empty_frame = empty_workflow.activation_history_frame()
    assert empty_frame.empty
    assert list(empty_frame.columns) == list(frame.columns)


def test_activation_cadence_stats_summarises_intervals() -> None:
    workflow = _workflow(
        [
            MarketRegime.TREND,
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.DAILY,
        ]
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[{"engine": "daily_trend_momentum"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.DAILY,
        name="daily-core",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    available = {"ohlcv", "technical_indicators", "order_book", "spread_history"}

    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 5, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 5, 9, 5, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 5, 9, 10, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 5, 9, 20, tzinfo=timezone.utc),
    )

    stats = workflow.activation_cadence_stats()
    assert isinstance(stats, ActivationCadenceStats)
    assert stats.total == 4
    assert stats.intervals == 3
    assert stats.min_interval == timedelta(minutes=5)
    assert stats.max_interval == timedelta(minutes=10)
    assert stats.mean_interval == timedelta(seconds=400)
    assert stats.median_interval == timedelta(minutes=5)
    assert stats.last_interval == timedelta(minutes=10)

    tail_stats = workflow.activation_cadence_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.intervals == 1
    assert tail_stats.min_interval == timedelta(minutes=10)
    assert tail_stats.max_interval == timedelta(minutes=10)
    assert tail_stats.mean_interval == timedelta(minutes=10)
    assert tail_stats.median_interval == timedelta(minutes=10)
    assert tail_stats.last_interval == timedelta(minutes=10)

    single_stats = workflow.activation_cadence_stats(limit=1)
    assert single_stats.total == 1
    assert single_stats.intervals == 0
    assert single_stats.min_interval is None
    assert single_stats.max_interval is None
    assert single_stats.mean_interval is None
    assert single_stats.median_interval is None
    assert single_stats.last_interval is None


def test_activation_uptime_stats_tracks_regime_and_fallback_time() -> None:
    workflow = _workflow(
        [
            MarketRegime.TREND,
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.DAILY,
        ]
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[{"engine": "daily_trend_momentum"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "scalping"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    full = {"ohlcv", "order_book", "spread_history", "technical_indicators"}

    workflow.activate(
        data,
        available_data=full,
        now=datetime(2024, 1, 6, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "order_book"},
        now=datetime(2024, 1, 6, 9, 10, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=full,
        now=datetime(2024, 1, 6, 9, 25, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=full,
        now=datetime(2024, 1, 6, 9, 45, tzinfo=timezone.utc),
    )

    history = workflow.activation_history()
    assert history[1].used_fallback
    assert history[3].used_fallback

    horizon = datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc)
    stats = workflow.activation_uptime_stats(until=horizon)
    assert isinstance(stats, ActivationUptimeStats)
    assert stats.total == 4
    assert stats.duration == timedelta(minutes=60)
    assert stats.regime_uptime[MarketRegime.TREND] == timedelta(minutes=25)
    assert stats.regime_uptime[MarketRegime.MEAN_REVERSION] == timedelta(minutes=20)
    assert stats.regime_uptime[MarketRegime.DAILY] == timedelta(minutes=15)
    assert stats.preset_uptime[MarketRegime.TREND] == timedelta(minutes=10)
    assert stats.preset_uptime[MarketRegime.MEAN_REVERSION] == timedelta(minutes=20)
    assert stats.preset_uptime[None] == timedelta(minutes=30)
    assert stats.fallback_uptime == timedelta(minutes=30)

    tail_stats = workflow.activation_uptime_stats(limit=2, until=horizon)
    assert tail_stats.total == 2
    assert tail_stats.duration == timedelta(minutes=35)
    assert tail_stats.regime_uptime[MarketRegime.MEAN_REVERSION] == timedelta(minutes=20)
    assert tail_stats.regime_uptime[MarketRegime.DAILY] == timedelta(minutes=15)
    assert tail_stats.preset_uptime[None] == timedelta(minutes=15)
    assert tail_stats.fallback_uptime == timedelta(minutes=15)

    single_stats = workflow.activation_uptime_stats(limit=1, until=horizon)
    assert single_stats.total == 1
    assert single_stats.duration == timedelta(minutes=15)
    assert single_stats.regime_uptime[MarketRegime.DAILY] == timedelta(minutes=15)
    assert single_stats.preset_uptime[None] == timedelta(minutes=15)
    assert single_stats.fallback_uptime == timedelta(minutes=15)


def test_activation_reliability_stats_reports_success_and_blocked() -> None:
    workflow = _workflow([
        MarketRegime.TREND,
        MarketRegime.TREND,
        MarketRegime.MEAN_REVERSION,
        MarketRegime.MEAN_REVERSION,
    ])
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-standard",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "parameters": {"window": 10},
                "required_data": ["ohlcv", "technical_indicators", "spread_history"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-pro",
        entries=[{"engine": "mean_reversion", "parameters": {"lookback": 15}}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    trend_full = {"ohlcv", "technical_indicators", "spread_history", "order_book"}
    trend_limited = {"ohlcv", "technical_indicators", "order_book"}
    mean_full = {"ohlcv", "spread_history", "order_book", "technical_indicators"}

    workflow.activate(
        data,
        available_data=trend_full,
        now=datetime(2024, 1, 7, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=trend_limited,
        now=datetime(2024, 1, 7, 9, 5, tzinfo=timezone.utc),
    )

    capabilities = build_capabilities_from_payload(
            {
                "edition": "standard",
                "strategies": {"daily_trend_momentum": True, "day_trading": True},
                "environments": ["paper"],
            },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        workflow.activate(
            data,
            available_data=mean_full,
            now=datetime(2024, 1, 7, 9, 10, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    workflow.activate(
        data,
        available_data=mean_full,
        now=datetime(2024, 1, 7, 9, 20, tzinfo=timezone.utc),
    )

    stats = workflow.activation_reliability_stats()
    assert isinstance(stats, ActivationReliabilityStats)
    assert stats.total == 4
    assert stats.completed == 2
    assert stats.fallback_count == 2
    assert stats.blocked_count == 2
    assert stats.completed_ratio == pytest.approx(0.5)
    assert stats.fallback_ratio == pytest.approx(0.5)
    assert stats.blocked_ratio == pytest.approx(0.5)

    tail_stats = workflow.activation_reliability_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.completed == 1
    assert tail_stats.fallback_count == 1
    assert tail_stats.blocked_count == 1
    assert tail_stats.completed_ratio == pytest.approx(0.5)
    assert tail_stats.fallback_ratio == pytest.approx(0.5)
    assert tail_stats.blocked_ratio == pytest.approx(0.5)

    empty_workflow = _workflow([MarketRegime.TREND])
    empty_stats = empty_workflow.activation_reliability_stats()
    assert empty_stats.total == 0
    assert empty_stats.completed_ratio == pytest.approx(0.0)
    assert empty_stats.fallback_ratio == pytest.approx(0.0)
    assert empty_stats.blocked_ratio == pytest.approx(0.0)


def test_activation_outcome_stats_groups_by_regime_and_preset() -> None:
    workflow = _workflow([
        MarketRegime.TREND,
        MarketRegime.TREND,
        MarketRegime.MEAN_REVERSION,
        MarketRegime.MEAN_REVERSION,
    ])
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-standard",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "parameters": {"window": 10},
                "required_data": ["ohlcv", "technical_indicators", "spread_history"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-pro",
        entries=[{"engine": "mean_reversion", "parameters": {"lookback": 15}}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    trend_full = {"ohlcv", "technical_indicators", "spread_history", "order_book"}
    trend_limited = {"ohlcv", "technical_indicators", "order_book"}
    mean_full = {"ohlcv", "spread_history", "order_book", "technical_indicators"}

    workflow.activate(
        data,
        available_data=trend_full,
        now=datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=trend_limited,
        now=datetime(2024, 1, 8, 9, 5, tzinfo=timezone.utc),
    )

    capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {"daily_trend_momentum": True, "day_trading": True},
            "environments": ["paper"],
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        workflow.activate(
            data,
            available_data=mean_full,
            now=datetime(2024, 1, 8, 9, 10, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    workflow.activate(
        data,
        available_data=mean_full,
        now=datetime(2024, 1, 8, 9, 20, tzinfo=timezone.utc),
    )

    stats = workflow.activation_outcome_stats()
    assert isinstance(stats, ActivationOutcomeStats)
    assert stats.total == 4
    assert stats.completed_total == 2
    assert stats.fallback_total == 2
    assert stats.blocked_total == 2
    assert stats.regime_completed[MarketRegime.TREND] == 1
    assert stats.regime_completed[MarketRegime.MEAN_REVERSION] == 1
    assert stats.regime_fallback[MarketRegime.TREND] == 1
    assert stats.regime_fallback[MarketRegime.MEAN_REVERSION] == 1
    assert stats.regime_blocked[MarketRegime.TREND] == 1
    assert stats.regime_blocked[MarketRegime.MEAN_REVERSION] == 1
    assert stats.preset_completed[MarketRegime.TREND] == 1
    assert stats.preset_completed[MarketRegime.MEAN_REVERSION] == 1
    assert stats.preset_fallback[None] == 2
    assert stats.preset_blocked[None] == 2

    tail_stats = workflow.activation_outcome_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.completed_total == 1
    assert tail_stats.fallback_total == 1
    assert tail_stats.blocked_total == 1
    assert tail_stats.regime_completed[MarketRegime.MEAN_REVERSION] == 1
    assert tail_stats.regime_fallback[MarketRegime.MEAN_REVERSION] == 1
    assert tail_stats.regime_blocked[MarketRegime.MEAN_REVERSION] == 1
    assert tail_stats.preset_completed[MarketRegime.MEAN_REVERSION] == 1
    assert tail_stats.preset_fallback[None] == 1
    assert tail_stats.preset_blocked[None] == 1

    empty_stats = StrategyRegimeWorkflow().activation_outcome_stats()
    assert empty_stats.total == 0
    assert empty_stats.completed_total == 0
    assert empty_stats.fallback_total == 0
    assert empty_stats.blocked_total == 0


def test_activation_preset_stats_reports_usage_and_versions() -> None:
    classifier = _SequenceClassifier(
        [
            MarketRegime.TREND,
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.TREND,
        ]
    )
    history = RegimeHistory(thresholds_loader=lambda: {})
    schedule = [ScheduleWindow(start=time(8, 0), end=time(18, 0), allow_trading=True)]
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
        schedule_windows=schedule,
    )
    signing_key = b"license-key"

    trend_version = workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": [
                    "ohlcv",
                    "technical_indicators",
                    "spread_history",
                ],
                "license_tier": "standard",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    mean_version = workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
                "license_tier": "professional",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    fallback_version = workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "day_trading", "required_data": ["ohlcv"]}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    full_data = {"ohlcv", "technical_indicators", "spread_history", "order_book"}
    trend_missing = {"ohlcv", "technical_indicators", "order_book"}

    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc),
    )

    workflow.activate(
        data,
        available_data=trend_missing,
        now=datetime(2024, 1, 3, 9, 5, tzinfo=timezone.utc),
    )

    limited_capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {"daily_trend_momentum": True, "day_trading": True},
            "environments": ["paper"],
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(limited_capabilities)
    try:
        workflow.activate(
            data,
            available_data=full_data,
            now=datetime(2024, 1, 3, 9, 10, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 3, 9, 15, tzinfo=timezone.utc),
    )

    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 3, 7, 55, tzinfo=timezone.utc),
    )

    stats = workflow.activation_preset_stats()
    assert isinstance(stats, ActivationPresetStats)
    assert stats.total == 5
    assert stats.preset_usage["trend-pro"] == 1
    assert stats.preset_usage["mean-core"] == 1
    assert stats.preset_usage["fallback"] == 3
    assert stats.version_usage[trend_version.hash] == 1
    assert stats.version_usage[mean_version.hash] == 1
    assert stats.version_usage[fallback_version.hash] == 3
    assert stats.versions_by_preset["fallback"][fallback_version.hash] == 3
    assert stats.regime_preset_usage[MarketRegime.TREND]["trend-pro"] == 1
    assert stats.regime_preset_usage[MarketRegime.TREND]["fallback"] == 2
    assert stats.regime_preset_usage[MarketRegime.MEAN_REVERSION]["mean-core"] == 1
    assert stats.regime_preset_usage[MarketRegime.MEAN_REVERSION]["fallback"] == 1
    assert stats.fallback_by_preset["fallback"] == 3
    assert stats.fallback_by_version[fallback_version.hash] == 3
    assert stats.fallback_by_regime[MarketRegime.TREND] == 2
    assert stats.fallback_by_regime[MarketRegime.MEAN_REVERSION] == 1
    assert stats.blocked_by_preset["fallback"] == 3
    assert stats.blocked_by_version[fallback_version.hash] == 3
    assert stats.blocked_by_regime[MarketRegime.TREND] == 2
    assert stats.blocked_by_regime[MarketRegime.MEAN_REVERSION] == 1

    tail_stats = workflow.activation_preset_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.preset_usage["mean-core"] == 1
    assert tail_stats.preset_usage["fallback"] == 1
    assert tail_stats.version_usage[mean_version.hash] == 1
    assert tail_stats.version_usage[fallback_version.hash] == 1
    assert tail_stats.regime_preset_usage[MarketRegime.MEAN_REVERSION]["mean-core"] == 1
    assert tail_stats.regime_preset_usage[MarketRegime.TREND]["fallback"] == 1
    assert tail_stats.fallback_by_preset["fallback"] == 1
    assert tail_stats.blocked_by_preset["fallback"] == 1

    empty_stats = StrategyRegimeWorkflow().activation_preset_stats()
    assert empty_stats.total == 0
    assert empty_stats.preset_usage == {}


def test_activation_license_stats_reports_regimes_and_presets() -> None:
    workflow = _workflow(
        [
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.TREND,
        ]
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "scalping",
                "license_tier": "professional",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-pro",
        entries=[
            {
                "engine": "mean_reversion",
                "license_tier": "professional",
                "capability": "mean_reversion_pro",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    fallback_version = workflow.register_emergency_preset(
        name="emergency",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    restricted_capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {
                "daily_trend_momentum": True,
                "day_trading": True,
            },
            "environments": ["paper"],
        },
        effective_date=date(2024, 1, 1),
    )

    data = _market_frame()
    install_capability_guard(restricted_capabilities)
    try:
        first = workflow.activate(
            data,
            available_data={
                "ohlcv",
                "technical_indicators",
                "spread_history",
                "order_book",
            },
            now=datetime(2024, 1, 4, 9, 0, tzinfo=timezone.utc),
        )
        second = workflow.activate(
            data,
            available_data={
                "ohlcv",
                "technical_indicators",
                "spread_history",
                "order_book",
            },
            now=datetime(2024, 1, 4, 9, 5, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    third = workflow.activate(
        data,
        available_data={
            "ohlcv",
            "technical_indicators",
            "spread_history",
            "order_book",
        },
        now=datetime(2024, 1, 4, 9, 10, tzinfo=timezone.utc),
    )

    assert first.used_fallback and first.blocked_reason == "license_blocked"
    assert second.used_fallback and second.blocked_reason == "license_blocked"
    assert third.used_fallback is False
    assert fallback_version.hash == first.version.hash
    assert fallback_version.hash == second.version.hash

    stats = workflow.activation_license_stats()
    assert isinstance(stats, ActivationLicenseStats)
    assert stats.total == 3
    assert stats.activations_with_issues == 2
    assert stats.blocked_by_license == 2
    assert stats.fallback_due_to_license == 2
    assert stats.regimes_with_issues[MarketRegime.TREND] == 1
    assert stats.regimes_with_issues[MarketRegime.MEAN_REVERSION] == 1
    assert stats.presets_with_issues["emergency"] == 2
    assert any("trend-pro" in issue or "licenc" in issue for issue in stats.issue_counts)
    assert any(
        "mean-pro" in issue or "licenc" in issue for issue in stats.issue_counts
    )
    assert any("scalping" in issue for issue in stats.issue_counts)
    assert any("mean_reversion" in issue for issue in stats.issue_counts)
    assert "emergency" in stats.issues_by_preset
    assert stats.issues_by_regime[MarketRegime.TREND]
    assert stats.issues_by_regime[MarketRegime.MEAN_REVERSION]

    tail_stats = workflow.activation_license_stats(limit=1)
    assert tail_stats.total == 1
    assert tail_stats.activations_with_issues == 0
    assert tail_stats.issue_counts == {}

    empty_stats = StrategyRegimeWorkflow().activation_license_stats()
    assert empty_stats.total == 0
    assert empty_stats.activations_with_issues == 0
    assert empty_stats.issue_counts == {}


def test_activation_assessment_stats_summarises_confidence_and_metrics() -> None:
    assessments = [
        MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.6,
            risk_score=0.3,
            metrics={"volatility": 0.01, "drawdown": 0.02},
        ),
        MarketRegimeAssessment(
            regime=MarketRegime.MEAN_REVERSION,
            confidence=0.3,
            risk_score=0.7,
            metrics={"volatility": 0.015, "drawdown": 0.03, "skew": -0.1},
        ),
        MarketRegimeAssessment(
            regime=MarketRegime.DAILY,
            confidence=0.9,
            risk_score=0.2,
            metrics={"volatility": 0.02, "drawdown": 0.01},
        ),
    ]
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_AssessmentSequenceClassifier(assessments),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=[ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)],
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[{"engine": "daily_trend_momentum"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.DAILY,
        name="daily-core",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    available = {"ohlcv", "technical_indicators", "order_book", "spread_history"}

    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 10, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 10, 9, 1, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 10, 9, 2, tzinfo=timezone.utc),
    )

    stats = workflow.activation_assessment_stats()
    assert isinstance(stats, ActivationAssessmentStats)
    assert stats.total == 3
    assert stats.regime_counts[MarketRegime.TREND] == 1
    assert stats.regime_counts[MarketRegime.MEAN_REVERSION] == 1
    assert stats.regime_counts[MarketRegime.DAILY] == 1
    assert stats.min_confidence == pytest.approx(0.3)
    assert stats.max_confidence == pytest.approx(0.9)
    assert stats.mean_confidence == pytest.approx(0.6)
    assert stats.min_risk_score == pytest.approx(0.2)
    assert stats.max_risk_score == pytest.approx(0.7)
    assert stats.mean_risk_score == pytest.approx(0.4)
    assert stats.metrics_mean["volatility"] == pytest.approx(0.015)
    assert stats.metrics_min["volatility"] == pytest.approx(0.01)
    assert stats.metrics_max["volatility"] == pytest.approx(0.02)
    assert stats.metrics_mean["drawdown"] == pytest.approx(0.02)
    assert stats.metrics_min["drawdown"] == pytest.approx(0.01)
    assert stats.metrics_max["drawdown"] == pytest.approx(0.03)
    assert stats.metrics_mean["skew"] == pytest.approx(-0.1)
    assert stats.metrics_min["skew"] == pytest.approx(-0.1)
    assert stats.metrics_max["skew"] == pytest.approx(-0.1)

    tail_stats = workflow.activation_assessment_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.regime_counts[MarketRegime.MEAN_REVERSION] == 1
    assert tail_stats.regime_counts[MarketRegime.DAILY] == 1
    assert tail_stats.min_confidence == pytest.approx(0.3)
    assert tail_stats.max_confidence == pytest.approx(0.9)
    assert tail_stats.mean_confidence == pytest.approx(0.6)
    assert tail_stats.min_risk_score == pytest.approx(0.2)
    assert tail_stats.max_risk_score == pytest.approx(0.7)
    assert tail_stats.mean_risk_score == pytest.approx(0.45)
    assert tail_stats.metrics_mean["volatility"] == pytest.approx(0.0175)
    assert tail_stats.metrics_min["volatility"] == pytest.approx(0.015)
    assert tail_stats.metrics_max["volatility"] == pytest.approx(0.02)
    assert tail_stats.metrics_mean["drawdown"] == pytest.approx(0.02)
    assert tail_stats.metrics_min["drawdown"] == pytest.approx(0.01)
    assert tail_stats.metrics_max["drawdown"] == pytest.approx(0.03)
    assert tail_stats.metrics_mean["skew"] == pytest.approx(-0.1)
    assert tail_stats.metrics_min["skew"] == pytest.approx(-0.1)
    assert tail_stats.metrics_max["skew"] == pytest.approx(-0.1)

    empty_stats = StrategyRegimeWorkflow().activation_assessment_stats()
    assert empty_stats.total == 0
    assert empty_stats.regime_counts == {}
    assert empty_stats.metrics_mean == {}


def test_activation_data_stats_captures_missing_inputs() -> None:
    workflow = _workflow([MarketRegime.TREND, MarketRegime.TREND, MarketRegime.TREND])
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "technical_indicators", "spread_history"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "scalping", "required_data": ["ohlcv"]}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()

    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "spread_history", "order_book"},
        now=datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "spread_history", "order_book"},
        now=datetime(2024, 1, 3, 12, 5, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "order_book"},
        now=datetime(2024, 1, 3, 12, 10, tzinfo=timezone.utc),
    )

    stats = workflow.activation_data_stats()
    assert isinstance(stats, ActivationDataStats)
    assert stats.total == 3
    assert stats.activations_with_missing == 2
    assert stats.blocked_due_to_missing == 2
    assert stats.fallback_due_to_missing == 2
    assert stats.missing_data_counts["technical_indicators"] == 1
    assert stats.missing_data_counts["spread_history"] == 1
    assert stats.missing_data_by_regime[MarketRegime.TREND]["technical_indicators"] == 1
    assert stats.missing_data_by_regime[MarketRegime.TREND]["spread_history"] == 1

    tail_stats = workflow.activation_data_stats(limit=1)
    assert tail_stats.total == 1
    assert tail_stats.activations_with_missing == 1
    assert tail_stats.missing_data_counts["spread_history"] == 1

    empty_stats = StrategyRegimeWorkflow().activation_data_stats()
    assert empty_stats.total == 0
    assert empty_stats.activations_with_missing == 0
    assert empty_stats.missing_data_counts == {}


def test_activation_decision_stats_aggregates_candidates() -> None:
    workflow = _workflow(
        [
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.TREND,
        ]
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "parameters": {"window": 20},
                "metadata": {
                    "expected_return_bps": 45,
                    "expected_probability": 0.7,
                    "notional": 5000,
                },
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-suite",
        entries=[
            {
                "engine": "mean_reversion",
                "parameters": {"lookback": 12},
                "metadata": {
                    "expected_return_bps": 25,
                    "expected_probability": 0.6,
                    "notional": 3000,
                },
            },
            {
                "engine": "statistical_arbitrage",
                "parameters": {"pairs": 4},
                "metadata": {
                    "expected_return_bps": 15,
                    "expected_probability": 0.5,
                    "notional": 2500,
                },
            },
        ],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    available = {"ohlcv", "technical_indicators", "spread_history"}

    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 4, 8, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 4, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=available,
        now=datetime(2024, 1, 4, 10, 0, tzinfo=timezone.utc),
    )

    stats = workflow.activation_decision_stats()
    assert isinstance(stats, ActivationDecisionStats)
    assert stats.total == 3
    assert stats.activations_with_candidates == 3
    assert stats.candidate_count == 4
    assert stats.strategy_counts["daily_trend_momentum"] == 2
    assert stats.strategy_counts["mean_reversion"] == 1
    assert stats.strategy_counts["statistical_arbitrage"] == 1
    assert (
        stats.strategy_counts_by_regime[MarketRegime.TREND]["daily_trend_momentum"]
        == 2
    )
    assert (
        stats.strategy_counts_by_regime[MarketRegime.MEAN_REVERSION]["mean_reversion"]
        == 1
    )
    assert (
        stats.strategy_counts_by_regime[MarketRegime.MEAN_REVERSION][
            "statistical_arbitrage"
        ]
        == 1
    )
    assert stats.mean_expected_return_bps == pytest.approx(32.5)
    assert stats.mean_expected_probability == pytest.approx(0.625)
    assert stats.expected_value_sum_bps == pytest.approx(85.5)
    assert stats.expected_value_by_strategy["daily_trend_momentum"] == pytest.approx(
        63.0
    )
    assert stats.expected_value_by_strategy["mean_reversion"] == pytest.approx(15.0)
    assert stats.expected_value_by_strategy["statistical_arbitrage"] == pytest.approx(
        7.5
    )
    assert stats.total_notional == pytest.approx(15500.0)
    assert stats.notional_by_strategy["daily_trend_momentum"] == pytest.approx(10000.0)
    assert stats.notional_by_strategy["mean_reversion"] == pytest.approx(3000.0)
    assert stats.notional_by_strategy["statistical_arbitrage"] == pytest.approx(2500.0)

    tail_stats = workflow.activation_decision_stats(limit=1)
    assert tail_stats.total == 1
    assert tail_stats.candidate_count == 1
    assert tail_stats.strategy_counts["daily_trend_momentum"] == 1
    assert tail_stats.mean_expected_return_bps == pytest.approx(45.0)
    assert tail_stats.mean_expected_probability == pytest.approx(0.7)
    assert tail_stats.expected_value_sum_bps == pytest.approx(31.5)
    assert tail_stats.total_notional == pytest.approx(5000.0)

    empty_stats = StrategyRegimeWorkflow().activation_decision_stats()
    assert empty_stats.total == 0
    assert empty_stats.candidate_count == 0
    assert empty_stats.strategy_counts == {}


def test_activation_recommendation_stats_tracks_matches_and_fallbacks() -> None:
    decision_engine = _StubDecisionEngine(
        ["daily_trend_momentum", "mean_reversion", "grid_trading"]
    )
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=_SequenceClassifier(
            [
                MarketRegime.TREND,
                MarketRegime.MEAN_REVERSION,
                MarketRegime.TREND,
            ]
        ),  # type: ignore[arg-type]
        history=RegimeHistory(thresholds_loader=lambda: {}),
        schedule_windows=[ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True)],
        decision_engine=decision_engine,
    )

    signing_key = b"license-key"
    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "parameters": {"window": 20},
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-suite",
        entries=[
            {
                "engine": "mean_reversion",
                "parameters": {"lookback": 12},
                "required_data": ["spread_history"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    full_data = {"ohlcv", "technical_indicators", "spread_history", "order_book"}

    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "order_book"},
        now=datetime(2024, 1, 5, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc),
    )

    stats = workflow.activation_recommendation_stats()
    assert isinstance(stats, ActivationRecommendationStats)
    assert stats.total == 3
    assert stats.with_recommendation == 3
    assert stats.matched_presets == 1
    assert stats.matched_candidates == 1
    assert stats.fallback_recommendations == 1
    assert stats.unmatched_recommendations == 2
    assert stats.recommendation_counts["daily_trend_momentum"] == 1
    assert stats.recommendation_counts["mean_reversion"] == 1
    assert stats.recommendation_counts["grid_trading"] == 1
    assert stats.recommendation_by_regime[MarketRegime.TREND]["daily_trend_momentum"] == 1
    assert stats.recommendation_by_regime[MarketRegime.TREND]["grid_trading"] == 1
    assert stats.recommendation_by_regime[MarketRegime.MEAN_REVERSION]["mean_reversion"] == 1

    tail_stats = workflow.activation_recommendation_stats(limit=1)
    assert tail_stats.total == 1
    assert tail_stats.with_recommendation == 1
    assert tail_stats.matched_presets == 0
    assert tail_stats.unmatched_recommendations == 1
    assert tail_stats.recommendation_counts == {"grid_trading": 1}

    empty_stats = StrategyRegimeWorkflow(decision_engine=_StubDecisionEngine([])).activation_recommendation_stats()
    assert empty_stats.total == 0
    assert empty_stats.with_recommendation == 0
    assert empty_stats.recommendation_counts == {}


def test_activation_tag_stats_aggregates_usage_and_blocks() -> None:
    classifier = _SequenceClassifier(
        [
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.TREND,
        ]
    )
    history = RegimeHistory(thresholds_loader=lambda: {})
    schedule = [ScheduleWindow(start=time(8, 0), end=time(18, 0), allow_trading=True)]
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
        schedule_windows=schedule,
    )

    signing_key = b"license-key"
    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "technical_indicators"],
                "tags": ["Momentum", "Trend"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-suite",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
                "tags": ["mean_reversion", "Spread"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[
            {
                "engine": "day_trading",
                "tags": ["Fallback", "Hedge"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    full_data = {"ohlcv", "technical_indicators", "spread_history"}

    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=full_data | {"order_book"},
        now=datetime(2024, 1, 3, 9, 30, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators"},
        now=datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators"},
        now=datetime(2024, 1, 3, 7, 0, tzinfo=timezone.utc),
    )

    stats = workflow.activation_tag_stats()
    assert isinstance(stats, ActivationTagStats)
    assert stats.total == 4
    assert stats.activations_with_tags == 4
    assert stats.tag_counts["momentum"] == 3
    assert stats.tag_counts["trend"] == 1
    assert stats.tag_counts["mean_reversion"] == 1
    assert stats.tag_counts["spread"] == 1
    assert stats.tag_counts["stat_arbitrage"] == 1
    assert stats.tag_counts["intraday"] == 2
    assert stats.tag_counts["fallback"] == 2
    assert stats.fallback_tag_counts["fallback"] == 2
    assert stats.fallback_tag_counts["hedge"] == 2
    assert stats.fallback_tag_counts["momentum"] == 2
    assert stats.blocked_tag_counts["fallback"] == 2
    assert stats.blocked_tag_counts["momentum"] == 2
    assert stats.tags_by_regime[MarketRegime.TREND]["momentum"] == 2
    assert stats.tags_by_regime[MarketRegime.TREND]["fallback"] == 1
    assert stats.tags_by_regime[MarketRegime.TREND]["intraday"] == 1
    assert stats.tags_by_regime[MarketRegime.MEAN_REVERSION]["mean_reversion"] == 1
    assert stats.tags_by_regime[MarketRegime.MEAN_REVERSION]["fallback"] == 1
    assert stats.tags_by_regime[MarketRegime.MEAN_REVERSION]["momentum"] == 1

    tail_stats = workflow.activation_tag_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.tag_counts["fallback"] == 2
    assert tail_stats.tag_counts["momentum"] == 2
    assert tail_stats.tag_counts["intraday"] == 2

    empty_stats = StrategyRegimeWorkflow().activation_tag_stats()
    assert empty_stats.total == 0
    assert empty_stats.tag_counts == {}


def test_activation_capability_stats_aggregates_usage_and_blocks() -> None:
    classifier = _SequenceClassifier(
        [
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.TREND,
        ]
    )
    history = RegimeHistory(thresholds_loader=lambda: {})
    schedule = [ScheduleWindow(start=time(8, 0), end=time(18, 0), allow_trading=True)]
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
        schedule_windows=schedule,
    )

    signing_key = b"license-key"
    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "technical_indicators", "order_book"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-suite",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
                "capability": "spread-maker",
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[
            {
                "engine": "day_trading",
                "required_data": ["ohlcv", "technical_indicators", "order_book"],
            },
            {
                "engine": "scalping",
                "required_data": ["ohlcv", "technical_indicators", "order_book"],
            },
        ],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    full_data = {"ohlcv", "technical_indicators", "spread_history", "order_book"}

    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 3, 9, 30, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "order_book"},
        now=datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "order_book"},
        now=datetime(2024, 1, 3, 7, 0, tzinfo=timezone.utc),
    )

    stats = workflow.activation_capability_stats()
    assert isinstance(stats, ActivationCapabilityStats)
    assert stats.total == 4
    assert stats.activations_with_capabilities == 4
    assert stats.capability_counts["trend_d1"] == 1
    assert stats.capability_counts["mean_reversion"] == 1
    assert stats.capability_counts["day_trading"] == 2
    assert stats.capability_counts["scalping"] == 2
    assert stats.capabilities_by_regime[MarketRegime.TREND]["trend_d1"] == 1
    assert stats.capabilities_by_regime[MarketRegime.TREND]["day_trading"] == 1
    assert stats.capabilities_by_regime[MarketRegime.TREND]["scalping"] == 1
    assert stats.capabilities_by_regime[MarketRegime.MEAN_REVERSION]["mean_reversion"] == 1
    assert stats.capabilities_by_regime[MarketRegime.MEAN_REVERSION]["day_trading"] == 1
    assert stats.capabilities_by_regime[MarketRegime.MEAN_REVERSION]["scalping"] == 1
    assert stats.fallback_capability_counts["day_trading"] == 2
    assert stats.fallback_capability_counts["scalping"] == 2
    assert stats.blocked_capability_counts["day_trading"] == 2
    assert stats.blocked_capability_counts["scalping"] == 2

    tail_stats = workflow.activation_capability_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.capability_counts["day_trading"] == 2
    assert tail_stats.capability_counts["scalping"] == 2
    assert "trend_d1" not in tail_stats.capability_counts

    empty_stats = StrategyRegimeWorkflow().activation_capability_stats()
    assert empty_stats.total == 0
    assert empty_stats.capability_counts == {}


def test_activation_risk_stats_summarises_classes_and_fallbacks() -> None:
    classifier = _SequenceClassifier(
        [
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.TREND,
        ]
    )
    history = RegimeHistory(thresholds_loader=lambda: {})
    schedule = [ScheduleWindow(start=time(8, 0), end=time(18, 0), allow_trading=True)]
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
        schedule_windows=schedule,
    )

    signing_key = b"license-key"
    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "technical_indicators"],
                "risk_classes": ["Directional", "Trend"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-suite",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
                "risk_classes": ["Statistical", "Spread"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[
            {
                "engine": "day_trading",
                "risk_classes": ["Fallback", "Defensive"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()
    full_data = {"ohlcv", "technical_indicators", "spread_history"}

    workflow.activate(
        data,
        available_data=full_data,
        now=datetime(2024, 1, 4, 9, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data=full_data | {"order_book"},
        now=datetime(2024, 1, 4, 9, 30, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators"},
        now=datetime(2024, 1, 4, 10, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators"},
        now=datetime(2024, 1, 4, 7, 0, tzinfo=timezone.utc),
    )

    stats = workflow.activation_risk_stats()
    assert isinstance(stats, ActivationRiskStats)
    assert stats.total == 4
    assert stats.activations_with_risk == 4
    assert stats.risk_counts["directional"] == 1
    assert stats.risk_counts["momentum"] == 3
    assert stats.risk_counts["trend"] == 1
    assert stats.risk_counts["statistical"] == 1
    assert stats.risk_counts["mean_reversion"] == 1
    assert stats.risk_counts["spread"] == 1
    assert stats.risk_counts["intraday"] == 2
    assert stats.risk_counts["fallback"] == 2
    assert stats.risk_counts["defensive"] == 2
    assert stats.fallback_risk_counts["intraday"] == 2
    assert stats.fallback_risk_counts["momentum"] == 2
    assert stats.fallback_risk_counts["fallback"] == 2
    assert stats.fallback_risk_counts["defensive"] == 2
    assert stats.blocked_risk_counts["intraday"] == 2
    assert stats.blocked_risk_counts["fallback"] == 2
    assert stats.blocked_risk_counts["defensive"] == 2
    assert stats.risk_by_regime[MarketRegime.TREND]["directional"] == 1
    assert stats.risk_by_regime[MarketRegime.TREND]["momentum"] == 2
    assert stats.risk_by_regime[MarketRegime.TREND]["defensive"] == 1
    assert stats.risk_by_regime[MarketRegime.MEAN_REVERSION]["statistical"] == 1
    assert stats.risk_by_regime[MarketRegime.MEAN_REVERSION]["momentum"] == 1
    assert stats.risk_by_regime[MarketRegime.MEAN_REVERSION]["defensive"] == 1

    tail_stats = workflow.activation_risk_stats(limit=2)
    assert tail_stats.total == 2
    assert tail_stats.risk_counts["intraday"] == 2
    assert tail_stats.risk_counts["fallback"] == 2
    assert tail_stats.risk_counts["defensive"] == 2
    assert tail_stats.fallback_risk_counts["intraday"] == 2
    assert tail_stats.fallback_risk_counts["defensive"] == 2

    empty_stats = StrategyRegimeWorkflow().activation_risk_stats()
    assert empty_stats.total == 0
    assert empty_stats.risk_counts == {}


def test_activation_blocker_stats_groups_reasons_and_issues() -> None:
    classifier = _SequenceClassifier(
        [
            MarketRegime.TREND,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.TREND,
        ]
    )
    history = RegimeHistory(thresholds_loader=lambda: {})
    schedule = [ScheduleWindow(start=time(8, 0), end=time(18, 0), allow_trading=True)]
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
        schedule_windows=schedule,
    )
    signing_key = b"license-key"

    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-pro",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ["ohlcv", "technical_indicators"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ["ohlcv", "spread_history"],
            }
        ],
        signing_key=signing_key,
        key_id="regime",
    )
    workflow.register_emergency_preset(
        name="fallback",
        entries=[{"engine": "day_trading"}],
        signing_key=signing_key,
        key_id="regime",
    )

    data = _market_frame()

    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "spread_history", "order_book"},
        now=datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc),
    )
    workflow.activate(
        data,
        available_data={"ohlcv", "technical_indicators", "order_book"},
        now=datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
    )

    capabilities = build_capabilities_from_payload(
        {
            "edition": "standard",
            "strategies": {"day_trading": True},
            "environments": ["paper"],
        },
        effective_date=date(2024, 1, 1),
    )
    install_capability_guard(capabilities)
    try:
        workflow.activate(
            data,
            available_data={"ohlcv", "technical_indicators", "spread_history", "order_book"},
            now=datetime(2024, 1, 2, 9, 5, tzinfo=timezone.utc),
        )
    finally:
        reset_capability_guard()

    stats = workflow.activation_blocker_stats()
    assert isinstance(stats, ActivationBlockerStats)
    assert stats.total == 3
    assert stats.reason_counts["schedule_blocked"] == 1
    assert stats.reason_counts["missing_data"] == 1
    assert stats.reason_counts["license_blocked"] == 1
    assert stats.regime_blocked[MarketRegime.TREND] == 2
    assert stats.regime_blocked[MarketRegime.MEAN_REVERSION] == 1
    assert stats.missing_data["spread_history"] == 1
    assert stats.missing_data_by_regime[MarketRegime.MEAN_REVERSION]["spread_history"] == 1
    assert stats.license_issues
    assert any("trend_d1" in issue or "licenc" in issue for issue in stats.license_issues)
    assert any(
        "trend_d1" in issue or "licenc" in issue
        for issue in stats.license_issues_by_regime[MarketRegime.TREND]
    )

    tail_stats = workflow.activation_blocker_stats(limit=2)
    assert tail_stats.total == 2
    assert "schedule_blocked" not in tail_stats.reason_counts
    assert tail_stats.regime_blocked[MarketRegime.TREND] == 1
    assert tail_stats.regime_blocked[MarketRegime.MEAN_REVERSION] == 1
    assert tail_stats.missing_data["spread_history"] == 1

    empty_stats = StrategyRegimeWorkflow().activation_blocker_stats()
    assert empty_stats.total == 0
    assert empty_stats.reason_counts == {}
