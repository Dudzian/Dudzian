from __future__ import annotations

from collections import deque
from datetime import date, datetime, time, timezone, timedelta

import numpy as np
import pandas as pd

from bot_core.ai import MarketRegime, MarketRegimeAssessment, RegimeHistory
from bot_core.auto_trader.schedule import ScheduleWindow
from bot_core.strategies import StrategyPresetWizard
from bot_core.strategies.regime_workflow import (
    ActivationCadenceStats,
    ActivationHistoryStats,
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
