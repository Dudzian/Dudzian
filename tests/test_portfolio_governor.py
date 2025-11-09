from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.runtime.journal import InMemoryTradingDecisionJournal

try:
    from bot_core.runtime.multi_strategy_scheduler import MultiStrategyScheduler  # type: ignore[attr-defined]
    _HAVE_MULTI_SCHEDULER = True
except Exception:  # pragma: no cover
    MultiStrategyScheduler = None  # type: ignore[assignment]
    _HAVE_MULTI_SCHEDULER = False

# ============================================================
#   Importy warunkowe – wspieramy dwa różne warianty API
#   1) Wariant "asset-based" z evaluate(), SLO, stress overrides
#   2) Wariant "scoring-based" z observe_strategy_metrics(), maybe_rebalance()
# ============================================================

# --- Wariant asset-based (HEAD) ---
try:
    from bot_core.market_intel import MarketIntelSnapshot  # type: ignore[attr-defined]
    _HAVE_MI_SNAPSHOT = True
except Exception:  # pragma: no cover
    MarketIntelSnapshot = None  # type: ignore[assignment]
    _HAVE_MI_SNAPSHOT = False

try:
    from bot_core.observability import SLOStatus  # type: ignore[attr-defined]
    _HAVE_SLO_STATUS = True
except Exception:  # pragma: no cover
    SLOStatus = None  # type: ignore[assignment]
    _HAVE_SLO_STATUS = False

try:
    # asset-based config wprost z bot_core.portfolio
    from bot_core.portfolio import (  # type: ignore[attr-defined]
        PortfolioAssetConfig,
        PortfolioDriftTolerance,
        PortfolioDecisionLog,
        PortfolioGovernor,
        PortfolioGovernorConfig as PortfolioGovernorConfig_Asset,
        PortfolioRiskBudgetConfig,
        PortfolioSloOverrideConfig,
    )
    _HAVE_PORTFOLIO_ASSET_CONFIG = True
except Exception:  # pragma: no cover
    PortfolioAssetConfig = None  # type: ignore[assignment]
    PortfolioDriftTolerance = None  # type: ignore[assignment]
    PortfolioDecisionLog = None  # type: ignore[assignment]
    PortfolioGovernor = None  # type: ignore[assignment]
    PortfolioGovernorConfig_Asset = None  # type: ignore[assignment]
    PortfolioRiskBudgetConfig = None  # type: ignore[assignment]
    PortfolioSloOverrideConfig = None  # type: ignore[assignment]
    _HAVE_PORTFOLIO_ASSET_CONFIG = False

try:
    from bot_core.risk import StressOverrideRecommendation  # type: ignore[attr-defined]
    _HAVE_STRESS_OVERRIDE = True
except Exception:  # pragma: no cover
    StressOverrideRecommendation = None  # type: ignore[assignment]
    _HAVE_STRESS_OVERRIDE = False


# --- Wariant scoring-based (main) ---
try:
    from bot_core.config.models import (  # type: ignore[attr-defined]
        PortfolioGovernorConfig as PortfolioGovernorConfig_Scoring,
        PortfolioGovernorScoringWeights,
        PortfolioGovernorStrategyConfig,
    )
    _HAVE_SCORING_CONFIG = True
except Exception:  # pragma: no cover
    PortfolioGovernorConfig_Scoring = None  # type: ignore[assignment]
    PortfolioGovernorScoringWeights = None  # type: ignore[assignment]
    PortfolioGovernorStrategyConfig = None  # type: ignore[assignment]
    _HAVE_SCORING_CONFIG = False

if 'PortfolioGovernor' not in globals():  # jeśli nie wczytany wyżej
    try:
        from bot_core.portfolio import PortfolioGovernor  # type: ignore
    except Exception:  # pragma: no cover
        PortfolioGovernor = None  # type: ignore[assignment]


# ============================================================
#   Pomocnicze wykrywanie możliwości
# ============================================================

def _supports_asset_api() -> bool:
    return all([
        _HAVE_MI_SNAPSHOT,
        _HAVE_SLO_STATUS,
        _HAVE_PORTFOLIO_ASSET_CONFIG,
        PortfolioGovernor is not None,
        hasattr(PortfolioGovernor, "evaluate"),
    ])


def _supports_scoring_api() -> bool:
    return all([
        _HAVE_SCORING_CONFIG,
        PortfolioGovernor is not None,
        hasattr(PortfolioGovernor, "observe_strategy_metrics"),
        hasattr(PortfolioGovernor, "maybe_rebalance"),
    ])


# ============================================================
#   Wspólne helpery dla wariantu asset-based
# ============================================================

def _snapshot(
    *,
    volatility: float | None = None,
    liquidity: float | None = None,
    drawdown: float | None = None,
) -> "MarketIntelSnapshot":
    if not _supports_asset_api():
        pytest.skip("Asset-based PortfolioGovernor API not available")
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return MarketIntelSnapshot(  # type: ignore[misc]
        symbol="BTC_USDT",
        interval="1h",
        start=now,
        end=now,
        bar_count=24,
        price_change_pct=5.0,
        volatility_pct=volatility,
        max_drawdown_pct=drawdown,
        average_volume=1_000.0,
        liquidity_usd=liquidity,
        momentum_score=2.0,
        metadata={},
    )


def _governor_config_asset() -> "PortfolioGovernorConfig_Asset":
    if not _supports_asset_api():
        pytest.skip("Asset-based PortfolioGovernor API not available")
    return PortfolioGovernorConfig_Asset(  # type: ignore[misc]
        name="core",
        portfolio_id="core",
        drift_tolerance=PortfolioDriftTolerance(absolute=0.02, relative=0.1),  # type: ignore[call-arg]
        min_rebalance_value=100.0,
        min_rebalance_weight=0.01,
        assets=(
            PortfolioAssetConfig(  # type: ignore[call-arg]
                symbol="BTC_USDT",
                target_weight=0.5,
                min_weight=0.1,
                max_weight=0.6,
                max_volatility_pct=20.0,
                min_liquidity_usd=500.0,
                risk_budget="balanced",
            ),
        ),
        risk_budgets={
            "balanced": PortfolioRiskBudgetConfig(  # type: ignore[call-arg]
                name="balanced",
                max_var_pct=25.0,
                max_drawdown_pct=35.0,
                max_leverage=1.0,
                severity="warning",
            )
        },
    )


# ============================================================
#   TESTY: wariant asset-based
# ============================================================

def test_portfolio_governor_detects_drift() -> None:
    if not _supports_asset_api():
        pytest.skip("Asset-based PortfolioGovernor API not available")

    config = _governor_config_asset()
    governor = PortfolioGovernor(config, clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))  # type: ignore[call-arg]

    decision = governor.evaluate(  # type: ignore[attr-defined]
        portfolio_value=100_000.0,
        allocations={"BTC_USDT": 0.3},
        market_data={"BTC_USDT": _snapshot(volatility=10.0, liquidity=10_000.0, drawdown=5.0)},
    )

    assert decision.rebalance_required is True
    assert len(decision.adjustments) == 1
    adjustment = decision.adjustments[0]
    assert adjustment.symbol == "BTC_USDT"
    assert adjustment.proposed_weight == pytest.approx(0.5)
    assert governor.last_rebalance_at == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_portfolio_governor_enforces_risk_budget() -> None:
    if not _supports_asset_api():
        pytest.skip("Asset-based PortfolioGovernor API not available")

    config = _governor_config_asset()
    governor = PortfolioGovernor(config)  # type: ignore[call-arg]

    decision = governor.evaluate(  # type: ignore[attr-defined]
        portfolio_value=50_000.0,
        allocations={"BTC_USDT": 0.5},
        market_data={"BTC_USDT": _snapshot(volatility=32.0, liquidity=10_000.0, drawdown=40.0)},
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.1)
    assert "volatility" in adjustment.reason
    assert decision.advisories
    advisory = decision.advisories[0]
    assert advisory.code == "risk_budget.balanced"
    assert "volatility" in advisory.message
    assert "drawdown" in advisory.message


def test_portfolio_governor_applies_slo_overrides() -> None:
    if not _supports_asset_api():
        pytest.skip("Asset-based PortfolioGovernor API not available")

    config = PortfolioGovernorConfig_Asset(  # type: ignore[misc]
        name="core",
        portfolio_id="core",
        drift_tolerance=PortfolioDriftTolerance(absolute=0.01, relative=0.05),  # type: ignore[call-arg]
        min_rebalance_value=0.0,
        min_rebalance_weight=0.0,
        assets=(
            PortfolioAssetConfig(  # type: ignore[call-arg]
                symbol="ETH_USDT",
                target_weight=0.4,
                min_weight=0.1,
                max_weight=0.6,
                tags=("core",),
            ),
        ),
        risk_budgets={},
        slo_overrides=(
            PortfolioSloOverrideConfig(  # type: ignore[call-arg]
                slo_name="latency",
                apply_on=("warning", "breach"),
                weight_multiplier=0.5,
                severity="critical",
                force_rebalance=True,
            ),
        ),
    )

    governor = PortfolioGovernor(config)  # type: ignore[call-arg]
    status = SLOStatus(  # type: ignore[call-arg]
        name="latency",
        indicator="router_latency_ms",
        value=320.0,
        target=250.0,
        comparison="<=",
        status="breach",
        severity="critical",
        warning_threshold=200.0,
        error_budget_pct=0.28,
        window_start=None,
        window_end=None,
        sample_size=7200,
    )

    decision = governor.evaluate(  # type: ignore[attr-defined]
        portfolio_value=100_000.0,
        allocations={"ETH_USDT": 0.4},
        market_data={"ETH_USDT": _snapshot(volatility=15.0, liquidity=20_000.0, drawdown=5.0)},
        slo_statuses={"latency": status},
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.2)
    assert adjustment.severity == "critical"
    assert adjustment.metadata["slo::latency"] == pytest.approx(0.28)
    assert adjustment.metadata["slo::latency::force_rebalance"] == pytest.approx(1.0)


def test_portfolio_governor_applies_stress_override_for_symbol() -> None:
    if not (_supports_asset_api() and _HAVE_STRESS_OVERRIDE):
        pytest.skip("Asset-based API or stress override recommendation not available")

    config = _governor_config_asset()
    config = PortfolioGovernorConfig_Asset(  # type: ignore[misc]
        name=config.name,
        portfolio_id=config.portfolio_id,
        drift_tolerance=config.drift_tolerance,
        min_rebalance_value=0.0,
        min_rebalance_weight=1.0,
        assets=config.assets,
        risk_budgets=config.risk_budgets,
    )
    governor = PortfolioGovernor(config)  # type: ignore[call-arg]
    overrides = [
        StressOverrideRecommendation(  # type: ignore[call-arg]
            severity="critical",
            reason="latency_spike",
            symbol="BTC_USDT",
            weight_multiplier=0.3,
            min_weight=0.1,
            force_rebalance=True,
        )
    ]

    decision = governor.evaluate(  # type: ignore[attr-defined]
        portfolio_value=150_000.0,
        allocations={"BTC_USDT": 0.5},
        market_data={"BTC_USDT": _snapshot(volatility=12.0, liquidity=50_000.0, drawdown=5.0)},
        stress_overrides=overrides,
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.15)
    assert adjustment.severity == "critical"
    assert "stress::latency_spike" in adjustment.reason
    assert adjustment.metadata["stress::count"] == pytest.approx(1.0)
    assert adjustment.metadata["stress::1::weight_multiplier"] == pytest.approx(0.3)
    assert adjustment.metadata["stress::1::force_rebalance"] == pytest.approx(1.0)


def test_portfolio_governor_applies_stress_override_for_risk_budget() -> None:
    if not (_supports_asset_api() and _HAVE_STRESS_OVERRIDE):
        pytest.skip("Asset-based API or stress override recommendation not available")

    config = _governor_config_asset()
    governor = PortfolioGovernor(config)  # type: ignore[call-arg]
    overrides = [
        StressOverrideRecommendation(  # type: ignore[call-arg]
            severity="warning",
            reason="drawdown_pressure",
            risk_budget="balanced",
            weight_multiplier=0.5,
        )
    ]

    decision = governor.evaluate(  # type: ignore[attr-defined]
        portfolio_value=90_000.0,
        allocations={"BTC_USDT": 0.45},
        market_data={"BTC_USDT": _snapshot(volatility=10.0, liquidity=60_000.0, drawdown=6.0)},
        stress_overrides=overrides,
    )

    assert decision.rebalance_required is True
    adjustment = decision.adjustments[0]
    assert adjustment.proposed_weight == pytest.approx(0.25)
    assert adjustment.severity == "warning"
    assert "stress::drawdown_pressure" in adjustment.reason


def test_portfolio_governor_writes_decision_log(tmp_path: Path) -> None:
    if not _supports_asset_api():
        pytest.skip("Asset-based PortfolioGovernor API not available")

    config = _governor_config_asset()
    log_path = (tmp_path / "audit").joinpath("portfolio_decisions.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    key = b"S" * 48
    log = PortfolioDecisionLog(jsonl_path=log_path, signing_key=key, signing_key_id="stage6")  # type: ignore[call-arg]
    clock_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    governor = PortfolioGovernor(config, clock=lambda: clock_time, decision_log=log)  # type: ignore[call-arg]

    overrides = []
    if _HAVE_STRESS_OVERRIDE:
        overrides = [
            StressOverrideRecommendation(  # type: ignore[call-arg]
                severity="critical",
                reason="stress_log_test",
                symbol="BTC_USDT",
                weight_multiplier=0.4,
                force_rebalance=True,
            )
        ]

    decision = governor.evaluate(  # type: ignore[attr-defined]
        portfolio_value=120_000.0,
        allocations={"BTC_USDT": 0.2},
        market_data={"BTC_USDT": _snapshot(volatility=12.0, liquidity=15_000.0, drawdown=10.0)},
        stress_overrides=overrides or None,
        log_context={"environment": "paper", "run_id": "unit-test"},
    )

    assert decision.rebalance_required is True
    assert log_path.exists()
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["portfolio_id"] == config.portfolio_id
    if overrides:
        assert entry["metadata"]["stress_overrides"][0]["reason"] == "stress_log_test"
    assert entry["metadata"]["environment"] == "paper"
    assert entry["metadata"]["adjustment_count"] == 1
    assert "signature" in entry and entry["signature"]["key_id"] == "stage6"


# ============================================================
#   TESTY: wariant scoring-based
# ============================================================

def _build_governor_scoring(*, enabled: bool = True, **overrides: object) -> "PortfolioGovernor":
    if not _supports_scoring_api():
        pytest.skip("Scoring-based PortfolioGovernor API not available")

    params = {
        "enabled": enabled,
        "rebalance_interval_minutes": 0.0,
        "smoothing": 1.0,
        "min_score_threshold": 0.0,
        "default_cost_bps": 0.5,
        "scoring": PortfolioGovernorScoringWeights(alpha=1.0, cost=0.5, slo=0.25, risk=0.0),  # type: ignore[call-arg]
        "strategies": {
            "trend": PortfolioGovernorStrategyConfig(  # type: ignore[call-arg]
                baseline_weight=0.5,
                min_weight=0.2,
                max_weight=0.8,
                baseline_max_signals=4,
                max_signal_factor=2.0,
            ),
            "mean_reversion": PortfolioGovernorStrategyConfig(  # type: ignore[call-arg]
                baseline_weight=0.5,
                min_weight=0.1,
                max_weight=0.6,
                baseline_max_signals=3,
                max_signal_factor=1.5,
            ),
        },
    }
    params.update(overrides)
    config = PortfolioGovernorConfig_Scoring(**params)  # type: ignore[call-arg]
    return PortfolioGovernor(config, clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))  # type: ignore[call-arg]


def test_portfolio_governor_rebalances_based_on_scores() -> None:
    if not _supports_scoring_api():
        pytest.skip("Scoring-based PortfolioGovernor API not available")

    governor = _build_governor_scoring(default_cost_bps=0.0)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    governor.observe_strategy_metrics(  # type: ignore[attr-defined]
        "trend",
        {"alpha_score": 2.0, "slo_violation_rate": 0.05, "risk_penalty": 0.0},
        timestamp=timestamp,
    )
    governor.observe_strategy_metrics(  # type: ignore[attr-defined]
        "mean_reversion",
        {"alpha_score": 0.4, "slo_violation_rate": 0.0, "risk_penalty": 0.0},
        timestamp=timestamp,
    )

    decision = governor.maybe_rebalance(timestamp=timestamp, force=True)  # type: ignore[attr-defined]
    assert decision is not None
    assert decision.weights["trend"] > decision.weights["mean_reversion"]
    allocation = governor.resolve_allocation("trend")  # type: ignore[attr-defined]
    assert allocation.max_signal_hint == 4
    assert allocation.signal_factor == pytest.approx(1.56, rel=1e-2)


def test_portfolio_governor_requires_complete_metrics_when_configured() -> None:
    if not _supports_scoring_api():
        pytest.skip("Scoring-based PortfolioGovernor API not available")

    governor = _build_governor_scoring(require_complete_metrics=True)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    governor.observe_strategy_metrics(  # type: ignore[attr-defined]
        "trend",
        {"alpha_score": 1.5, "slo_violation_rate": 0.0, "risk_penalty": 0.0},
        timestamp=timestamp,
    )
    assert governor.maybe_rebalance(timestamp=timestamp, force=True) is None  # type: ignore[attr-defined]

    governor.observe_strategy_metrics(  # type: ignore[attr-defined]
        "mean_reversion",
        {"alpha_score": 1.0, "slo_violation_rate": 0.0, "risk_penalty": 0.0},
        timestamp=timestamp,
    )
    decision = governor.maybe_rebalance(timestamp=timestamp, force=True)  # type: ignore[attr-defined]
    assert decision is not None
    assert all(weight >= 0.0 for weight in decision.weights.values())


def test_portfolio_governor_uses_cost_report_updates() -> None:
    if not _supports_scoring_api():
        pytest.skip("Scoring-based PortfolioGovernor API not available")

    governor = _build_governor_scoring(
        scoring=PortfolioGovernorScoringWeights(alpha=1.0, cost=1.0, slo=0.0, risk=0.0),  # type: ignore[call-arg]
        default_cost_bps=10.0,
    )
    report = {
        "strategies": {
            "trend": {"total": {"cost_bps": 1.0}},
            "mean_reversion": {"total": {"cost_bps": 8.0}},
        },
        "total": {"cost_bps": 12.0},
    }
    governor.update_costs_from_report(report)  # type: ignore[attr-defined]

    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    payload = {"alpha_score": 2.0, "slo_violation_rate": 0.0, "risk_penalty": 0.0}
    governor.observe_strategy_metrics("trend", payload, timestamp=timestamp)  # type: ignore[attr-defined]
    governor.observe_strategy_metrics("mean_reversion", payload, timestamp=timestamp)  # type: ignore[attr-defined]

    decision = governor.maybe_rebalance(timestamp=timestamp, force=True)  # type: ignore[attr-defined]
    assert decision is not None
    assert decision.cost_components["trend"] == pytest.approx(1.0)
    assert decision.cost_components["mean_reversion"] == pytest.approx(8.0)
    assert decision.weights["trend"] > decision.weights["mean_reversion"]


def test_portfolio_governor_scheduler_journals_decision() -> None:
    if not (_supports_scoring_api() and _HAVE_MULTI_SCHEDULER):
        pytest.skip("Scoring PortfolioGovernor or scheduler not available")

    governor = _build_governor_scoring(default_cost_bps=0.0)
    timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
    payload = {"alpha_score": 1.8, "slo_violation_rate": 0.0, "risk_penalty": 0.0}
    governor.observe_strategy_metrics("trend", payload, timestamp=timestamp)  # type: ignore[attr-defined]
    governor.observe_strategy_metrics("mean_reversion", payload, timestamp=timestamp)  # type: ignore[attr-defined]
    decision = governor.maybe_rebalance(timestamp=timestamp, force=True)  # type: ignore[attr-defined]
    assert decision is not None

    class _Engine:
        def warm_up(self, history):
            return None

        def on_data(self, snapshot):
            return ()

    class _Feed:
        def load_history(self, strategy_name: str, bars: int):
            return ()

        def fetch_latest(self, strategy_name: str):
            return ()

    class _Sink:
        def submit(self, **kwargs):
            return None

    journal = InMemoryTradingDecisionJournal()
    scheduler = MultiStrategyScheduler(  # type: ignore[misc]
        environment="paper",
        portfolio="core",
        clock=lambda: timestamp,
        decision_journal=journal,
        portfolio_governor=governor,
    )

    scheduler.register_schedule(  # type: ignore[attr-defined]
        name="trend_schedule",
        strategy_name="trend",
        strategy=_Engine(),
        feed=_Feed(),
        sink=_Sink(),
        cadence_seconds=60,
        max_drift_seconds=10,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=4,
    )
    scheduler.register_schedule(  # type: ignore[attr-defined]
        name="mean_schedule",
        strategy_name="mean_reversion",
        strategy=_Engine(),
        feed=_Feed(),
        sink=_Sink(),
        cadence_seconds=60,
        max_drift_seconds=10,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    captured: list[object] = []
    scheduler.add_portfolio_decision_listener(captured.append)  # type: ignore[attr-defined]
    scheduler._apply_portfolio_decision(decision)  # pylint: disable=protected-access

    assert captured and captured[0] is decision
    exported = list(journal.export())
    assert exported
    last_event = exported[-1]
    assert last_event["event"] == "portfolio_review"
    assert json.loads(last_event["weights"])  # ensures weights are logged as JSON payload
    assert last_event["rebalance_required"] == "0"
