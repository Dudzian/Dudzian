"""Testy jednostkowe dla silnika zarządzania ryzykiem."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.ai import ModelScore
from bot_core.config.models import (
    DecisionEngineConfig,
    DecisionOrchestratorThresholds,
    DecisionStressTestConfig,
)
from bot_core.decision import DecisionCandidate, DecisionEvaluation, DecisionOrchestrator
from bot_core.decision.models import ModelSelectionDetail, ModelSelectionMetadata
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import InMemoryRiskRepository, ThresholdRiskEngine
from bot_core.risk.events import RiskDecisionLog
from bot_core.risk.repository import FileRiskRepository
from bot_core.risk.profiles.manual import ManualProfile


@pytest.fixture()
def manual_profile() -> ManualProfile:
    return ManualProfile(
        name="test-profile",
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.2,
        daily_loss_limit=0.005,
        max_position_pct=0.5,
        target_volatility=0.1,
        stop_loss_atr_multiple=2.0,
    )


def _snapshot(equity: float) -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": equity},
        total_equity=equity,
        available_margin=equity,
        maintenance_margin=0.0,
    )


def _order(
    price: float,
    *,
    atr: float = 100.0,
    side: str = "buy",
    stop_multiple: float = 2.0,
    quantity: float = 0.01,
) -> OrderRequest:
    side_normalized = side.lower()
    stop_distance = atr * stop_multiple
    stop_price = price - stop_distance if side_normalized == "buy" else price + stop_distance
    return OrderRequest(
        symbol="BTCUSDT",
        side=side_normalized,
        quantity=quantity,
        order_type="limit",
        price=price,
        stop_price=stop_price,
        atr=atr,
        metadata={"atr": atr, "stop_price": stop_price},
    )


def _decision_engine_config() -> DecisionEngineConfig:
    return DecisionEngineConfig(
        orchestrator=DecisionOrchestratorThresholds(
            max_cost_bps=15.0,
            min_net_edge_bps=1.0,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.1,
            max_position_ratio=0.6,
            max_open_positions=10,
            max_latency_ms=400.0,
        ),
        profile_overrides={},
        stress_tests=DecisionStressTestConfig(
            cost_shock_bps=1.0,
            latency_spike_ms=25.0,
            slippage_multiplier=1.1,
        ),
        min_probability=0.4,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


class _StubInference:
    def __init__(self, score: ModelScore) -> None:
        self._score = score
        self.is_ready = True

    def score(self, features: Mapping[str, float]) -> ModelScore:
        return self._score


class _SequencedOrchestrator:
    def __init__(self, evaluations: Sequence[DecisionEvaluation]) -> None:
        self._evaluations = iter(evaluations)

    def evaluate_candidate(
        self, candidate: DecisionCandidate, snapshot: Mapping[str, object]
    ) -> DecisionEvaluation:
        try:
            return next(self._evaluations)
        except StopIteration as exc:  # pragma: no cover - test konfiguracji
            raise AssertionError("Brak kolejnej zaplanowanej ewaluacji") from exc


class _FailingOrchestrator:
    def evaluate_candidate(
        self, candidate: DecisionCandidate, snapshot: Mapping[str, object]
    ) -> DecisionEvaluation:
        raise RuntimeError("evaluation failed")


def test_daily_loss_limit_blocks_on_first_day(manual_profile: ManualProfile) -> None:
    clock_value = datetime(2024, 1, 1, 12, 0, 0)
    engine = ThresholdRiskEngine(clock=lambda: clock_value)
    engine.register_profile(manual_profile)

    first_snapshot = _snapshot(1_000.0)
    request = _order(30_000.0)

    first_result = engine.apply_pre_trade_checks(
        request,
        account=first_snapshot,
        profile_name=manual_profile.name,
    )
    assert first_result.allowed is True

    state = engine._states[manual_profile.name]
    assert state.start_of_day_equity == pytest.approx(1_000.0)
    assert state.peak_equity == pytest.approx(1_000.0)

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol=request.symbol,
        side="buy",
        position_value=0.0,
        pnl=-10.0,
        timestamp=clock_value,
    )

    second_snapshot = _snapshot(990.0)
    second_result = engine.apply_pre_trade_checks(
        request,
        account=second_snapshot,
        profile_name=manual_profile.name,
    )

    assert second_result.allowed is False
    assert "Przekroczono dzienny limit straty." in (second_result.reason or "")


def test_margin_check_blocks_when_available_margin_is_too_low() -> None:
    low_leverage_profile = ManualProfile(
        name="low-margin",
        max_positions=5,
        max_leverage=1.0,
        drawdown_limit=0.5,
        daily_loss_limit=0.5,
        max_position_pct=1.0,
        target_volatility=0.2,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(low_leverage_profile)

    snapshot = AccountSnapshot(
        balances={"USDT": 1_000.0},
        total_equity=1_000.0,
        available_margin=500.0,
        maintenance_margin=150.0,
    )

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=20_000.0,
        stop_price=20_000.0 - 2.0 * 200.0,
        atr=200.0,
    )

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=low_leverage_profile.name,
    )

    assert result.allowed is False
    assert result.reason == "Niewystarczający wolny margines na otwarcie lub powiększenie pozycji."
    assert result.adjustments is not None
    max_quantity = result.adjustments.get("max_quantity") if result.adjustments else None
    assert max_quantity is not None
    # Dostępny margines po uwzględnieniu maintenance to 350 USDT.
    assert max_quantity == pytest.approx(350.0 / 20_000.0, rel=1e-6)


def test_margin_check_passes_when_leverage_covers_notional() -> None:
    leveraged_profile = ManualProfile(
        name="leveraged",
        max_positions=5,
        max_leverage=4.0,
        drawdown_limit=0.5,
        daily_loss_limit=0.5,
        max_position_pct=1.0,
        target_volatility=0.2,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(leveraged_profile)

    snapshot = AccountSnapshot(
        balances={"USDT": 1_000.0},
        total_equity=1_000.0,
        available_margin=500.0,
        maintenance_margin=150.0,
    )

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=20_000.0,
        stop_price=20_000.0 - 2.0 * 200.0,
        atr=200.0,
    )

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=leveraged_profile.name,
    )

    assert result.allowed is True


def test_decision_log_records_allowed_and_denied_events(tmp_path, manual_profile):
    timestamps = iter(
        [
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ]
    )

    def _next_timestamp() -> datetime:
        try:
            return next(timestamps)
        except StopIteration:
            return datetime(2024, 1, 1, 12, 0, 59, tzinfo=timezone.utc)

    log_path = tmp_path / "risk_decisions.jsonl"
    decision_log = RiskDecisionLog(max_entries=10, jsonl_path=log_path, clock=_next_timestamp)
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0), decision_log=decision_log)
    engine.register_profile(manual_profile)

    snapshot = _snapshot(1_000.0)
    allowed_request = _order(30_000.0)
    denied_request = _order(30_000.0, stop_multiple=0.0)

    allowed_result = engine.apply_pre_trade_checks(
        allowed_request,
        account=snapshot,
        profile_name=manual_profile.name,
    )
    assert allowed_result.allowed is True

    denied_result = engine.apply_pre_trade_checks(
        denied_request,
        account=snapshot,
        profile_name=manual_profile.name,
    )
    assert denied_result.allowed is False
    assert denied_result.reason is not None

    events = engine.recent_decisions(profile_name=manual_profile.name, limit=5)
    assert len(events) == 2
    allowed_event, denied_event = events
    assert allowed_event["allowed"] is True
    assert allowed_event["symbol"] == allowed_request.symbol
    assert "metadata" in allowed_event
    assert allowed_event["metadata"]["state"]["gross_notional"] >= 0
    assert denied_event["allowed"] is False
    assert "stop loss" in denied_event["reason"].lower()

    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 2


def test_decision_log_generates_hmac_signature(tmp_path, manual_profile):
    timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def _clock() -> datetime:
        return timestamp

    log_path = tmp_path / "risk_signed.jsonl"
    decision_log = RiskDecisionLog(
        max_entries=5,
        jsonl_path=log_path,
        clock=_clock,
        signing_key=b"very-secret-key",
        signing_key_id="risk-ci",
    )
    engine = ThresholdRiskEngine(clock=_clock, decision_log=decision_log)
    engine.register_profile(manual_profile)

    account = _snapshot(10_000.0)
    request = _order(5_000.0)
    result = engine.apply_pre_trade_checks(
        request,
        account=account,
        profile_name=manual_profile.name,
    )

    assert result.allowed is True
    recent = engine.recent_decisions(profile_name=manual_profile.name, limit=1)
    assert len(recent) == 1
    entry = recent[0]
    assert entry["signature"]["algorithm"] == "HMAC-SHA256"
    assert entry["signature"]["key_id"] == "risk-ci"

    serialized = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(serialized) == 1
    payload = json.loads(serialized[0])
    assert payload["signature"]["value"]


def test_combined_strategy_orders_respect_max_position_pct(manual_profile: ManualProfile) -> None:
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    engine.register_profile(manual_profile)

    account = _snapshot(1_000.0)
    first_order = _order(30_000.0, quantity=0.01)
    first_result = engine.apply_pre_trade_checks(first_order, account=account, profile_name=manual_profile.name)
    assert first_result.allowed

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol=first_order.symbol,
        side="buy",
        position_value=first_order.price * first_order.quantity,
        pnl=0.0,
        timestamp=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
    )

    second_order = _order(30_000.0, quantity=0.05)
    second_result = engine.apply_pre_trade_checks(
        second_order,
        account=account,
        profile_name=manual_profile.name,
    )
    assert not second_result.allowed
    message = (second_result.reason or "").lower()
    assert "limit" in message and "ekspozycji" in message


def test_force_liquidation_due_to_drawdown_allows_only_reducing_orders() -> None:
    clock_values = [
        datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 9, 10, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 9, 15, 0, tzinfo=timezone.utc),
    ]
    timestamps = iter(clock_values)

    def _clock() -> datetime:
        try:
            return next(timestamps)
        except StopIteration:
            return clock_values[-1]

    profile = ManualProfile(
        name="drawdown-limited",
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.1,
        daily_loss_limit=0.5,
        max_position_pct=0.5,
        target_volatility=0.02,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=_clock)
    engine.register_profile(profile)

    opening_snapshot = _snapshot(50_000.0)
    opening_order = _order(25_000.0, quantity=0.5)
    opening_result = engine.apply_pre_trade_checks(
        opening_order,
        account=opening_snapshot,
        profile_name=profile.name,
    )
    assert opening_result.allowed is True

    engine.on_fill(
        profile_name=profile.name,
        symbol=opening_order.symbol,
        side="buy",
        position_value=opening_order.price * opening_order.quantity,
        pnl=0.0,
        timestamp=datetime(2024, 1, 1, 9, 6, 0, tzinfo=timezone.utc),
    )

    drawdown_snapshot = _snapshot(45_000.0)
    expansion_order = _order(25_000.0, quantity=0.1)
    expansion_result = engine.apply_pre_trade_checks(
        expansion_order,
        account=drawdown_snapshot,
        profile_name=profile.name,
    )

    assert expansion_result.allowed is False
    assert expansion_result.reason is not None
    assert "Przekroczono limit obsunięcia portfela" in expansion_result.reason
    assert "redukujące" in expansion_result.reason

    reducing_order = _order(25_000.0, side="sell", quantity=0.25)
    reducing_result = engine.apply_pre_trade_checks(
        reducing_order,
        account=drawdown_snapshot,
        profile_name=profile.name,
    )

    assert reducing_result.allowed is True


@pytest.mark.parametrize(
    ("atr", "target_vol", "equity", "price"),
    [
        (150.0, 0.02, 20_000.0, 28_000.0),
        (250.0, 0.01, 15_000.0, 24_000.0),
    ],
)
def test_target_volatility_limits_position_size(
    atr: float, target_vol: float, equity: float, price: float
) -> None:
    profile = ManualProfile(
        name="volatility-aware",
        max_positions=5,
        max_leverage=10.0,
        drawdown_limit=0.5,
        daily_loss_limit=0.5,
        max_position_pct=5.0,
        target_volatility=target_vol,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(profile)

    snapshot = _snapshot(equity)
    risk_budget = target_vol * equity
    expected_quantity = risk_budget / (atr * profile.stop_loss_atr_multiple())

    oversized_order = _order(
        price,
        atr=atr,
        quantity=expected_quantity * 1.5,
        stop_multiple=profile.stop_loss_atr_multiple(),
    )
    result = engine.apply_pre_trade_checks(
        oversized_order,
        account=snapshot,
        profile_name=profile.name,
    )

    assert result.allowed is False
    assert result.adjustments is not None
    assert result.adjustments["max_quantity"] == pytest.approx(expected_quantity, rel=1e-6)

    allowed_order = _order(
        price,
        atr=atr,
        quantity=expected_quantity * 0.95,
        stop_multiple=profile.stop_loss_atr_multiple(),
    )
    allowed_result = engine.apply_pre_trade_checks(
        allowed_order,
        account=snapshot,
        profile_name=profile.name,
    )

    assert allowed_result.allowed is True


def test_stop_loss_must_not_be_tighter_than_minimum(manual_profile: ManualProfile) -> None:
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(manual_profile)

    snapshot = _snapshot(5_000.0)
    price = 25_000.0
    atr = 120.0

    too_tight_stop = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=price,
        stop_price=price - atr * manual_profile.stop_loss_atr_multiple() * 0.8,  # celowo za ciasny
        atr=atr,
    )

    result = engine.apply_pre_trade_checks(
        too_tight_stop,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert result.allowed is False
    assert "stop loss" in (result.reason or "").lower()

    wider_stop = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=price,
        stop_price=price - atr * manual_profile.stop_loss_atr_multiple() * 1.6,
        atr=atr,
    )

    wider_result = engine.apply_pre_trade_checks(
        wider_stop,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert wider_result.allowed is True

    missing_stop = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=price,
        stop_price=None,
        atr=atr,
    )

    missing_result = engine.apply_pre_trade_checks(
        missing_stop,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert missing_result.allowed is False
    assert "stop_price" in (missing_result.reason or "").lower()


def test_stop_loss_wider_than_minimum_limits_position_size() -> None:
    profile = ManualProfile(
        name="paper-profile",
        max_positions=5,
        max_leverage=15.0,
        drawdown_limit=0.2,
        daily_loss_limit=0.01,
        max_position_pct=2.0,
        target_volatility=0.03,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(profile)

    equity = 120_000.0
    price = 31_000.0
    atr = 140.0
    snapshot = _snapshot(equity)

    wider_multiple = profile.stop_loss_atr_multiple() * 1.8
    stop_distance = atr * wider_multiple
    max_risk_capital = profile.target_volatility() * equity
    allowed_quantity = max_risk_capital / stop_distance

    baseline_order = _order(
        price,
        atr=atr,
        stop_multiple=wider_multiple,
        quantity=allowed_quantity * 0.95,
    )

    allowed = engine.apply_pre_trade_checks(
        baseline_order,
        account=snapshot,
        profile_name=profile.name,
    )

    assert allowed.allowed is True

    oversized_order = _order(
        price,
        atr=atr,
        stop_multiple=wider_multiple,
        quantity=allowed_quantity * 2,
    )

    denial = engine.apply_pre_trade_checks(
        oversized_order,
        account=snapshot,
        profile_name=profile.name,
    )

    assert denial.allowed is False
    assert denial.adjustments is not None
    assert denial.adjustments["max_quantity"] == pytest.approx(allowed_quantity, rel=1e-6)


def test_daily_loss_limit_resets_after_new_trading_day() -> None:
    clock_values = [
        datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 9, 5, 0, tzinfo=timezone.utc),
    ]
    timestamps = iter(clock_values)

    def _clock() -> datetime:
        try:
            return next(timestamps)
        except StopIteration:
            return clock_values[-1]

    profile = ManualProfile(
        name="daily-loss-reset",
        max_positions=3,
        max_leverage=4.0,
        drawdown_limit=0.5,
        daily_loss_limit=0.01,
        max_position_pct=1.0,
        target_volatility=0.02,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=_clock)
    engine.register_profile(profile)

    day_one_snapshot = _snapshot(10_000.0)
    opening_order = _order(25_000.0, quantity=0.2)
    first_result = engine.apply_pre_trade_checks(
        opening_order,
        account=day_one_snapshot,
        profile_name=profile.name,
    )
    assert first_result.allowed is True

    engine.on_fill(
        profile_name=profile.name,
        symbol=opening_order.symbol,
        side="buy",
        position_value=opening_order.price * opening_order.quantity,
        pnl=-150.0,
        timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
    )

    denial_snapshot = _snapshot(9_850.0)
    blocked_order = _order(25_000.0, quantity=0.05)
    blocked_result = engine.apply_pre_trade_checks(
        blocked_order,
        account=denial_snapshot,
        profile_name=profile.name,
    )

    assert blocked_result.allowed is False
    assert blocked_result.reason is not None
    assert "dzienny limit straty" in blocked_result.reason.lower()

    engine.on_fill(
        profile_name=profile.name,
        symbol=opening_order.symbol,
        side="sell",
        position_value=0.0,
        pnl=0.0,
        timestamp=datetime(2024, 1, 1, 16, 0, 0, tzinfo=timezone.utc),
    )

    day_two_snapshot = _snapshot(9_850.0)
    reset_order = _order(25_000.0, quantity=0.05)
    reset_result = engine.apply_pre_trade_checks(
        reset_order,
        account=day_two_snapshot,
        profile_name=profile.name,
    )

    assert reset_result.allowed is True


def test_on_fill_normalizes_position_side_and_allows_growth(manual_profile: ManualProfile) -> None:
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(manual_profile)

    snapshot = _snapshot(10_000.0)
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=30_000.0,
        stop_price=30_000.0 - manual_profile.stop_loss_atr_multiple() * 200.0,
        atr=200.0,
    )

    first_check = engine.apply_pre_trade_checks(request, account=snapshot, profile_name=manual_profile.name)
    assert first_check.allowed is True

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol=request.symbol,
        side="buy",
        position_value=request.quantity * request.price,
        pnl=0.0,
        timestamp=datetime(2024, 1, 1, 13, 0, 0),
    )

    state = engine._states[manual_profile.name]
    assert state.positions[request.symbol].side == "long"

    second_check = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert second_check.allowed is True


def test_target_volatility_caps_position_size() -> None:
    profile = ManualProfile(
        name="vol-profile",
        max_positions=5,
        max_leverage=5.0,
        drawdown_limit=0.2,
        daily_loss_limit=0.05,
        max_position_pct=1.0,
        target_volatility=0.02,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(profile)

    snapshot = _snapshot(50_000.0)

    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=2.0,
        order_type="limit",
        price=25_000.0,
        atr=500.0,
        stop_price=24_000.0,
    )

    result = engine.apply_pre_trade_checks(order, account=snapshot, profile_name=profile.name)

    assert result.allowed is False
    assert "target volatility" in (result.reason or "").lower()
    assert result.adjustments is not None
    assert result.adjustments.get("max_quantity") == pytest.approx(1.0)


def test_rejects_stop_loss_tighter_than_required() -> None:
    profile = ManualProfile(
        name="strict-stop",
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.2,
        daily_loss_limit=0.05,
        max_position_pct=1.0,
        target_volatility=0.05,
        stop_loss_atr_multiple=3.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(profile)

    snapshot = _snapshot(20_000.0)

    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.5,
        order_type="limit",
        price=30_000.0,
        atr=200.0,
        stop_price=30_000.0 - 200.0 * 2.0,  # < wymagane 3x ATR
    )

    result = engine.apply_pre_trade_checks(order, account=snapshot, profile_name=profile.name)

    assert result.allowed is False
    assert "stop loss" in (result.reason or "").lower()


def test_register_profile_normalizes_persisted_state(manual_profile: ManualProfile) -> None:
    repository = InMemoryRiskRepository()
    repository.store(
        manual_profile.name,
        {
            "profile": manual_profile.name,
            "current_day": "2024-01-01",
            "start_of_day_equity": 1_000.0,
            "daily_realized_pnl": 0.0,
            "peak_equity": 1_000.0,
            "force_liquidation": False,
            "last_equity": 1_000.0,
            "positions": {
                "BTCUSDT": {"side": "buy", "notional": 500.0},
            },
        },
    )

    engine = ThresholdRiskEngine(repository=repository, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(manual_profile)

    state = engine._states[manual_profile.name]
    assert state.positions["BTCUSDT"].side == "long"


def test_file_risk_repository_persists_state(tmp_path: Path, manual_profile: ManualProfile) -> None:
    repository = FileRiskRepository(tmp_path)

    clock = lambda: datetime(2024, 1, 1, 12, 0, 0)
    engine = ThresholdRiskEngine(repository=repository, clock=clock)
    engine.register_profile(manual_profile)

    snapshot = _snapshot(1_000.0)
    request = _order(20_000.0)

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=manual_profile.name,
    )
    assert result.allowed is True

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol="BTCUSDT",
        side="buy",
        position_value=500.0,
        pnl=-15.0,
        timestamp=datetime(2024, 1, 1, 13, 0, 0),
    )

    # Now instantiate a new engine sharing the same repository – state should persist.
    new_engine = ThresholdRiskEngine(repository=repository, clock=lambda: datetime(2024, 1, 1, 14, 0, 0))
    new_engine.register_profile(manual_profile)

    state = new_engine._states[manual_profile.name]
    assert state.start_of_day_equity == pytest.approx(1_000.0)
    assert state.daily_realized_pnl == pytest.approx(-15.0)
    assert "BTCUSDT" in state.positions
    btc_position = state.positions["BTCUSDT"]
    assert btc_position.side == "long"
    assert btc_position.notional == pytest.approx(500.0)


def test_snapshot_state_returns_enriched_metrics(manual_profile: ManualProfile) -> None:
    clock_value = datetime(2024, 1, 1, 12, 0, 0)
    engine = ThresholdRiskEngine(clock=lambda: clock_value)
    engine.register_profile(manual_profile)

    baseline = _snapshot(1_000.0)
    order = _order(30_000.0)

    engine.apply_pre_trade_checks(
        order,
        account=baseline,
        profile_name=manual_profile.name,
    )

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol=order.symbol,
        side="buy",
        position_value=6_000.0,
        pnl=-50.0,
        timestamp=clock_value,
    )

    snapshot = engine.snapshot_state(manual_profile.name)
    assert snapshot is not None
    assert snapshot["profile"] == manual_profile.name
    assert snapshot["gross_notional"] == pytest.approx(6_000.0)
    assert snapshot["active_positions"] == 1
    assert snapshot["daily_loss_pct"] == pytest.approx(0.05)
    limits = snapshot.get("limits")
    assert isinstance(limits, dict)
    assert limits["max_positions"] == manual_profile.max_positions()
    assert limits["max_leverage"] == manual_profile.max_leverage()
    assert limits["daily_loss_limit"] == manual_profile.daily_loss_limit()
    assert limits["max_position_pct"] == manual_profile.max_position_exposure()


def test_decision_orchestrator_metadata_when_candidate_passes(
    manual_profile: ManualProfile,
) -> None:
    orchestrator = DecisionOrchestrator(
        _decision_engine_config(),
        inference=_StubInference(
            ModelScore(expected_return_bps=14.0, success_probability=0.8)
        ),
    )
    clock_value = datetime(2024, 7, 1, 9, 0, 0)
    engine = ThresholdRiskEngine(
        clock=lambda: clock_value,
        decision_orchestrator=orchestrator,
    )
    engine.register_profile(manual_profile)

    snapshot = _snapshot(1_000.0)
    request = _order(25_000.0)
    assert isinstance(request.metadata, dict)
    request.metadata["decision_candidate"] = {
        "strategy": "mean_reversion",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": request.symbol,
        "notional": 200.0,
        "expected_return_bps": 15.0,
        "expected_probability": 0.9,
        "cost_bps_override": 3.0,
        "latency_ms": 150.0,
        "metadata": {"model_features": {"momentum": 1.0, "volatility": 0.2}},
    }

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert result.allowed is True
    assert result.metadata is not None
    decision_meta = result.metadata.get("decision_orchestrator")  # type: ignore[index]
    assert isinstance(decision_meta, dict)
    assert decision_meta.get("status") == "evaluated"
    assert decision_meta.get("accepted") is True
    assert decision_meta.get("candidate", {}).get("strategy") == "mean_reversion"
    assert decision_meta.get("model_name") == "__default__"
    assert decision_meta.get("model_success_probability") == pytest.approx(0.8)
    selection_meta = decision_meta.get("model_selection")
    assert isinstance(selection_meta, dict)
    assert selection_meta.get("selected") == "__default__"
    candidates = selection_meta.get("candidates")
    assert isinstance(candidates, list) and candidates
    default_detail = next(
        (detail for detail in candidates if detail.get("name") == "__default__"),
        None,
    )
    assert default_detail is not None
    assert default_detail.get("available") is True
    assert default_detail.get("reason") is None
    thresholds_meta = decision_meta.get("thresholds")
    assert isinstance(thresholds_meta, dict)
    assert thresholds_meta.get("min_probability") == pytest.approx(0.4)


def test_decision_orchestrator_blocks_when_thresholds_exceeded(
    manual_profile: ManualProfile,
) -> None:
    orchestrator = DecisionOrchestrator(
        _decision_engine_config(),
        inference=_StubInference(
            ModelScore(expected_return_bps=6.0, success_probability=0.7)
        ),
    )
    engine = ThresholdRiskEngine(
        clock=lambda: datetime(2024, 7, 1, 9, 0, 0),
        decision_orchestrator=orchestrator,
    )
    engine.register_profile(manual_profile)

    snapshot = _snapshot(1_000.0)
    request = _order(25_000.0)
    assert isinstance(request.metadata, dict)
    request.metadata["decision_candidate"] = {
        "strategy": "cross_exchange",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": request.symbol,
        "notional": 400.0,
        "expected_return_bps": 6.0,
        "expected_probability": 0.7,
        "cost_bps_override": 20.0,
        "latency_ms": 150.0,
        "metadata": {"model_features": {"momentum": 0.5}},
    }

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert result.allowed is False
    assert result.reason is not None and "DecisionOrchestrator" in result.reason
    assert result.metadata is not None
    decision_meta = result.metadata.get("decision_orchestrator")  # type: ignore[index]
    assert isinstance(decision_meta, dict)
    assert decision_meta.get("accepted") is False
    reasons_text = " ".join(str(item) for item in decision_meta.get("reasons", []))
    assert "koszt" in reasons_text or "edge" in reasons_text
    assert decision_meta.get("model_name") == "__default__"
    selection_meta = decision_meta.get("model_selection")
    assert isinstance(selection_meta, dict)
    assert selection_meta.get("selected") == "__default__"
    candidates = selection_meta.get("candidates")
    assert isinstance(candidates, list) and candidates
    default_detail = next(
        (detail for detail in candidates if detail.get("name") == "__default__"),
        None,
    )
    assert default_detail is not None
    assert default_detail.get("available") is True
    thresholds_meta = decision_meta.get("thresholds")
    assert isinstance(thresholds_meta, dict)
    assert thresholds_meta.get("min_probability") == pytest.approx(0.4)
    assert default_detail.get("reason") is None


def test_decision_orchestrator_reports_invalid_payload(
    manual_profile: ManualProfile,
) -> None:
    orchestrator = DecisionOrchestrator(_decision_engine_config())
    engine = ThresholdRiskEngine(
        clock=lambda: datetime(2024, 7, 1, 9, 0, 0),
        decision_orchestrator=orchestrator,
    )
    engine.register_profile(manual_profile)

    snapshot = _snapshot(1_000.0)
    request = _order(25_000.0)
    assert isinstance(request.metadata, dict)
    request.metadata["decision_candidate"] = "INVALID"

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert result.allowed is False
    assert result.reason is not None
    assert "nie mógł ocenić" in result.reason
    assert result.metadata is not None
    decision_meta = result.metadata.get("decision_orchestrator")  # type: ignore[index]
    assert isinstance(decision_meta, dict)
    assert decision_meta.get("status") == "error"


def test_decision_model_outcomes_track_accepts_and_rejections(
    manual_profile: ManualProfile,
) -> None:
    candidate_payload = {
        "strategy": "mean_reversion",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": "BTCUSDT",
        "notional": 250.0,
        "expected_return_bps": 12.0,
        "expected_probability": 0.65,
    }
    candidate = DecisionCandidate.from_mapping(candidate_payload)
    selection = ModelSelectionMetadata(
        selected="fast",
        candidates=(
            ModelSelectionDetail(
                name="fast",
                score=0.8,
                weight=1.0,
                effective_score=0.8,
            ),
        ),
    )
    accepted_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=True,
        cost_bps=5.0,
        net_edge_bps=2.0,
        reasons=(),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=14.0,
        model_success_probability=0.6,
        model_name="fast",
        model_selection=selection,
    )
    rejected_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=False,
        cost_bps=5.0,
        net_edge_bps=2.0,
        reasons=("edge_below_threshold",),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=14.0,
        model_success_probability=0.6,
        model_name="fast",
        model_selection=selection,
    )

    orchestrator = _SequencedOrchestrator((accepted_eval, rejected_eval))
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 7, 1, 9, 0, 0))
    engine.register_profile(manual_profile)
    engine.attach_decision_orchestrator(orchestrator)

    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=25_000.0,
        stop_price=25_000.0 - 2.0 * 100.0,
        atr=100.0,
        metadata={
            "decision_candidate": candidate_payload,
            "atr": 100.0,
            "stop_price": 25_000.0 - 2.0 * 100.0,
        },
    )
    account = _snapshot(1_000.0)

    first_result = engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )
    assert first_result.allowed is True

    stats = engine.decision_model_outcomes()
    assert stats == {"fast": {"accepted": 1, "rejected": 0}}

    second_result = engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )
    assert second_result.allowed is False

    stats = engine.decision_model_outcomes()
    assert stats == {"fast": {"accepted": 1, "rejected": 1}}

    snapshot = engine.decision_model_outcomes(reset=True)
    assert snapshot == {"fast": {"accepted": 1, "rejected": 1}}
    assert engine.decision_model_outcomes() == {}


def test_decision_model_rejection_reasons_aggregate_by_model(
    manual_profile: ManualProfile,
) -> None:
    candidate_payload = {
        "strategy": "momentum",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": "ETHUSDT",
        "notional": 150.0,
        "expected_return_bps": 8.0,
        "expected_probability": 0.55,
    }
    candidate = DecisionCandidate.from_mapping(candidate_payload)
    selection = ModelSelectionMetadata(
        selected="fast",
        candidates=(
            ModelSelectionDetail(
                name="fast",
                score=0.75,
                weight=1.0,
                effective_score=0.75,
            ),
        ),
    )
    accepted_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=True,
        cost_bps=4.0,
        net_edge_bps=3.0,
        reasons=(),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=9.0,
        model_success_probability=0.58,
        model_name="fast",
        model_selection=selection,
    )
    rejected_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=False,
        cost_bps=6.0,
        net_edge_bps=-1.0,
        reasons=("edge_below_threshold",),
        risk_flags=("risk_limit",),
        stress_failures=("latency_spike",),
        model_expected_return_bps=7.0,
        model_success_probability=0.5,
        model_name="fast",
        model_selection=selection,
    )

    orchestrator = _SequencedOrchestrator((accepted_eval, rejected_eval))
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 7, 1, 9, 0, 0))
    engine.register_profile(manual_profile)
    engine.attach_decision_orchestrator(orchestrator)

    order = OrderRequest(
        symbol="ETHUSDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=1_800.0,
        stop_price=1_800.0 - 2.0 * 15.0,
        atr=15.0,
        metadata={
            "decision_candidate": candidate_payload,
            "atr": 15.0,
            "stop_price": 1_800.0 - 2.0 * 15.0,
        },
    )
    account = _snapshot(1_000.0)

    engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )

    # Pierwsza ewaluacja jest przyjęta – powody odrzuceń powinny być puste.
    assert engine.decision_model_rejection_reasons() == {}

    result = engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )

    assert result.allowed is False

    stats = engine.decision_model_rejection_reasons()
    assert stats == {
        "fast": {
            "edge_below_threshold": 1,
            "risk_limit": 1,
            "latency_spike": 1,
        }
    }

    snapshot = engine.decision_model_rejection_reasons(reset=True)
    assert snapshot == stats
    assert engine.decision_model_rejection_reasons() == {}


def test_decision_model_metrics_aggregate_and_reset(
    manual_profile: ManualProfile,
) -> None:
    candidate_payload = {
        "strategy": "momentum",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": "BTCUSDT",
        "notional": 250.0,
        "expected_return_bps": 12.0,
        "expected_probability": 0.6,
    }
    candidate = DecisionCandidate.from_mapping(candidate_payload)
    selection = ModelSelectionMetadata(
        selected="fast",
        candidates=(
            ModelSelectionDetail(
                name="fast",
                score=0.8,
                weight=1.0,
                effective_score=0.8,
            ),
        ),
    )
    accepted_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=True,
        cost_bps=5.0,
        net_edge_bps=2.0,
        reasons=(),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=14.0,
        model_success_probability=0.6,
        model_name="fast",
        model_selection=selection,
    )
    rejected_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=False,
        cost_bps=6.0,
        net_edge_bps=-1.0,
        reasons=("edge_below_threshold",),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=7.0,
        model_success_probability=0.5,
        model_name="fast",
        model_selection=selection,
    )

    orchestrator = _SequencedOrchestrator((accepted_eval, rejected_eval))
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 7, 1, 9, 0, 0))
    engine.register_profile(manual_profile)
    engine.attach_decision_orchestrator(orchestrator)

    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=25_000.0,
        stop_price=25_000.0 - 2.0 * 100.0,
        atr=100.0,
        metadata={
            "decision_candidate": candidate_payload,
            "atr": 100.0,
            "stop_price": 25_000.0 - 2.0 * 100.0,
        },
    )
    account = _snapshot(1_000.0)

    first_result = engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )
    assert first_result.allowed is True

    metrics = engine.decision_model_metrics()
    assert set(metrics.keys()) == {"fast"}
    fast_stats = metrics["fast"]
    assert fast_stats["evaluations"] == 1
    assert fast_stats["accepted"] == 1
    assert fast_stats["rejected"] == 0
    assert fast_stats["cost_bps"]["sum"] == pytest.approx(5.0)
    assert fast_stats["cost_bps"]["average"] == pytest.approx(5.0)
    assert fast_stats["cost_bps"]["count"] == 1
    assert fast_stats["net_edge_bps"]["average"] == pytest.approx(2.0)
    assert fast_stats["model_expected_return_bps"]["sum"] == pytest.approx(14.0)
    assert fast_stats["model_success_probability"]["average"] == pytest.approx(0.6)

    second_result = engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )
    assert second_result.allowed is False

    metrics = engine.decision_model_metrics()
    fast_stats = metrics["fast"]
    assert fast_stats["evaluations"] == 2
    assert fast_stats["accepted"] == 1
    assert fast_stats["rejected"] == 1
    assert fast_stats["cost_bps"]["sum"] == pytest.approx(11.0)
    assert fast_stats["cost_bps"]["average"] == pytest.approx(5.5)
    assert fast_stats["cost_bps"]["count"] == 2
    assert fast_stats["net_edge_bps"]["average"] == pytest.approx(0.5)
    assert fast_stats["model_expected_return_bps"]["average"] == pytest.approx(10.5)
    assert fast_stats["model_success_probability"]["average"] == pytest.approx(0.55)

    snapshot = engine.decision_model_metrics(reset=True)
    assert snapshot == metrics
    assert engine.decision_model_metrics() == {}


def test_decision_model_selection_stats_track_candidates_and_reset(
    manual_profile: ManualProfile,
) -> None:
    candidate_payload = {
        "strategy": "momentum",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": "BTCUSDT",
        "notional": 250.0,
    }
    candidate = DecisionCandidate.from_mapping(candidate_payload)
    first_selection = ModelSelectionMetadata(
        selected="fast",
        candidates=(
            ModelSelectionDetail(
                name="fast",
                score=0.9,
                weight=0.7,
                effective_score=0.63,
                available=True,
            ),
            ModelSelectionDetail(
                name="slow",
                score=0.6,
                weight=0.2,
                effective_score=0.12,
                available=False,
                reason="stale_model",
            ),
        ),
    )
    second_selection = ModelSelectionMetadata(
        selected="slow",
        candidates=(
            ModelSelectionDetail(
                name="fast",
                score=0.4,
                weight=0.3,
                effective_score=0.12,
                available=True,
            ),
            ModelSelectionDetail(
                name="slow",
                score=0.85,
                weight=0.65,
                effective_score=0.5525,
                available=True,
            ),
        ),
    )

    first_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=True,
        cost_bps=5.0,
        net_edge_bps=2.0,
        reasons=(),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=14.0,
        model_success_probability=0.6,
        model_name="fast",
        model_selection=first_selection,
    )
    second_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=False,
        cost_bps=6.0,
        net_edge_bps=-1.0,
        reasons=("edge_below_threshold",),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=7.0,
        model_success_probability=0.5,
        model_name="slow",
        model_selection=second_selection,
    )
    third_eval = {
        "candidate": candidate.to_mapping(),
        "accepted": True,
        "cost_bps": 4.0,
        "net_edge_bps": 1.5,
        "model_name": "backup",
        "model_selection": {
            "selected": "backup",
            "candidates": [
                {
                    "name": "fast",
                    "score": 0.55,
                    "weight": 0.45,
                    "effective_score": 0.2475,
                    "available": True,
                },
                {
                    "name": "backup",
                    "score": 0.7,
                    "weight": 0.55,
                    "effective_score": 0.385,
                    "available": True,
                    "reason": "fresh_model",
                },
            ],
        },
    }

    orchestrator = _SequencedOrchestrator((first_eval, second_eval, third_eval))
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 7, 1, 9, 0, 0))
    engine.register_profile(manual_profile)
    engine.attach_decision_orchestrator(orchestrator)

    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=25_000.0,
        stop_price=25_000.0 - 2.0 * 100.0,
        atr=100.0,
        metadata={
            "decision_candidate": candidate_payload,
            "atr": 100.0,
            "stop_price": 25_000.0 - 2.0 * 100.0,
        },
    )
    account = _snapshot(1_000.0)

    engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )
    engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )
    engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )

    stats = engine.decision_model_selection_stats()
    assert set(stats.keys()) == {"fast", "slow", "backup"}

    fast = stats["fast"]
    assert fast["considered"] == 3
    assert fast["selected"] == 1
    assert fast["available"] == 3
    assert fast["unavailable"] == 0
    assert fast["reasons"] == {}
    assert fast["score"]["sum"] == pytest.approx(1.85)
    assert fast["score"]["average"] == pytest.approx(1.85 / 3)
    assert fast["score"]["count"] == 3
    assert fast["weight"]["average"] == pytest.approx(1.45 / 3)
    assert fast["effective_score"]["sum"] == pytest.approx(0.9975)

    slow = stats["slow"]
    assert slow["considered"] == 2
    assert slow["selected"] == 1
    assert slow["available"] == 1
    assert slow["unavailable"] == 1
    assert slow["reasons"] == {"stale_model": 1}
    assert slow["weight"]["sum"] == pytest.approx(0.85)
    assert slow["score"]["average"] == pytest.approx(1.45 / 2)
    assert slow["effective_score"]["average"] == pytest.approx(0.6725 / 2)

    backup = stats["backup"]
    assert backup["considered"] == 1
    assert backup["selected"] == 1
    assert backup["available"] == 1
    assert backup["unavailable"] == 0
    assert backup["reasons"] == {"fresh_model": 1}
    assert backup["score"]["average"] == pytest.approx(0.7)
    assert backup["weight"]["average"] == pytest.approx(0.55)
    assert backup["effective_score"]["average"] == pytest.approx(0.385)

    snapshot = engine.decision_model_selection_stats(reset=True)
    assert snapshot == stats
    assert engine.decision_model_selection_stats() == {}


def test_decision_orchestrator_activity_tracks_counts_and_reset(
    manual_profile: ManualProfile,
) -> None:
    candidate_payload = {
        "strategy": "mean_reversion",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": "ETHUSDT",
        "notional": 150.0,
    }
    candidate = DecisionCandidate.from_mapping(candidate_payload)
    accepted_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=True,
        cost_bps=3.0,
        net_edge_bps=1.5,
        reasons=(),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=12.0,
        model_success_probability=0.55,
        model_name="alpha",
        model_selection=None,
    )
    rejected_eval = DecisionEvaluation(
        candidate=candidate,
        accepted=False,
        cost_bps=4.5,
        net_edge_bps=-0.5,
        reasons=("edge_negative",),
        risk_flags=(),
        stress_failures=(),
        model_expected_return_bps=6.0,
        model_success_probability=0.42,
        model_name="beta",
        model_selection=None,
    )

    orchestrator = _SequencedOrchestrator((accepted_eval, rejected_eval))
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 7, 1, 9, 0, 0))
    engine.register_profile(manual_profile)
    engine.attach_decision_orchestrator(orchestrator)

    order = _order(25_000.0)
    order.metadata = dict(order.metadata)
    order.metadata["decision_candidate"] = candidate_payload
    account = _snapshot(2_000.0)

    engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )
    engine.apply_pre_trade_checks(
        order,
        account=account,
        profile_name=manual_profile.name,
    )

    skip_order = _order(25_000.0)
    engine.apply_pre_trade_checks(
        skip_order,
        account=account,
        profile_name=manual_profile.name,
    )

    stats = engine.decision_orchestrator_activity()
    assert stats["attempts"] == 2
    assert stats["evaluated"] == 2
    assert stats["accepted"] == 1
    assert stats["rejected"] == 1
    assert stats["errors"] == 0
    assert stats["skipped"] == 1
    duration = stats["duration_ms"]
    assert duration["count"] == 2
    assert duration["average"] is not None
    assert stats["error_reasons"] == {}

    snapshot = engine.decision_orchestrator_activity(reset=True)
    assert snapshot == stats

    reset_stats = engine.decision_orchestrator_activity()
    assert reset_stats["attempts"] == 0
    assert reset_stats["errors"] == 0
    assert reset_stats["duration_ms"]["count"] == 0
    assert reset_stats["duration_ms"]["average"] is None


def test_decision_orchestrator_activity_records_errors(
    manual_profile: ManualProfile,
) -> None:
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 7, 1, 9, 0, 0))
    engine.register_profile(manual_profile)
    engine.attach_decision_orchestrator(_FailingOrchestrator())

    invalid_order = _order(25_000.0)
    invalid_order.metadata = dict(invalid_order.metadata)
    invalid_order.metadata["decision_candidate"] = "invalid"
    account = _snapshot(2_000.0)

    result_invalid = engine.apply_pre_trade_checks(
        invalid_order,
        account=account,
        profile_name=manual_profile.name,
    )
    assert result_invalid.allowed is False
    assert "invalid_candidate" in (result_invalid.reason or "")

    valid_candidate = {
        "strategy": "carry",
        "action": "enter",
        "risk_profile": manual_profile.name,
        "symbol": "BTCUSDT",
        "notional": 300.0,
    }
    failing_order = _order(25_000.0)
    failing_order.metadata = dict(failing_order.metadata)
    failing_order.metadata["decision_candidate"] = valid_candidate

    result_failed = engine.apply_pre_trade_checks(
        failing_order,
        account=account,
        profile_name=manual_profile.name,
    )
    assert result_failed.allowed is False
    assert "evaluation_failed" in (result_failed.reason or "")

    stats = engine.decision_orchestrator_activity()
    assert stats["attempts"] == 1
    assert stats["evaluated"] == 0
    assert stats["errors"] == 2
    assert stats["skipped"] == 0
    assert stats["accepted"] == 0
    assert stats["rejected"] == 0
    duration = stats["duration_ms"]
    assert duration["count"] == 1
    assert duration["average"] is not None
    assert stats["error_reasons"] == {
        "invalid_candidate_payload": 1,
        "evaluation_failed": 1,
    }

    snapshot = engine.decision_orchestrator_activity(reset=True)
    assert snapshot == stats
    reset_stats = engine.decision_orchestrator_activity()
    assert reset_stats["errors"] == 0
    assert reset_stats["duration_ms"]["count"] == 0


def test_decision_model_outcomes_empty_without_orchestrator(
    manual_profile: ManualProfile,
) -> None:
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 7, 1, 9, 0, 0))
    engine.register_profile(manual_profile)

    assert engine.decision_model_outcomes() == {}
