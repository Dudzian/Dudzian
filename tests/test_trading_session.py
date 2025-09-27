from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from bot_core.alerts import DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel, AlertMessage
from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.runtime.session import InstrumentConfig, TradingSession
from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


class _StaticStrategy(StrategyEngine):
    def __init__(self, signals: Sequence[StrategySignal]) -> None:
        self._signals = list(signals)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:  # noqa: ARG002
        return list(self._signals)

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:  # noqa: D401, ARG002 - brak dodatkowych działań
        """Strategia statyczna nie wymaga rozgrzewania."""
        return None


class _MemoryExecutionService(ExecutionService):
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []
        self.contexts: list[ExecutionContext] = []

    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        self.requests.append(request)
        self.contexts.append(context)
        return OrderResult(
            order_id="exec-1",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={"source": "test"},
        )

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # noqa: ARG002 - tylko interfejs
        return None

    def flush(self) -> None:
        return None


class _StubAdapter(ExchangeAdapter):
    def __init__(self, snapshot: AccountSnapshot) -> None:
        super().__init__(ExchangeCredentials(key_id="stub"))
        self._snapshot = snapshot

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401, ARG002
        """Konfiguracja sieciowa nie jest potrzebna w testach."""

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return self._snapshot

    def fetch_symbols(self) -> Iterable[str]:
        return []

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # noqa: D401, ARG002
        """Nie korzystamy z danych OHLCV w testach runtime."""
        return []

    def place_order(self, request: OrderRequest) -> OrderResult:  # pragma: no cover - nieużywane w tym teście
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # pragma: no cover - nieużywane
        raise NotImplementedError

    def stream_public_data(self, *, channels: Sequence[str]):  # pragma: no cover - nieużywane
        raise NotImplementedError

    def stream_private_data(self, *, channels: Sequence[str]):  # pragma: no cover - nieużywane
        raise NotImplementedError


class _DummyChannel(AlertChannel):
    def __init__(self) -> None:
        self.name = "dummy"
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> Mapping[str, str]:
        return {"status": "ok"}


def _build_profile(name: str) -> ManualProfile:
    return ManualProfile(
        name=name,
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.10,
        daily_loss_limit=0.015,
        max_position_pct=0.05,
        target_volatility=0.11,
        stop_loss_atr_multiple=1.5,
    )


def test_trading_session_executes_signal_and_sends_alert() -> None:
    profile = _build_profile("balanced")
    risk_engine = ThresholdRiskEngine()
    risk_engine.register_profile(profile)

    execution = _MemoryExecutionService()
    adapter = _StubAdapter(
        AccountSnapshot(
            balances={"USDT": 60000.0, "BTC": 0.0},
            total_equity=100000.0,
            available_margin=60000.0,
            maintenance_margin=0.0,
        )
    )

    channel = _DummyChannel()
    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    router.register(channel)

    strategy = _StaticStrategy(
        [
            StrategySignal(
                symbol="BTCUSDT",
                side="buy",
                confidence=0.8,
                metadata={
                    "stop_price": 19000.0,
                    "atr": 1000.0,
                },
            )
        ]
    )

    session = TradingSession(
        strategy=strategy,
        strategy_name="daily_trend",
        adapter=adapter,
        risk_engine=risk_engine,
        risk_profile=profile,
        execution=execution,
        alert_router=router,
        instruments={
            "BTCUSDT": InstrumentConfig(
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                min_quantity=0.001,
                min_notional=10.0,
                step_size=0.001,
            )
        },
        environment=Environment.PAPER,
        portfolio_id="core",
    )

    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=1_700_000_000,
        open=20000.0,
        high=20500.0,
        low=19800.0,
        close=20000.0,
        volume=123.0,
    )

    results = session.process_snapshot(snapshot)

    assert len(results) == 1
    request = execution.requests[0]
    assert pytest.approx(request.quantity, rel=1e-3) == 0.2
    assert channel.messages, "Kanał powinien otrzymać alert po realizacji zlecenia."
    message = channel.messages[-1]
    assert message.category == "trade"
    assert message.severity == "info"


def test_trading_session_reports_risk_rejection() -> None:
    profile = ManualProfile(
        name="tight",
        max_positions=0,
        max_leverage=1.0,
        drawdown_limit=0.05,
        daily_loss_limit=0.01,
        max_position_pct=0.05,
        target_volatility=0.05,
        stop_loss_atr_multiple=1.0,
    )
    risk_engine = ThresholdRiskEngine()
    risk_engine.register_profile(profile)

    execution = _MemoryExecutionService()
    adapter = _StubAdapter(
        AccountSnapshot(
            balances={"USDT": 60000.0},
            total_equity=100000.0,
            available_margin=60000.0,
            maintenance_margin=0.0,
        )
    )

    channel = _DummyChannel()
    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    router.register(channel)

    strategy = _StaticStrategy(
        [
            StrategySignal(
                symbol="BTCUSDT",
                side="buy",
                confidence=0.9,
                metadata={"stop_price": 19000.0},
            )
        ]
    )

    session = TradingSession(
        strategy=strategy,
        strategy_name="daily_trend",
        adapter=adapter,
        risk_engine=risk_engine,
        risk_profile=profile,
        execution=execution,
        alert_router=router,
        instruments={
            "BTCUSDT": InstrumentConfig(
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                min_quantity=0.001,
                min_notional=10.0,
                step_size=0.001,
            )
        },
        environment=Environment.PAPER,
        portfolio_id="core",
    )

    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=1_700_000_100,
        open=20000.0,
        high=20200.0,
        low=19900.0,
        close=20000.0,
        volume=50.0,
    )

    results = session.process_snapshot(snapshot)

    assert results == []
    assert execution.requests == []
    assert channel.messages, "Oczekiwano alertu ostrzegającego o odrzuceniu przez ryzyko."
    assert channel.messages[-1].severity == "warning"
