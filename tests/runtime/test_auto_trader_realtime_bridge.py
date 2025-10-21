from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest.mock import MagicMock

import pandas as pd
import pytest

from bot_core.alerts import DefaultAlertRouter
from bot_core.alerts.audit import InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel, AlertMessage
from bot_core.auto_trader.app import AutoTrader
from bot_core.config.models import (
    ControllerRuntimeConfig,
    CoreConfig,
    EnvironmentConfig,
    InstrumentUniverseConfig,
    RiskProfileConfig,
)
from bot_core.data.base import CacheStorage, DataSource, OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv.backfill import OHLCVBackfillService
from bot_core.data.ohlcv.cache import CachedOHLCVSource
from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import AccountSnapshot, Environment
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.runtime.controller import DailyTrendController, TradingController
from bot_core.runtime.realtime import DailyTrendRealtimeRunner
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy


class _Emitter:
    def __init__(self) -> None:
        self.logs: list[tuple[str, dict[str, object]]] = []
        self.events: list[tuple[str, Mapping[str, object]]] = []

    def log(self, message: str, *_, **payload: object) -> None:
        self.logs.append((message, dict(payload)))

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, dict(payload)))


class _Var:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _GUI:
    def __init__(self) -> None:
        self.timeframe_var = _Var("1d")

    def is_demo_mode_active(self) -> bool:
        return True


class _CollectingChannel(AlertChannel):
    name = "collector"

    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> Mapping[str, str]:  # pragma: no cover - interface compliance
        return {"status": "ok"}


class _MemoryStorage(CacheStorage):
    def __init__(self) -> None:
        self._store: dict[str, Mapping[str, Sequence[Sequence[float]]]] = {}
        self._metadata: dict[str, str] = {}

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._store[key] = payload

    def metadata(self) -> MutableMapping[str, str]:
        return self._metadata

    def latest_timestamp(self, key: str) -> float | None:
        try:
            rows = self._store[key]["rows"]
        except KeyError:
            return None
        if not rows:
            return None
        return float(rows[-1][0])


@dataclass(slots=True)
class _FixtureSource(DataSource):
    rows: Sequence[Sequence[float]]

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        filtered = [row for row in self.rows if request.start <= float(row[0]) <= request.end]
        limit = request.limit or len(filtered)
        return OHLCVResponse(
            columns=("open_time", "open", "high", "low", "close", "volume"),
            rows=filtered[:limit],
        )

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:  # pragma: no cover - optional
        del symbols, intervals


def _build_daily_trend_controller(position_size: float = 0.2) -> tuple[
    DailyTrendController,
    TradingController,
    PaperTradingExecutionService,
    Sequence[Sequence[float]],
]:
    day_ms = 86_400_000
    start_time = 1_700_000_000_000
    candles = [
        [float(start_time + i * day_ms), 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 10.0]
        for i in range(5)
    ]
    candles.append([float(start_time + 5 * day_ms), 107.0, 110.0, 106.0, 108.0, 12.0])

    storage = _MemoryStorage()
    source = _FixtureSource(rows=candles)
    cached = CachedOHLCVSource(storage=storage, upstream=source)
    backfill = OHLCVBackfillService(cached, chunk_limit=10)

    settings = DailyTrendMomentumSettings(
        fast_ma=3,
        slow_ma=5,
        breakout_lookback=4,
        momentum_window=3,
        atr_window=3,
        atr_multiplier=1.5,
        min_trend_strength=0.0,
        min_momentum=0.0,
    )
    strategy = DailyTrendMomentumStrategy(settings)

    runtime_cfg = ControllerRuntimeConfig(tick_seconds=60.0, interval="1d")
    core_cfg = CoreConfig(
        environments={
            "paper": EnvironmentConfig(
                name="paper",
                exchange="paper",
                environment=Environment.PAPER,
                keychain_key="paper",
                data_cache_path="./var/data",
                risk_profile="paper_risk",
                alert_channels=(),
            )
        },
        risk_profiles={
            "paper_risk": RiskProfileConfig(
                name="paper_risk",
                max_daily_loss_pct=1.0,
                max_position_pct=1.0,
                target_volatility=0.0,
                max_leverage=10.0,
                stop_loss_atr_multiple=2.0,
                max_open_positions=10,
                hard_drawdown_pct=1.0,
            )
        },
        instrument_universes={
            "default": InstrumentUniverseConfig(name="default", description="", instruments=())
        },
        strategies={},
        reporting={},
        sms_providers={},
        telegram_channels={},
        email_channels={},
        signal_channels={},
        whatsapp_channels={},
        messenger_channels={},
        runtime_controllers={"daily_trend": runtime_cfg},
    )

    risk_engine = ThresholdRiskEngine()
    profile = ManualProfile(
        name="paper_risk",
        max_positions=5,
        max_leverage=5.0,
        drawdown_limit=1.0,
        daily_loss_limit=1.0,
        max_position_pct=1.0,
        target_volatility=0.0,
        stop_loss_atr_multiple=2.0,
    )
    risk_engine.register_profile(profile)

    execution_service = PaperTradingExecutionService(
        {"BTCUSDT": MarketMetadata(base_asset="BTC", quote_asset="USDT", min_notional=0.0)},
        initial_balances={"USDT": 100_000.0},
        maker_fee=0.0,
        taker_fee=0.0,
        slippage_bps=0.0,
    )

    account_snapshot = AccountSnapshot(
        balances={"USDT": 100_000.0},
        total_equity=100_000.0,
        available_margin=100_000.0,
        maintenance_margin=0.0,
    )

    controller = DailyTrendController(
        core_config=core_cfg,
        environment_name="paper",
        controller_name="daily_trend",
        symbols=("BTCUSDT",),
        backfill_service=backfill,
        data_source=cached,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_service=execution_service,
        account_loader=lambda: account_snapshot,
        execution_context=ExecutionContext(
            portfolio_id="paper-demo",
            risk_profile="paper_risk",
            environment=Environment.PAPER.value,
            metadata={},
        ),
        position_size=position_size,
    )

    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = _CollectingChannel()
    router.register(channel)

    trading_controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution_service,
        alert_router=router,
        account_snapshot_provider=controller.account_loader,
        portfolio_id=controller.execution_context.portfolio_id,
        environment=controller.execution_context.environment,
        risk_profile=controller.execution_context.risk_profile,
        health_check_interval=0.0,
    )

    return controller, trading_controller, execution_service, candles


@pytest.mark.integration
def test_auto_trader_executes_realtime_controller_cycle() -> None:
    controller, trading_controller, execution_service, candles = _build_daily_trend_controller()

    risk_engine = trading_controller.risk_engine
    original_apply = risk_engine.apply_pre_trade_checks
    spy = MagicMock(wraps=original_apply)
    risk_engine.apply_pre_trade_checks = spy  # type: ignore[assignment]

    emitter = _Emitter()
    gui = _GUI()

    runner = DailyTrendRealtimeRunner(
        controller=controller,
        trading_controller=trading_controller,
        history_bars=10,
        clock=lambda: datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc),
        sleep=lambda _: None,
    )

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        auto_trade_interval_s=0.05,
        controller_runner=runner,
    )

    try:
        trader.start()
        trader.confirm_auto_trade(True)

        for _ in range(80):
            ledger_entries = list(execution_service.ledger())
            if ledger_entries:
                break
            time.sleep(0.05)

        assert list(execution_service.ledger()), "AutoTrader did not trigger execution via controller runner"
        assert spy.called, "Risk engine was not invoked during auto-trade cycle"
        assert runner.last_cycle_signals, "Runner metadata should capture last controller signals"
        assert runner.last_cycle_results, "Runner metadata should capture last order results"
        assert trader._last_signal == "buy", "AutoTrader should expose last signal generated by the controller"

        telemetry = trader.get_last_controller_cycle()
        assert telemetry is not None, "AutoTrader should expose telemetry for the last controller cycle"
        assert telemetry["signals"], "Telemetry should contain controller signals"
        assert telemetry["results"], "Telemetry should contain execution results"
        assert isinstance(telemetry["started_at"], (float, type(None)))
        assert isinstance(telemetry.get("finished_at"), float)
        assert telemetry.get("sequence", 0) >= 1
        assert telemetry.get("orders") == len(telemetry["results"])
        duration = telemetry.get("duration_s")
        assert duration is None or duration >= 0.0

        history = trader.get_controller_cycle_history()
        assert history, "History should contain at least one controller cycle"
        assert history[-1]["sequence"] == telemetry["sequence"]
        assert history[-1]["orders"] == telemetry["orders"]
        assert history[-1]["signals"] == telemetry["signals"]
        assert history[-1]["results"] == telemetry["results"]
        assert trader.get_controller_cycle_history(limit=1)
        reversed_history = trader.get_controller_cycle_history(reverse=True)
        assert reversed_history[0]["sequence"] == history[-1]["sequence"]
        assert trader.get_controller_cycle_history(limit="1")

        event = next((item for item in emitter.events if item[0] == "auto_trader.controller_cycle"), None)
        assert event is not None, "AutoTrader should emit controller cycle telemetry event"
        payload = event[1]
        assert isinstance(payload.get("signals"), tuple)
        assert isinstance(payload.get("results"), tuple)
        assert payload["signals"], "Event payload should contain controller signals"
        assert payload["results"], "Event payload should contain execution results"
        assert payload["started_at"] == telemetry["started_at"]
        assert payload["sequence"] == telemetry["sequence"]
        assert payload["orders"] == telemetry["orders"]
        assert payload.get("finished_at") == telemetry["finished_at"]
    finally:
        trader.stop()
        risk_engine.apply_pre_trade_checks = original_apply  # type: ignore[assignment]


def test_auto_trader_controller_cycle_history_limit() -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _StubRunner:
        def __init__(self) -> None:
            self.counter = 0
            self.last_cycle_signals: Sequence[object] = ()
            self.last_cycle_results: Sequence[object] = ()
            self.last_cycle_started_at = datetime.fromtimestamp(0, tz=timezone.utc)

        def run_once(self) -> list[object]:
            self.counter += 1
            side = "buy" if self.counter % 2 else "sell"
            signal = SimpleNamespace(signal=SimpleNamespace(side=side))
            self.last_cycle_signals = (signal,)
            result = SimpleNamespace(order_id=f"order-{self.counter}")
            self.last_cycle_results = (result,)
            self.last_cycle_started_at = datetime.fromtimestamp(
                1_700_000_000 + self.counter,
                tz=timezone.utc,
            )
            return list(self.last_cycle_results)

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        controller_cycle_history_limit=2,
    )

    runner = _StubRunner()
    trader._execute_controller_runner_cycle(runner)
    trader._execute_controller_runner_cycle(runner)
    trader._execute_controller_runner_cycle(runner)

    history = trader.get_controller_cycle_history()
    assert len(history) == 2
    assert history[0]["sequence"] + 1 == history[1]["sequence"]
    assert history[-1]["sequence"] == trader.get_last_controller_cycle()["sequence"]
    assert history[-1]["orders"] == len(history[-1]["results"])
    assert history[0]["sequence"] >= 1
    assert history[-1]["sequence"] >= history[0]["sequence"]
    assert len(trader.get_controller_cycle_history(limit=5)) == 2
    assert trader.get_controller_cycle_history(limit=0) == []


def test_auto_trader_controller_cycle_history_mutators() -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _StubRunner:
        def __init__(self) -> None:
            self.counter = 0
            self.last_cycle_signals: Sequence[object] = ()
            self.last_cycle_results: Sequence[object] = ()
            self.last_cycle_started_at = datetime.fromtimestamp(0, tz=timezone.utc)

        def run_once(self) -> list[object]:
            self.counter += 1
            signal = SimpleNamespace(signal=SimpleNamespace(side="buy"))
            self.last_cycle_signals = (signal,)
            result = SimpleNamespace(order_id=f"order-{self.counter}")
            self.last_cycle_results = (result,)
            self.last_cycle_started_at = datetime.fromtimestamp(
                1_800_000_000 + self.counter,
                tz=timezone.utc,
            )
            return list(self.last_cycle_results)

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        controller_cycle_history_limit=4,
    )

    runner = _StubRunner()
    for _ in range(3):
        trader._execute_controller_runner_cycle(runner)

    history = trader.get_controller_cycle_history()
    assert len(history) == 3

    effective = trader.set_controller_cycle_history_limit(2)
    assert effective == 2
    trimmed_history = trader.get_controller_cycle_history()
    assert len(trimmed_history) == 2
    assert trimmed_history[0]["sequence"] + 1 == trimmed_history[1]["sequence"]

    unlimited = trader.set_controller_cycle_history_limit(0)
    assert unlimited == -1
    assert len(trader.get_controller_cycle_history()) == 2

    trader.clear_controller_cycle_history()
    assert trader.get_controller_cycle_history() == []
    assert trader.get_last_controller_cycle() is not None


def test_auto_trader_controller_cycle_history_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _StubRunner:
        def __init__(self) -> None:
            self._starts = [100.0, 110.0, 112.0]
            self._index = 0
            self.last_cycle_signals: Sequence[object] = ()
            self.last_cycle_results: Sequence[object] = ()
            self.last_cycle_started_at = datetime.fromtimestamp(
                self._starts[0],
                tz=timezone.utc,
            )

        def run_once(self) -> list[object]:
            timestamp = self._starts[self._index]
            self.last_cycle_started_at = datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            )
            signal = SimpleNamespace(signal=SimpleNamespace(side="buy"))
            self.last_cycle_signals = (signal,)
            result = SimpleNamespace(order_id=f"order-{self._index + 1}")
            self.last_cycle_results = (result,)
            if self._index < len(self._starts) - 1:
                self._index += 1
            return list(self.last_cycle_results)

    timeline = iter([100.5, 101.0, 110.5, 111.0, 112.5, 113.0, 120.0])

    def fake_time() -> float:
        try:
            return next(timeline)
        except StopIteration:
            return 120.0

    monkeypatch.setattr("bot_core.auto_trader.app.time.time", fake_time)

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        controller_cycle_history_limit=16,
        controller_cycle_history_ttl_s=5.0,
    )

    runner = _StubRunner()

    trader._execute_controller_runner_cycle(runner)
    first_history = trader.get_controller_cycle_history()
    assert len(first_history) == 1
    assert trader.get_controller_cycle_history_ttl() == pytest.approx(5.0)

    trader._execute_controller_runner_cycle(runner)
    second_history = trader.get_controller_cycle_history()
    assert len(second_history) == 1
    assert second_history[0]["sequence"] == 2

    trader._execute_controller_runner_cycle(runner)
    third_history = trader.get_controller_cycle_history()
    assert len(third_history) == 2
    assert {entry["sequence"] for entry in third_history} == {2, 3}

    new_ttl = trader.set_controller_cycle_history_ttl(1.0)
    assert new_ttl == pytest.approx(1.0)
    assert trader.get_controller_cycle_history() == []

    disabled_ttl = trader.set_controller_cycle_history_ttl(None)
    assert disabled_ttl is None
    assert trader.get_controller_cycle_history_ttl() is None

    ttl_trim_logs = [payload for message, payload in emitter.logs if payload.get("trimmed_by_ttl")]
    assert ttl_trim_logs, "TTL pruning should emit telemetry in logs"
    assert any(payload.get("trimmed_by_ttl") >= 1 for payload in ttl_trim_logs)


def test_auto_trader_controller_cycle_history_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _StubRunner:
        def __init__(self) -> None:
            self._starts = [200.0, 210.0, 220.0]
            self._sides = ["buy", "sell", "hold"]
            self._statuses = ["filled", "rejected", None]
            self._index = 0
            self.last_cycle_signals: Sequence[object] = ()
            self.last_cycle_results: Sequence[object] = ()
            self.last_cycle_started_at = datetime.fromtimestamp(
                self._starts[0],
                tz=timezone.utc,
            )

        def run_once(self) -> list[object]:
            timestamp = self._starts[self._index]
            self.last_cycle_started_at = datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            )
            side = self._sides[self._index]
            signal = SimpleNamespace(signal=SimpleNamespace(side=side))
            self.last_cycle_signals = (signal,)
            status = self._statuses[self._index]
            result = SimpleNamespace(
                order_id=f"order-{self._index + 1}",
                status=status,
            )
            self.last_cycle_results = (result,)
            if self._index < len(self._starts) - 1:
                self._index += 1
            return list(self.last_cycle_results)

    timeline = iter([200.0, 201.5, 210.0, 212.0, 220.0, 223.5, 230.0])

    def fake_time() -> float:
        try:
            return next(timeline)
        except StopIteration:
            return 230.0

    monkeypatch.setattr("bot_core.auto_trader.app.time.time", fake_time)

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        controller_cycle_history_limit=10,
    )

    runner = _StubRunner()

    trader._execute_controller_runner_cycle(runner)
    trader._execute_controller_runner_cycle(runner)
    trader._execute_controller_runner_cycle(runner)

    summary = trader.summarize_controller_cycle_history()
    assert summary["total"] == 3
    assert summary["config"]["limit"] == 10
    assert summary["orders"]["total"] == 3
    assert summary["orders"]["average"] == pytest.approx(1.0)
    assert summary["signals"]["by_side"] == {"buy": 1, "sell": 1, "hold": 1}
    assert summary["results"]["status_counts"] == {"filled": 1, "rejected": 1}
    assert summary["first_sequence"] == 1
    assert summary["last_sequence"] == 3
    assert summary["duration"]["total"] == pytest.approx(1.5 + 2.0 + 3.5)
    assert summary["duration"]["max"] == pytest.approx(3.5)
    assert summary["first_timestamp"] == pytest.approx(201.5)
    assert summary["last_timestamp"] == pytest.approx(223.5)

    summary_limited = trader.summarize_controller_cycle_history(limit=2)
    assert summary_limited["total"] == 2
    assert summary_limited["first_sequence"] == 2
    assert summary_limited["last_sequence"] == 3

    summary_since = trader.summarize_controller_cycle_history(since=212.0)
    assert summary_since["total"] == 2
    assert summary_since["first_sequence"] == 2
    assert summary_since["last_sequence"] == 3

    summary_until = trader.summarize_controller_cycle_history(until=212.0)
    assert summary_until["total"] == 2
    assert summary_until["last_sequence"] == 2

    summary_window = trader.summarize_controller_cycle_history(since=220.0, until=223.6)
    assert summary_window["total"] == 1
    assert summary_window["first_sequence"] == 3
    assert summary_window["last_sequence"] == 3

    summary_empty = trader.summarize_controller_cycle_history(limit=0, since=100.0)
    assert summary_empty["total"] == 0
    assert summary_empty["filters"]["limit"] == 0
    assert summary_empty["filters"]["since"] == pytest.approx(100.0)
    assert summary_empty["orders"]["total"] == 0


def test_auto_trader_controller_cycle_history_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _StubRunner:
        def __init__(self) -> None:
            self._starts = (200.0, 205.0, 210.0)
            self._sides = ("buy", "sell", "hold")
            self._statuses = ("filled", "rejected", None)
            self._index = 0
            self.last_cycle_signals: Sequence[object] = ()
            self.last_cycle_results: Sequence[object] = ()
            self.last_cycle_started_at = datetime.fromtimestamp(
                self._starts[0],
                tz=timezone.utc,
            )

        def run_once(self) -> list[object]:
            timestamp = self._starts[self._index]
            self.last_cycle_started_at = datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            )
            signal = SimpleNamespace(signal=SimpleNamespace(side=self._sides[self._index]))
            self.last_cycle_signals = (signal,)
            result = SimpleNamespace(
                order_id=f"order-{self._index + 1}",
                status=self._statuses[self._index],
            )
            self.last_cycle_results = (result,)
            if self._index < len(self._starts) - 1:
                self._index += 1
            return list(self.last_cycle_results)

    timeline = iter([200.0, 201.5, 212.0, 215.0, 222.0, 230.0])

    def fake_time() -> float:
        try:
            return next(timeline)
        except StopIteration:
            return 230.0

    monkeypatch.setattr("bot_core.auto_trader.app.time.time", fake_time)

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        controller_cycle_history_limit=10,
    )

    runner = _StubRunner()

    trader._execute_controller_runner_cycle(runner)
    trader._execute_controller_runner_cycle(runner)
    trader._execute_controller_runner_cycle(runner)

    frame = trader.controller_cycle_history_to_dataframe()
    assert list(frame.columns) == [
        "sequence",
        "started_at",
        "finished_at",
        "duration_s",
        "orders",
        "signals_count",
        "results_count",
        "signals",
        "results",
    ]
    assert len(frame) == 3
    assert frame.iloc[0]["sequence"] == 1
    assert isinstance(frame.iloc[0]["finished_at"], pd.Timestamp)
    assert frame.iloc[0]["signals_count"] == 1
    assert frame.iloc[0]["results_count"] == 1

    limited = trader.controller_cycle_history_to_dataframe(limit=2, reverse=True)
    assert list(limited["sequence"]) == [3, 2]

    window = trader.controller_cycle_history_to_dataframe(since=215.0)
    assert list(window["sequence"]) == [2, 3]

    cutoff = trader.controller_cycle_history_to_dataframe(until=212.0)
    assert list(cutoff["sequence"]) == [1]

    raw = trader.controller_cycle_history_to_dataframe(coerce_timestamps=False)
    assert isinstance(raw.iloc[0]["finished_at"], float)

    slim = trader.controller_cycle_history_to_dataframe(
        include_sequences=False,
        include_counts=False,
    )
    assert list(slim.columns) == [
        "sequence",
        "started_at",
        "finished_at",
        "duration_s",
        "orders",
    ]

    empty = trader.controller_cycle_history_to_dataframe(limit=0)
    assert empty.empty

    records = trader.controller_cycle_history_to_records()
    assert len(records) == 3
    assert records[0]["sequence"] == 1
    assert records[0]["signals_count"] == 1
    assert isinstance(records[0]["signals"], tuple)
    assert isinstance(records[0]["finished_at"], float)

    records_window = trader.controller_cycle_history_to_records(since=215.0)
    assert [item["sequence"] for item in records_window] == [2, 3]

    limited_records = trader.controller_cycle_history_to_records(
        limit=1,
        reverse=True,
        include_sequences=False,
        include_counts=False,
    )
    assert len(limited_records) == 1
    assert limited_records[0]["sequence"] == 3
    assert set(limited_records[0]) == {
        "sequence",
        "started_at",
        "finished_at",
        "duration_s",
        "orders",
    }

    coerced = trader.controller_cycle_history_to_records(coerce_timestamps=True)
    assert isinstance(coerced[0]["finished_at"], datetime)
    assert coerced[0]["finished_at"].tzinfo == timezone.utc

    empty_records = trader.controller_cycle_history_to_records(limit=0)
    assert empty_records == []
