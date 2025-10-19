import importlib
import sys
import time

try:
    _auto_trader_module = importlib.import_module("bot_core.auto_trader")
except ModuleNotFoundError:  # pragma: no cover - legacy fallback for local runs
    runtime_module = importlib.import_module("bot_core.runtime")
    if not hasattr(runtime_module, "resolve_core_config_path"):
        runtime_module.resolve_core_config_path = importlib.import_module(  # type: ignore[attr-defined]
            "bot_core.runtime.paths"
        ).resolve_core_config_path
    if not hasattr(runtime_module, "PaperTradingAdapter"):
        runtime_module.PaperTradingAdapter = importlib.import_module(  # type: ignore[attr-defined]
            "bot_core.runtime.paper_trading"
        ).PaperTradingAdapter

    legacy_auto_trader = importlib.import_module("KryptoLowca.auto_trader")
    sys.modules.setdefault("bot_core.auto_trader", legacy_auto_trader)
    _auto_trader_module = legacy_auto_trader

AutoTrader = _auto_trader_module.AutoTrader


class _DummyEmitter:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def on(self, *_args, **_kwargs) -> None:  # pragma: no cover - hooks nie są potrzebne
        return

    def off(self, *_args, **_kwargs) -> None:  # pragma: no cover
        return

    def emit(self, *_args, **_kwargs) -> None:  # pragma: no cover
        return

    def log(self, message: str, *_, **__) -> None:
        self.logs.append(message)


class _DummyVar:
    def get(self) -> str:
        return "1m"


class _DummyGui:
    def __init__(self) -> None:
        self.timeframe_var = _DummyVar()
        self.ai_mgr = None
        self.ex_mgr = None

    def is_demo_mode_active(self) -> bool:
        return True


def test_auto_trader_requires_manual_confirmation() -> None:
    emitter = _DummyEmitter()
    gui = _DummyGui()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=True,
        walkforward_interval_s=None,
        auto_trade_interval_s=0.01,
        market_data_provider=None,
    )

    trader._auto_trade_loop = lambda: trader._stop.wait(0.05)  # type: ignore[attr-defined]

    trader.start()
    assert trader.enable_auto_trade is True
    assert trader._started is True
    assert trader._auto_trade_thread_active is False
    assert any(
        "Auto-trade awaiting explicit activation" in message for message in emitter.logs
    )

    trader.confirm_auto_trade(True)
    time.sleep(0.02)
    assert trader._auto_trade_thread_active is True

    trader.confirm_auto_trade(False)
    time.sleep(0.01)
    assert trader._auto_trade_user_confirmed is False

    trader.stop()
