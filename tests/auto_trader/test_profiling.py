from types import SimpleNamespace

from bot_core.auto_trader import AutoTrader
from core.perf import profile_block


class _Emitter:
    def emit(self, *_args, **_kwargs) -> None:
        return None

    def log(self, *_args, **_kwargs) -> None:
        return None


def _cpu_intensive_task() -> int:
    total = 0
    for value in range(200):
        total += value * value
    return total


def test_auto_trader_summarize_hotspots_merges_reports() -> None:
    trader = AutoTrader(
        _Emitter(),
        SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
    )

    for index in range(2):
        with profile_block(f"test.session.{index}", enable_gpu=False, limit=10) as session:
            for _ in range(5):
                _cpu_intensive_task()
        report = session.report
        assert report is not None
        trader._store_profile(report)

    hotspots = trader.summarize_hotspots(limit=5)
    assert hotspots, "Hot spot summary should not be empty"
    top = hotspots[0]
    assert top["function"], "Hot spot entry should include function name"
    assert top["total_calls"] >= top["primitive_calls"]
    assert top["avg_cumulative_time"] >= 0.0


def test_auto_trader_keeps_multiple_reports_per_section() -> None:
    trader = AutoTrader(
        _Emitter(),
        SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
    )

    for _ in range(2):
        with profile_block("test.session", enable_gpu=False, limit=5) as session:
            for _ in range(3):
                _cpu_intensive_task()
        report = session.report
        assert report is not None
        trader._store_profile(report)

    reports = trader.get_profile_reports()
    assert "auto_trader.test.session" not in reports  # guard against aliasing
    stored = reports.get("test.session")
    assert stored is not None
    assert len(stored) == 2
    assert stored[0] is not stored[1]
