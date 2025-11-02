from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from bot_core.backtest.walk_forward import (
    WalkForwardBacktester,
    WalkForwardSegment,
)
from bot_core.strategies.catalog import StrategyDefinition


class _WarningEngine:
    """Minimalny silnik strategii emitujący ostrzeżenie pandas."""

    def __init__(self) -> None:
        self._warned = False

    def warm_up(self, snapshots: list[object]) -> None:  # pragma: no cover - nieistotne
        del snapshots

    def on_data(self, snapshot: object) -> list[SimpleNamespace]:
        if not self._warned:
            warnings.warn("vectorized fallback", pd.errors.PerformanceWarning)
            self._warned = True
        return [SimpleNamespace(side="buy", confidence=1.0)]


class _StubCatalog:
    def create(self, _: StrategyDefinition) -> _WarningEngine:
        return _WarningEngine()


def test_walk_forward_segment_captures_pandas_warnings(caplog) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [1, 1, 1, 1, 1, 1],
        },
        index=index,
    )

    segment = WalkForwardSegment(
        train_start=index[0],
        train_end=index[2],
        test_start=index[2],
        test_end=index[-1],
    )

    definition = StrategyDefinition(
        name="stub",
        engine="stub",
        license_tier="standard",
        risk_classes=("retail",),
        required_data=("close",),
    )

    backtester = WalkForwardBacktester(_StubCatalog())

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        caplog.at_level(logging.WARNING, logger="bot_core.backtest.walk_forward"),
    ):
        report = backtester.run(definition, {"TEST": frame}, [segment], initial_balance=1_000.0)

    assert report.segments, "Powinien zostać wygenerowany co najmniej jeden segment raportu."
    assert observe_warning.call_count == 1
    kwargs = observe_warning.call_args.kwargs
    assert kwargs["component"] == "backtest.walk_forward.segment"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "vectorized fallback"
    assert any(
        "Pandas warning captured in backtest.walk_forward.segment" in message
        for message in caplog.messages
    )
