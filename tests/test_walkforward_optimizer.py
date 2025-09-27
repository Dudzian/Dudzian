"""Testy modułu walk-forward."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.strategies import MarketSnapshot
from bot_core.strategies.walkforward import (
    RollingWindowWalkForwardOptimizer,
    WalkForwardError,
    WalkForwardWindow,
)


def _build_snapshot(index: int, symbol: str = "BTC/USDT") -> MarketSnapshot:
    price = float(index + 1)
    return MarketSnapshot(
        symbol=symbol,
        timestamp=index,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price,
        volume=1.0,
    )


def _build_series(length: int, symbol: str = "BTC/USDT") -> Sequence[MarketSnapshot]:
    return tuple(_build_snapshot(i, symbol) for i in range(length))


def test_split_generates_overlapping_segments() -> None:
    series = _build_series(12)
    window = WalkForwardWindow(training_size=5, testing_size=2)
    optimizer = RollingWindowWalkForwardOptimizer(
        window,
        parameter_grid=[{"alpha": 0.1}],
        scorer=lambda data, params: 0.0,
    )

    segments = optimizer.split(series)

    assert len(segments) == 3
    assert all(len(in_sample) == 5 for in_sample, _ in segments)
    assert all(len(out_sample) == 2 for _, out_sample in segments)
    # upewniamy się, że segmenty przesuwają się o długość testową
    assert segments[0][0][0].timestamp == 0
    assert segments[1][0][0].timestamp == 2
    assert segments[2][0][0].timestamp == 4


def test_split_requires_enough_data() -> None:
    series = _build_series(6)
    window = WalkForwardWindow(training_size=5, testing_size=2)
    optimizer = RollingWindowWalkForwardOptimizer(
        window,
        parameter_grid=[{"alpha": 0.1}],
        scorer=lambda data, params: 0.0,
    )

    with pytest.raises(WalkForwardError):
        optimizer.split(series)


def test_select_parameters_respects_objective_direction() -> None:
    series = _build_series(10)
    window = WalkForwardWindow(training_size=6, testing_size=2, step_size=2)

    def scorer(data: Sequence[MarketSnapshot], params: dict[str, float]) -> float:
        total = sum(snapshot.close for snapshot in data)
        return total * params["alpha"]

    optimizer = RollingWindowWalkForwardOptimizer(
        window,
        parameter_grid=[{"alpha": 0.5}, {"alpha": 1.0}],
        scorer=scorer,
        maximize=True,
    )

    best = optimizer.select_parameters(series[:6])
    assert best == {"alpha": 1.0}

    optimizer_min = RollingWindowWalkForwardOptimizer(
        window,
        parameter_grid=[{"alpha": 0.5}, {"alpha": 1.0}],
        scorer=scorer,
        maximize=False,
    )

    best_min = optimizer_min.select_parameters(series[:6])
    assert best_min == {"alpha": 0.5}


def test_walkforward_window_validates_arguments() -> None:
    with pytest.raises(WalkForwardError):
        WalkForwardWindow(training_size=0, testing_size=2)
    with pytest.raises(WalkForwardError):
        WalkForwardWindow(training_size=5, testing_size=0)
    with pytest.raises(WalkForwardError):
        WalkForwardWindow(training_size=5, testing_size=2, step_size=0)
