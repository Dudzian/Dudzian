from __future__ import annotations

import numpy as np
import pytest

from bot_core.data.pipelines.ml_features import MLFeaturePipeline
from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.ml.base import MLModelAdapter, MLStrategyEngine


def _build_history(size: int, base_price: float = 100.0) -> list[MarketSnapshot]:
    return [
        MarketSnapshot(
            symbol="TEST",
            timestamp=index,
            open=base_price + index,
            high=base_price + index + 1,
            low=base_price + index - 1,
            close=base_price + index,
            volume=1_000 + index,
        )
        for index in range(size)
    ]


class DummyAdapter(MLModelAdapter):
    def __init__(self) -> None:
        super().__init__(name="dummy")
        self.fitted = False
        self.last_train_shape: tuple[int, ...] | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.fitted = True
        self.last_train_shape = features.shape

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full((features.shape[0],), 0.7, dtype=float)


def test_strategy_engine_generates_signal_after_warmup() -> None:
    adapter = DummyAdapter()
    pipeline = MLFeaturePipeline(window=5, forecast_horizon=1)
    engine = MLStrategyEngine(model=adapter, feature_pipeline=pipeline, threshold=0.6)
    history = _build_history(20)
    engine.warm_up(history)
    assert adapter.fitted is True
    assert adapter.last_train_shape is not None

    new_snapshot = MarketSnapshot(
        symbol="TEST",
        timestamp=999,
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1_500,
    )
    signal = engine.on_data(new_snapshot)[0]
    assert signal.side == "BUY"
    assert signal.confidence == pytest.approx(0.1)
    assert signal.metadata["prediction"] == pytest.approx(0.7)
