from __future__ import annotations

import pytest

from bot_core.data.pipelines.ml_features import MLFeaturePipeline
from bot_core.strategies.base import MarketSnapshot


def _build_history(size: int, base_price: float = 100.0) -> list[MarketSnapshot]:
    history: list[MarketSnapshot] = []
    for idx in range(size):
        price = base_price + idx
        history.append(
            MarketSnapshot(
                symbol="TEST",
                timestamp=idx,
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000 + idx,
            )
        )
    return history


def test_ml_feature_pipeline_generates_training_set() -> None:
    pipeline = MLFeaturePipeline(window=5, forecast_horizon=1, normalise=False)
    history = _build_history(16)
    features, target = pipeline.build_training_set(history)
    assert features.shape[1] == len(pipeline.feature_names)
    assert features.shape[0] == len(history) - pipeline.window - pipeline.forecast_horizon + 1
    assert target.shape[0] == features.shape[0]
    first_target = (history[pipeline.window].close - history[pipeline.window - 1].close) / history[pipeline.window - 1].close
    assert target[0] == pytest.approx(first_target)


def test_transform_requires_sufficient_history() -> None:
    pipeline = MLFeaturePipeline(window=3)
    with pytest.raises(ValueError):
        pipeline.transform_features(MarketSnapshot(
            symbol="TEST",
            timestamp=999,
            open=1,
            high=1,
            low=1,
            close=1,
            volume=1,
        ))
    history = _build_history(3)
    pipeline.fit(history)
    for snap in history:
        pipeline.transform_features(snap)
    features = pipeline.transform_features(_build_history(1, base_price=200.0)[0])
    assert features.shape == (len(pipeline.feature_names),)
