"""Fixtures pomocnicze dla testów modułu core.ml."""

from __future__ import annotations

from typing import Iterable

import pytest

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector


def _build_vectors() -> Iterable[FeatureVector]:
    base_timestamp = 1_700_000_000.0
    for idx in range(6):
        yield FeatureVector(
            timestamp=base_timestamp + idx * 60.0,
            symbol="BTCUSDT",
            features={
                "momentum": float(idx) * 0.1,
                "volatility": 0.5 + float(idx) * 0.05,
            },
            target_bps=0.02 * (idx - 2),
        )


@pytest.fixture()
def synthetic_feature_dataset() -> FeatureDataset:
    """Niewielki syntetyczny zbiór cech wykorzystywany w testach."""

    vectors = tuple(_build_vectors())
    metadata = {"source": "synthetic", "target_scale": 1.0}
    return FeatureDataset(vectors=vectors, metadata=metadata)

