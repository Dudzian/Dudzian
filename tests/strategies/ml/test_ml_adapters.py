from __future__ import annotations

import numpy as np
import pytest

from bot_core.strategies.ml import classical


class DummyEstimator:
    def __init__(self, **params: object) -> None:
        self.params = params
        self.fit_called = False

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.fit_called = True
        self.last_fit_shape = features.shape

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.ones((features.shape[0],), dtype=float) * 0.75


def test_random_forest_adapter_delegates_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classical, "RandomForestRegressor", DummyEstimator)
    adapter = classical.RandomForestAdapter(n_estimators=10)
    features = np.ones((4, 3))
    target = np.ones((4,))
    adapter.fit(features, target)
    prediction = adapter.predict(features)
    assert adapter.estimator.fit_called is True
    assert prediction.shape == (4,)
    assert prediction[0] == pytest.approx(0.75)


def test_xgboost_adapter_requires_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classical, "xgboost", None)
    with pytest.raises(ImportError):
        classical.XGBoostAdapter()
