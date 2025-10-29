"""Adaptery dla klasycznych modeli uczenia nadzorowanego."""
from __future__ import annotations

from typing import Any

from .base import ClassicalModelAdapter

try:  # pragma: no cover - import opcjonalny
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # noqa: BLE001
    RandomForestRegressor = None  # type: ignore[assignment]

try:  # pragma: no cover - import opcjonalny
    import xgboost
except Exception:  # noqa: BLE001
    xgboost = None  # type: ignore[assignment]


class RandomForestAdapter(ClassicalModelAdapter):
    """Adapter RandomForest z domyślnymi hiperparametrami giełdowymi."""

    def __init__(self, **hyperparameters: Any):
        if RandomForestRegressor is None:
            raise ImportError("Wymagana jest biblioteka scikit-learn")
        estimator = RandomForestRegressor(**hyperparameters)
        super().__init__(estimator=estimator, name="random_forest", hyperparameters=hyperparameters)


class XGBoostAdapter(ClassicalModelAdapter):
    """Adapter modelu gradient boosting XGBoost."""

    def __init__(self, **hyperparameters: Any):
        if xgboost is None:
            raise ImportError("Wymagana jest biblioteka xgboost")
        booster = xgboost.XGBRegressor(**hyperparameters)
        super().__init__(estimator=booster, name="xgboost", hyperparameters=hyperparameters)


__all__ = ["RandomForestAdapter", "XGBoostAdapter"]
