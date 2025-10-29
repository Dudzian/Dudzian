"""Pipeline przygotowujący cechy dla strategii ML."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Sequence

import numpy as np

from bot_core.strategies.base import MarketSnapshot


@dataclass(slots=True)
class MLFeaturePipeline:
    """Buduje cechy oparte o prostą analizę okien czasowych."""

    window: int = 20
    forecast_horizon: int = 1
    normalise: bool = True
    feature_names: tuple[str, ...] = ("close_last", "close_change", "close_mean", "close_std", "volume_last")
    _history: Deque[MarketSnapshot] = field(default_factory=lambda: deque(maxlen=256), init=False)
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _std: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, snapshots: Sequence[MarketSnapshot]) -> None:
        self._history.clear()
        for snap in snapshots[-self.window :]:
            self._history.append(snap)
        features, _ = self.build_training_set(snapshots)
        if self.normalise and len(features):
            self._mean = features.mean(axis=0)
            self._std = features.std(axis=0)

    def transform_features(self, snapshot: MarketSnapshot) -> np.ndarray:
        self._append_snapshot(snapshot)
        if len(self._history) < self.window:
            raise ValueError("Za mało danych historycznych do zbudowania cech")
        window_snaps = list(self._history)[-self.window :]
        features = self._compute_features(window_snaps)
        return self._normalise(features)

    def build_training_set(self, history: Sequence[MarketSnapshot]) -> tuple[np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        targets: list[float] = []
        last_index = len(history) - self.forecast_horizon + 1
        for idx in range(self.window, max(self.window, last_index)):
            window_snaps = history[idx - self.window : idx]
            future_snapshot = history[idx + self.forecast_horizon - 1]
            features.append(self._compute_features(window_snaps))
            current_close = window_snaps[-1].close
            future_close = future_snapshot.close
            target_return = (future_close - current_close) / current_close if current_close else 0.0
            targets.append(target_return)
        if not features:
            return np.empty((0, len(self.feature_names))), np.empty((0,))
        features_matrix = np.vstack(features)
        targets_array = np.asarray(targets, dtype=float)
        if self.normalise:
            self._mean = features_matrix.mean(axis=0)
            self._std = features_matrix.std(axis=0)
            features_matrix = self._normalise(features_matrix)
        return features_matrix, targets_array

    # ------------------------------------------------------------------ helpers --
    def _append_snapshot(self, snapshot: MarketSnapshot) -> None:
        maxlen = max(self.window + self.forecast_horizon, 2 * self.window)
        if self._history.maxlen != maxlen:
            self._history = deque(self._history, maxlen=maxlen)
        self._history.append(snapshot)

    def _compute_features(self, window_snaps: Sequence[MarketSnapshot]) -> np.ndarray:
        closes = np.array([snap.close for snap in window_snaps], dtype=float)
        volumes = np.array([snap.volume for snap in window_snaps], dtype=float)
        change = closes[-1] - closes[0]
        features = np.array(
            [
                closes[-1],
                change,
                float(closes.mean()),
                float(closes.std(ddof=0)),
                volumes[-1],
            ],
            dtype=float,
        )
        return features

    def _normalise(self, features: np.ndarray) -> np.ndarray:
        if not self.normalise:
            return features
        if self._mean is None or self._std is None:
            return features
        return (features - self._mean) / np.where(self._std == 0, 1.0, self._std)


__all__ = ["MLFeaturePipeline"]
