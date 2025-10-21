"""Pomocnicze stuby do testów modułu AI managera.

Moduł ``tests/test_ai_manager_multimodel.py`` zawierał kilka niemal
identycznych definicji klas ``StubModel``. Każda z nich realizowała ten sam
kontrakt – minimalną implementację modeli wykorzystywanych przez
``AIManager`` – a różniła się jedynie wyzwalaniem prostych hooków. W ramach
uprzątania duplikatów udostępniamy wspólną fabrykę stubów, która pozwala na
parametryzację zachowania bez powielania kodu.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class _TrainHook(Protocol):
    def __call__(self, model_type: str) -> None:  # pragma: no cover - protokół
        ...


class _PredictHook(Protocol):
    def __call__(self, model_type: str) -> None:  # pragma: no cover - protokół
        ...


class _InitHook(Protocol):
    def __call__(self, model_type: str, model_dir: Path | None) -> None:  # pragma: no cover - protokół
        ...


PredictFn = Callable[[str], float]


def positive_negative_predict(model_type: str) -> float:
    """Zwraca wartość ``+1`` dla modelu ``alpha`` oraz ``-1`` dla pozostałych."""

    return 1.0 if model_type == "alpha" else -1.0


def make_stub_model(
    *,
    predict_fn: PredictFn | float = positive_negative_predict,
    init_hook: _InitHook | None = None,
    train_hook: _TrainHook | None = None,
    predict_hook: _PredictHook | None = None,
    predict_series_hook: _PredictHook | None = None,
):
    """Zwraca klasę stubu zgodną z oczekiwaniami ``AIManager``.

    Parametry pozwalają w prosty sposób skonfigurować zachowanie stubu w
    poszczególnych testach – od logowania wywołań po wymuszenie konkretnej
    wartości predykcji.
    """

    def _resolve_prediction(model_type: str) -> float:
        value = predict_fn(model_type) if callable(predict_fn) else predict_fn
        return float(value)

    class StubModel:
        def __init__(
            self,
            input_size: int,
            seq_len: int,
            model_type: str,
            *,
            model_dir: Path | None = None,
        ) -> None:
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None
            if init_hook is not None:
                init_hook(model_type, self.model_dir)

        def train(self, X, y, **_: object) -> None:  # noqa: N803 - podpis zgodny z realnym API
            if train_hook is not None:
                train_hook(self.model_type)

        def predict(self, X):  # noqa: N803 - podpis zgodny z realnym API
            if predict_hook is not None:
                predict_hook(self.model_type)
            return np.full((len(X),), _resolve_prediction(self.model_type), dtype=float)

        def predict_series(self, df, feature_cols):  # noqa: N803 - podpis zgodny z realnym API
            if predict_series_hook is not None:
                predict_series_hook(self.model_type)
            data = np.full(len(df), _resolve_prediction(self.model_type), dtype=float)
            return pd.Series(data, index=df.index)

    return StubModel


__all__ = ["make_stub_model", "positive_negative_predict"]
