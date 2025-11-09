# test_ai_manager.py
# -*- coding: utf-8 -*-
"""
Unit tests for ai_manager.py.
"""
import asyncio
import pytest
import pandas as pd
import numpy as np
from bot_core.ai.manager import AIManager, TrainResult

@pytest.fixture
async def ai_manager(tmp_path):
    class MockAIModels:
        def __init__(self, input_size: int, seq_len: int, model_type: str = "rf"):
            self.model_type = model_type
            self.input_size = input_size
            self.seq_len = seq_len

        async def train(self, X, y, epochs, batch_size, progress_callback=None, model_out=None, verbose=False):
            if model_out:
                with open(model_out, "wb") as f:
                    joblib.dump(self, f)
            return None

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.random.randn(len(X))

        async def predict_series(self, df, feature_cols):
            return pd.Series(np.random.randn(len(df)), index=df.index)

        @staticmethod
        def load_model(path):
            return joblib.load(path)

    global _AIModels
    _AIModels = MockAIModels

    def mock_windowize(df, feature_cols, seq_len, target_col):
        return np.random.randn(len(df) - seq_len, seq_len, len(feature_cols)), np.random.randn(len(df) - seq_len)

    global _windowize
    _windowize = mock_windowize

    manager = AIManager(ai_threshold_bps=5.0, model_dir=str(tmp_path))
    return manager

@pytest.mark.asyncio
async def test_train_all_models(ai_manager, tmp_path):
    df = pd.DataFrame({
        "open": np.random.randn(100), "high": np.random.randn(100),
        "low": np.random.randn(100), "close": np.random.randn(100),
        "volume": np.random.randn(100), "rsi_14": np.random.randn(100)
    })
    result = await ai_manager.train_all_models("BTCUSDT", df, ["rf", "lstm"], seq_len=10, epochs=5)
    assert "rf" in result and "lstm" in result
    assert isinstance(result["rf"], TrainResult)
    assert (tmp_path / "btcusdt:rf.joblib").exists()

@pytest.mark.asyncio
async def test_predict_series(ai_manager):
    df = pd.DataFrame({
        "open": np.random.randn(100), "high": np.random.randn(100),
        "low": np.random.randn(100), "close": np.random.randn(100),
        "volume": np.random.randn(100), "rsi_14": np.random.randn(100)
    })
    preds = await ai_manager.predict_series("BTCUSDT", df, model_types=["rf"])
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(df)

@pytest.mark.asyncio
async def test_invalid_input(ai_manager):
    df = pd.DataFrame({"invalid": np.random.randn(100)})
    with pytest.raises(ValueError):
        await ai_manager.train_all_models("BTCUSDT", df, ["rf"])
    df = pd.DataFrame({"open": [np.nan] * 100, "high": np.random.randn(100),
                       "low": np.random.randn(100), "close": np.random.randn(100), "volume": np.random.randn(100)})
    with pytest.raises(ValueError):
        await ai_manager.train_all_models("BTCUSDT", df, ["rf"])

@pytest.mark.asyncio
async def test_sanitize_predictions(ai_manager):
    series = pd.Series([1000, -1000, 0.1, 0.2, 10])
    sanitized = ai_manager._sanitize_predictions(series)
    assert abs(sanitized).max() <= 1.0
    assert len(sanitized) == len(series)
    series = pd.Series([1.0] * 5)  # Constant series
    sanitized = ai_manager._sanitize_predictions(series)
    assert abs(sanitized).max() <= 1.0

@pytest.mark.asyncio
async def test_import_model(ai_manager, tmp_path):
    df = pd.DataFrame({
        "open": np.random.randn(100), "high": np.random.randn(100),
        "low": np.random.randn(100), "close": np.random.randn(100),
        "volume": np.random.randn(100), "rsi_14": np.random.randn(100)
    })
    await ai_manager.train_all_models("BTCUSDT", df, ["rf"], seq_len=10, epochs=5)
    model_path = tmp_path / "btcusdt:rf.joblib"
    await ai_manager.import_model("BTCUSDT", "rf", str(model_path))
    assert "btcusdt:rf" in ai_manager.models