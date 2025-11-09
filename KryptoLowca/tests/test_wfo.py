import numpy as np
import pandas as pd
import pytest

from bot_core.ai.manager import AIManager
from KryptoLowca.services.wfo import WalkForwardOptimizer


def _build_df(rows: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    base = np.linspace(25000, 26000, rows) + np.random.randn(rows) * 15
    return pd.DataFrame(
        {
            "open": base + np.random.randn(rows),
            "high": base + np.abs(np.random.randn(rows)),
            "low": base - np.abs(np.random.randn(rows)),
            "close": base + np.random.randn(rows),
            "volume": np.random.rand(rows) * 500,
        },
        index=index,
    )


@pytest.mark.asyncio
async def test_walk_forward_optimizer_generates_report(tmp_path):
    manager = AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    optimizer = WalkForwardOptimizer(manager, seq_len=12, folds=2)
    df = _build_df(420)
    report = await optimizer.optimize("BTCUSDT", df, ["rf", "lstm"], window=160, step=40)
    assert report.slices
    assert report.best_model in {"rf", "lstm"}
    assert report.aggregate_hit_rate >= 0.0
