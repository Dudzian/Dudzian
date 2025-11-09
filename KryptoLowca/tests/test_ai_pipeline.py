import asyncio

import numpy as np
import pandas as pd
import pytest

from bot_core.ai.manager import AIManager


def _build_dataframe(rows: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    base = np.linspace(10000, 10500, rows) + np.random.randn(rows) * 10
    df = pd.DataFrame(
        {
            "open": base + np.random.randn(rows),
            "high": base + np.abs(np.random.randn(rows)),
            "low": base - np.abs(np.random.randn(rows)),
            "close": base + np.random.randn(rows),
            "volume": np.random.rand(rows) * 1000,
        },
        index=index,
    )
    return df


@pytest.mark.asyncio
async def test_select_best_model_ranks_and_returns(tmp_path):
    manager = AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _build_dataframe(220)
    result = await manager.select_best_model("BTCUSDT", df, ["rf", "lstm"], seq_len=16, folds=3)
    assert result.best_model in {"rf", "lstm"}
    assert result.evaluations
    scores = [ev.composite_score() for ev in result.evaluations]
    assert scores == sorted(scores, reverse=True)


def test_detect_drift_flags_high_variance(tmp_path):
    manager = AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    baseline = _build_dataframe(200)
    recent = baseline.copy()
    recent["close"] = recent["close"] * 1.05 + np.random.randn(len(recent)) * 50
    report = manager.detect_drift(baseline, recent, threshold=0.05)
    assert report.triggered
    assert report.feature_drift >= 0.0


@pytest.mark.asyncio
async def test_schedule_periodic_training_registers_and_cancels(tmp_path):
    manager = AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _build_dataframe(180)

    async def _run():
        schedule = manager.schedule_periodic_training(
            "BTCUSDT",
            lambda: df,
            ["rf"],
            interval_seconds=0.1,
            seq_len=12,
            epochs=1,
        )
        await asyncio.sleep(0.25)
        manager.cancel_schedule("BTCUSDT")
        return schedule

    schedule = await _run()
    assert schedule.symbol == "btcusdt"
    assert "btcusdt:rf" in manager.models


@pytest.mark.asyncio
async def test_run_pipeline_returns_predictions_and_drift(tmp_path):
    manager = AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    baseline = _build_dataframe(180)
    df = _build_dataframe(200)
    result = await manager.run_pipeline(
        "BTCUSDT",
        df,
        ["rf", "lstm"],
        seq_len=12,
        folds=2,
        baseline=baseline,
    )
    assert result.predictions is not None
    assert len(result.predictions) == len(df)
    assert result.drift_report is not None
