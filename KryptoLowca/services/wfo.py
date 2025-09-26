"""Walk-forward optimization (WFO) dla strategii AI."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from KryptoLowca.ai_manager import AIManager

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass(slots=True)
class WalkForwardSliceResult:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    model_type: str
    hit_rate: float
    pnl: float
    sharpe: float


@dataclass(slots=True)
class WalkForwardReport:
    symbol: str
    slices: List[WalkForwardSliceResult] = field(default_factory=list)
    aggregate_hit_rate: float = 0.0
    aggregate_pnl: float = 0.0
    best_model: Optional[str] = None


class WalkForwardOptimizer:
    """Realizuje walk-forward optimization na danych historycznych."""

    def __init__(self, ai_manager: AIManager, *, seq_len: int = 64, folds: int = 3) -> None:
        self.ai_manager = ai_manager
        self.seq_len = max(8, int(seq_len))
        self.folds = max(2, int(folds))

    async def optimize(
        self,
        symbol: str,
        df: pd.DataFrame,
        model_types: Iterable[str],
        *,
        window: int = 300,
        step: int = 50,
    ) -> WalkForwardReport:
        if df is None or df.empty:
            raise ValueError("DataFrame wejściowy jest pusty")
        if "close" not in df.columns:
            raise ValueError("DataFrame musi zawierać kolumnę 'close'")
        if len(df) < window + step + self.seq_len:
            raise ValueError("Za mało danych do przeprowadzenia WFO")

        report = WalkForwardReport(symbol=symbol)
        slices: List[WalkForwardSliceResult] = []
        pnl_acc: List[float] = []
        hit_acc: List[float] = []
        best_models: List[str] = []

        for start in range(window, len(df) - step, step):
            train_df = df.iloc[start - window : start]
            test_df = df.iloc[start : start + step]
            selection = await self.ai_manager.select_best_model(
                symbol,
                train_df,
                model_types,
                seq_len=self.seq_len,
                folds=self.folds,
            )
            await self.ai_manager.train_all_models(
                symbol,
                train_df,
                [selection.best_model],
                seq_len=self.seq_len,
            )
            preds = await self.ai_manager.predict_series(
                symbol,
                test_df,
                model_types=[selection.best_model],
            )
            returns = test_df["close"].pct_change().fillna(0.0)
            shifted_preds = preds.shift(1).fillna(0.0)
            pnl = float(np.sum(shifted_preds.to_numpy(dtype=float) * returns.to_numpy(dtype=float)))
            target = np.sign(returns.to_numpy(dtype=float))
            guesses = np.sign(shifted_preds.to_numpy(dtype=float))
            hit_rate = float((guesses == target).mean())
            sharpe = self.ai_manager._sharpe_ratio(shifted_preds.to_numpy(dtype=float) * returns.to_numpy(dtype=float))

            slice_result = WalkForwardSliceResult(
                train_start=pd.to_datetime(train_df.index[0]).to_pydatetime().replace(tzinfo=timezone.utc),
                train_end=pd.to_datetime(train_df.index[-1]).to_pydatetime().replace(tzinfo=timezone.utc),
                test_start=pd.to_datetime(test_df.index[0]).to_pydatetime().replace(tzinfo=timezone.utc),
                test_end=pd.to_datetime(test_df.index[-1]).to_pydatetime().replace(tzinfo=timezone.utc),
                model_type=selection.best_model,
                hit_rate=hit_rate,
                pnl=pnl,
                sharpe=sharpe,
            )
            slices.append(slice_result)
            pnl_acc.append(pnl)
            hit_acc.append(hit_rate)
            best_models.append(selection.best_model)

        if slices:
            report.aggregate_pnl = float(np.sum(pnl_acc))
            report.aggregate_hit_rate = float(np.mean(hit_acc))
            report.best_model = max(
                ((model, best_models.count(model)) for model in set(best_models)),
                key=lambda item: item[1],
            )[0]
        report.slices = slices
        return report


__all__ = [
    "WalkForwardOptimizer",
    "WalkForwardReport",
    "WalkForwardSliceResult",
]
